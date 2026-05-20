// SPDX-License-Identifier: GPL-3.0-or-later
#include "aboba/lowlatency.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <stdexcept>

namespace aboba {

namespace {

constexpr float kPi    = 3.14159265358979323846f;
constexpr float kTwoPi = 2.0f * kPi;
constexpr float kMaxSemitones = 24.0f;

inline float sanitize(float x) noexcept {
    return std::isfinite(x) ? x : 0.0f;
}

}  // namespace

LowLatencyPipeline::LowLatencyPipeline(LowLatencyConfig cfg, Backend* backend)
    : cfg_(cfg), backend_(backend) {
    if (!(cfg_.sample_rate > 0.0))
        throw std::invalid_argument("LowLatencyPipeline: sample_rate must be > 0");
    if (cfg_.window_size < 32 || cfg_.window_size > 2048)
        throw std::invalid_argument("LowLatencyPipeline: window_size out of range [32..2048]");
    if (cfg_.hop_size == 0 || cfg_.hop_size > cfg_.window_size)
        throw std::invalid_argument("LowLatencyPipeline: hop_size out of range");

    dc_.configure(cfg_.sample_rate, 20.0f);
    hp_.configure(cfg_.sample_rate, cfg_.highpass_cutoff_hz);
    gate_.configure(cfg_.sample_rate, -45.0f, -55.0f, 0.005f, 0.150f);
    limiter_.configure(cfg_.sample_rate, 0.95f, 0.050f);

    // SOLA history: enough to read back ~2 windows worth (for both
    // pitch-down extrapolation and pitch-up read-ahead). Plus headroom.
    const std::size_t history_cap = cfg_.window_size * 4;
    in_history_.assign(history_cap, 0.0f);
    in_history_pos_ = 0;

    // Output buffer: at least window_size headroom in case SOLA produces
    // ahead of the read pointer. Plus enough for typical block sizes.
    out_buf_.assign(cfg_.window_size * 4, 0.0f);
    out_read_  = 0;
    out_write_ = cfg_.window_size;  // pre-fill latency
}

LowLatencyPipeline::~LowLatencyPipeline() = default;

void LowLatencyPipeline::set_pitch_semitones(float st) {
    if (!std::isfinite(st)) st = 0.0f;
    if (st >  kMaxSemitones) st =  kMaxSemitones;
    if (st < -kMaxSemitones) st = -kMaxSemitones;
    pitch_semitones_ = st;
    pitch_ratio_ = std::pow(2.0f, st / 12.0f);
}

void LowLatencyPipeline::set_max_block_us(int us) noexcept {
    if (us < 0) us = 0;
    cfg_.max_block_us = us;
}

void LowLatencyPipeline::set_bypass_policy(BypassPolicy p) noexcept {
    bypass_policy_ = p;
    if (p == BypassPolicy::Log) {
        currently_bypassed_.store(false);
    }
}

void LowLatencyPipeline::reset() {
    dc_.reset();
    hp_.reset();
    gate_.reset();
    limiter_.reset();
    std::fill(in_history_.begin(), in_history_.end(), 0.0f);
    std::fill(out_buf_.begin(),    out_buf_.end(),    0.0f);
    in_history_pos_ = 0;
    out_read_  = 0;
    out_write_ = cfg_.window_size;
    sola_phase_ = 0.0f;
    recovery_count_ = 0;
    currently_bypassed_.store(false);
}

std::size_t LowLatencyPipeline::latency_samples() const noexcept {
    // SOLA latency = window_size (the pre-fill we do in constructor)
    return cfg_.window_size;
}

LowLatencyStats LowLatencyPipeline::stats() const noexcept {
    LowLatencyStats s;
    s.total_blocks       = stat_total_blocks_.load();
    s.bypassed_blocks    = stat_bypassed_blocks_.load();
    s.exception_recovers = stat_exception_recov_.load();
    s.last_block_us      = stat_last_us_.load();
    s.p99_block_us       = stat_p99_us_.load();
    s.currently_bypassed = currently_bypassed_.load();
    return s;
}

// SOLA pitch shift, time-domain.
//
// Algorithm:
//   * Maintain a circular history of past input
//   * For each output frame, read from history at a fractional position
//     advancing by `pitch_ratio_` per sample (so playback at >1.0 = higher
//     pitch but faster); we counter-compensate by overlapping frames
//   * Linear-interpolation read for sub-sample positions
//
// This is a simplified SOLA that does not search for best alignment
// (proper SOLA searches in a small window for cross-correlation max). We
// skip that for deterministic per-block cost; quality is slightly worse
// than full SOLA but predictable.
void LowLatencyPipeline::sola_pitch_shift(const float* in, float* out,
                                          std::size_t n) {
    if (std::fabs(pitch_ratio_ - 1.0f) < 1e-5f) {
        // Identity passthrough — exact
        std::memcpy(out, in, n * sizeof(float));
        return;
    }

    const std::size_t H = in_history_.size();

    // Append input to history (circular)
    for (std::size_t i = 0; i < n; ++i) {
        in_history_[in_history_pos_] = in[i];
        in_history_pos_ = (in_history_pos_ + 1) % H;
    }

    // Read from history at fractional positions. The "current" position
    // (== in_history_pos_) is the most recent input. We read at
    // (current - latency_samples + i * pitch_ratio) for output sample i.
    // Adjusted modulo H for circular wrap.
    const std::size_t latency = cfg_.window_size;

    for (std::size_t i = 0; i < n; ++i) {
        // Fractional read position (in samples back from "now")
        const float back = static_cast<float>(latency) - sola_phase_;
        sola_phase_ += pitch_ratio_;

        // Compute integer + frac indices into history
        if (back < 1.0f || back > static_cast<float>(H - 2)) {
            // Out of safe range — emit zero rather than read OOB
            out[i] = 0.0f;
            continue;
        }
        const std::size_t back_int = static_cast<std::size_t>(back);
        const float       back_frac = back - static_cast<float>(back_int);

        // History index = (in_history_pos_ - back_int) mod H
        const std::size_t idx_a = (in_history_pos_ + H - back_int)     % H;
        const std::size_t idx_b = (in_history_pos_ + H - back_int - 1) % H;
        const float a = in_history_[idx_a];
        const float b = in_history_[idx_b];
        out[i] = a * (1.0f - back_frac) + b * back_frac;
    }

    // Wrap phase to avoid float precision drift over very long sessions
    if (sola_phase_ > static_cast<float>(H)) {
        sola_phase_ = std::fmod(sola_phase_, static_cast<float>(latency));
    }
}

void LowLatencyPipeline::process_block_inner(const float* in, float* out,
                                             std::size_t n) {
    // 1. DC + HP + Gate (sample-by-sample, no latency)
    // We use `out` as scratch since we're going to write to it anyway.
    if (cfg_.enable_dc_blocker) dc_.process_block(in, out, n);
    else                         std::memcpy(out, in, n * sizeof(float));

    if (cfg_.enable_highpass) hp_.process_block(out, out, n);
    if (cfg_.enable_gate)     gate_.process_block(out, out, n);

    // 2. Pitch shift via SOLA (in-place — read from `out`, write to `out`).
    //    Because SOLA reads from internal history (not `out` directly), it's
    //    safe to alias input and output here.
    {
        // Need a temporary because sola_pitch_shift reads `in[i]` for the
        // history append BEFORE writing `out[i]`. With in==out that would
        // contaminate the history. Use stack buffer if small, heap else.
        // For typical game block sizes (64-1024) the stack copy is fine.
        std::vector<float> tmp(out, out + n);
        sola_pitch_shift(tmp.data(), out, n);
    }

    // 3. Limiter (last line of defense — sample-by-sample)
    if (cfg_.enable_limiter) limiter_.process_block(out, out, n);

    // 4. NaN/Inf sanitize on output edge
    for (std::size_t i = 0; i < n; ++i) {
        out[i] = sanitize(out[i]);
    }
}

void LowLatencyPipeline::process(const float* input, float* output,
                                 std::size_t n) {
    if (n == 0) return;
    if (!input || !output)
        throw std::invalid_argument("LowLatencyPipeline::process: null buffer");

    stat_total_blocks_.fetch_add(1, std::memory_order_relaxed);

    // If currently bypassed (watchdog-induced), just passthrough.
    if (currently_bypassed_.load(std::memory_order_relaxed)) {
        std::memcpy(output, input, n * sizeof(float));
        stat_bypassed_blocks_.fetch_add(1, std::memory_order_relaxed);
        // Try to recover by tentatively running one more block. If it
        // fits the budget for kRecoveryBlocksNeeded consecutive tries
        // we'll exit bypass.
        const auto t0 = std::chrono::steady_clock::now();
        try {
            std::vector<float> probe_out(n);
            process_block_inner(input, probe_out.data(), n);
            const auto t1 = std::chrono::steady_clock::now();
            const auto us = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
            stat_last_us_.store(static_cast<float>(us));
            if (cfg_.max_block_us > 0 && us <= cfg_.max_block_us) {
                if (++recovery_count_ >= kRecoveryBlocksNeeded) {
                    currently_bypassed_.store(false);
                    recovery_count_ = 0;
                    // Adopt the probe result for THIS block since we
                    // managed to compute it in budget.
                    std::memcpy(output, probe_out.data(), n * sizeof(float));
                }
            } else {
                recovery_count_ = 0;
            }
        } catch (...) {
            stat_exception_recov_.fetch_add(1, std::memory_order_relaxed);
            recovery_count_ = 0;
        }
        return;
    }

    // Normal path: process with timing + exception guard.
    const auto t0 = std::chrono::steady_clock::now();
    try {
        process_block_inner(input, output, n);
    } catch (...) {
        // Pathological: process_block_inner threw. Don't crash; passthrough.
        stat_exception_recov_.fetch_add(1, std::memory_order_relaxed);
        std::memcpy(output, input, n * sizeof(float));
        return;
    }
    const auto t1 = std::chrono::steady_clock::now();
    const auto us = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
    const float us_f = static_cast<float>(us);
    stat_last_us_.store(us_f);

    // Update approximate p99: take the running max but decay slowly so
    // it represents recent peak rather than ever-grown high.
    {
        float prev = stat_p99_us_.load(std::memory_order_relaxed);
        const float decay = 0.995f;
        const float new_p99 = (us_f > prev) ? us_f : (prev * decay);
        stat_p99_us_.store(new_p99, std::memory_order_relaxed);
    }

    // Budget watchdog
    if (cfg_.max_block_us > 0 && us > cfg_.max_block_us) {
        if (bypass_policy_ == BypassPolicy::Bypass) {
            currently_bypassed_.store(true, std::memory_order_relaxed);
            recovery_count_ = 0;
            // Note: we already wrote `output` (the processing succeeded
            // within wallclock terms, it just exceeded budget). Keep it —
            // glitching the audio mid-block is worse than letting one
            // over-budget block through.
        }
        stat_bypassed_blocks_.fetch_add(1, std::memory_order_relaxed);
    }
}

}  // namespace aboba
