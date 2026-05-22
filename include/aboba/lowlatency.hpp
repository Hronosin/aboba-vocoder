// SPDX-License-Identifier: GPL-3.0-or-later
//
// LowLatencyPipeline — sub-2ms per-block voice processing.
//
// ========================================================================
//                              QUALITY WARNING
// ========================================================================
//
// THE LOW-LATENCY PIPELINE TRADES AUDIO QUALITY FOR LATENCY.
//
// The normal pipeline uses a Phase-Vocoder formant-preserving pitch shifter
// with an FFT size of 2048 samples (~42ms latency at 48kHz). This produces
// professional-grade voice changing.
//
// The low-latency pipeline CANNOT use that engine — it would not fit in a
// 2ms budget. Instead, it uses:
//
//   * Time-domain SOLA (Synchronous Overlap-Add) for pitch shifting.
//     SOLA is sample-by-sample; very low latency; QUALITY-WISE:
//       - Works well for ±3 semitones
//       - Becomes audibly artefacty beyond ±5 semitones
//       - Does NOT preserve formants — voice will sound chipmunky/giant
//         with large pitch shifts (this is unavoidable in time-domain
//         methods)
//   * Single-sample IIR DC blocker, high-pass, gate, limiter (no latency)
//   * NO noise reduction (FFT-based, doesn't fit the budget)
//   * NO autotune (needs >40ms window for YIN F0 estimation)
//   * NO reverb (long-tail FIR convolution doesn't fit the budget)
//
// ========================================================================
//                         WHEN TO USE LOW-LATENCY
// ========================================================================
//
// USE the low-latency pipeline for:
//   * Game engines where 2ms is the audio thread budget
//   * VR/AR applications where end-to-end mouth-to-speaker latency must be
//     under 20ms
//   * Walkie-talkie / radio simulation effects in games (where SOLA's
//     telephone-y artifacts are actually atmospheric)
//
// USE the normal pipeline (aboba_pipeline_create) for:
//   * Stream / OBS / Discord overlays where 50ms latency is fine
//   * Offline batch processing
//   * Any case where you need formant preservation, autotune, reverb,
//     spectral noise reduction
//
// ========================================================================
//                          BUDGET ENFORCEMENT
// ========================================================================
//
// The low-latency pipeline AUTOMATICALLY enables:
//   * max_block_us = 2000 (configurable down to 1000)
//   * budget_policy = BYPASS (on overrun, immediately passthrough for the
//     next blocks until we recover headroom)
//
// This is the policy "stability > speed". A glitch is worse than a brief
// passthrough; passthrough is just unprocessed voice for one block.
//
// ========================================================================
//                              ALGORITHMS
// ========================================================================
//
// Time-domain pitch shift (SOLA):
//   The classic SOLA algorithm: extract overlapping frames, optionally
//   resample to change pitch, find best alignment via cross-correlation,
//   overlap-add into output. Window size 256 samples (~5.3ms) is enough
//   for one full pitch period at 200 Hz; we use 192 samples to keep
//   per-block work small.
//
// Per-block deterministic cost:
//   Every operation is O(block_size) — no FFTs, no allocations, no
//   variable-time code paths. The same input always takes the same time.
#pragma once

#include "backend.hpp"
#include "dsp_blocks.hpp"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace aboba {

struct LowLatencyConfig {
    double sample_rate = 48000.0;

    // SOLA window size in samples. Smaller = lower latency but more
    // pitchy artifacts. Larger = smoother but eats budget.
    // Default: 192 samples (~4ms @ 48kHz)
    std::size_t window_size = 192;

    // Hop size — how much SOLA advances per output. Smaller = smoother
    // but more CPU. Default 64 (~3x overlap).
    std::size_t hop_size = 64;

    // Hard budget in microseconds. Watchdog will trigger bypass past this.
    int max_block_us = 2000;

    // Effects toggles
    bool enable_dc_blocker = true;
    bool enable_highpass   = true;
    float highpass_cutoff_hz = 80.0f;
    bool enable_gate       = true;
    bool enable_limiter    = true;
};

struct LowLatencyStats {
    std::uint64_t total_blocks       = 0;
    std::uint64_t bypassed_blocks    = 0;
    std::uint64_t exception_recovers = 0;
    float         last_block_us      = 0.0f;
    float         p99_block_us       = 0.0f;  // approximate
    bool          currently_bypassed = false;
};

class LowLatencyPipeline {
public:
    LowLatencyPipeline(LowLatencyConfig cfg, Backend* backend);
    ~LowLatencyPipeline();

    void process(const float* input, float* output, std::size_t n);
    void reset();

    void set_pitch_semitones(float st);
    float get_pitch_semitones() const noexcept { return pitch_semitones_; }

    void set_max_block_us(int us) noexcept;
    int  get_max_block_us() const noexcept { return cfg_.max_block_us; }

    enum class BypassPolicy : std::uint8_t {
        // Log when budget exceeded but keep processing.
        Log    = 0,
        // Switch to passthrough on overrun. Re-enable when N consecutive
        // blocks fit the budget.
        Bypass = 1,
    };
    void set_bypass_policy(BypassPolicy p) noexcept;

    LowLatencyStats stats() const noexcept;

    std::size_t latency_samples() const noexcept;

private:
    void process_block_inner(const float* in, float* out, std::size_t n);
    void sola_pitch_shift(const float* in, float* out, std::size_t n);

    LowLatencyConfig cfg_;
    Backend*         backend_;   // unused but accepted for API consistency

    // Pitch
    float pitch_semitones_ = 0.0f;
    float pitch_ratio_     = 1.0f;

    // DSP blocks (single-sample, zero-latency)
    DcBlocker       dc_;
    OnePoleHighPass hp_;
    NoiseGate       gate_;
    SoftLimiter     limiter_;

    // SOLA state
    std::vector<float> in_history_;   // sliding window of past input
    std::size_t        in_history_pos_ = 0;
    std::vector<float> out_buf_;
    std::size_t        out_read_  = 0;
    std::size_t        out_write_ = 0;
    float              sola_phase_ = 0.0f;  // fractional read pointer

    // Pre-allocated scratch buffers used during process() — these are
    // sized at construction and may be grown by process() if the caller
    // passes a block bigger than expected. ALL allocations happen on
    // construction or first oversized block; we never reallocate mid-block.
    std::vector<float> sola_input_scratch_;
    std::vector<float> probe_output_scratch_;

    // Watchdog
    BypassPolicy bypass_policy_ = BypassPolicy::Bypass;
    int          recovery_count_ = 0;
    static constexpr int kRecoveryBlocksNeeded = 8;
    std::atomic<bool> currently_bypassed_ {false};

    // Stats
    std::atomic<std::uint64_t> stat_total_blocks_     {0};
    std::atomic<std::uint64_t> stat_bypassed_blocks_  {0};
    std::atomic<std::uint64_t> stat_exception_recov_  {0};
    std::atomic<float>         stat_last_us_          {0.0f};
    // Reservoir-style approximate p99 — simple running max with slow decay
    std::atomic<float>         stat_p99_us_           {0.0f};
};

}  // namespace aboba
