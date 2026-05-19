// SPDX-License-Identifier: GPL-3.0-or-later
#include "aboba/pitch_corrector.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>

namespace aboba {

namespace {
inline float clamp(float v, float lo, float hi) noexcept {
    return v < lo ? lo : (v > hi ? hi : v);
}
}

PitchCorrector::PitchCorrector(PitchCorrectorConfig cfg) : cfg_(cfg) {
    if (!(cfg_.sample_rate > 0.0))
        throw std::invalid_argument("PitchCorrector: sample_rate must be > 0");
    if (!(cfg_.f0_max_hz > cfg_.f0_min_hz) || !(cfg_.f0_min_hz > 0.0f))
        throw std::invalid_argument("PitchCorrector: invalid f0 range");
    if (cfg_.analysis_hop == 0)
        throw std::invalid_argument("PitchCorrector: analysis_hop must be > 0");

    cfg_.strength = clamp(cfg_.strength, 0.0f, 1.0f);
    cfg_.glide_ms = clamp(cfg_.glide_ms, 0.0f, 2000.0f);
    cfg_.max_correction_semitones =
        clamp(cfg_.max_correction_semitones, 0.0f, 60.0f);

    YinConfig yc;
    yc.sample_rate = cfg_.sample_rate;
    yc.f0_min_hz   = cfg_.f0_min_hz;
    yc.f0_max_hz   = cfg_.f0_max_hz;
    yin_ = std::make_unique<YinDetector>(yc);
    yin_buf_.assign(yin_->window_size() + yin_->tau_max(), 0.0f);
    yin_buf_fill_ = 0;

    // Initialize active scale mask / root.
    if (cfg_.scale == MusicalScale::Custom) {
        active_mask_ = cfg_.custom_mask;
    } else {
        active_mask_ = scale_mask(cfg_.scale);
    }
    int root = cfg_.root_semis % 12;
    if (root < 0) root += 12;
    active_root_ = root;

    recompute_smoothing_coef();
}

void PitchCorrector::recompute_smoothing_coef() noexcept {
    // One-pole IIR: out = coef * out_prev + (1-coef) * target
    // Approach 63% of the target in `glide_ms`. tau in samples = ms * sr / 1000.
    if (cfg_.glide_ms <= 0.0f) {
        smoothing_coef_ = 0.0f;  // instant
        return;
    }
    const float tau = cfg_.glide_ms * 0.001f * static_cast<float>(cfg_.sample_rate);
    smoothing_coef_ = std::exp(-1.0f / std::max(tau, 1.0f));
}

void PitchCorrector::set_scale(MusicalScale s, int root_semis) noexcept {
    cfg_.scale = s;
    if (s != MusicalScale::Custom) active_mask_ = scale_mask(s);
    int root = root_semis % 12;
    if (root < 0) root += 12;
    cfg_.root_semis = root;
    active_root_    = root;
}

void PitchCorrector::set_custom_scale(std::uint16_t mask, int root_semis) noexcept {
    cfg_.scale       = MusicalScale::Custom;
    cfg_.custom_mask = mask;
    active_mask_     = mask & 0x0FFFu;
    int root = root_semis % 12;
    if (root < 0) root += 12;
    cfg_.root_semis = root;
    active_root_    = root;
}

void PitchCorrector::set_strength(float s) noexcept {
    cfg_.strength = clamp(s, 0.0f, 1.0f);
}

void PitchCorrector::set_glide_ms(float ms) noexcept {
    cfg_.glide_ms = clamp(ms, 0.0f, 2000.0f);
    recompute_smoothing_coef();
}

void PitchCorrector::reset() noexcept {
    std::fill(yin_buf_.begin(), yin_buf_.end(), 0.0f);
    yin_buf_fill_ = 0;
    samples_since_analysis_ = 0;
    current_st_ = 0.0f;
    target_st_  = 0.0f;
    cnt_total_.store(0); cnt_voiced_.store(0); cnt_unvoiced_.store(0);
    last_in_hz_.store(0.0f); last_tgt_hz_.store(0.0f); last_corr_.store(0.0f);
    smoothed_correction_st_.store(0.0f);
}

PitchCorrectorStats PitchCorrector::stats() const noexcept {
    PitchCorrectorStats s;
    s.analyses_total    = cnt_total_.load();
    s.analyses_voiced   = cnt_voiced_.load();
    s.analyses_unvoiced = cnt_unvoiced_.load();
    s.last_input_f0_hz  = last_in_hz_.load();
    s.last_target_f0_hz = last_tgt_hz_.load();
    s.last_correction_st = last_corr_.load();
    s.smoothed_correction_st = smoothed_correction_st_.load();
    return s;
}

void PitchCorrector::run_yin_if_ready() {
    const std::size_t needed = yin_->window_size() + yin_->tau_max();
    if (yin_buf_fill_ < needed) return;

    cnt_total_.fetch_add(1, std::memory_order_relaxed);
    auto r = yin_->detect(yin_buf_.data(), needed);

    if (r.is_voiced && r.f0_hz > 0.0f) {
        cnt_voiced_.fetch_add(1, std::memory_order_relaxed);

        // Snap to scale
        const float midi_in = hz_to_midi(r.f0_hz);
        const float midi_target = snap_to_scale(midi_in, active_mask_, active_root_);
        const float f0_target = midi_to_hz(midi_target);

        // Correction in semitones (positive = shift UP)
        float correction = midi_target - midi_in;
        // Apply strength: 0=no correction, 1=full snap
        correction *= cfg_.strength;
        // Clamp
        correction = clamp(correction,
                           -cfg_.max_correction_semitones,
                           +cfg_.max_correction_semitones);

        target_st_ = correction;
        last_in_hz_.store(r.f0_hz);
        last_tgt_hz_.store(f0_target);
        last_corr_.store(correction);
    } else {
        cnt_unvoiced_.fetch_add(1, std::memory_order_relaxed);
        if (cfg_.bypass_unvoiced) {
            target_st_ = 0.0f;
        }
        // else: keep target_st_ at its previous value
    }

    // Shift buffer: slide by analysis_hop samples so we accumulate fresh
    // samples for the next analysis.
    const std::size_t hop = std::min(cfg_.analysis_hop, yin_buf_fill_);
    if (hop > 0 && yin_buf_fill_ > hop) {
        std::memmove(yin_buf_.data(),
                     yin_buf_.data() + hop,
                     (yin_buf_fill_ - hop) * sizeof(float));
        yin_buf_fill_ -= hop;
    } else if (hop >= yin_buf_fill_) {
        yin_buf_fill_ = 0;
    }
}

float PitchCorrector::apply_glide_step(float target_st) noexcept {
    if (smoothing_coef_ <= 0.0f) {
        current_st_ = target_st;
    } else {
        current_st_ = smoothing_coef_ * current_st_
                    + (1.0f - smoothing_coef_) * target_st;
    }
    return current_st_;
}

float PitchCorrector::analyze(const float* samples, std::size_t n) {
    if (!samples) return current_st_;

    // 1. Accumulate into yin_buf_
    const std::size_t cap = yin_buf_.size();
    std::size_t i = 0;
    while (i < n) {
        const std::size_t space = cap - yin_buf_fill_;
        const std::size_t take  = std::min(space, n - i);
        std::memcpy(yin_buf_.data() + yin_buf_fill_, samples + i,
                    take * sizeof(float));
        yin_buf_fill_ += take;
        i += take;
        samples_since_analysis_ += take;

        // 2. Trigger detection at the hop interval
        if (samples_since_analysis_ >= cfg_.analysis_hop) {
            samples_since_analysis_ = 0;
            run_yin_if_ready();
        }

        // 3. If buffer full but we never ran (rare — buffer smaller than hop),
        //    force a run.
        if (yin_buf_fill_ == cap && samples_since_analysis_ > 0) {
            run_yin_if_ready();
        }
    }

    // 4. Smooth current_st_ toward target_st_ once per call (NOT per sample;
    //    we'd lose the time constant in fragmented small-block calls). The
    //    glide tau IS in samples but we step by sample count instead so it
    //    survives jitter in n.
    if (smoothing_coef_ <= 0.0f) {
        current_st_ = target_st_;
    } else {
        // n-step IIR: out = coef^n * out_prev + (1 - coef^n) * target
        const float k = std::pow(smoothing_coef_, static_cast<float>(n));
        current_st_ = k * current_st_ + (1.0f - k) * target_st_;
    }
    smoothed_correction_st_.store(current_st_);
    return current_st_;
}

void PitchCorrector::set_external_f0(float f0_hz, bool voiced) noexcept {
    cnt_total_.fetch_add(1, std::memory_order_relaxed);
    if (voiced && f0_hz > 0.0f) {
        cnt_voiced_.fetch_add(1, std::memory_order_relaxed);
        const float midi_in = hz_to_midi(f0_hz);
        const float midi_tg = snap_to_scale(midi_in, active_mask_, active_root_);
        float corr = (midi_tg - midi_in) * cfg_.strength;
        corr = clamp(corr,
                     -cfg_.max_correction_semitones,
                     +cfg_.max_correction_semitones);
        target_st_ = corr;
        last_in_hz_.store(f0_hz);
        last_tgt_hz_.store(midi_to_hz(midi_tg));
        last_corr_.store(corr);
    } else {
        cnt_unvoiced_.fetch_add(1, std::memory_order_relaxed);
        if (cfg_.bypass_unvoiced) target_st_ = 0.0f;
    }
}

}  // namespace aboba
