// SPDX-License-Identifier: GPL-3.0-or-later
//
// Pitch corrector (autotune-style).
//
// This is an ANALYSIS module. It:
//   1. Buffers samples in a window of YIN's size.
//   2. Periodically (every analysis_hop samples) detects F0.
//   3. Snaps F0 to a musical scale.
//   4. Returns the *suggested pitch shift in semitones* to apply.
//
// The caller (typically VoicePipeline) feeds the returned value into the
// FormantVocoder's set_pitch_semitones(). This separation keeps the pitch
// shifting math in one place (the vocoder) and the musical logic in another
// (here).
//
// Glide:
//   Snapping instantly creates the "T-Pain" robotic effect. Glide adds a
//   smoothing time constant: the actual returned correction is a one-pole
//   IIR'd version of the snap target. Set glide_ms larger for natural,
//   smaller for robotic.
//
// Strength:
//   0.0 = no correction (always returns 0 shift), 1.0 = fully snap to the
//   scale. Values in between interpolate (so you can get "10% nudge toward
//   the note" for subtle tuning).
//
// Bypass on unvoiced:
//   Consonants/sibilance/breath have no meaningful pitch. By default we
//   return 0 on unvoiced frames, so they pass through dry. Disable if you
//   want the corrector to keep applying the last seen ratio (useful for
//   "frozen" effects).
//
// Paranoia:
//   * All buffers pre-allocated; no allocations after construction.
//   * NaN/Inf safe (uses YIN which is itself safe).
//   * Clamps the returned shift to ±24 semitones (one really should not
//     try to autotune a kazoo into a clarinet).
//   * Detection RMS gate: if input is too quiet, returns 0 and doesn't
//     try to detect from noise.
#pragma once

#include "musical_scale.hpp"
#include "yin.hpp"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace aboba {

struct PitchCorrectorConfig {
    double sample_rate    = 48000.0;

    // Note range YIN should try to identify.
    float f0_min_hz       = 60.0f;
    float f0_max_hz       = 1200.0f;

    // Re-analyse every N samples. Smaller = snappier response but more CPU.
    std::size_t analysis_hop = 512;

    MusicalScale scale       = MusicalScale::Chromatic;
    int          root_semis  = 0;     // 0=C, 9=A
    std::uint16_t custom_mask = 0xFFF; // used when scale == Custom

    float strength  = 1.0f;            // 0..1
    float glide_ms  = 50.0f;           // smoothing time
    bool  bypass_unvoiced = true;

    // Maximum correction we will ever ask for. Beyond this we clamp.
    float max_correction_semitones = 24.0f;
};

struct PitchCorrectorStats {
    std::uint64_t analyses_total      = 0;
    std::uint64_t analyses_voiced     = 0;
    std::uint64_t analyses_unvoiced   = 0;
    float         last_input_f0_hz    = 0.0f;
    float         last_target_f0_hz   = 0.0f;
    float         last_correction_st  = 0.0f;
    float         smoothed_correction_st = 0.0f;
};

class PitchCorrector {
public:
    explicit PitchCorrector(PitchCorrectorConfig cfg);

    // Feed a block of input samples. Updates internal state and returns the
    // CURRENT smoothed correction in semitones (the same value will be
    // returned for any further calls until enough new samples have been
    // analysed).
    float analyze(const float* samples, std::size_t n);

    // For users who already have an F0 from elsewhere (e.g. their own
    // detector). Skips YIN; updates state.
    void set_external_f0(float f0_hz, bool voiced) noexcept;

    // Configure / reconfigure.
    void set_scale(MusicalScale s, int root_semis) noexcept;
    void set_custom_scale(std::uint16_t mask, int root_semis) noexcept;
    void set_strength(float s) noexcept;
    void set_glide_ms(float ms) noexcept;
    void set_bypass_unvoiced(bool b) noexcept { cfg_.bypass_unvoiced = b; }

    // Reset internal state (smoothed correction, last F0, etc).
    void reset() noexcept;

    PitchCorrectorStats stats() const noexcept;

    // Most recent smoothed correction, queryable for UI/meters.
    float current_correction_semitones() const noexcept {
        return smoothed_correction_st_.load(std::memory_order_relaxed);
    }

private:
    void recompute_smoothing_coef() noexcept;
    void run_yin_if_ready();
    float apply_glide_step(float target_st) noexcept;

    PitchCorrectorConfig cfg_;
    std::uint16_t        active_mask_ = 0xFFF;
    int                  active_root_ = 0;

    std::unique_ptr<YinDetector> yin_;
    std::vector<float>           yin_buf_;
    std::size_t                  yin_buf_fill_ = 0;
    std::size_t                  samples_since_analysis_ = 0;

    // One-pole smoothing toward target
    float smoothing_coef_ = 0.99f;       // per-sample
    float current_st_     = 0.0f;        // current applied correction
    float target_st_      = 0.0f;        // most recent snap target

    // Diagnostics (atomic for cross-thread read)
    std::atomic<std::uint64_t> cnt_total_    {0};
    std::atomic<std::uint64_t> cnt_voiced_   {0};
    std::atomic<std::uint64_t> cnt_unvoiced_ {0};
    std::atomic<float>         last_in_hz_   {0.0f};
    std::atomic<float>         last_tgt_hz_  {0.0f};
    std::atomic<float>         last_corr_    {0.0f};
    std::atomic<float>         smoothed_correction_st_ {0.0f};
};

}  // namespace aboba
