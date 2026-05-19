// SPDX-License-Identifier: GPL-3.0-or-later
//
// Spectral noise reducer.
//
// Algorithm: Wiener-style spectral subtraction with automatic noise floor
// learning + minimum-statistics tracking (Martin 2001, simplified).
//
//   1. Warm-up phase (first `learning_seconds`): assume input is noise,
//      collect |X(k)|² average per bin -> initial noise PSD N(k).
//   2. After warm-up: per-bin running minimum of |X(k)|² over a sliding
//      window updates N(k). This implicitly tracks noise without needing
//      an external VAD (the minimum across ~1.5 s of speech includes the
//      gaps between syllables, which is noise).
//   3. Each frame: compute gain G(k) = sqrt(max(1 - α N(k)/|X(k)|², β))
//      where α is oversubtraction, β is the spectral floor.
//   4. Smooth G(k) over time to suppress "musical noise" artifacts.
//   5. Apply Y(k) = G(k) * X(k), iFFT, OLA, output.
//
// Notes:
//   * Phase is preserved (only magnitudes are touched). This avoids the
//     "underwater" sound of phase-modifying NR.
//   * Operates independently of the vocoder. Place it BEFORE the vocoder
//     in the pipeline so the F0 detector sees a cleaner signal.
//   * fft_size defaults to 1024 (~21ms at 48 kHz) — a separate, smaller
//     FFT from the vocoder's 2048 keeps total pipeline latency reasonable.
//
// Paranoia:
//   * All buffers pre-allocated; no allocations on the audio path.
//   * NaN/Inf in -> 0 (sanitized).
//   * Gain clamped to [spectral_floor, 1.0]; can never amplify.
//   * Noise floor floored to 1e-12 to avoid divide-by-zero blowup.
//   * Backend pointer must outlive the reducer.
#pragma once

#include "backend.hpp"

#include <complex>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace aboba {

struct NoiseReductionConfig {
    double      sample_rate = 48000.0;
    std::size_t fft_size    = 1024;
    std::size_t hop_size    = 256;          // 75% overlap

    // Aggressiveness, 1.0 = standard Wiener, >1 = oversubtract (more
    // reduction, more risk of musical noise / speech damage).
    float oversubtraction   = 1.5f;

    // Spectral floor — minimum gain allowed. Lower = more reduction but
    // more musical noise. Typical 0.02..0.10.
    float spectral_floor    = 0.05f;

    // Gain smoothing across frames (0 = no smoothing, 0.95 = heavy).
    // Smoothing trades transient sharpness for less musical noise.
    float gain_smoothing    = 0.6f;

    // Initial "learn from this assumed-quiet section" duration in seconds.
    // The first frames are assumed to contain only noise; their average
    // PSD becomes the initial noise profile.
    float learning_seconds  = 0.5f;

    // Adaptive minimum-statistics window in seconds.
    float min_track_seconds = 1.5f;

    // If true, never let the adaptive noise floor exceed the current
    // signal PSD by too much (prevents speech from being learned as noise
    // when the user starts talking immediately and learning_seconds wasn't
    // really silence).
    bool clamp_noise_to_signal = true;
};

struct NoiseReductionStats {
    std::uint64_t frames_total          = 0;
    std::uint64_t frames_in_learning    = 0;
    std::uint64_t frames_noise_updated  = 0;
    float         last_mean_gain        = 1.0f;  // [0..1], mean across bins
    float         last_min_gain         = 1.0f;  // most-attenuated bin
    float         last_max_gain         = 1.0f;
    float         estimated_snr_db      = 0.0f;
};

class NoiseReducer {
public:
    NoiseReducer(NoiseReductionConfig cfg, Backend* backend);

    void process(const float* input, float* output, std::size_t n_samples);

    // Manually feed a known-silent buffer to (re)initialize the noise
    // profile. Useful for "calibration" workflows where the user records
    // a few seconds of room tone before going live.
    void learn_noise_profile(const float* silence, std::size_t n);

    // Clear all state including learned noise floor. Returns the reducer
    // to the "I know nothing" startup state — it will re-learn on next
    // process() call.
    void reset();

    void set_oversubtraction(float v) noexcept;
    void set_spectral_floor(float v) noexcept;
    void set_gain_smoothing(float v) noexcept;

    std::size_t latency_samples() const noexcept { return cfg_.fft_size; }
    NoiseReductionStats stats() const noexcept;

private:
    void process_one_frame();

    NoiseReductionConfig cfg_;
    Backend*    backend_ = nullptr;

    std::size_t n_bins_   = 0;
    float       ola_norm_ = 1.0f;

    // I/O buffers
    std::vector<float> window_;
    std::vector<float> in_buf_;
    std::size_t        in_fill_ = 0;
    std::vector<float> out_buf_;
    std::size_t        out_write_ = 0;
    std::size_t        out_read_  = 0;

    // STFT scratch
    std::vector<float>                frame_;
    std::vector<std::complex<float>>  spec_;

    // Noise profile and tracking
    std::vector<float> noise_psd_;        // running noise PSD per bin
    std::vector<float> signal_psd_;       // smoothed |X(k)|² for stats
    std::vector<float> min_tracker_;      // sliding min of |X(k)|²
    std::vector<float> gain_smooth_;      // smoothed gain per bin

    std::size_t learning_frames_remaining_ = 0;
    std::size_t total_learning_frames_     = 0;
    std::size_t min_track_window_frames_   = 0;
    std::size_t frames_since_min_reset_    = 0;

    std::uint64_t cnt_total_         = 0;
    std::uint64_t cnt_in_learning_   = 0;
    std::uint64_t cnt_noise_updated_ = 0;
    float         last_mean_gain_    = 1.0f;
    float         last_min_gain_     = 1.0f;
    float         last_max_gain_     = 1.0f;
    float         last_snr_db_       = 0.0f;
};

}  // namespace aboba
