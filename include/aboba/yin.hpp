// SPDX-License-Identifier: GPL-3.0-or-later
//
// YIN pitch detector (de Cheveigné & Kawahara, 2002).
//
// Strategy:
//   1. Compute the "difference function": d(tau) = sum_n (x[n] - x[n+tau])^2
//      over a window. Pitch period -> d(tau) ~ 0.
//   2. Cumulative mean normalization: d'(tau) = d(tau) / (mean of d over [1..tau]).
//      This makes the absolute threshold (~0.10-0.15) robust across signal
//      levels.
//   3. Find first dip below threshold; refine with parabolic interpolation.
//   4. Aperiodicity = d'(tau_best) is a great voiced/unvoiced indicator:
//      below 0.1 -> very voiced, above 0.4 -> definitely not pitched.
//
// Range: ~60 Hz to ~1500 Hz at 48 kHz with the defaults below. The lower
// bound is determined by the window size; we use 2x the longest expected
// period.
//
// CPU: O(window * (tau_max - tau_min)). For 48 kHz and 60 Hz min, that's
// ~25-30 µs per frame on commodity hardware. We compute at hop intervals,
// not per-sample.
//
// Paranoia:
//   - All buffers pre-allocated. No allocations after construction.
//   - Inputs sanitized for NaN/Inf.
//   - Aperiodicity returned alongside F0 — caller decides voiced/unvoiced
//     using its own threshold.
//   - Returns 0.0 Hz for unvoiced/silent inputs; never garbage.
#pragma once

#include <cstddef>
#include <vector>

namespace aboba {

struct YinConfig {
    double sample_rate    = 48000.0;
    float  f0_min_hz      = 60.0f;     // lowest detectable pitch
    float  f0_max_hz      = 1500.0f;   // highest detectable pitch
    float  threshold      = 0.15f;     // aperiodicity cut for "found pitch"
    // Window size = round(2 * sample_rate / f0_min). Auto-computed in ctor
    // unless overridden.
    std::size_t window_size = 0;
};

struct YinResult {
    float f0_hz;          // 0.0 if not pitched
    float aperiodicity;   // d'(tau_best), 0..1+. Low = confident pitch.
    bool  is_voiced;      // shorthand for aperiodicity < threshold
};

class YinDetector {
public:
    explicit YinDetector(YinConfig cfg);

    // Estimate F0 from a window of `n` samples. If n < window_size, returns
    // unvoiced (we need a full window). If n > window_size, only the first
    // window_size samples are used.
    YinResult detect(const float* samples, std::size_t n);

    std::size_t window_size() const noexcept { return window_size_; }
    const YinConfig& config()  const noexcept { return cfg_; }

    // For diagnostics: returns the smallest tau (in samples) considered.
    std::size_t tau_min() const noexcept { return tau_min_; }
    std::size_t tau_max() const noexcept { return tau_max_; }

private:
    void compute_difference(const float* x);
    void cumulative_mean_normalize();
    int  absolute_threshold();          // returns tau_best, or -1
    float parabolic_interpolation(int tau_best) const;

    YinConfig   cfg_;
    std::size_t window_size_ = 0;
    std::size_t tau_min_     = 0;
    std::size_t tau_max_     = 0;

    // Pre-allocated working buffers. Size = tau_max_ + 1.
    std::vector<float> d_;    // difference function
    std::vector<float> cmn_;  // cumulative mean normalized
};

}  // namespace aboba
