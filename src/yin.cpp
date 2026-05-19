// SPDX-License-Identifier: GPL-3.0-or-later
#include "aboba/yin.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace aboba {

namespace {
inline float safe_sample(float x) noexcept {
    return std::isfinite(x) ? x : 0.0f;
}
}  // namespace

YinDetector::YinDetector(YinConfig cfg) : cfg_(cfg) {
    if (!(cfg_.sample_rate > 0.0)) {
        throw std::invalid_argument("YinDetector: sample_rate must be > 0");
    }
    if (!(cfg_.f0_min_hz > 0.0f) || !(cfg_.f0_max_hz > cfg_.f0_min_hz)) {
        throw std::invalid_argument("YinDetector: invalid f0 range");
    }
    if (cfg_.threshold < 0.01f) cfg_.threshold = 0.01f;
    if (cfg_.threshold > 1.0f)  cfg_.threshold = 1.0f;

    // Tau range in samples
    tau_min_ = static_cast<std::size_t>(std::floor(
        cfg_.sample_rate / static_cast<double>(cfg_.f0_max_hz)));
    tau_max_ = static_cast<std::size_t>(std::ceil(
        cfg_.sample_rate / static_cast<double>(cfg_.f0_min_hz)));

    if (tau_min_ < 2) tau_min_ = 2;
    if (tau_max_ <= tau_min_ + 2) tau_max_ = tau_min_ + 3;

    // Window: 2x the longest expected period.
    // d(tau) computed for n in [0, window) and (x[n+tau] in [tau, window+tau)),
    // so total samples needed = window + tau_max.
    if (cfg_.window_size > 0) {
        window_size_ = cfg_.window_size;
    } else {
        window_size_ = 2 * tau_max_;
    }
    // Safety cap to avoid absurd allocations
    if (window_size_ > 16384) window_size_ = 16384;

    d_.assign(tau_max_ + 1, 0.0f);
    cmn_.assign(tau_max_ + 1, 1.0f);
}

void YinDetector::compute_difference(const float* x) {
    // d(tau) = sum_{n=0}^{window-1} (x[n] - x[n+tau])^2
    //
    // We need x[n + tau] for n in [0, window) and tau up to tau_max_, so
    // the caller's buffer must have window_size_ + tau_max_ samples.
    // We can't enforce that at this layer; the caller (detect) is checked.
    const std::size_t W = window_size_;
    for (std::size_t tau = 1; tau <= tau_max_; ++tau) {
        float sum = 0.0f;
        // The straightforward double-loop. Vectorizes well under -O2.
        for (std::size_t n = 0; n < W; ++n) {
            const float dx = x[n] - x[n + tau];
            sum += dx * dx;
        }
        d_[tau] = sum;
    }
    d_[0] = 0.0f;
}

void YinDetector::cumulative_mean_normalize() {
    // d'(tau) = d(tau) * tau / sum_{i=1}^{tau} d(i)
    // First value defined as 1.
    cmn_[0] = 1.0f;
    float running_sum = 0.0f;
    for (std::size_t tau = 1; tau <= tau_max_; ++tau) {
        running_sum += d_[tau];
        if (running_sum > 1e-20f) {
            cmn_[tau] = d_[tau] * static_cast<float>(tau) / running_sum;
        } else {
            cmn_[tau] = 1.0f;  // signal effectively silent in this region
        }
    }
}

int YinDetector::absolute_threshold() {
    // Walk from tau_min_ to tau_max_. The first dip below threshold is the
    // pitch — but we keep going through the LOCAL minimum to refine the
    // estimate (classic YIN trick to avoid latching onto the very first
    // sub-threshold sample).
    const float th = cfg_.threshold;
    for (std::size_t tau = tau_min_; tau <= tau_max_; ++tau) {
        if (cmn_[tau] < th) {
            // Step forward while cmn_ keeps decreasing
            while (tau + 1 <= tau_max_ && cmn_[tau + 1] < cmn_[tau]) ++tau;
            return static_cast<int>(tau);
        }
    }
    // Fallback: if nothing crossed threshold, return tau with smallest cmn_.
    // Caller will check aperiodicity and decide what to do.
    std::size_t best = tau_min_;
    float best_v = cmn_[tau_min_];
    for (std::size_t tau = tau_min_ + 1; tau <= tau_max_; ++tau) {
        if (cmn_[tau] < best_v) {
            best_v = cmn_[tau];
            best   = tau;
        }
    }
    return -static_cast<int>(best);  // sign marks "no threshold cross"
}

float YinDetector::parabolic_interpolation(int tau_best) const {
    // Refine tau_best by fitting a parabola through (tau-1, tau, tau+1).
    // Returns the refined tau (fractional sample).
    const int t = std::abs(tau_best);
    if (t <= 0 || t >= static_cast<int>(tau_max_)) {
        return static_cast<float>(t);
    }
    const float y0 = cmn_[static_cast<std::size_t>(t - 1)];
    const float y1 = cmn_[static_cast<std::size_t>(t)];
    const float y2 = cmn_[static_cast<std::size_t>(t + 1)];
    const float denom = y0 - 2.0f * y1 + y2;
    if (std::fabs(denom) < 1e-12f) return static_cast<float>(t);
    const float delta = 0.5f * (y0 - y2) / denom;
    // Clamp to one sample to avoid wild extrapolation when the parabola is bad
    float clamped = delta;
    if (clamped < -1.0f) clamped = -1.0f;
    if (clamped >  1.0f) clamped =  1.0f;
    return static_cast<float>(t) + clamped;
}

YinResult YinDetector::detect(const float* samples, std::size_t n) {
    YinResult r{0.0f, 1.0f, false};

    // Need at least window_size_ + tau_max_ samples for d(tau) up to tau_max_.
    const std::size_t needed = window_size_ + tau_max_;
    if (!samples || n < needed) return r;

    // Quick RMS check: if signal is essentially silent, skip the work.
    // This is both an optimization and a paranoia guard — d/cmn on tiny
    // values can produce noisy minima.
    {
        double ss = 0.0;
        const std::size_t N = std::min(needed, std::size_t{4096});
        for (std::size_t i = 0; i < N; ++i) {
            const float s = safe_sample(samples[i]);
            ss += static_cast<double>(s) * s;
        }
        const float rms = static_cast<float>(std::sqrt(ss / static_cast<double>(N)));
        if (rms < 1e-5f) {
            r.aperiodicity = 1.0f;
            return r;
        }
    }

    compute_difference(samples);
    cumulative_mean_normalize();

    const int tau_signed = absolute_threshold();
    const int tau_best   = std::abs(tau_signed);

    if (tau_best < static_cast<int>(tau_min_) ||
        tau_best > static_cast<int>(tau_max_)) {
        r.aperiodicity = 1.0f;
        return r;
    }

    const float refined_tau = parabolic_interpolation(tau_signed);
    if (refined_tau <= 0.0f || !std::isfinite(refined_tau)) {
        r.aperiodicity = 1.0f;
        return r;
    }

    r.aperiodicity = cmn_[static_cast<std::size_t>(tau_best)];
    // Sub-threshold cross => is_voiced, else unvoiced but we still return the
    // best F0 estimate so the caller can decide.
    r.is_voiced = (tau_signed > 0) && (r.aperiodicity < cfg_.threshold);
    r.f0_hz = static_cast<float>(cfg_.sample_rate / static_cast<double>(refined_tau));

    // Sanity-clamp F0 to configured range
    if (r.f0_hz < cfg_.f0_min_hz || r.f0_hz > cfg_.f0_max_hz) {
        r.is_voiced = false;
        r.f0_hz = 0.0f;
        if (r.aperiodicity < cfg_.threshold) r.aperiodicity = 0.5f;
    }

    return r;
}

}  // namespace aboba
