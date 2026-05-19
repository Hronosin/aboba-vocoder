// SPDX-License-Identifier: GPL-3.0-or-later
//
// Small DSP building blocks used by the audio pipeline.
//
// All blocks share these contracts:
//   * Sample-by-sample API (process_one) AND block API (process_block).
//   * No allocations after construction. Ever.
//   * NaN/Inf-safe: pathological inputs become 0.0 without poisoning state.
//   * Coefficients are recomputed only when setters are called.
//   * Thread-safe to call from the audio thread; not thread-safe to
//     configure from other threads concurrently.
#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>

namespace aboba {

namespace detail {
// Used in many places. Hot path; inline.
inline float sanitize_sample(float x) noexcept {
    return std::isfinite(x) ? x : 0.0f;
}
}  // namespace detail


// ----------------------------------------------------------------------
// DC blocker: y[n] = x[n] - x[n-1] + R * y[n-1]
// One-pole high-pass at ~10 Hz by default. Removes mic offset / drift.
// ----------------------------------------------------------------------
class DcBlocker {
public:
    void reset() noexcept { x1_ = 0.0f; y1_ = 0.0f; }

    // R close to 1 = lower cutoff. R=0.995 -> ~50 Hz @ 48 kHz.
    // We tune by sample rate so cutoff stays around `hz` regardless.
    void configure(double sample_rate, float cutoff_hz = 20.0f) noexcept {
        if (!(sample_rate > 0.0) || !std::isfinite(cutoff_hz)) {
            R_ = 0.995f;
            return;
        }
        const double w = 2.0 * 3.14159265358979 * cutoff_hz / sample_rate;
        // Approx: R = 1 - w. Clamp.
        double r = 1.0 - w;
        if (r < 0.0)  r = 0.0;
        if (r > 0.9999) r = 0.9999;
        R_ = static_cast<float>(r);
    }

    inline float process_one(float x) noexcept {
        x = detail::sanitize_sample(x);
        const float y = x - x1_ + R_ * y1_;
        x1_ = x;
        y1_ = y;
        return y;
    }

    void process_block(const float* in, float* out, std::size_t n) noexcept {
        for (std::size_t i = 0; i < n; ++i) out[i] = process_one(in[i]);
    }

private:
    float R_  = 0.995f;
    float x1_ = 0.0f;
    float y1_ = 0.0f;
};


// ----------------------------------------------------------------------
// One-pole IIR high-pass for rumble removal (typically ~80 Hz for voice).
// Topology: difference equation derived from bilinear transform of an
// analog one-pole. Stable for cutoff < Nyquist/2.
// ----------------------------------------------------------------------
class OnePoleHighPass {
public:
    void reset() noexcept { z_ = 0.0f; }

    void configure(double sample_rate, float cutoff_hz) noexcept {
        if (!(sample_rate > 0.0) || !std::isfinite(cutoff_hz) || cutoff_hz <= 0.0f) {
            a_ = 1.0f;  // passthrough
            return;
        }
        const double rc = 1.0 / (2.0 * 3.14159265358979 * cutoff_hz);
        const double dt = 1.0 / sample_rate;
        double a = rc / (rc + dt);
        if (a < 0.0) a = 0.0;
        if (a > 0.9999) a = 0.9999;
        a_ = static_cast<float>(a);
    }

    inline float process_one(float x) noexcept {
        x = detail::sanitize_sample(x);
        // y[n] = a * (y[n-1] + x[n] - x_prev). We keep state as "running HP".
        const float y = a_ * (z_ + x - x1_);
        x1_ = x;
        z_  = y;
        return y;
    }

    void process_block(const float* in, float* out, std::size_t n) noexcept {
        for (std::size_t i = 0; i < n; ++i) out[i] = process_one(in[i]);
    }

private:
    float a_  = 1.0f;
    float z_  = 0.0f;
    float x1_ = 0.0f;
};


// ----------------------------------------------------------------------
// Noise gate with attack/release envelope follower.
// Below threshold -> attenuate; above -> let through. Hysteresis via
// separate open/close thresholds prevents fluttering on borderline signals.
// ----------------------------------------------------------------------
class NoiseGate {
public:
    void reset() noexcept {
        env_  = 0.0f;
        gain_ = 1.0f;
        open_ = false;
    }

    // open_threshold_db:  signal must exceed this to "open"
    // close_threshold_db: must drop below this to "close" (hysteresis)
    // attack/release in seconds
    void configure(double sample_rate,
                   float  open_threshold_db  = -45.0f,
                   float  close_threshold_db = -55.0f,
                   float  attack_sec         = 0.005f,
                   float  release_sec        = 0.150f) noexcept {
        if (!(sample_rate > 0.0)) sample_rate = 48000.0;

        // dB -> linear amplitude
        open_thresh_  = std::pow(10.0f, open_threshold_db  / 20.0f);
        close_thresh_ = std::pow(10.0f, close_threshold_db / 20.0f);

        // Single-pole envelope time constants
        att_coef_ = (attack_sec  > 0.0f)
            ? std::exp(-1.0f / (attack_sec  * static_cast<float>(sample_rate)))
            : 0.0f;
        rel_coef_ = (release_sec > 0.0f)
            ? std::exp(-1.0f / (release_sec * static_cast<float>(sample_rate)))
            : 0.0f;

        // For the gain ramp itself (smoother than the envelope detector).
        gain_att_ = std::exp(-1.0f / (0.001f * static_cast<float>(sample_rate)));
        gain_rel_ = std::exp(-1.0f / (0.020f * static_cast<float>(sample_rate)));
    }

    inline float process_one(float x) noexcept {
        x = detail::sanitize_sample(x);
        const float a = std::fabs(x);

        // Envelope follower (peak)
        if (a > env_) env_ = att_coef_ * env_ + (1.0f - att_coef_) * a;
        else          env_ = rel_coef_ * env_ + (1.0f - rel_coef_) * a;

        // Hysteresis decision
        if (env_ > open_thresh_)  open_ = true;
        if (env_ < close_thresh_) open_ = false;

        // Gain ramp to target
        const float target = open_ ? 1.0f : 0.0f;
        if (target > gain_) gain_ = gain_att_ * gain_ + (1.0f - gain_att_) * target;
        else                gain_ = gain_rel_ * gain_ + (1.0f - gain_rel_) * target;

        return x * gain_;
    }

    void process_block(const float* in, float* out, std::size_t n) noexcept {
        for (std::size_t i = 0; i < n; ++i) out[i] = process_one(in[i]);
    }

    bool is_open() const noexcept { return open_; }

private:
    float open_thresh_  = 0.005f;
    float close_thresh_ = 0.002f;
    float att_coef_     = 0.0f;
    float rel_coef_     = 0.0f;
    float gain_att_     = 0.0f;
    float gain_rel_     = 0.0f;

    float env_  = 0.0f;
    float gain_ = 1.0f;
    bool  open_ = false;
};


// ----------------------------------------------------------------------
// Soft clipper / brick-wall limiter combo.
//   * Soft knee using tanh-like function for musical limiting up to threshold
//   * Hard clip at +-1.0 as last resort (after gain ramping fails)
//
// Lookahead is NOT implemented (would add latency). For voice this is fine
// because the gain ramp is fast enough.
// ----------------------------------------------------------------------
class SoftLimiter {
public:
    void reset() noexcept { gain_ = 1.0f; }

    void configure(double sample_rate, float threshold = 0.95f,
                   float release_sec = 0.050f) noexcept {
        if (!std::isfinite(threshold)) threshold = 0.95f;
        if (threshold < 0.1f)  threshold = 0.1f;
        if (threshold > 0.99f) threshold = 0.99f;
        threshold_ = threshold;
        if (!(sample_rate > 0.0)) sample_rate = 48000.0;
        // Attack: very fast (one block worth); release: musical.
        att_coef_ = std::exp(-1.0f / (0.0005f * static_cast<float>(sample_rate)));
        rel_coef_ = (release_sec > 0.0f)
            ? std::exp(-1.0f / (release_sec * static_cast<float>(sample_rate)))
            : 0.0f;
    }

    inline float process_one(float x) noexcept {
        x = detail::sanitize_sample(x);
        const float a = std::fabs(x);

        // Target gain to keep peak at threshold
        const float target = (a > threshold_) ? (threshold_ / a) : 1.0f;

        // Asymmetric ramp: clamp fast, recover slow
        if (target < gain_) gain_ = att_coef_ * gain_ + (1.0f - att_coef_) * target;
        else                gain_ = rel_coef_ * gain_ + (1.0f - rel_coef_) * target;

        float y = x * gain_;
        // Last-resort hard clip in case of math anomalies
        if (y >  1.0f) y =  1.0f;
        if (y < -1.0f) y = -1.0f;
        return y;
    }

    void process_block(const float* in, float* out, std::size_t n) noexcept {
        for (std::size_t i = 0; i < n; ++i) out[i] = process_one(in[i]);
    }

private:
    float threshold_ = 0.95f;
    float att_coef_  = 0.0f;
    float rel_coef_  = 0.0f;
    float gain_      = 1.0f;
};


// ----------------------------------------------------------------------
// Simple de-esser: detect sibilance band energy (~4-9 kHz) with a one-pole
// HP, and duck the output proportionally when it spikes. Not surgical — for
// surgical use a multiband compressor — but good enough to keep "s" sounds
// from getting harshened by pitch-shift.
// ----------------------------------------------------------------------
class DeEsser {
public:
    void reset() noexcept {
        hp1_x1_ = hp1_y1_ = 0.0f;
        env_ = 0.0f;
    }

    void configure(double sample_rate, float band_hz = 5500.0f,
                   float threshold_db = -22.0f,
                   float reduction_db = 6.0f) noexcept {
        if (!(sample_rate > 0.0)) sample_rate = 48000.0;
        // One-pole HP at band_hz for sidechain
        const double w = 2.0 * 3.14159265358979 * band_hz / sample_rate;
        const double r = std::exp(-w);
        sc_a_ = static_cast<float>(1.0 - r);
        sc_b_ = static_cast<float>(r);

        thresh_ = std::pow(10.0f, threshold_db / 20.0f);
        // reduction_db is the maximum gain reduction we'll apply
        max_red_ = std::pow(10.0f, -std::fabs(reduction_db) / 20.0f);

        const float env_rel = 0.020f;
        env_coef_ = std::exp(-1.0f / (env_rel * static_cast<float>(sample_rate)));
    }

    inline float process_one(float x) noexcept {
        x = detail::sanitize_sample(x);
        // Sidechain: one-pole HP then envelope
        const float hp = x - hp1_x1_ + sc_b_ * hp1_y1_;
        hp1_x1_ = x;
        hp1_y1_ = hp;

        const float a = std::fabs(hp) * sc_a_;
        if (a > env_) env_ = a;
        else          env_ = env_coef_ * env_;

        // Reduce when sidechain envelope is above threshold
        float gain = 1.0f;
        if (env_ > thresh_) {
            // linear interp between 1.0 and max_red_ based on how far over
            const float over = std::min(1.0f, (env_ - thresh_) / thresh_);
            gain = 1.0f - (1.0f - max_red_) * over;
        }
        return x * gain;
    }

    void process_block(const float* in, float* out, std::size_t n) noexcept {
        for (std::size_t i = 0; i < n; ++i) out[i] = process_one(in[i]);
    }

private:
    float sc_a_     = 0.0f;
    float sc_b_     = 0.0f;
    float thresh_   = 0.07f;
    float max_red_  = 0.5f;
    float env_coef_ = 0.0f;

    float hp1_x1_ = 0.0f;
    float hp1_y1_ = 0.0f;
    float env_    = 0.0f;
};

}  // namespace aboba
