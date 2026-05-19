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

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

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
        const double w = 2.0 * 3.14159265358979 * static_cast<double>(cutoff_hz) / sample_rate;
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
        const double rc = 1.0 / (2.0 * 3.14159265358979 * static_cast<double>(cutoff_hz));
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
        const double w = 2.0 * 3.14159265358979 * static_cast<double>(band_hz) / sample_rate;
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

// ----------------------------------------------------------------------
// AutoGain: automatic gain control that keeps perceived loudness stable
// regardless of how far the speaker is from the mic.
//
// Maintains an RMS-like envelope, then computes the gain needed to bring
// that envelope to the target. Gain is clamped (won't amplify above
// max_gain or attenuate below min_gain) and smoothed (slow attack to
// avoid pumping, slower release for stability). Below noise_gate level
// the gain is HELD — we never amplify silence into roaring background.
//
// Use it BEFORE the vocoder so the vocoder gets consistent input level.
// ----------------------------------------------------------------------
class AutoGain {
public:
    void reset() noexcept {
        env_  = 0.0f;
        gain_ = 1.0f;
    }

    void configure(double sample_rate,
                   float target_dbfs       = -16.0f,
                   float max_gain_db       = 24.0f,
                   float min_gain_db       = -12.0f,
                   float attack_ms         = 50.0f,
                   float release_ms        = 1000.0f,
                   float noise_gate_dbfs   = -55.0f) noexcept {
        if (!(sample_rate > 0.0)) sample_rate = 48000.0;

        target_level_ = std::pow(10.0f, target_dbfs / 20.0f);
        max_gain_     = std::pow(10.0f, max_gain_db / 20.0f);
        min_gain_     = std::pow(10.0f, min_gain_db / 20.0f);
        noise_gate_   = std::pow(10.0f, noise_gate_dbfs / 20.0f);

        const float sr_f = static_cast<float>(sample_rate);

        // Envelope follower
        att_env_ = (attack_ms > 0.0f) ? std::exp(-1.0f / (0.005f * sr_f)) : 0.0f;
        rel_env_ = (release_ms > 0.0f) ? std::exp(-1.0f / (0.150f * sr_f)) : 0.0f;

        // Gain smoothing
        att_gain_ = (attack_ms > 0.0f)
            ? std::exp(-1.0f / (attack_ms * 0.001f * sr_f)) : 0.0f;
        rel_gain_ = (release_ms > 0.0f)
            ? std::exp(-1.0f / (release_ms * 0.001f * sr_f)) : 0.0f;
    }

    inline float process_one(float x) noexcept {
        x = detail::sanitize_sample(x);
        const float a = std::fabs(x);

        // Envelope follower (peak-RMS hybrid)
        if (a > env_) env_ = att_env_ * env_ + (1.0f - att_env_) * a;
        else          env_ = rel_env_ * env_ + (1.0f - rel_env_) * a;

        // Required gain to hit the target
        float required = 1.0f;
        if (env_ > noise_gate_) {
            required = target_level_ / env_;
            if (required > max_gain_) required = max_gain_;
            if (required < min_gain_) required = min_gain_;
        } else {
            // Hold current gain to avoid amplifying silence
            required = gain_;
        }

        // Smooth gain toward required
        if (required < gain_) {
            gain_ = att_gain_ * gain_ + (1.0f - att_gain_) * required;
        } else {
            gain_ = rel_gain_ * gain_ + (1.0f - rel_gain_) * required;
        }

        return x * gain_;
    }

    void process_block(const float* in, float* out, std::size_t n) noexcept {
        for (std::size_t i = 0; i < n; ++i) out[i] = process_one(in[i]);
    }

    float current_gain() const noexcept { return gain_; }
    float current_gain_db() const noexcept {
        return 20.0f * std::log10(std::max(gain_, 1e-6f));
    }
    float current_envelope() const noexcept { return env_; }

private:
    float target_level_ = 0.158f;   // -16 dBFS
    float max_gain_     = 15.85f;   // +24 dB
    float min_gain_     = 0.251f;   // -12 dB
    float noise_gate_   = 0.00178f; // -55 dBFS
    float att_env_      = 0.0f;
    float rel_env_      = 0.0f;
    float att_gain_     = 0.0f;
    float rel_gain_     = 0.0f;

    float env_  = 0.0f;
    float gain_ = 1.0f;
};


// ----------------------------------------------------------------------
// LookaheadLimiter: brick-wall limiter with sample-accurate lookahead.
//
// Standard SoftLimiter (above) reacts AFTER the peak hits, so transients
// can sneak past. LookaheadLimiter delays the signal by `lookahead_ms` so
// it can ramp gain down BEFORE the peak arrives at the output. Cost:
// adds latency equal to the lookahead window.
//
// Choose this when broadcast-grade ceiling is needed (e.g. streaming);
// stick with SoftLimiter when latency must be zero.
// ----------------------------------------------------------------------
class LookaheadLimiter {
public:
    void reset() noexcept {
        std::fill(delay_.begin(), delay_.end(), 0.0f);
        head_ = 0;
        gain_ = 1.0f;
    }

    void configure(double sample_rate,
                   float threshold_dbfs    = -1.0f,
                   float lookahead_ms      = 2.0f,
                   float release_ms        = 50.0f) {
        if (!(sample_rate > 0.0)) sample_rate = 48000.0;
        if (!std::isfinite(threshold_dbfs)) threshold_dbfs = -1.0f;
        if (threshold_dbfs > 0.0f)  threshold_dbfs = 0.0f;
        if (threshold_dbfs < -30.0f) threshold_dbfs = -30.0f;
        threshold_ = std::pow(10.0f, threshold_dbfs / 20.0f);

        if (!std::isfinite(lookahead_ms) || lookahead_ms < 0.1f) {
            lookahead_ms = 0.1f;
        }
        if (lookahead_ms > 20.0f) lookahead_ms = 20.0f;

        std::size_t ls = static_cast<std::size_t>(
            std::ceil(lookahead_ms * 0.001f * static_cast<float>(sample_rate)));
        if (ls < 1) ls = 1;
        lookahead_samples_ = ls;
        delay_.assign(ls, 0.0f);
        head_ = 0;
        gain_ = 1.0f;

        // Attack ramp: reach target gain over the lookahead window.
        // Linear step approach: we never need to go below 0 (gain stays
        // positive). Step per sample is ~1/lookahead.
        attack_step_ = 1.0f / static_cast<float>(ls);

        if (!std::isfinite(release_ms) || release_ms < 1.0f) release_ms = 1.0f;
        if (release_ms > 5000.0f) release_ms = 5000.0f;
        release_coef_ = std::exp(-1.0f / (release_ms * 0.001f
                                          * static_cast<float>(sample_rate)));
    }

    inline float process_one(float x) noexcept {
        x = detail::sanitize_sample(x);
        const float a = std::fabs(x);
        const float target = (a > threshold_) ? (threshold_ / a) : 1.0f;

        if (target < gain_) {
            // Attack: linear step toward target (fast). The step is sized
            // so a worst-case 0->1 gain change completes in lookahead
            // samples.
            const float step = attack_step_;
            if (gain_ - step > target) {
                gain_ -= step;
            } else {
                gain_ = target;
            }
        } else {
            // Release: exponential decay back toward unity
            gain_ = release_coef_ * gain_ + (1.0f - release_coef_) * 1.0f;
        }

        // Read delayed sample, then push new sample
        const float delayed = delay_[head_];
        delay_[head_] = x;
        head_ = (head_ + 1) % lookahead_samples_;

        float y = delayed * gain_;
        // Last-resort hard clip in case lookahead wasn't quite enough
        if (y >  1.0f) y =  1.0f;
        if (y < -1.0f) y = -1.0f;
        return y;
    }

    void process_block(const float* in, float* out, std::size_t n) noexcept {
        for (std::size_t i = 0; i < n; ++i) out[i] = process_one(in[i]);
    }

    std::size_t latency_samples() const noexcept { return lookahead_samples_; }
    float current_gain() const noexcept { return gain_; }

private:
    std::vector<float> delay_;
    std::size_t head_              = 0;
    std::size_t lookahead_samples_ = 1;

    float threshold_     = 0.891f;  // -1 dBFS
    float gain_          = 1.0f;
    float attack_step_   = 1.0f;
    float release_coef_  = 0.0f;
};

}  // namespace aboba
