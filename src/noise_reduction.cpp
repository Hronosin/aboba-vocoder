// SPDX-License-Identifier: GPL-3.0-or-later
#include "aboba/noise_reduction.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <stdexcept>

namespace aboba {

namespace {

constexpr float kPi    = 3.14159265358979323846f;
constexpr float kTwoPi = 2.0f * kPi;

// Tiny floor for noise PSD to avoid divide-by-zero.
constexpr float kNoiseFloor = 1e-12f;

inline float sanitize(float x) noexcept {
    return std::isfinite(x) ? x : 0.0f;
}

}  // namespace

NoiseReducer::NoiseReducer(NoiseReductionConfig cfg, Backend* backend)
    : cfg_(cfg), backend_(backend) {
    if (!backend_) {
        throw std::invalid_argument("NoiseReducer: null backend");
    }
    if (cfg_.fft_size < 64 || cfg_.fft_size > (1u << 20)) {
        throw std::invalid_argument("NoiseReducer: fft_size out of range");
    }
    if (cfg_.hop_size == 0 || cfg_.hop_size > cfg_.fft_size) {
        throw std::invalid_argument("NoiseReducer: hop_size out of range");
    }
    if (!(cfg_.sample_rate > 0.0)) {
        throw std::invalid_argument("NoiseReducer: invalid sample_rate");
    }

    // Clamp tunables to safe ranges
    if (!std::isfinite(cfg_.oversubtraction) || cfg_.oversubtraction < 1.0f) {
        cfg_.oversubtraction = 1.0f;
    }
    if (cfg_.oversubtraction > 5.0f) cfg_.oversubtraction = 5.0f;
    if (!std::isfinite(cfg_.spectral_floor) || cfg_.spectral_floor < 0.001f) {
        cfg_.spectral_floor = 0.001f;
    }
    if (cfg_.spectral_floor > 1.0f) cfg_.spectral_floor = 1.0f;
    if (!std::isfinite(cfg_.gain_smoothing) || cfg_.gain_smoothing < 0.0f) {
        cfg_.gain_smoothing = 0.0f;
    }
    if (cfg_.gain_smoothing > 0.99f) cfg_.gain_smoothing = 0.99f;

    n_bins_ = cfg_.fft_size / 2 + 1;

    // Pre-allocate everything
    window_.assign(cfg_.fft_size, 0.0f);
    in_buf_.assign(cfg_.fft_size, 0.0f);
    out_buf_.assign(cfg_.fft_size * 4, 0.0f);

    frame_.assign(cfg_.fft_size, 0.0f);
    spec_.assign(n_bins_, std::complex<float>(0.0f, 0.0f));

    noise_psd_.assign(n_bins_, 0.0f);
    signal_psd_.assign(n_bins_, 0.0f);
    min_tracker_.assign(n_bins_, std::numeric_limits<float>::max());
    gain_smooth_.assign(n_bins_, 1.0f);

    // Hann window + COLA² gain
    for (std::size_t n = 0; n < cfg_.fft_size; ++n) {
        window_[n] = 0.5f * (1.0f - std::cos(kTwoPi * static_cast<float>(n)
                             / static_cast<float>(cfg_.fft_size - 1)));
    }
    float sum_w2 = 0.0f;
    for (float w : window_) sum_w2 += w * w;
    ola_norm_ = sum_w2 / static_cast<float>(cfg_.hop_size);
    if (ola_norm_ < 1e-20f) ola_norm_ = 1.0f;

    // Frame counters from durations
    const float frame_rate = static_cast<float>(cfg_.sample_rate)
                             / static_cast<float>(cfg_.hop_size);
    total_learning_frames_ = static_cast<std::size_t>(
        std::ceil(cfg_.learning_seconds * frame_rate));
    if (total_learning_frames_ < 1) total_learning_frames_ = 1;
    learning_frames_remaining_ = total_learning_frames_;

    min_track_window_frames_ = static_cast<std::size_t>(
        std::ceil(cfg_.min_track_seconds * frame_rate));
    if (min_track_window_frames_ < 8) min_track_window_frames_ = 8;
}

void NoiseReducer::set_oversubtraction(float v) noexcept {
    if (!std::isfinite(v) || v < 1.0f) v = 1.0f;
    if (v > 5.0f) v = 5.0f;
    cfg_.oversubtraction = v;
}

void NoiseReducer::set_spectral_floor(float v) noexcept {
    if (!std::isfinite(v) || v < 0.001f) v = 0.001f;
    if (v > 1.0f) v = 1.0f;
    cfg_.spectral_floor = v;
}

void NoiseReducer::set_gain_smoothing(float v) noexcept {
    if (!std::isfinite(v) || v < 0.0f) v = 0.0f;
    if (v > 0.99f) v = 0.99f;
    cfg_.gain_smoothing = v;
}

void NoiseReducer::reset() {
    std::fill(in_buf_.begin(),     in_buf_.end(),     0.0f);
    std::fill(out_buf_.begin(),    out_buf_.end(),    0.0f);
    std::fill(noise_psd_.begin(),  noise_psd_.end(),  0.0f);
    std::fill(signal_psd_.begin(), signal_psd_.end(), 0.0f);
    std::fill(min_tracker_.begin(), min_tracker_.end(),
              std::numeric_limits<float>::max());
    std::fill(gain_smooth_.begin(), gain_smooth_.end(), 1.0f);
    in_fill_   = 0;
    out_write_ = 0;
    out_read_  = 0;
    learning_frames_remaining_ = total_learning_frames_;
    frames_since_min_reset_    = 0;
    cnt_total_ = cnt_in_learning_ = cnt_noise_updated_ = 0;
    last_mean_gain_ = last_min_gain_ = last_max_gain_ = 1.0f;
    last_snr_db_    = 0.0f;
}

NoiseReductionStats NoiseReducer::stats() const noexcept {
    NoiseReductionStats s;
    s.frames_total         = cnt_total_;
    s.frames_in_learning   = cnt_in_learning_;
    s.frames_noise_updated = cnt_noise_updated_;
    s.last_mean_gain       = last_mean_gain_;
    s.last_min_gain        = last_min_gain_;
    s.last_max_gain        = last_max_gain_;
    s.estimated_snr_db     = last_snr_db_;
    return s;
}

void NoiseReducer::learn_noise_profile(const float* silence, std::size_t n) {
    if (!silence || n < cfg_.fft_size) return;

    // Average |X|² over as many full FFT windows as fit (no overlap for
    // this offline routine — we want independent samples)
    std::fill(noise_psd_.begin(), noise_psd_.end(), 0.0f);
    std::size_t count = 0;

    std::vector<float> wf(cfg_.fft_size);
    std::vector<std::complex<float>> sp(n_bins_);

    for (std::size_t off = 0; off + cfg_.fft_size <= n; off += cfg_.fft_size) {
        for (std::size_t k = 0; k < cfg_.fft_size; ++k) {
            wf[k] = sanitize(silence[off + k]) * window_[k];
        }
        backend_->fft_r2c(wf.data(), sp.data(), cfg_.fft_size);
        for (std::size_t b = 0; b < n_bins_; ++b) {
            const float re = sp[b].real();
            const float im = sp[b].imag();
            noise_psd_[b] += re * re + im * im;
        }
        ++count;
    }
    if (count == 0) return;

    const float inv = 1.0f / static_cast<float>(count);
    for (std::size_t b = 0; b < n_bins_; ++b) {
        noise_psd_[b] *= inv;
        if (noise_psd_[b] < kNoiseFloor) noise_psd_[b] = kNoiseFloor;
    }

    // We're calibrated — skip the learning phase
    learning_frames_remaining_ = 0;
}

void NoiseReducer::process_one_frame() {
    ++cnt_total_;

    // Window + FFT
    float* __restrict__ frame      = frame_.data();
    const float* __restrict__ win  = window_.data();
    const float* __restrict__ in   = in_buf_.data();
    for (std::size_t n = 0; n < cfg_.fft_size; ++n) {
        frame[n] = sanitize(in[n]) * win[n];
    }
    backend_->fft_r2c(frame, spec_.data(), cfg_.fft_size);

    // |X(k)|²
    float frame_signal_pow = 0.0f;
    float frame_noise_pow  = 0.0f;
    for (std::size_t k = 0; k < n_bins_; ++k) {
        const float re = spec_[k].real();
        const float im = spec_[k].imag();
        const float p  = re * re + im * im;
        signal_psd_[k] = p;
        frame_signal_pow += p;
    }

    // Noise floor update logic
    bool noise_updated = false;
    if (learning_frames_remaining_ > 0) {
        // Warm-up phase: average |X|² into noise_psd_
        const float w = 1.0f / static_cast<float>(total_learning_frames_);
        for (std::size_t k = 0; k < n_bins_; ++k) {
            noise_psd_[k] += signal_psd_[k] * w;
        }
        --learning_frames_remaining_;
        ++cnt_in_learning_;
        noise_updated = true;
        if (learning_frames_remaining_ == 0) {
            // Floor everything to avoid pathological zeros
            for (std::size_t k = 0; k < n_bins_; ++k) {
                if (noise_psd_[k] < kNoiseFloor) noise_psd_[k] = kNoiseFloor;
            }
        }
    } else {
        // Adaptive phase: minimum tracking.
        // Reset the per-bin minimum every min_track_window_frames_; let it
        // capture the quietest moment in each window. Leak the noise
        // estimate gently toward that minimum.
        ++frames_since_min_reset_;
        if (frames_since_min_reset_ >= min_track_window_frames_) {
            for (std::size_t k = 0; k < n_bins_; ++k) {
                // Adopt the tracked minimum as the new noise estimate
                // (with safety floor)
                float m = min_tracker_[k];
                if (!std::isfinite(m) || m > 1e30f) m = noise_psd_[k];
                if (m < kNoiseFloor) m = kNoiseFloor;
                // Slow blend toward minimum (smoother than hard replace)
                noise_psd_[k] = 0.7f * noise_psd_[k] + 0.3f * m;
                if (cfg_.clamp_noise_to_signal) {
                    // Don't let estimate exceed signal — protects against
                    // mistaking sustained speech for noise
                    if (noise_psd_[k] > signal_psd_[k] * 1.5f) {
                        noise_psd_[k] = signal_psd_[k] * 1.5f;
                    }
                    if (noise_psd_[k] < kNoiseFloor) noise_psd_[k] = kNoiseFloor;
                }
                // Reset tracker
                min_tracker_[k] = signal_psd_[k];
            }
            frames_since_min_reset_ = 0;
            ++cnt_noise_updated_;
            noise_updated = true;
        } else {
            // Just update the running minimum
            for (std::size_t k = 0; k < n_bins_; ++k) {
                if (signal_psd_[k] < min_tracker_[k]) {
                    min_tracker_[k] = signal_psd_[k];
                }
            }
        }
    }
    (void)noise_updated;

    // Compute gain per bin: G(k) = sqrt(max(1 - α N/|X|², β))
    float gain_min = std::numeric_limits<float>::max();
    float gain_max = 0.0f;
    double gain_sum = 0.0;
    for (std::size_t k = 0; k < n_bins_; ++k) {
        const float sp_pow = std::max(signal_psd_[k], kNoiseFloor);
        const float ratio = cfg_.oversubtraction * noise_psd_[k] / sp_pow;
        float g2 = 1.0f - ratio;
        if (g2 < cfg_.spectral_floor) g2 = cfg_.spectral_floor;
        if (g2 > 1.0f) g2 = 1.0f;
        const float g = std::sqrt(g2);

        // Temporal smoothing
        const float smoothed = cfg_.gain_smoothing * gain_smooth_[k]
                             + (1.0f - cfg_.gain_smoothing) * g;
        gain_smooth_[k] = smoothed;

        if (smoothed < gain_min) gain_min = smoothed;
        if (smoothed > gain_max) gain_max = smoothed;
        gain_sum += smoothed;

        // Apply
        spec_[k] = std::complex<float>(spec_[k].real() * smoothed,
                                        spec_[k].imag() * smoothed);
        frame_noise_pow += noise_psd_[k];
    }
    last_mean_gain_ = static_cast<float>(gain_sum / static_cast<double>(n_bins_));
    last_min_gain_  = gain_min;
    last_max_gain_  = gain_max;

    // SNR estimate (dB): 10 log10(signal_pow / noise_pow)
    if (frame_noise_pow > 0.0f && frame_signal_pow > 0.0f) {
        const float snr = 10.0f * std::log10(frame_signal_pow / frame_noise_pow);
        last_snr_db_ = std::isfinite(snr) ? snr : 0.0f;
    }

    // iFFT + OLA
    backend_->fft_c2r(spec_.data(), frame, cfg_.fft_size);
    const float ifft_scale = 1.0f / (static_cast<float>(cfg_.fft_size) * ola_norm_);

    assert(out_write_ + cfg_.fft_size <= out_buf_.size());
    float* __restrict__ out_ola = out_buf_.data() + out_write_;
    for (std::size_t n = 0; n < cfg_.fft_size; ++n) {
        out_ola[n] += frame[n] * win[n] * ifft_scale;
    }
    out_write_ += cfg_.hop_size;
}

void NoiseReducer::process(const float* input, float* output, std::size_t n) {
    if (n == 0) return;
    if (!input || !output) {
        throw std::invalid_argument("NoiseReducer::process: null buffer");
    }

    for (std::size_t i = 0; i < n; ++i) {
        in_buf_[in_fill_++] = input[i];

        if (in_fill_ == cfg_.fft_size) {
            // Ensure out_buf_ has room for another full window
            if (out_write_ + cfg_.fft_size > out_buf_.size()) {
                const std::size_t active_end =
                    std::min(out_write_ + cfg_.fft_size, out_buf_.size());
                const std::size_t active =
                    (active_end > out_read_) ? (active_end - out_read_) : 0;
                if (active > 0 && out_read_ > 0) {
                    std::memmove(out_buf_.data(),
                                 out_buf_.data() + out_read_,
                                 active * sizeof(float));
                    std::fill(out_buf_.begin() + static_cast<std::ptrdiff_t>(active),
                              out_buf_.end(), 0.0f);
                    out_write_ -= out_read_;
                    out_read_   = 0;
                }
                if (out_write_ + cfg_.fft_size > out_buf_.size()) {
                    out_buf_.resize(out_write_ + cfg_.fft_size * 2, 0.0f);
                }
            }

            process_one_frame();

            const std::size_t keep = cfg_.fft_size - cfg_.hop_size;
            std::memmove(in_buf_.data(),
                         in_buf_.data() + cfg_.hop_size,
                         keep * sizeof(float));
            in_fill_ = keep;
        }

        if (out_read_ < out_write_) {
            output[i] = sanitize(out_buf_[out_read_]);
            out_buf_[out_read_] = 0.0f;
            ++out_read_;
        } else {
            output[i] = 0.0f;
        }
    }

    // Periodic compaction
    if (out_read_ > out_buf_.size() / 2) {
        const std::size_t active_end =
            std::min(out_write_ + cfg_.fft_size, out_buf_.size());
        const std::size_t active =
            (active_end > out_read_) ? (active_end - out_read_) : 0;
        if (active > 0) {
            std::memmove(out_buf_.data(),
                         out_buf_.data() + out_read_,
                         active * sizeof(float));
        }
        std::fill(out_buf_.begin() + static_cast<std::ptrdiff_t>(active),
                  out_buf_.end(), 0.0f);
        out_write_ -= out_read_;
        out_read_   = 0;
    }
}

}  // namespace aboba
