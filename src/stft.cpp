// SPDX-License-Identifier: GPL-3.0-or-later
#include "aboba/stft.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>

namespace aboba {

namespace {
constexpr float kPi = 3.14159265358979323846f;
constexpr float kTwoPi = 2.0f * kPi;
}  // namespace

STFT::STFT(std::size_t fft_size,
           std::size_t hop_size,
           WindowType window_type,
           Backend* backend)
    : fft_size_(fft_size),
      hop_size_(hop_size),
      backend_(backend) {
    if (!backend_) throw std::invalid_argument("STFT: backend is null");
    if (fft_size == 0 || (fft_size & (fft_size - 1)) != 0) {
        // Non-power-of-2 sizes work with FFTW, but we recommend pow2 for GPU.
        // We don't hard-error; just a soft contract.
    }
    if (hop_size == 0 || hop_size > fft_size) {
        throw std::invalid_argument("STFT: hop_size must be in (0, fft_size]");
    }
    build_window(window_type);
    compute_ola_norm();
    frame_buf_.resize(fft_size_);
    spec_buf_.resize(n_bins());
}

STFT::~STFT() = default;

void STFT::build_window(WindowType type) {
    window_.resize(fft_size_);
    const float N = static_cast<float>(fft_size_);
    for (std::size_t n = 0; n < fft_size_; ++n) {
        const float x = static_cast<float>(n);
        switch (type) {
            case WindowType::Hann:
                window_[n] = 0.5f * (1.0f - std::cos(kTwoPi * x / (N - 1.0f)));
                break;
            case WindowType::Hamming:
                window_[n] = 0.54f - 0.46f * std::cos(kTwoPi * x / (N - 1.0f));
                break;
            case WindowType::Blackman:
                window_[n] = 0.42f
                           - 0.5f  * std::cos(kTwoPi * x / (N - 1.0f))
                           + 0.08f * std::cos(2.0f * kTwoPi * x / (N - 1.0f));
                break;
        }
    }
}

void STFT::compute_ola_norm() {
    // For OLA reconstruction we divide by the windowed-overlap energy at each
    // output sample. With Hann + hop=fft_size/4 the sum-of-squares is constant
    // (COLA condition), but we compute it explicitly to support any window.
    ola_norm_.assign(fft_size_, 0.0f);
    for (std::size_t n = 0; n < fft_size_; ++n) {
        ola_norm_[n] = window_[n] * window_[n];
    }
}

std::size_t STFT::num_frames(std::size_t n_samples) const {
    if (n_samples < fft_size_) return 0;
    return 1 + (n_samples - fft_size_) / hop_size_;
}

std::size_t STFT::forward(const float* input,
                          std::size_t n_samples,
                          std::complex<float>* output) {
    if (n_samples == 0) return 0;
    if (!input || !output) {
        throw std::invalid_argument("STFT::forward: null buffer");
    }

    const std::size_t n_frames = num_frames(n_samples);
    const std::size_t bins     = n_bins();

    for (std::size_t f = 0; f < n_frames; ++f) {
        const float* src = input + f * hop_size_;
        for (std::size_t n = 0; n < fft_size_; ++n) {
            frame_buf_[n] = src[n] * window_[n];
        }
        backend_->fft_r2c(frame_buf_.data(), spec_buf_.data(), fft_size_);
        std::memcpy(output + f * bins,
                    spec_buf_.data(),
                    bins * sizeof(std::complex<float>));
    }
    return n_frames;
}

void STFT::inverse(const std::complex<float>* input,
                   std::size_t n_frames,
                   float* output) {
    if (n_frames == 0) return;
    if (!input || !output) {
        throw std::invalid_argument("STFT::inverse: null buffer");
    }

    const std::size_t bins = n_bins();
    const std::size_t out_len = (n_frames - 1) * hop_size_ + fft_size_;

    // Grow norm_accum_ only if needed (avoids reallocation across calls
    // of similar size). Caller's `output` buffer is assumed pre-zeroed in
    // the sense that we explicitly fill it here.
    if (norm_accum_.size() < out_len) {
        norm_accum_.resize(out_len);
    }
    std::fill(output, output + out_len, 0.0f);
    std::fill(norm_accum_.begin(), norm_accum_.begin() + out_len, 0.0f);

    const float ifft_scale = 1.0f / static_cast<float>(fft_size_);

    for (std::size_t f = 0; f < n_frames; ++f) {
        std::memcpy(spec_buf_.data(),
                    input + f * bins,
                    bins * sizeof(std::complex<float>));
        backend_->fft_c2r(spec_buf_.data(), frame_buf_.data(), fft_size_);

        const std::size_t offset = f * hop_size_;
        for (std::size_t n = 0; n < fft_size_; ++n) {
            output[offset + n]      += frame_buf_[n] * window_[n] * ifft_scale;
            norm_accum_[offset + n] += ola_norm_[n];
        }
    }

    constexpr float eps = 1e-8f;
    for (std::size_t i = 0; i < out_len; ++i) {
        if (norm_accum_[i] > eps) {
            output[i] /= norm_accum_[i];
        }
    }
}

}  // namespace aboba
