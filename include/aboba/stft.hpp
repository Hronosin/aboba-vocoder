// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once

#include "backend.hpp"
#include <complex>
#include <vector>

namespace aboba {

enum class WindowType {
    Hann,
    Hamming,
    Blackman,
};

// STFT / iSTFT engine.
// Designed for both offline (whole signal) and streaming (chunk by chunk).
//
// Layout convention:
//   spectrogram = [n_frames][n_bins], row-major
//   where n_bins = fft_size / 2 + 1
class STFT {
public:
    STFT(std::size_t fft_size,
         std::size_t hop_size,
         WindowType window_type,
         Backend* backend);

    ~STFT();

    // Offline forward STFT.
    // Returns number of frames written to `output`.
    // `output` must hold at least num_frames(n_samples) * n_bins() complex values.
    std::size_t forward(const float* input,
                        std::size_t n_samples,
                        std::complex<float>* output);

    // Offline inverse STFT with overlap-add (OLA).
    // Output length: (n_frames - 1) * hop_size + fft_size
    void inverse(const std::complex<float>* input,
                 std::size_t n_frames,
                 float* output);

    std::size_t num_frames(std::size_t n_samples) const;

    std::size_t fft_size() const { return fft_size_; }
    std::size_t hop_size() const { return hop_size_; }
    std::size_t n_bins() const   { return fft_size_ / 2 + 1; }

private:
    void build_window(WindowType type);
    void compute_ola_norm();

    std::size_t fft_size_;
    std::size_t hop_size_;
    std::vector<float> window_;     // analysis window
    std::vector<float> ola_norm_;   // overlap-add normalization
    Backend* backend_;              // not owned

    // Scratch buffers (avoid alloc in hot path)
    std::vector<float> frame_buf_;
    std::vector<std::complex<float>> spec_buf_;
    std::vector<float> norm_accum_;  // used by inverse(); grown as needed
};

}  // namespace aboba
