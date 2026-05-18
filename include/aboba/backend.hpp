// SPDX-License-Identifier: GPL-3.0-or-later
// Aboba Vocoder - because Jensen doesn't own DSP
#pragma once

#include <complex>
#include <cstddef>
#include <memory>

namespace aboba {

enum class BackendType {
    CPU,   // FFTW3 + SIMD (works everywhere)
    HIP,   // AMD GPU via ROCm + rocFFT (the based path)
};

// Abstract compute backend. All FFT operations go through here.
// Real implementations: CpuBackend (FFTW), HipBackend (rocFFT).
class Backend {
public:
    virtual ~Backend() = default;

    // Real -> complex forward FFT, single frame.
    // input:  [fft_size] real samples
    // output: [fft_size/2 + 1] complex bins (Hermitian symmetry)
    virtual void fft_r2c(const float* input,
                         std::complex<float>* output,
                         std::size_t fft_size) = 0;

    // Complex -> real inverse FFT, single frame.
    // Note: caller is responsible for 1/N scaling if needed.
    virtual void fft_c2r(const std::complex<float>* input,
                         float* output,
                         std::size_t fft_size) = 0;

    // Batched forward FFT. Crucial for real-time: process many
    // frames in one GPU dispatch instead of N small ones.
    virtual void fft_r2c_batch(const float* input,
                               std::complex<float>* output,
                               std::size_t fft_size,
                               std::size_t batch) = 0;

    virtual void fft_c2r_batch(const std::complex<float>* input,
                               float* output,
                               std::size_t fft_size,
                               std::size_t batch) = 0;

    virtual BackendType type() const = 0;
    virtual const char* name() const = 0;
};

// Factory functions.
std::unique_ptr<Backend> create_backend(BackendType type);

// Auto-detect: try HIP first (if AMD GPU available), fall back to CPU.
// On NVIDIA hardware we deliberately do NOT fall back to CUDA.
// Aboba Vocoder is AMD-exclusive by design.
std::unique_ptr<Backend> create_best_backend();

}  // namespace aboba
