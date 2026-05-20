// SPDX-License-Identifier: GPL-3.0-or-later
//
// VulkanBackend — cross-vendor GPU FFT via Vulkan compute shaders.
//
// Why Vulkan:
//   * Cross-vendor: AMD, NVIDIA, Intel, Qualcomm (Adreno), ARM (Mali), all
//     ship Vulkan drivers. The HIP backend is AMD-only and the only path
//     to NVIDIA is CUDA (which we refuse to use philosophically).
//   * Works on Linux, Windows, Android. macOS via MoltenVK (Metal under).
//   * Works on Lavapipe (software Vulkan via LLVM) for development/CI on
//     machines without a real GPU. This actually means we can TEST the
//     entire Vulkan codepath in sandbox without hardware!
//
// Algorithm: Radix-2 Stockham FFT, computed stage-by-stage via repeated
// dispatches of the fft_stage compute shader. For an N-point transform
// we do log2(N) stages, each a single GPU dispatch. Ping-pong between
// two buffers; final result ends up in one or the other depending on
// log2(N) parity.
//
// Sizes:
//   * Power-of-2 only. fft_size must be in {64, 128, 256, ..., 16384}.
//   * For non-power-of-2 fall back to CPU.
//
// Memory:
//   * Per-fft_size we cache a "plan": staging buffers, descriptor sets,
//     pipelines. Once created, we never reallocate. Cache is bounded.
//   * Input/output cross PCI on each call. For our small N this is a
//     fraction of total cost. Real GPU win starts at fft_size >= 1024.
//
// Latency profile (typical, AMD discrete GPU, fft_size=2048):
//   ~50-200us per call dominated by dispatch + PCIe round-trip
//   CPU FFTW does the same in ~20-50us. THE VULKAN BACKEND IS SLOWER FOR
//   SMALL FFTS. It wins on:
//     * Very large FFTs (>= 8192)
//     * Batched FFTs (we don't expose batched API yet — TODO)
//     * Cases where the CPU is busy with other work
//
// Paranoia:
//   * Instance creation, device selection, all error-checked. On any
//     Vulkan failure during construction, we throw and the caller falls
//     back to CPU via the hybrid backend.
//   * Buffer overflow checks: input/output size verified against plan.
//   * Validation layer optional (off by default; can be enabled via env
//     var ABOBA_VULKAN_VALIDATE=1).
//   * All Vulkan resources destroyed in correct order in destructor.
//
// Threading:
//   * Not thread-safe per-instance. Use one VulkanBackend per thread.
//
// Build:
//   The backend is compiled only when ABOBA_ENABLE_VULKAN is ON in CMake.
//   Even when not built, the factory will refuse VULKAN backend type.
#pragma once

#include "backend.hpp"

#include <cstddef>
#include <memory>
#include <vector>

namespace aboba {

// Opaque implementation. Vulkan headers leak everywhere if exposed here;
// we keep them in the .cpp.
class VulkanBackendImpl;

class VulkanBackend : public Backend {
public:
    // Throws std::runtime_error if Vulkan initialization fails (no driver,
    // no compatible device, instance/device creation error, etc).
    //
    // Pass enable_validation=true to load the VK_LAYER_KHRONOS_validation
    // layer (must be installed). Useful for development; spammy & slow.
    VulkanBackend(bool enable_validation = false,
                  bool prefer_discrete_gpu = true);
    ~VulkanBackend() override;

    VulkanBackend(const VulkanBackend&) = delete;
    VulkanBackend& operator=(const VulkanBackend&) = delete;

    void fft_r2c(const float* input, std::complex<float>* output,
                 std::size_t fft_size) override;
    void fft_c2r(const std::complex<float>* input, float* output,
                 std::size_t fft_size) override;
    void fft_r2c_batch(const float* input, std::complex<float>* output,
                       std::size_t fft_size, std::size_t batch) override;
    void fft_c2r_batch(const std::complex<float>* input, float* output,
                       std::size_t fft_size, std::size_t batch) override;

    BackendType type() const override { return BackendType::HIP; /* GPU-like */ }
    const char* name() const override;

    // Device introspection (returned strings stable for backend lifetime)
    const char* device_name() const;
    const char* driver_info() const;
    bool is_software_renderer() const;  // true if Lavapipe / etc.

private:
    std::unique_ptr<VulkanBackendImpl> impl_;
};

// Test whether a Vulkan backend can be created on this system. Returns
// false if Vulkan isn't built into the library, no driver loaded, or no
// suitable device exists. Does NOT throw.
bool vulkan_backend_available() noexcept;

}  // namespace aboba
