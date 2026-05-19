// SPDX-License-Identifier: GPL-3.0-or-later
#include "aboba/mock_backend.hpp"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <stdexcept>
#include <thread>
namespace aboba {

namespace {
// LCG step. Deterministic, no global state.
inline std::uint64_t lcg_step(std::uint64_t s) noexcept {
    return s * 6364136223846793005ULL + 1442695040888963407ULL;
}
inline std::int32_t lcg_int(std::uint64_t s) noexcept {
    return static_cast<std::int32_t>((s >> 33) & 0x7FFFFFFFu);
}
}  // namespace

MockBackend::MockBackend(MockBackendConfig cfg, Backend* wrapped)
    : cfg_(std::move(cfg)), wrapped_(wrapped) {
    if (!wrapped_ && !cfg_.produce_zero_output) {
        throw std::invalid_argument(
            "MockBackend: wrapped backend is null and produce_zero_output is false");
    }
}

void MockBackend::simulate_overhead() {
    const auto current_call = total_calls_.fetch_add(1, std::memory_order_relaxed);

    int latency_us = cfg_.base_latency_us;

    if (cfg_.warmup_calls > 0 &&
        current_call < static_cast<std::uint64_t>(cfg_.warmup_calls)) {
        latency_us += cfg_.warmup_latency_us;
    }

    if (cfg_.jitter_us > 0) {
        // Advance our PRNG. relaxed load+store is fine — we don't need
        // strict ordering, just per-call variation.
        std::uint64_t s = rng_state_.load(std::memory_order_relaxed);
        s = lcg_step(s);
        rng_state_.store(s, std::memory_order_relaxed);
        const std::int32_t r = lcg_int(s);
        const int j = (r % (2 * cfg_.jitter_us + 1)) - cfg_.jitter_us;
        latency_us += j;
        if (latency_us < 0) latency_us = 0;
    }

    if (latency_us > 0) {
        std::this_thread::sleep_for(std::chrono::microseconds(latency_us));
    }
}

void MockBackend::maybe_fail() {
    if (cfg_.failure_period <= 0) return;
    // Check using the current call counter (already incremented in simulate_overhead).
    // We use (count % period == 0) as the failure trigger — but skip count 0
    // so we always have at least one successful call to anchor expectations.
    const std::uint64_t count = total_calls_.load(std::memory_order_relaxed);
    if (count > 0 && (count % static_cast<std::uint64_t>(cfg_.failure_period) == 0)) {
        failed_calls_.fetch_add(1, std::memory_order_relaxed);
        throw MockBackendInjectedFailure(
            std::string("MockBackend('") + cfg_.name + "'): injected failure on call " +
            std::to_string(count));
    }
}

void MockBackend::fft_r2c(const float* input, std::complex<float>* output,
                          std::size_t fft_size) {
    simulate_overhead();
    maybe_fail();
    if (cfg_.produce_zero_output) {
        std::fill_n(output, fft_size / 2 + 1, std::complex<float>(0.0f, 0.0f));
        return;
    }
    wrapped_->fft_r2c(input, output, fft_size);
}

void MockBackend::fft_c2r(const std::complex<float>* input, float* output,
                          std::size_t fft_size) {
    simulate_overhead();
    maybe_fail();
    if (cfg_.produce_zero_output) {
        std::memset(output, 0, fft_size * sizeof(float));
        return;
    }
    wrapped_->fft_c2r(input, output, fft_size);
}

void MockBackend::fft_r2c_batch(const float* input, std::complex<float>* output,
                                std::size_t fft_size, std::size_t batch) {
    simulate_overhead();
    maybe_fail();
    if (cfg_.produce_zero_output) {
        std::fill_n(output, batch * (fft_size / 2 + 1), std::complex<float>(0.0f, 0.0f));
        return;
    }
    wrapped_->fft_r2c_batch(input, output, fft_size, batch);
}

void MockBackend::fft_c2r_batch(const std::complex<float>* input, float* output,
                                std::size_t fft_size, std::size_t batch) {
    simulate_overhead();
    maybe_fail();
    if (cfg_.produce_zero_output) {
        std::memset(output, 0, batch * fft_size * sizeof(float));
        return;
    }
    wrapped_->fft_c2r_batch(input, output, fft_size, batch);
}

}  // namespace aboba
