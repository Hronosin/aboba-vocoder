// SPDX-License-Identifier: GPL-3.0-or-later
//
// MockBackend — a configurable Backend implementation for testing the
// hybrid/routing infrastructure without real GPU hardware.
//
// It wraps any real Backend (typically the CPU one) and adds:
//   * Configurable artificial latency (per-call sleep)
//   * Optional latency jitter (uniformly distributed delta)
//   * Optional failure injection (every Nth call throws)
//   * Optional warmup behavior (first K calls slower)
//   * Atomic counters of calls / failures
//
// This lets us simulate a GPU's characteristics in unit tests:
//   * GPU = lower throughput floor for big batches but higher per-call cost
//     for small ones (because of launch overhead). The CPU backend has
//     near-zero per-call overhead.
//   * GPU sometimes fails (driver reset, OOM, etc) — we want our hybrid
//     scheduler to recover gracefully.
//
// Paranoia:
//   * Failure injection is deterministic: failure modulo gives the same
//     pattern between runs, so test failures are reproducible.
//   * Counters are atomic; safe to read from a watchdog thread.
//   * The wrapped backend MUST outlive the MockBackend (we don't take
//     ownership — call ownership is the caller's choice).
#pragma once

#include "backend.hpp"

#include <atomic>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>

namespace aboba {

struct MockBackendConfig {
    // Pretend we're a GPU. If false, this is "another CPU" — useful for
    // multi-channel testing where you want a second independent backend.
    bool pretend_gpu = true;

    // Sleep this many microseconds at the start of every call (simulates
    // a GPU's kernel launch + sync overhead).
    int  base_latency_us = 0;

    // Add a uniformly-distributed random jitter in [-jitter_us, +jitter_us]
    // to each call. Useful for stress-testing schedulers.
    int  jitter_us = 0;

    // First `warmup_calls` calls add `warmup_latency_us` extra. Models
    // GPU JIT compilation / pipeline cache warm-up.
    int  warmup_calls       = 0;
    int  warmup_latency_us  = 0;

    // Inject a failure (throw std::runtime_error) every Nth call. 0 = never.
    // The throw happens BEFORE the wrapped backend is called, so no real
    // work is performed for the failing call.
    int  failure_period = 0;

    // For testing: skip the inner backend's actual computation. The output
    // buffer is filled with zeros. Use only in scheduler/routing tests
    // where you don't care about the FFT result.
    bool produce_zero_output = false;

    // Custom name string (so tests can distinguish multiple mocks).
    std::string name = "mock-gpu";
};

class MockBackend : public Backend {
public:
    // The wrapped backend must outlive `this`. Pass nullptr only if
    // produce_zero_output=true.
    MockBackend(MockBackendConfig cfg, Backend* wrapped);

    void fft_r2c(const float* input, std::complex<float>* output,
                 std::size_t fft_size) override;
    void fft_c2r(const std::complex<float>* input, float* output,
                 std::size_t fft_size) override;
    void fft_r2c_batch(const float* input, std::complex<float>* output,
                       std::size_t fft_size, std::size_t batch) override;
    void fft_c2r_batch(const std::complex<float>* input, float* output,
                       std::size_t fft_size, std::size_t batch) override;

    BackendType type() const override {
        return cfg_.pretend_gpu ? BackendType::HIP : BackendType::CPU;
    }
    const char* name() const override { return cfg_.name.c_str(); }

    // Diagnostics
    std::uint64_t total_calls()    const noexcept { return total_calls_.load(); }
    std::uint64_t failed_calls()   const noexcept { return failed_calls_.load(); }
    std::uint64_t completed_calls() const noexcept {
        return total_calls_.load() - failed_calls_.load();
    }

    // Allow tests to change failure period at runtime
    void set_failure_period(int p) noexcept { cfg_.failure_period = p; }
    void set_base_latency_us(int us) noexcept { cfg_.base_latency_us = us; }

    // Reset all counters
    void reset_counters() noexcept {
        total_calls_.store(0);
        failed_calls_.store(0);
    }

private:
    void simulate_overhead();
    void maybe_fail();

    MockBackendConfig cfg_;
    Backend* wrapped_;
    std::atomic<std::uint64_t> total_calls_  {0};
    std::atomic<std::uint64_t> failed_calls_ {0};
    // Simple LCG for deterministic jitter without locks
    std::atomic<std::uint64_t> rng_state_    {0x123456789ABCDEFull};
};

// Exception thrown by MockBackend on injected failure. Callers can catch
// this specifically to distinguish from real backend errors.
class MockBackendInjectedFailure : public std::runtime_error {
public:
    explicit MockBackendInjectedFailure(const std::string& msg)
        : std::runtime_error(msg) {}
};

}  // namespace aboba
