// SPDX-License-Identifier: GPL-3.0-or-later
//
// HybridBackend — wraps multiple Backend implementations with three
// orthogonal capabilities:
//
//   1. ADAPTIVE ROUTING (per-call selection)
//      Measure cost (latency) of each child backend over a rolling window
//      and route each call to the one currently fastest for the call's
//      size class. Size classes are bucketed by fft_size so the GPU's
//      penalty for small calls and benefit for big calls are properly
//      accounted for.
//
//   2. FAILOVER CHAIN (resilience)
//      Each backend has a position in priority order. A call goes to the
//      highest-priority backend whose health is OK. On exception, the
//      backend is marked unhealthy, we record the failure, and we retry
//      on the next priority. Unhealthy backends are re-tried after a
//      cooldown (10 seconds by default).
//
//   3. MULTI-CHANNEL (parallelism)
//      Batched calls (fft_r2c_batch / fft_c2r_batch) can be split across
//      backends and processed in parallel via std::async. For batches
//      smaller than the split-threshold we just send the whole thing to
//      the current best backend (parallelism overhead > work).
//
// Configuration:
//   * Modes are independent: you can enable any subset.
//   * In pure-adaptive mode (no failover), exceptions propagate as usual.
//   * In failover-only mode (no adaptive), calls always start at the
//     highest-priority backend and walk down on failure.
//   * In adaptive+failover, the routing decides which backend to PREFER,
//     but if it fails we still walk the chain.
//
// Threading:
//   * The class is THREAD-SAFE for concurrent fft_* calls.
//   * Health state is atomic. Cost samples use a mutex (rare write path).
//   * Multi-channel split uses std::async (launch::async forces a thread).
//     We do NOT release the GIL here; the Python bindings handle that.
//
// Paranoia:
//   * If ALL backends in the chain are unhealthy and we have no fallback,
//     we throw rather than corrupt the output buffer.
//   * Cost measurements use steady_clock (monotonic, no NTP jumps).
//   * Window size is bounded; old samples expire. No unbounded memory.
//   * Failover does NOT retry on the SAME backend — we don't want to
//     hammer a broken driver.
//   * Health cooldown timer starts at the moment of failure, not first
//     post-failure call.
#pragma once

#include "backend.hpp"

#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace aboba {

enum class HybridMode : std::uint8_t {
    // Try only the first healthy backend in order. Equivalent to plain
    // failover. Simplest, no cost tracking.
    StrictPriority = 0,

    // Pick the fastest backend per size class. No failover; an exception
    // propagates out. Use only when you trust all backends.
    AdaptiveNoFailover = 1,

    // Default. Pick the fastest, but on failure walk the priority chain.
    AdaptiveWithFailover = 2,
};

struct HybridChild {
    // Non-owning pointer. The Backend must outlive the HybridBackend.
    Backend*    backend = nullptr;
    // Display name (defaults to backend->name() if empty).
    std::string label;
    // Lower numbers = higher priority. Ties broken by adaptive cost.
    int priority = 100;
    // If true, this backend is allowed for adaptive routing. If false,
    // it's a pure failover target (e.g. CPU fallback for a GPU stack).
    bool eligible_for_adaptive = true;
};

struct HybridBackendConfig {
    HybridMode mode = HybridMode::AdaptiveWithFailover;

    // How many recent latency samples to average per (backend, size class).
    std::size_t cost_window = 32;

    // How long an unhealthy backend stays out before we retry it.
    std::chrono::milliseconds health_cooldown{10000};

    // Multi-channel: batches with >= this many frames may be split across
    // backends. Below this they go to a single backend (avoid thread
    // launch overhead).
    std::size_t split_batch_threshold = 32;

    // If non-zero, max number of concurrent worker threads for batch
    // splitting. 0 = use all eligible backends.
    int max_parallel_workers = 0;

    // When debug logging is enabled, scheduler decisions print to stderr.
    bool log_decisions = false;
};

struct HybridStats {
    std::vector<std::string>    backend_names;
    std::vector<std::uint64_t>  calls_routed;       // total calls assigned
    std::vector<std::uint64_t>  calls_completed;    // successful
    std::vector<std::uint64_t>  calls_failed;       // threw exception
    std::vector<bool>           healthy;            // current state
    std::vector<float>          avg_latency_us;     // overall (across sizes)
    std::uint64_t failover_invocations = 0;         // times we walked the chain
    std::uint64_t total_calls = 0;
};

class HybridBackend : public Backend {
public:
    HybridBackend(std::vector<HybridChild> children,
                  HybridBackendConfig cfg = {});

    void fft_r2c(const float* input, std::complex<float>* output,
                 std::size_t fft_size) override;
    void fft_c2r(const std::complex<float>* input, float* output,
                 std::size_t fft_size) override;
    void fft_r2c_batch(const float* input, std::complex<float>* output,
                       std::size_t fft_size, std::size_t batch) override;
    void fft_c2r_batch(const std::complex<float>* input, float* output,
                       std::size_t fft_size, std::size_t batch) override;

    BackendType type() const override { return BackendType::HIP; }  // claim GPU-ish
    const char* name() const override { return "Hybrid"; }

    HybridStats stats() const;

    // Force a backend's health flag. Useful for tests, also for users
    // who want to drain a backend before maintenance.
    void set_backend_healthy(std::size_t idx, bool healthy);

    // Convenience: number of children
    std::size_t num_children() const noexcept { return children_.size(); }

private:
    // FFT-size buckets: powers of two from 64 to 65536 -> 11 buckets. Larger
    // sizes go in the last bucket.
    static constexpr std::size_t kNumSizeBuckets = 11;
    static std::size_t size_bucket(std::size_t fft_size) noexcept;

    struct PerChild {
        HybridChild meta;
        // Per-bucket rolling latency averages (us). NaN = no data.
        std::array<std::atomic<float>, kNumSizeBuckets> avg_us;
        std::array<std::atomic<std::uint32_t>, kNumSizeBuckets> sample_count;
        std::atomic<bool>           healthy;
        std::atomic<std::chrono::steady_clock::time_point::rep> unhealthy_since;
        std::atomic<std::uint64_t>  calls_routed;
        std::atomic<std::uint64_t>  calls_completed;
        std::atomic<std::uint64_t>  calls_failed;

        PerChild();
        // Copy/move are non-trivial because of atomics; we'll just construct
        // in place and use indices.
        PerChild(const PerChild&) = delete;
        PerChild& operator=(const PerChild&) = delete;
    };

    // Pick the preferred backend index for a call of `fft_size`. Returns
    // -1 if none are eligible/healthy.
    int pick_preferred(std::size_t fft_size);

    // Build an ordered list of backend indices to try, starting with the
    // preferred and continuing down the priority chain. Excludes
    // unhealthy backends unless ALL are unhealthy (in which case we try
    // them all anyway as a last resort).
    void build_try_order(std::size_t fft_size, std::vector<int>& out);

    // Check health cooldown — if an unhealthy backend's cooldown has
    // expired, mark it healthy. Called from every fft_*.
    void refresh_health();

    void record_latency(std::size_t child_idx, std::size_t fft_size,
                        float latency_us);
    void record_failure(std::size_t child_idx);

    // Dispatch helpers — call the actual backend method.
    enum class Op { R2C, C2R, R2C_BATCH, C2R_BATCH };
    void invoke(std::size_t child_idx, Op op,
                const void* input, void* output,
                std::size_t fft_size, std::size_t batch);

    // Try the ordered chain. Throws if all backends fail.
    void dispatch_chain(Op op, const void* input, void* output,
                        std::size_t fft_size, std::size_t batch);

    HybridBackendConfig cfg_;
    std::vector<std::unique_ptr<PerChild>> children_;
    std::atomic<std::uint64_t> failover_invocations_ {0};
    std::atomic<std::uint64_t> total_calls_ {0};
    mutable std::mutex log_mtx_;  // serialize debug log output only
};

}  // namespace aboba
