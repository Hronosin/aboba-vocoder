// SPDX-License-Identifier: GPL-3.0-or-later
//
// Self-check: synthetic load test before going live.
//
// Before opening a real audio stream, run this on the same processor with
// the same block size and sample rate. We measure the realistic
// per-call time and compare it to the wall-clock budget. If we can't
// reliably stay under budget, the host application should refuse to start
// the audio stream rather than try and hang the system.
//
// Returns recommended action plus measured numbers so the caller can make
// an informed decision (or just check `.safe_to_run`).
#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>

namespace aboba {

struct SelfCheckResult {
    // Measured per-call times in microseconds.
    std::uint32_t min_us       = 0;
    std::uint32_t median_us    = 0;
    std::uint32_t p95_us       = 0;
    std::uint32_t p99_us       = 0;
    std::uint32_t max_us       = 0;

    // Configured budget for one call (block_size / sample_rate seconds).
    std::uint32_t budget_us    = 0;

    // Total iterations actually run.
    std::uint32_t iterations   = 0;

    // Overall verdict.
    bool safe_to_run           = false;

    // If !safe_to_run, this points at a short human-readable reason.
    // Owned by the library; do not free.
    const char* reason         = "ok";
};

struct SelfCheckConfig {
    std::size_t   block_size      = 256;
    double        sample_rate     = 48000.0;
    std::uint32_t iterations      = 200;
    // Run at most this many calls regardless of iterations, to keep startup
    // snappy even with absurd settings.
    std::uint32_t max_iterations  = 1000;
    // Wall-clock budget fraction we require (1.0 = exact budget, 0.5 = use
    // at most half the budget at p99 so there's headroom for OS jitter).
    double        budget_fraction = 0.5;
};

using SelfCheckProcessor =
    std::function<void(const float* in, float* out, std::size_t n)>;

// Drive `processor` with a synthetic mid-energy signal for some iterations,
// time each call, and decide whether we can safely run it as a real-time
// audio callback. The processor is invoked sequentially in the calling
// thread — this is intentionally *not* concurrent, because we want to
// measure the worst case for a single audio callback.
SelfCheckResult run_self_check(SelfCheckProcessor processor,
                               SelfCheckConfig    cfg);

// Pretty-print the result to stderr.
void print_self_check(const SelfCheckResult& r);

}  // namespace aboba
