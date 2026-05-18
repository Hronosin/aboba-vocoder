// SPDX-License-Identifier: GPL-3.0-or-later
#include "aboba/self_check.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

namespace aboba {

namespace {

constexpr float kPi = 3.14159265358979323846f;

std::uint32_t percentile(const std::vector<std::uint32_t>& sorted,
                         double p) {
    if (sorted.empty()) return 0;
    const std::size_t n = sorted.size();
    // "Type-7" percentile (R default). Clamp.
    const double h = (static_cast<double>(n) - 1.0) * p;
    const std::size_t lo = static_cast<std::size_t>(std::floor(h));
    const std::size_t hi = std::min(lo + 1, n - 1);
    const double frac = h - static_cast<double>(lo);
    const double v = static_cast<double>(sorted[lo]) * (1.0 - frac)
                   + static_cast<double>(sorted[hi]) * frac;
    return static_cast<std::uint32_t>(v + 0.5);
}

}  // namespace

SelfCheckResult run_self_check(SelfCheckProcessor processor,
                               SelfCheckConfig    cfg) {
    SelfCheckResult r;

    if (!processor) {
        r.reason = "processor is null";
        return r;
    }
    if (cfg.block_size == 0 || cfg.sample_rate <= 0.0) {
        r.reason = "invalid block size or sample rate";
        return r;
    }

    const std::uint32_t iters = std::min(cfg.iterations, cfg.max_iterations);
    if (iters < 10) {
        r.reason = "iterations too low (need >= 10)";
        return r;
    }

    // Budget: (block_size / sample_rate) * 1e6 microseconds.
    const double budget_seconds = static_cast<double>(cfg.block_size)
                                / cfg.sample_rate;
    const std::uint32_t budget_us =
        static_cast<std::uint32_t>(budget_seconds * 1e6);
    r.budget_us = budget_us;

    // Generate a mid-energy test signal: voice-like fundamental + harmonics
    // + tiny noise. Avoid pure silence (which would hit the silence fast-
    // path) and avoid white noise alone (no spectral structure to phase-
    // track).
    std::vector<float> in_buf(cfg.block_size);
    std::vector<float> out_buf(cfg.block_size);
    std::mt19937 rng(12345);
    std::normal_distribution<float> noise(0.0f, 0.005f);

    std::vector<std::uint32_t> samples;
    samples.reserve(iters);

    // Warm-up: run a few iterations before measuring. First calls hit cold
    // FFTW plans, cold caches, etc. We don't include them in stats.
    constexpr std::uint32_t kWarmup = 5;
    for (std::uint32_t k = 0; k < iters + kWarmup; ++k) {
        // Refresh input every iteration (different phases)
        const float phase_off = static_cast<float>(k) * 0.13f;
        for (std::size_t i = 0; i < cfg.block_size; ++i) {
            const float t = (static_cast<float>(i) + phase_off)
                          / static_cast<float>(cfg.sample_rate);
            in_buf[i] = 0.2f * std::sin(2.0f * kPi * 180.0f * t)
                      + 0.1f * std::sin(2.0f * kPi * 360.0f * t)
                      + 0.05f* std::sin(2.0f * kPi * 720.0f * t)
                      + noise(rng);
        }

        const auto t0 = std::chrono::steady_clock::now();
        try {
            processor(in_buf.data(), out_buf.data(), cfg.block_size);
        } catch (...) {
            r.reason = "processor threw during self-check";
            return r;
        }
        const auto t1 = std::chrono::steady_clock::now();

        if (k >= kWarmup) {
            const auto us =
                std::chrono::duration_cast<std::chrono::microseconds>(
                    t1 - t0).count();
            samples.push_back(static_cast<std::uint32_t>(
                std::max<long long>(us, 0)));
        }
    }

    std::sort(samples.begin(), samples.end());
    r.iterations = static_cast<std::uint32_t>(samples.size());
    r.min_us     = samples.front();
    r.max_us     = samples.back();
    r.median_us  = percentile(samples, 0.50);
    r.p95_us     = percentile(samples, 0.95);
    r.p99_us     = percentile(samples, 0.99);

    // Verdict: p99 must be under (budget * fraction).
    const std::uint32_t allowed_us =
        static_cast<std::uint32_t>(static_cast<double>(budget_us)
                                   * cfg.budget_fraction);

    if (allowed_us == 0) {
        r.reason = "computed budget is zero";
        return r;
    }
    if (r.p99_us > allowed_us) {
        r.reason = "p99 latency exceeds safe fraction of budget";
        r.safe_to_run = false;
        return r;
    }
    if (r.max_us > budget_us) {
        // p99 was OK but a single spike blew the whole budget. Risky.
        r.reason = "max latency exceeds budget (jitter spike)";
        r.safe_to_run = false;
        return r;
    }

    r.reason = "ok";
    r.safe_to_run = true;
    return r;
}

void print_self_check(const SelfCheckResult& r) {
    std::fprintf(stderr,
        "  [self-check] %s\n"
        "    iterations : %u\n"
        "    budget_us  : %u (per call)\n"
        "    min_us     : %u\n"
        "    median_us  : %u\n"
        "    p95_us     : %u\n"
        "    p99_us     : %u\n"
        "    max_us     : %u\n"
        "    reason     : %s\n",
        r.safe_to_run ? "PASS" : "FAIL",
        r.iterations,
        r.budget_us,
        r.min_us, r.median_us, r.p95_us, r.p99_us, r.max_us,
        r.reason);
}

}  // namespace aboba
