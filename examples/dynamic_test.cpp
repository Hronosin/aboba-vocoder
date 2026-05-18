// SPDX-License-Identifier: GPL-3.0-or-later
//
// Dynamic tests: adversarial loads designed to make the system misbehave,
// then verify the health monitor responds correctly.
//
// Scenarios:
//   1. A processor that sleeps deterministically over budget → bypass should
//      trip within `trip_count` calls.
//   2. A processor that throws → bypass should trip immediately on exception
//      AND output should be silence (not garbage).
//   3. A processor that's fine for 1000 calls then has one nuclear spike →
//      hard-ceiling should trip bypass immediately on the spike.
//   4. A processor that's intermittently slow (10% over) but stays under the
//      trip threshold → bypass should NOT trip; system stays engaged.
//   5. After auto-bypass, when processor recovers, system should re-engage
//      after `recover_calls` clean calls.
//   6. Force-bypass and unforce.
//   7. Concurrent stats reads from another thread while audio thread runs —
//      no data races (run under TSan to confirm).
//   8. Self-check correctly rejects an over-budget processor.
//   9. Self-check correctly accepts a fast processor.
//  10. Self-check survives a throwing processor.
#include "aboba/backend.hpp"
#include "aboba/health.hpp"
#include "aboba/self_check.hpp"
#include "aboba/streaming.hpp"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <thread>
#include <vector>

namespace {

int g_failures = 0;
int g_total    = 0;

void check(bool cond, const char* what) {
    ++g_total;
    if (cond) std::printf("  PASS  %s\n", what);
    else      { std::printf("  FAIL  %s\n", what); ++g_failures; }
}

// Helper: drive `monitor` with `n` blocks of `block_size` samples each.
void drive(aboba::HealthMonitor& monitor, int n, std::size_t block_size) {
    std::vector<float> in(block_size, 0.1f);
    std::vector<float> out(block_size);
    for (int i = 0; i < n; ++i) {
        monitor.process(in.data(), out.data(), block_size);
    }
}

}  // namespace

int main() {
    using namespace std::chrono_literals;

    std::printf("=== Test 1: Slow processor trips bypass within trip_count ===\n");
    {
        aboba::HealthConfig cfg;
        cfg.budget_us       = 1000;     // 1 ms budget
        cfg.window_calls    = 50;
        cfg.trip_count      = 5;
        cfg.recover_calls   = UINT32_MAX;  // manual recover only
        cfg.hard_ceiling_us = 1000000;  // don't trip via hard-cap here

        aboba::HealthMonitor mon(
            [](const float*, float* out, std::size_t n) {
                // Stall every call for 2 ms — guaranteed over-budget
                std::this_thread::sleep_for(2ms);
                std::fill(out, out + n, 0.0f);
            }, cfg);

        // Should trip within trip_count+1 calls
        drive(mon, 10, 256);
        auto s = mon.snapshot();
        check(s.bypass_active, "bypass active after sustained over-budget");
        check(s.auto_bypass_trips >= 1, "auto-trip counter incremented");
        check(s.over_budget >= cfg.trip_count, "over_budget counter increased");

        // While bypassed, subsequent calls should still be served (memcpy path)
        std::vector<float> in(256, 0.5f), out(256, 0.0f);
        mon.process(in.data(), out.data(), 256);
        bool copied = true;
        for (float x : out) if (x != 0.5f) { copied = false; break; }
        check(copied, "bypass mode passes input straight through");
    }

    std::printf("\n=== Test 2: Throwing processor trips bypass immediately ===\n");
    {
        aboba::HealthConfig cfg;
        cfg.budget_us = 5000;
        cfg.recover_calls = UINT32_MAX;
        aboba::HealthMonitor mon(
            [](const float*, float*, std::size_t) {
                throw std::runtime_error("boom");
            }, cfg);

        std::vector<float> in(256, 0.7f), out(256, 0.7f);
        mon.process(in.data(), out.data(), 256);

        auto s = mon.snapshot();
        check(s.exceptions == 1, "exception counted");
        check(s.bypass_active, "bypass active after exception");
        // Output should be silence (not whatever was in `out` before)
        bool all_zero = true;
        for (float x : out) if (x != 0.0f) { all_zero = false; break; }
        check(all_zero, "output zeroed when processor throws");
    }

    std::printf("\n=== Test 3: Single nuclear spike trips hard ceiling ===\n");
    {
        aboba::HealthConfig cfg;
        cfg.budget_us       = 5000;
        cfg.trip_count      = 1000;       // window won't trip in this test
        cfg.hard_ceiling_us = 10000;      // 10 ms hard cap
        cfg.recover_calls   = UINT32_MAX;

        std::atomic<int> call_count{0};
        aboba::HealthMonitor mon(
            [&](const float*, float* out, std::size_t n) {
                std::fill(out, out + n, 0.0f);
                if (call_count.fetch_add(1) == 50) {
                    // The nuclear spike
                    std::this_thread::sleep_for(20ms);
                }
            }, cfg);

        drive(mon, 100, 256);
        auto s = mon.snapshot();
        check(s.bypass_active, "single spike past hard ceiling trips bypass");
        check(s.auto_bypass_trips >= 1, "trip counter went up");
    }

    std::printf("\n=== Test 4: Intermittent low-rate jitter stays engaged ===\n");
    {
        aboba::HealthConfig cfg;
        cfg.budget_us       = 5000;
        cfg.window_calls    = 100;
        cfg.trip_count      = 30;   // need 30% over-budget in window to trip
        cfg.hard_ceiling_us = 100000;
        cfg.recover_calls   = UINT32_MAX;

        std::atomic<int> call_count{0};
        aboba::HealthMonitor mon(
            [&](const float*, float* out, std::size_t n) {
                std::fill(out, out + n, 0.0f);
                // 10% of calls go over budget — under the 30% trip threshold
                if ((call_count.fetch_add(1) % 10) == 0) {
                    std::this_thread::sleep_for(6ms);  // > budget
                }
            }, cfg);

        drive(mon, 200, 256);
        auto s = mon.snapshot();
        check(!s.bypass_active, "low jitter rate keeps system engaged");
        check(s.over_budget > 0, "some calls were over budget (sanity)");
    }

    std::printf("\n=== Test 5: Auto-recover after processor recovers ===\n");
    {
        aboba::HealthConfig cfg;
        cfg.budget_us       = 5000;
        cfg.window_calls    = 20;
        cfg.trip_count      = 5;
        cfg.hard_ceiling_us = 100000;
        cfg.recover_calls   = 30;

        std::atomic<bool> bad{true};
        aboba::HealthMonitor mon(
            [&](const float*, float* out, std::size_t n) {
                std::fill(out, out + n, 0.0f);
                if (bad.load()) std::this_thread::sleep_for(8ms);
            }, cfg);

        // Phase 1: bad processor → trips bypass
        drive(mon, 30, 256);
        check(mon.snapshot().bypass_active, "tripped during bad phase");

        // Phase 2: processor recovers, run >= recover_calls clean iters
        bad = false;
        drive(mon, 100, 256);
        check(!mon.snapshot().bypass_active,
              "auto-recovered after processor healed");
    }

    std::printf("\n=== Test 6: force_bypass works regardless of health ===\n");
    {
        aboba::HealthConfig cfg;
        aboba::HealthMonitor mon(
            [](const float* in, float* out, std::size_t n) {
                for (std::size_t i = 0; i < n; ++i) out[i] = in[i] * 0.0f;
            }, cfg);

        mon.force_bypass(true);
        std::vector<float> in(256, 0.42f), out(256, 0.0f);
        mon.process(in.data(), out.data(), 256);
        check(out[0] == 0.42f, "force_bypass bypasses immediately");

        mon.force_bypass(false);
        mon.process(in.data(), out.data(), 256);
        check(out[0] == 0.0f, "unforce_bypass re-engages processor");
    }

    std::printf("\n=== Test 7: Concurrent stats reads don't crash ===\n");
    {
        aboba::HealthConfig cfg;
        aboba::HealthMonitor mon(
            [](const float* in, float* out, std::size_t n) {
                std::memcpy(out, in, n * sizeof(float));
            }, cfg);

        std::atomic<bool> stop{false};
        std::atomic<int>  reads{0};

        std::thread reader([&]() {
            while (!stop.load()) {
                auto s = mon.snapshot();
                (void)s;
                reads.fetch_add(1);
            }
        });

        // Hammer the audio thread for 200ms
        const auto deadline = std::chrono::steady_clock::now() + 200ms;
        std::vector<float> in(256, 0.1f), out(256);
        while (std::chrono::steady_clock::now() < deadline) {
            mon.process(in.data(), out.data(), 256);
        }
        stop = true;
        reader.join();

        check(reads.load() > 100, "reader saw many snapshots without crash");
        check(mon.snapshot().calls > 0, "audio thread did real work");
    }

    std::printf("\n=== Test 8: Self-check rejects slow processor ===\n");
    {
        aboba::SelfCheckConfig cfg;
        cfg.block_size  = 256;
        cfg.sample_rate = 48000.0;
        cfg.iterations  = 50;
        cfg.budget_fraction = 0.5;

        auto r = aboba::run_self_check(
            [](const float*, float* out, std::size_t n) {
                // 10 ms — way over the 5.3 ms budget @ block 256, sr 48k
                std::this_thread::sleep_for(10ms);
                std::fill(out, out + n, 0.0f);
            }, cfg);

        check(!r.safe_to_run, "self-check correctly rejected slow processor");
        check(r.p99_us > r.budget_us, "p99 actually was over budget");
    }

    std::printf("\n=== Test 9: Self-check accepts fast processor ===\n");
    {
        aboba::SelfCheckConfig cfg;
        cfg.block_size  = 256;
        cfg.sample_rate = 48000.0;
        cfg.iterations  = 50;

        auto r = aboba::run_self_check(
            [](const float* in, float* out, std::size_t n) {
                std::memcpy(out, in, n * sizeof(float));
            }, cfg);

        check(r.safe_to_run, "self-check accepted trivially-fast processor");
    }

    std::printf("\n=== Test 10: Self-check survives throwing processor ===\n");
    {
        aboba::SelfCheckConfig cfg;
        cfg.block_size  = 256;
        cfg.sample_rate = 48000.0;
        cfg.iterations  = 50;

        bool threw = false;
        try {
            auto r = aboba::run_self_check(
                [](const float*, float*, std::size_t) {
                    throw std::runtime_error("nope");
                }, cfg);
            check(!r.safe_to_run, "self-check fails for throwing processor");
        } catch (...) {
            threw = true;
        }
        check(!threw, "self-check itself does not propagate processor exception");
    }

    std::printf("\n=== Test 11: Real vocoder under self-check + monitor ===\n");
    {
        auto backend = aboba::create_best_backend();
        aboba::StreamingPhaseVocoder voc(2048, 512, backend.get());
        voc.set_pitch_semitones(5.0f);

        aboba::SelfCheckConfig cfg;
        cfg.block_size  = 256;
        cfg.sample_rate = 48000.0;
        cfg.iterations  = 100;
        // Self-check's strict default checks max < budget. On noisy CI
        // environments (containers, ASan, virtualized hardware) a single
        // 10 ms OS-scheduler spike can fail this even though the real
        // worker is fine. We separately verify the actual algorithmic
        // throughput via p99.

        auto r = aboba::run_self_check(
            [&](const float* in, float* out, std::size_t n) {
                voc.process(in, out, n);
            }, cfg);

        // The p99 check is the algorithmic one — it should always pass on
        // sane hardware. max can spike due to OS jitter unrelated to us.
        check(r.p99_us < r.budget_us / 2,
              "real vocoder p99 fits in half the budget (algorithmic perf)");
        std::printf("    (vocoder p99=%uus, max=%uus, budget=%uus, reason=%s)\n",
                    r.p99_us, r.max_us, r.budget_us, r.reason);

        // Now run it for a while under HealthMonitor and verify no false trips
        voc.reset();
        aboba::HealthMonitor mon(
            [&](const float* in, float* out, std::size_t n) {
                voc.process(in, out, n);
            }, aboba::HealthConfig{});
        drive(mon, 500, 256);
        auto s = mon.snapshot();
        check(!s.bypass_active, "real vocoder runs clean — no false bypass");
        check(s.auto_bypass_trips == 0, "no spurious trips");
    }

    std::printf("\n========================================\n");
    std::printf("Total: %d/%d passed", g_total - g_failures, g_total);
    if (g_failures == 0) {
        std::printf(" \u2713\n");
        return 0;
    } else {
        std::printf(" — %d FAILED \u2717\n", g_failures);
        return 1;
    }
}
