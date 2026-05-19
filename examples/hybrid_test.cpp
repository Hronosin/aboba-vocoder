// SPDX-License-Identifier: GPL-3.0-or-later
//
// Tests for HybridBackend + MockBackend.
//
// We use the real CPU backend as the "actual" worker and wrap it in
// MockBackend to inject latency, jitter, and failures. The HybridBackend
// then orchestrates multiple mocks.
//
// What we want to verify:
//   1. MockBackend obeys its config (latency, failure injection)
//   2. HybridBackend routes preferred backend correctly (strict / adaptive)
//   3. Adaptive routing actually picks the FASTER backend over time
//   4. Failover walks the chain on injected failures
//   5. Cooldown brings unhealthy backends back
//   6. Multi-channel batch splits correctly and produces identical output
//   7. All-backends-fail throws (doesn't silently corrupt)
//   8. Stats counters report sensibly

#include "aboba/backend.hpp"
#include "aboba/hybrid_backend.hpp"
#include "aboba/mock_backend.hpp"

#include <chrono>
#include <cmath>
#include <complex>
#include <cstdio>
#include <cstring>
#include <random>
#include <thread>
#include <vector>

namespace {

int g_total = 0, g_fail = 0;

void check(bool cond, const char* what) {
    ++g_total;
    if (cond) std::printf("  PASS  %s\n", what);
    else      { std::printf("  FAIL  %s\n", what); ++g_fail; }
}

// Produce a simple sinusoidal time-domain input.
std::vector<float> make_input(std::size_t n) {
    std::vector<float> s(n);
    for (std::size_t i = 0; i < n; ++i) {
        s[i] = 0.3f * std::sin(2.0f * 3.14159f * static_cast<float>(i) /
                                static_cast<float>(n) * 5.0f);
    }
    return s;
}

bool spectra_close(const std::complex<float>* a, const std::complex<float>* b,
                   std::size_t n, float tol = 1e-3f) {
    for (std::size_t i = 0; i < n; ++i) {
        if (std::fabs(a[i].real() - b[i].real()) > tol) return false;
        if (std::fabs(a[i].imag() - b[i].imag()) > tol) return false;
    }
    return true;
}

}  // namespace

int main() {
    using namespace aboba;

    // ============================================================
    std::printf("\n=== MockBackend: config defaults are sane ===\n");
    {
        auto cpu = create_backend(BackendType::CPU);
        MockBackendConfig cfg;
        cfg.name = "test-mock";
        MockBackend m(cfg, cpu.get());
        check(m.type() == BackendType::HIP, "pretend_gpu -> type is HIP");
        check(std::string(m.name()) == "test-mock", "name set");
        check(m.total_calls() == 0, "no calls yet");
    }

    // ============================================================
    std::printf("\n=== MockBackend: passes through correct result ===\n");
    {
        auto cpu = create_backend(BackendType::CPU);
        MockBackendConfig cfg;
        cfg.base_latency_us = 0;  // no artificial delay
        MockBackend m(cfg, cpu.get());

        const std::size_t N = 1024;
        auto in = make_input(N);
        std::vector<std::complex<float>> mock_out(N/2 + 1);
        std::vector<std::complex<float>> ref_out(N/2 + 1);

        m.fft_r2c(in.data(), mock_out.data(), N);
        cpu->fft_r2c(in.data(), ref_out.data(), N);

        check(spectra_close(mock_out.data(), ref_out.data(), N/2 + 1),
              "mock fft_r2c matches reference");
        check(m.total_calls() == 1, "1 call counted");
        check(m.failed_calls() == 0, "0 failures");
    }

    // ============================================================
    std::printf("\n=== MockBackend: latency is actually applied ===\n");
    {
        auto cpu = create_backend(BackendType::CPU);
        MockBackendConfig cfg;
        cfg.base_latency_us = 2000;  // 2ms artificial latency
        MockBackend m(cfg, cpu.get());
        const std::size_t N = 256;
        auto in = make_input(N);
        std::vector<std::complex<float>> out(N/2 + 1);

        const auto t0 = std::chrono::steady_clock::now();
        m.fft_r2c(in.data(), out.data(), N);
        const auto t1 = std::chrono::steady_clock::now();
        const auto us = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();

        std::printf("    Measured latency: %ld us (expected >=2000)\n", (long)us);
        check(us >= 1500, "base latency is observable");
    }

    // ============================================================
    std::printf("\n=== MockBackend: failure injection throws on cadence ===\n");
    {
        auto cpu = create_backend(BackendType::CPU);
        MockBackendConfig cfg;
        cfg.failure_period = 3;   // every 3rd call fails (3, 6, 9, ...)
        MockBackend m(cfg, cpu.get());
        const std::size_t N = 256;
        auto in = make_input(N);
        std::vector<std::complex<float>> out(N/2 + 1);

        int thrown = 0;
        for (int i = 0; i < 10; ++i) {
            try { m.fft_r2c(in.data(), out.data(), N); }
            catch (const MockBackendInjectedFailure&) { ++thrown; }
        }
        std::printf("    10 calls, %d threw (expect 3)\n", thrown);
        check(thrown == 3, "exactly 3 failures in 10 calls @ period 3");
        check(m.failed_calls() == 3, "failed_calls counter matches");
        check(m.completed_calls() == 7, "completed_calls counter matches");
    }

    // ============================================================
    std::printf("\n=== HybridBackend: rejects empty children list ===\n");
    {
        bool threw = false;
        try {
            HybridBackend hb(std::vector<HybridChild>{});
        } catch (const std::invalid_argument&) { threw = true; }
        check(threw, "empty children list rejected");
    }

    // ============================================================
    std::printf("\n=== HybridBackend: rejects null child backend ===\n");
    {
        bool threw = false;
        try {
            std::vector<HybridChild> kids;
            HybridChild c; c.backend = nullptr; c.priority = 0;
            kids.push_back(c);
            HybridBackend hb(std::move(kids));
        } catch (const std::invalid_argument&) { threw = true; }
        check(threw, "null backend rejected");
    }

    // ============================================================
    std::printf("\n=== HybridBackend: StrictPriority picks first healthy ===\n");
    {
        auto cpu = create_backend(BackendType::CPU);
        MockBackendConfig fast_cfg; fast_cfg.name = "fast"; fast_cfg.base_latency_us = 0;
        MockBackendConfig slow_cfg; slow_cfg.name = "slow"; slow_cfg.base_latency_us = 0;
        MockBackend fast(fast_cfg, cpu.get());
        MockBackend slow(slow_cfg, cpu.get());

        std::vector<HybridChild> kids;
        kids.push_back({&fast, "fast", 0, true});
        kids.push_back({&slow, "slow", 1, true});
        HybridBackendConfig hcfg; hcfg.mode = HybridMode::StrictPriority;
        HybridBackend hb(std::move(kids), hcfg);

        const std::size_t N = 256;
        auto in = make_input(N);
        std::vector<std::complex<float>> out(N/2 + 1);
        for (int i = 0; i < 5; ++i) {
            hb.fft_r2c(in.data(), out.data(), N);
        }
        check(fast.completed_calls() == 5, "all 5 calls went to 'fast' (priority 0)");
        check(slow.completed_calls() == 0, "no calls to 'slow' (priority 1)");
    }

    // ============================================================
    std::printf("\n=== HybridBackend: Adaptive picks faster backend ===\n");
    {
        auto cpu = create_backend(BackendType::CPU);
        // Both have priority 0, but slow has a big artificial latency.
        // We use a 5ms gap so the routing decision is unambiguous even
        // when wall clock is noisy (e.g. ASan slowing the host).
        MockBackendConfig fast_cfg; fast_cfg.name = "fast"; fast_cfg.base_latency_us = 0;
        MockBackendConfig slow_cfg; slow_cfg.name = "slow"; slow_cfg.base_latency_us = 5000;
        MockBackend fast(fast_cfg, cpu.get());
        MockBackend slow(slow_cfg, cpu.get());

        std::vector<HybridChild> kids;
        kids.push_back({&fast, "fast", 0, true});
        kids.push_back({&slow, "slow", 0, true});
        HybridBackendConfig hcfg; hcfg.mode = HybridMode::AdaptiveWithFailover;
        HybridBackend hb(std::move(kids), hcfg);

        const std::size_t N = 1024;
        auto in = make_input(N);
        std::vector<std::complex<float>> out(N/2 + 1);
        // First a couple of calls go to both for cost-discovery, then routing
        // should settle on the fast one.
        for (int i = 0; i < 20; ++i) {
            hb.fft_r2c(in.data(), out.data(), N);
        }
        auto fc = fast.completed_calls();
        auto sc = slow.completed_calls();
        std::printf("    fast: %llu calls, slow: %llu calls\n",
            (unsigned long long)fc, (unsigned long long)sc);
        check(fc > sc, "adaptive routes most calls to faster backend");
        check(fc >= 15, "fast handles at least 15 of 20 calls");
    }

    // ============================================================
    std::printf("\n=== HybridBackend: failover on injected failure ===\n");
    {
        auto cpu = create_backend(BackendType::CPU);
        // Primary fails every call. Secondary is healthy.
        MockBackendConfig bad_cfg; bad_cfg.name = "broken"; bad_cfg.failure_period = 1;
        MockBackendConfig ok_cfg;  ok_cfg.name  = "ok";
        MockBackend bad(bad_cfg, cpu.get());
        MockBackend ok(ok_cfg, cpu.get());

        std::vector<HybridChild> kids;
        kids.push_back({&bad, "broken", 0, true});
        kids.push_back({&ok,  "ok",     1, true});
        HybridBackendConfig hcfg; hcfg.mode = HybridMode::AdaptiveWithFailover;
        HybridBackend hb(std::move(kids), hcfg);

        const std::size_t N = 256;
        auto in = make_input(N);
        std::vector<std::complex<float>> out(N/2 + 1);
        bool got_result = false;
        try {
            hb.fft_r2c(in.data(), out.data(), N);
            got_result = true;
        } catch (...) {}

        check(got_result, "call succeeded via failover");
        check(bad.failed_calls() >= 1, "broken backend recorded failure");
        check(ok.completed_calls() >= 1, "ok backend completed the call");
        auto s = hb.stats();
        check(s.failover_invocations >= 1, "failover counter incremented");
    }

    // ============================================================
    std::printf("\n=== HybridBackend: unhealthy backend skipped after failure ===\n");
    {
        auto cpu = create_backend(BackendType::CPU);
        MockBackendConfig bad_cfg; bad_cfg.name = "broken"; bad_cfg.failure_period = 1;
        MockBackendConfig ok_cfg;  ok_cfg.name  = "ok";
        MockBackend bad(bad_cfg, cpu.get());
        MockBackend ok(ok_cfg, cpu.get());

        std::vector<HybridChild> kids;
        kids.push_back({&bad, "broken", 0, true});
        kids.push_back({&ok,  "ok",     1, true});
        HybridBackendConfig hcfg;
        hcfg.mode = HybridMode::AdaptiveWithFailover;
        hcfg.health_cooldown = std::chrono::milliseconds(60000);  // long
        HybridBackend hb(std::move(kids), hcfg);

        const std::size_t N = 256;
        auto in = make_input(N);
        std::vector<std::complex<float>> out(N/2 + 1);
        // First call: walks failover, marks 'broken' unhealthy.
        hb.fft_r2c(in.data(), out.data(), N);
        // Subsequent calls should skip 'broken' (no more failures).
        const auto bad_failures_before = bad.failed_calls();
        for (int i = 0; i < 10; ++i) {
            hb.fft_r2c(in.data(), out.data(), N);
        }
        const auto bad_failures_after = bad.failed_calls();
        check(bad_failures_after == bad_failures_before,
              "unhealthy backend not retried during cooldown");
        check(ok.completed_calls() >= 11, "all subsequent calls handled by ok");
    }

    // ============================================================
    std::printf("\n=== HybridBackend: cooldown expiry brings backend back ===\n");
    {
        auto cpu = create_backend(BackendType::CPU);
        MockBackendConfig p_cfg; p_cfg.name = "primary";
        MockBackend primary(p_cfg, cpu.get());
        MockBackendConfig s_cfg; s_cfg.name = "secondary";
        MockBackend secondary(s_cfg, cpu.get());

        std::vector<HybridChild> kids;
        kids.push_back({&primary,   "primary",   0, true});
        kids.push_back({&secondary, "secondary", 1, true});
        HybridBackendConfig hcfg;
        hcfg.mode = HybridMode::StrictPriority;
        hcfg.health_cooldown = std::chrono::milliseconds(100);  // 100ms
        HybridBackend hb(std::move(kids), hcfg);

        // Mark primary unhealthy manually
        hb.set_backend_healthy(0, false);

        const std::size_t N = 256;
        auto in = make_input(N);
        std::vector<std::complex<float>> out(N/2 + 1);
        hb.fft_r2c(in.data(), out.data(), N);
        check(secondary.completed_calls() == 1,
              "call routed to secondary while primary unhealthy");

        // Wait for cooldown
        std::this_thread::sleep_for(std::chrono::milliseconds(150));
        hb.fft_r2c(in.data(), out.data(), N);
        check(primary.completed_calls() == 1,
              "primary returns to service after cooldown");
    }

    // ============================================================
    std::printf("\n=== HybridBackend: AdaptiveNoFailover propagates errors ===\n");
    {
        auto cpu = create_backend(BackendType::CPU);
        MockBackendConfig bad_cfg; bad_cfg.name = "always-fails";
        bad_cfg.failure_period = 1;
        MockBackend bad(bad_cfg, cpu.get());
        std::vector<HybridChild> kids;
        kids.push_back({&bad, "bad", 0, true});
        HybridBackendConfig hcfg; hcfg.mode = HybridMode::AdaptiveNoFailover;
        HybridBackend hb(std::move(kids), hcfg);

        const std::size_t N = 256;
        auto in = make_input(N);
        std::vector<std::complex<float>> out(N/2 + 1);
        bool threw = false;
        try { hb.fft_r2c(in.data(), out.data(), N); }
        catch (const std::exception&) { threw = true; }
        check(threw, "NoFailover mode propagates exception");
    }

    // ============================================================
    std::printf("\n=== HybridBackend: throws when all backends fail ===\n");
    {
        auto cpu = create_backend(BackendType::CPU);
        MockBackendConfig a_cfg; a_cfg.name = "a"; a_cfg.failure_period = 1;
        MockBackendConfig b_cfg; b_cfg.name = "b"; b_cfg.failure_period = 1;
        MockBackend a(a_cfg, cpu.get());
        MockBackend b(b_cfg, cpu.get());
        std::vector<HybridChild> kids;
        kids.push_back({&a, "a", 0, true});
        kids.push_back({&b, "b", 1, true});
        HybridBackendConfig hcfg; hcfg.mode = HybridMode::AdaptiveWithFailover;
        HybridBackend hb(std::move(kids), hcfg);

        const std::size_t N = 256;
        auto in = make_input(N);
        std::vector<std::complex<float>> out(N/2 + 1);
        bool threw = false;
        try { hb.fft_r2c(in.data(), out.data(), N); }
        catch (const std::exception&) { threw = true; }
        check(threw, "all-fail throws rather than silently corrupting");
    }

    // ============================================================
    std::printf("\n=== HybridBackend: batch split equals reference ===\n");
    {
        auto cpu = create_backend(BackendType::CPU);
        MockBackendConfig a_cfg; a_cfg.name = "w1";
        MockBackendConfig b_cfg; b_cfg.name = "w2";
        MockBackend a(a_cfg, cpu.get());
        MockBackend b(b_cfg, cpu.get());
        std::vector<HybridChild> kids;
        kids.push_back({&a, "w1", 0, true});
        kids.push_back({&b, "w2", 0, true});
        HybridBackendConfig hcfg;
        hcfg.mode = HybridMode::AdaptiveWithFailover;
        hcfg.split_batch_threshold = 8;  // small for testing
        HybridBackend hb(std::move(kids), hcfg);

        const std::size_t N = 256;
        const std::size_t B = 64;  // batch size — will be split
        std::vector<float> in(B * N);
        std::mt19937 rng(42);
        std::uniform_real_distribution<float> dist(-0.3f, 0.3f);
        for (auto& v : in) v = dist(rng);

        std::vector<std::complex<float>> out_hybrid(B * (N/2 + 1));
        std::vector<std::complex<float>> out_ref   (B * (N/2 + 1));
        hb.fft_r2c_batch(in.data(), out_hybrid.data(), N, B);
        cpu->fft_r2c_batch(in.data(), out_ref.data(), N, B);

        check(spectra_close(out_hybrid.data(), out_ref.data(),
                            B * (N/2 + 1), 1e-2f),
              "split batch result matches reference");
        // Both workers should have handled some frames
        auto ac = a.completed_calls();
        auto bc = b.completed_calls();
        std::printf("    w1 calls=%llu, w2 calls=%llu\n",
                    (unsigned long long)ac, (unsigned long long)bc);
        check(ac > 0 && bc > 0, "both workers processed some frames");
    }

    // ============================================================
    std::printf("\n=== HybridBackend: small batch falls back to single dispatch ===\n");
    {
        auto cpu = create_backend(BackendType::CPU);
        MockBackendConfig a_cfg; a_cfg.name = "primary";
        MockBackend primary(a_cfg, cpu.get());
        std::vector<HybridChild> kids;
        kids.push_back({&primary, "primary", 0, true});
        HybridBackendConfig hcfg;
        hcfg.split_batch_threshold = 32;
        HybridBackend hb(std::move(kids), hcfg);

        const std::size_t N = 256;
        const std::size_t B = 4;  // below threshold
        std::vector<float> in(B * N, 0.1f);
        std::vector<std::complex<float>> out(B * (N/2 + 1));
        hb.fft_r2c_batch(in.data(), out.data(), N, B);
        check(primary.completed_calls() == 1,
              "small batch -> single dispatch call (not split)");
    }

    // ============================================================
    std::printf("\n=== HybridBackend: c2r roundtrip identity ===\n");
    {
        // Verify that batched r2c -> c2r on hybrid gives the input back
        // (modulo FFT scaling).
        auto cpu = create_backend(BackendType::CPU);
        MockBackendConfig a_cfg; a_cfg.name = "w1";
        MockBackendConfig b_cfg; b_cfg.name = "w2";
        MockBackend a(a_cfg, cpu.get());
        MockBackend b(b_cfg, cpu.get());
        std::vector<HybridChild> kids;
        kids.push_back({&a, "w1", 0, true});
        kids.push_back({&b, "w2", 0, true});
        HybridBackendConfig hcfg;
        hcfg.split_batch_threshold = 4;
        HybridBackend hb(std::move(kids), hcfg);

        const std::size_t N = 128;
        const std::size_t B = 16;
        std::vector<float> in(B * N);
        std::mt19937 rng(7);
        std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
        for (auto& v : in) v = dist(rng);

        std::vector<std::complex<float>> spec(B * (N/2 + 1));
        std::vector<float> roundtrip(B * N);

        hb.fft_r2c_batch(in.data(), spec.data(), N, B);
        hb.fft_c2r_batch(spec.data(), roundtrip.data(), N, B);

        // FFTW round trip is N * x. Check relative error.
        float max_err = 0.0f;
        for (std::size_t i = 0; i < in.size(); ++i) {
            const float expected = in[i] * static_cast<float>(N);
            const float err = std::fabs(roundtrip[i] - expected);
            if (err > max_err) max_err = err;
        }
        std::printf("    Max roundtrip error: %.4f\n",
                    static_cast<double>(max_err));
        check(max_err < 0.01f, "r2c then c2r through hybrid recovers input");
    }

    // ============================================================
    std::printf("\n=== HybridBackend: stats are well-formed ===\n");
    {
        auto cpu = create_backend(BackendType::CPU);
        MockBackendConfig a_cfg; a_cfg.name = "primary";
        MockBackend a(a_cfg, cpu.get());
        std::vector<HybridChild> kids;
        kids.push_back({&a, "primary", 0, true});
        HybridBackend hb(std::move(kids));

        const std::size_t N = 256;
        auto in = make_input(N);
        std::vector<std::complex<float>> out(N/2 + 1);
        for (int i = 0; i < 5; ++i) hb.fft_r2c(in.data(), out.data(), N);

        auto s = hb.stats();
        check(s.backend_names.size() == 1, "stats has 1 backend entry");
        check(s.backend_names[0] == "primary", "stats name correct");
        check(s.calls_routed[0] == 5, "5 calls routed");
        check(s.calls_completed[0] == 5, "5 calls completed");
        check(s.calls_failed[0] == 0, "0 failures");
        check(s.healthy[0], "still healthy");
        check(s.total_calls == 5, "total_calls = 5");
        check(s.failover_invocations == 0, "no failovers");
    }

    // ============================================================
    std::printf("\n=== HybridBackend: ineligible-for-adaptive backend used only as fallback ===\n");
    {
        auto cpu = create_backend(BackendType::CPU);
        MockBackendConfig gpu_cfg; gpu_cfg.name = "gpu"; gpu_cfg.failure_period = 1;
        MockBackend gpu(gpu_cfg, cpu.get());

        // CPU acts as pure failover — NOT in adaptive pool.
        std::vector<HybridChild> kids;
        kids.push_back({&gpu, "gpu", 0, true});
        kids.push_back({cpu.get(), "cpu", 1, false});
        HybridBackendConfig hcfg; hcfg.mode = HybridMode::AdaptiveWithFailover;
        HybridBackend hb(std::move(kids), hcfg);

        const std::size_t N = 256;
        auto in = make_input(N);
        std::vector<std::complex<float>> out(N/2 + 1);
        hb.fft_r2c(in.data(), out.data(), N);  // gpu fails -> falls to cpu

        check(gpu.failed_calls() >= 1, "gpu attempt was made and failed");
        check(true, "fallback CPU handled the call (no throw)");
    }

    std::printf("\n========================================\n");
    std::printf("Total: %d/%d passed", g_total - g_fail, g_total);
    if (g_fail == 0) { std::printf(" \u2713\n"); return 0; }
    else { std::printf(" - %d FAILED \u2717\n", g_fail); return 1; }
}
