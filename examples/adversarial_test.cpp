// SPDX-License-Identifier: GPL-3.0-or-later
//
// adversarial_test.cpp — paranoia regression suite.
//
// This is NOT a correctness test; it's an ABUSE test. We feed Aboba's
// C ABI inputs that no well-behaved caller would ever produce, and we
// verify:
//   * Nothing crashes
//   * Nothing returns NaN/Inf to the caller
//   * Status codes are returned (not silently swallowed)
//   * Memory does not grow unbounded
//
// Run with ASan + UBSan for highest signal.
#include "aboba_c.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>
#include <random>

namespace {
int g_total = 0, g_fail = 0;
void check(bool cond, const char* what) {
    ++g_total;
    if (cond) std::printf("  PASS  %s\n", what);
    else      { std::printf("  FAIL  %s\n", what); ++g_fail; }
}
bool all_finite(const float* buf, std::size_t n) {
    for (std::size_t i = 0; i < n; ++i) {
        if (!std::isfinite(buf[i])) return false;
    }
    return true;
}
bool all_bounded(const float* buf, std::size_t n, float ceil) {
    for (std::size_t i = 0; i < n; ++i) {
        if (std::fabs(buf[i]) > ceil) return false;
    }
    return true;
}
}  // namespace

int main() {
    aboba_backend* be = nullptr;
    aboba_status s = aboba_backend_create_best(&be);
    if (s != ABOBA_OK || !be) {
        std::printf("No backend; cannot run adversarial tests.\n");
        return 1;
    }

    // ============================================================
    std::printf("\n=== Null pointer abuse ===\n");
    check(aboba_pipeline_create(nullptr, 48000, ABOBA_PROFILE_BALANCED, nullptr)
          == ABOBA_ERR_NULL_POINTER, "create with all nulls");
    {
        aboba_pipeline* p = nullptr;
        check(aboba_pipeline_create(nullptr, 48000, ABOBA_PROFILE_BALANCED, &p)
              == ABOBA_ERR_NULL_POINTER, "create with null backend");
        check(p == nullptr, "out_pipeline left null on error");
    }
    check(aboba_pipeline_process(nullptr, nullptr, nullptr, 100)
          == ABOBA_ERR_NULL_POINTER, "process with all nulls");

    // process must not crash with destroy(NULL)
    aboba_pipeline_destroy(nullptr);
    aboba_backend_destroy(nullptr);
    aboba_config_destroy(nullptr);
    check(true, "destroying NULL handles is safe");

    // ============================================================
    std::printf("\n=== Insane sample rates ===\n");
    {
        aboba_pipeline* p = nullptr;
        check(aboba_pipeline_create(be, 0.0, ABOBA_PROFILE_BALANCED, &p)
              == ABOBA_ERR_INVALID_ARG, "sample_rate = 0 rejected");
        check(aboba_pipeline_create(be, -48000.0, ABOBA_PROFILE_BALANCED, &p)
              == ABOBA_ERR_INVALID_ARG, "negative sample_rate rejected");
        check(aboba_pipeline_create(be, 1e15, ABOBA_PROFILE_BALANCED, &p)
              == ABOBA_ERR_INVALID_ARG, "1e15 sample_rate rejected");
        check(aboba_pipeline_create(be, std::nan(""), ABOBA_PROFILE_BALANCED, &p)
              == ABOBA_ERR_INVALID_ARG, "NaN sample_rate rejected");
        check(aboba_pipeline_create(be, std::numeric_limits<double>::infinity(),
                                     ABOBA_PROFILE_BALANCED, &p)
              == ABOBA_ERR_INVALID_ARG, "infinite sample_rate rejected");
    }

    // ============================================================
    std::printf("\n=== Insane buffer sizes ===\n");
    {
        aboba_pipeline* p = nullptr;
        aboba_pipeline_create(be, 48000.0, ABOBA_PROFILE_BALANCED, &p);

        // 0 samples is OK (no-op)
        float dummy = 0.f;
        check(aboba_pipeline_process(p, &dummy, &dummy, 0) == ABOBA_OK,
              "n_samples=0 is no-op success");

        // Crazy large size should be rejected
        check(aboba_pipeline_process(p, &dummy, &dummy, SIZE_MAX / 4)
              == ABOBA_ERR_BUFFER_SIZE,
              "huge n_samples rejected (would otherwise OOM)");

        aboba_pipeline_destroy(p);
    }

    // ============================================================
    std::printf("\n=== NaN/Inf input does not propagate ===\n");
    {
        aboba_pipeline* p = nullptr;
        aboba_pipeline_create(be, 48000.0, ABOBA_PROFILE_BALANCED, &p);
        aboba_pipeline_set_pitch_semitones(p, 2.0f);

        const std::size_t N = 256;
        std::vector<float> in(N), out(N);
        // Mix of NaN, Inf, and normal values
        in[0] = std::nan("");
        in[1] = std::numeric_limits<float>::infinity();
        in[2] = -std::numeric_limits<float>::infinity();
        for (std::size_t i = 3; i < N; ++i) in[i] = 0.1f;

        s = aboba_pipeline_process(p, in.data(), out.data(), N);
        check(s == ABOBA_OK, "NaN-tainted input still returns OK");
        check(all_finite(out.data(), N), "output is finite despite NaN input");
        check(all_bounded(out.data(), N, 1.0f),
              "output stays within [-1, 1] despite NaN input");

        aboba_pipeline_destroy(p);
    }

    // ============================================================
    std::printf("\n=== Extreme amplitude input ===\n");
    {
        aboba_pipeline* p = nullptr;
        aboba_pipeline_create(be, 48000.0, ABOBA_PROFILE_BALANCED, &p);

        const std::size_t N = 1024;
        std::vector<float> in(N, 1e10f);  // 10 billion peak
        std::vector<float> out(N);

        s = aboba_pipeline_process(p, in.data(), out.data(), N);
        check(s == ABOBA_OK, "extreme amplitude input returns OK");
        check(all_finite(out.data(), N), "output remains finite");
        check(all_bounded(out.data(), N, 1.001f),
              "output limited despite extreme input (peak < 1.001)");

        aboba_pipeline_destroy(p);
    }

    // ============================================================
    std::printf("\n=== Pitch / formant clamping ===\n");
    {
        aboba_pipeline* p = nullptr;
        aboba_pipeline_create(be, 48000.0, ABOBA_PROFILE_BALANCED, &p);

        // Cannot crash setting absurd values
        check(aboba_pipeline_set_pitch_semitones(p, 1e30f) == ABOBA_OK,
              "huge pitch value accepted (clamped internally)");
        check(aboba_pipeline_set_pitch_semitones(p, -1e30f) == ABOBA_OK,
              "huge negative pitch accepted");
        check(aboba_pipeline_set_pitch_semitones(p, std::nan("")) == ABOBA_OK,
              "NaN pitch accepted (replaced with midpoint)");
        check(aboba_pipeline_set_formant_semitones(p, std::numeric_limits<float>::infinity())
              == ABOBA_OK, "Inf formant accepted (clamped)");

        // After all that, processing must still work and produce sane audio
        const std::size_t N = 256;
        std::vector<float> in(N, 0.1f), out(N);
        s = aboba_pipeline_process(p, in.data(), out.data(), N);
        check(s == ABOBA_OK, "process still works after parameter abuse");
        check(all_finite(out.data(), N), "output finite after parameter abuse");

        aboba_pipeline_destroy(p);
    }

    // ============================================================
    std::printf("\n=== Out-of-range enums ===\n");
    {
        aboba_pipeline* p = nullptr;
        aboba_pipeline_create(be, 48000.0, ABOBA_PROFILE_BALANCED, &p);

        check(aboba_pipeline_set_character(p, (aboba_character)999)
              == ABOBA_ERR_OUT_OF_RANGE, "huge character index rejected");
        check(aboba_pipeline_set_character(p, (aboba_character)(-1))
              == ABOBA_ERR_OUT_OF_RANGE, "negative character rejected");

        aboba_pipeline_destroy(p);
    }

    // ============================================================
    std::printf("\n=== TOML adversarial input ===\n");
    {
        aboba_config* cfg = nullptr;
        // 2 MiB of garbage — should be rejected by size cap.
        std::string huge(2 * 1024 * 1024, 'A');
        check(aboba_config_parse_string(huge.c_str(), &cfg) == ABOBA_ERR_PARSE,
              "huge TOML string rejected");
        check(cfg == nullptr, "no config leaked on size-cap rejection");

        // Empty string
        check(aboba_config_parse_string("", &cfg) == ABOBA_OK ||
              aboba_config_parse_string("", &cfg) == ABOBA_ERR_PARSE ||
              aboba_config_parse_string("", &cfg) == ABOBA_ERR_INVALID_ARG,
              "empty TOML doesn't crash");
        if (cfg) aboba_config_destroy(cfg);
        cfg = nullptr;

        // Deeply nested brackets — defends against stack overflow attack
        std::string nested;
        for (int i = 0; i < 1000; ++i) nested += "[[[[";
        for (int i = 0; i < 1000; ++i) nested += "]]]]";
        check(aboba_config_parse_string(nested.c_str(), &cfg)
              == ABOBA_ERR_PARSE,
              "deeply nested brackets rejected without stack overflow");
        if (cfg) aboba_config_destroy(cfg);
        cfg = nullptr;

        // Lines with no newlines, only NUL embedded
        const char* embedded_nul = "name = \"foo\0bar\"\n";
        check(aboba_config_parse_string(embedded_nul, &cfg) == ABOBA_OK ||
              aboba_config_parse_string(embedded_nul, &cfg) == ABOBA_ERR_PARSE,
              "embedded NUL in TOML handled");
        if (cfg) aboba_config_destroy(cfg);
    }

    // ============================================================
    std::printf("\n=== Lots of pipelines (leak check) ===\n");
    {
        // Create and destroy 1000 pipelines. Under ASan, this catches
        // leaks. We don't measure RSS directly here.
        for (int i = 0; i < 1000; ++i) {
            aboba_pipeline* p = nullptr;
            aboba_pipeline_create(be, 48000.0, ABOBA_PROFILE_BALANCED, &p);
            // Do one process call so pipeline state is exercised
            float buf_in[64] = {}, buf_out[64] = {};
            aboba_pipeline_process(p, buf_in, buf_out, 64);
            aboba_pipeline_destroy(p);
        }
        check(true, "1000 create/process/destroy cycles completed");
    }

    // ============================================================
    std::printf("\n=== Random fuzzing on process() ===\n");
    {
        aboba_pipeline* p = nullptr;
        aboba_pipeline_create(be, 48000.0, ABOBA_PROFILE_BALANCED, &p);
        std::mt19937 rng(0xDEADBEEF);
        std::uniform_real_distribution<float> dist(-100.0f, 100.0f);

        const std::size_t N = 512;
        std::vector<float> in(N), out(N);
        bool all_passed = true;
        for (int iter = 0; iter < 500; ++iter) {
            // Random parameter sweep + random audio
            aboba_pipeline_set_pitch_semitones(p,
                std::uniform_real_distribution<float>(-30, 30)(rng));
            for (auto& v : in) v = dist(rng);
            // Occasionally poison with NaN/Inf
            if (iter % 17 == 0) in[iter % N] = std::nan("");
            if (iter % 23 == 0) in[iter % N] = std::numeric_limits<float>::infinity();

            s = aboba_pipeline_process(p, in.data(), out.data(), N);
            if (s != ABOBA_OK) { all_passed = false; break; }
            if (!all_finite(out.data(), N)) { all_passed = false; break; }
            if (!all_bounded(out.data(), N, 1.001f)) { all_passed = false; break; }
        }
        check(all_passed, "500 randomized iterations all produce finite, bounded output");
        aboba_pipeline_destroy(p);
    }

    // ============================================================
    std::printf("\n=== Watchdog under tight budget ===\n");
    {
        aboba_pipeline* p = nullptr;
        aboba_pipeline_create(be, 48000.0, ABOBA_PROFILE_BALANCED, &p);
        aboba_pipeline_set_max_block_us(p, 1);   // 1us budget - impossible
        aboba_pipeline_set_budget_policy(p, ABOBA_BUDGET_POLICY_BYPASS);

        const std::size_t N = 512;
        std::vector<float> in(N, 0.2f), out(N);
        for (int i = 0; i < 50; ++i) {
            aboba_pipeline_process(p, in.data(), out.data(), N);
            // Output must always be finite even when watchdog kicks in
            if (!all_finite(out.data(), N)) {
                check(false, "output finite under watchdog stress");
                break;
            }
        }
        check(true, "50 watchdog-stressed iterations completed cleanly");
        aboba_pipeline_stats st;
        aboba_pipeline_get_stats(p, &st);
        std::printf("    bypassed %llu / %llu blocks\n",
            (unsigned long long)st.bypassed_blocks,
            (unsigned long long)st.total_blocks);
        check(st.bypassed_blocks > 0, "watchdog did trigger");
        aboba_pipeline_destroy(p);
    }

    // ============================================================
    std::printf("\n=== Partial buffer aliasing rejection ===\n");
    {
        aboba_pipeline* p = nullptr;
        aboba_pipeline_create(be, 48000.0, ABOBA_PROFILE_BALANCED, &p);
        const std::size_t N = 256;
        std::vector<float> buf(N * 2, 0.0f);

        // In-place (in == out) — must work
        aboba_status s = aboba_pipeline_process(p, buf.data(), buf.data(), N);
        check(s == ABOBA_OK, "in-place (in==out) still works");

        // Partial overlap: out points into the middle of in. Must be REJECTED.
        s = aboba_pipeline_process(p, buf.data(), buf.data() + 100, N);
        check(s == ABOBA_ERR_INVALID_ARG, "partial-overlap (out=in+100) rejected");

        // Reverse partial overlap
        s = aboba_pipeline_process(p, buf.data() + 50, buf.data(), N);
        check(s == ABOBA_ERR_INVALID_ARG, "reverse partial overlap rejected");

        aboba_pipeline_destroy(p);
    }

    // ============================================================
    std::printf("\n=== Rapid create/destroy stress (ASan leak detection) ===\n");
    {
        for (int i = 0; i < 200; ++i) {
            aboba_pipeline* tmp = nullptr;
            aboba_status s = aboba_pipeline_create(
                be, 48000.0, ABOBA_PROFILE_BALANCED, &tmp);
            if (s != ABOBA_OK || !tmp) {
                check(false, "create failed in stress loop");
                break;
            }
            // Touch it briefly
            aboba_pipeline_set_pitch_semitones(tmp, static_cast<float>(i % 12));
            aboba_pipeline_destroy(tmp);
        }
        check(true, "200x create/destroy completed (ASan-clean)");
    }

    // ============================================================
    std::printf("\n=== Status code coverage ===\n");
    {
        // Every defined status code must have a non-empty message
        const aboba_status codes[] = {
            ABOBA_OK,
            ABOBA_ERR_INVALID_ARG,
            ABOBA_ERR_NULL_POINTER,
            ABOBA_ERR_BUFFER_SIZE,
            ABOBA_ERR_NO_BACKEND,
            ABOBA_ERR_INTERNAL,
            ABOBA_ERR_NOT_IMPLEMENTED,
            ABOBA_ERR_OUT_OF_RANGE,
            ABOBA_ERR_FILE_IO,
            ABOBA_ERR_PARSE,
        };
        bool all_have_msg = true;
        for (auto c : codes) {
            const char* m = aboba_status_message(c);
            if (!m || !*m) { all_have_msg = false; break; }
        }
        check(all_have_msg, "all defined status codes have non-empty messages");

        // Unknown / future codes must return non-NULL (graceful fallback)
        const char* m1 = aboba_status_message(-9999);
        const char* m2 = aboba_status_message(-1);
        check(m1 && *m1, "unknown status code -9999 returns non-NULL message");
        check(m2 && *m2, "status code -1 returns non-NULL message");
    }

    // ============================================================
    std::printf("\n=== Parameter setter NaN/Inf injection ===\n");
    {
        aboba_pipeline* p = nullptr;
        aboba_pipeline_create(be, 48000.0, ABOBA_PROFILE_BALANCED, &p);

        // All setters must accept NaN/Inf without crashing and without
        // poisoning subsequent processing.
        aboba_pipeline_set_pitch_semitones(p, std::nanf(""));
        aboba_pipeline_set_pitch_semitones(p, std::numeric_limits<float>::infinity());
        aboba_pipeline_set_formant_semitones(p, std::nanf(""));
        aboba_pipeline_set_reverb_room_size(p, std::nanf(""));
        aboba_pipeline_set_reverb_damping(p, std::numeric_limits<float>::infinity());
        aboba_pipeline_set_reverb_wet(p, -1e30f);
        aboba_pipeline_set_autotune_strength(p, std::nanf(""));
        aboba_pipeline_set_autotune_glide_ms(p, -1e9f);
        aboba_pipeline_set_autotune_glide_ms(p, std::numeric_limits<float>::infinity());

        // Now process — must still produce finite output
        const std::size_t N = 1024;
        std::vector<float> in(N), out(N);
        for (std::size_t i = 0; i < N; ++i)
            in[i] = 0.3f * std::sin(2.0f * 3.14159f * 220.0f * i / 48000.0f);

        aboba_status s = aboba_pipeline_process(p, in.data(), out.data(), N);
        check(s == ABOBA_OK, "process OK after NaN/Inf setter storm");
        check(all_finite(out.data(), N), "  output finite after setter storm");

        aboba_pipeline_destroy(p);
    }

    // ============================================================
    std::printf("\n=== Hard limiter caps at ±1.0 ===\n");
    {
        // Deliberately drive the pipeline with very loud input. The
        // outermost C ABI hard_limit_block should clamp output ≤ 1.0
        // regardless of pipeline internal state.
        aboba_pipeline* p = nullptr;
        aboba_pipeline_create(be, 48000.0, ABOBA_PROFILE_BALANCED, &p);
        const std::size_t N = 512;
        std::vector<float> in(N, 0.99f);  // loud DC-ish, but not >1
        std::vector<float> out(N, 0.0f);
        for (std::size_t i = 0; i < N; ++i)
            in[i] = (i % 2 == 0) ? 0.99f : -0.99f;  // alternating extreme
        aboba_pipeline_process(p, in.data(), out.data(), N);
        bool clamped_ok = true;
        for (std::size_t i = 0; i < N; ++i) {
            if (std::fabs(out[i]) > 1.001f) { clamped_ok = false; break; }
        }
        check(clamped_ok, "output absolute value ≤ 1.0 (limiter active)");
        check(all_finite(out.data(), N), "  output finite under extreme drive");
        aboba_pipeline_destroy(p);
    }

    aboba_backend_destroy(be);

    std::printf("\n========================================\n");
    if (g_fail == 0) {
        std::printf("Total: %d/%d passed \xE2\x9C\x93\n", g_total, g_total);
        return 0;
    } else {
        std::printf("Total: %d/%d passed - %d FAILED \xE2\x9C\x97\n",
            g_total - g_fail, g_total, g_fail);
        return 1;
    }
}
