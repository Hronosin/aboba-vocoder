/* SPDX-License-Identifier: GPL-3.0-or-later
 *
 * c_abi_test.c — pure C test program for the aboba_c.h ABI.
 *
 * Compiled as C99. This verifies that:
 *   * The header is valid C (not just C++)
 *   * Every entry point can be called from C
 *   * Status codes are correctly returned
 *   * Memory ownership is clean
 *   * Audio processing produces sensible output
 *   * Watchdog kicks in when budget is exceeded
 *
 * Build:
 *   cc -std=c99 -Iinclude -o c_abi_test examples/c_abi_test.c build/libaboba_c.so
 * Run:
 *   LD_LIBRARY_PATH=build ./c_abi_test
 */

#include "aboba_c.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int g_total = 0, g_fail = 0;

static void check(int cond, const char* what) {
    ++g_total;
    if (cond) printf("  PASS  %s\n", what);
    else      { printf("  FAIL  %s\n", what); ++g_fail; }
}

#define PI 3.14159265358979f

static void fill_sine(float* buf, size_t n, float hz, float sr, float amp) {
    for (size_t i = 0; i < n; ++i) {
        buf[i] = amp * sinf(2.0f * PI * hz * (float)i / sr);
    }
}

static float compute_rms(const float* buf, size_t n) {
    double ss = 0;
    for (size_t i = 0; i < n; ++i) ss += (double)buf[i] * buf[i];
    return (float)sqrt(ss / (n > 0 ? n : 1));
}

static int is_finite_bounded(const float* buf, size_t n, float bound) {
    for (size_t i = 0; i < n; ++i) {
        if (!isfinite(buf[i])) return 0;
        if (fabsf(buf[i]) > bound) return 0;
    }
    return 1;
}

int main(void) {
    aboba_status s;

    /* ============================================================ */
    printf("\n=== Version + status codes ===\n");
    check(aboba_runtime_abi_version_major() == ABOBA_C_API_VERSION_MAJOR,
          "major version matches header");
    check(aboba_runtime_abi_version_minor() == ABOBA_C_API_VERSION_MINOR,
          "minor version matches header");
    check(aboba_runtime_version_string() != NULL && strlen(aboba_runtime_version_string()) > 0,
          "version string non-empty");
    check(strcmp(aboba_status_message(ABOBA_OK), "OK") == 0,
          "status message for OK");
    check(strstr(aboba_status_message(ABOBA_ERR_INVALID_ARG), "Invalid") != NULL,
          "status message for INVALID_ARG");

    /* ============================================================ */
    printf("\n=== Backend creation ===\n");
    aboba_backend* be = NULL;
    s = aboba_backend_create_best(&be);
    check(s == ABOBA_OK, "create_best returns OK");
    check(be != NULL, "backend pointer non-null");
    if (be) {
        const char* name = aboba_backend_name(be);
        check(name && strlen(name) > 0, "backend name non-empty");
        printf("    backend: %s\n", name);
        aboba_backend_type t = aboba_backend_type_get(be);
        check(t == ABOBA_BACKEND_TYPE_CPU || t == ABOBA_BACKEND_TYPE_HIP,
              "backend type is valid");
    }

    /* ============================================================ */
    printf("\n=== Null pointer rejection ===\n");
    s = aboba_backend_create_best(NULL);
    check(s == ABOBA_ERR_NULL_POINTER, "create_best(NULL) returns NULL_POINTER");
    s = aboba_pipeline_create(NULL, 48000, ABOBA_PROFILE_BALANCED, NULL);
    check(s == ABOBA_ERR_NULL_POINTER, "pipeline_create with null backend rejected");

    /* ============================================================ */
    printf("\n=== Pipeline creation + destroy ===\n");
    aboba_pipeline* pipe = NULL;
    s = aboba_pipeline_create(be, 48000.0, ABOBA_PROFILE_BALANCED, &pipe);
    check(s == ABOBA_OK, "pipeline_create succeeds");
    check(pipe != NULL, "pipeline pointer non-null");
    size_t latency = aboba_pipeline_latency_samples(pipe);
    printf("    latency: %zu samples (%.1f ms @ 48kHz)\n",
           latency, (double)latency / 48.0);
    check(latency > 0 && latency < 10000,
          "latency is plausible");

    /* ============================================================ */
    printf("\n=== Process basic sine ===\n");
    {
        const size_t N = 48000;  /* 1 second */
        float* in  = (float*)calloc(N, sizeof(float));
        float* out = (float*)calloc(N, sizeof(float));
        fill_sine(in, N, 220.0f, 48000.0f, 0.3f);
        s = aboba_pipeline_set_pitch_semitones(pipe, 3.0f);
        check(s == ABOBA_OK, "set_pitch_semitones returns OK");
        s = aboba_pipeline_process(pipe, in, out, N);
        check(s == ABOBA_OK, "process returns OK");
        check(is_finite_bounded(out, N, 2.0f), "output is finite/bounded");
        float in_rms  = compute_rms(in,  N);
        float out_rms = compute_rms(out, N);
        printf("    in_rms=%.4f out_rms=%.4f\n", in_rms, out_rms);
        check(out_rms > 0.01f, "output has signal");
        free(in); free(out);
    }

    /* ============================================================ */
    printf("\n=== Character preset ===\n");
    s = aboba_pipeline_set_character(pipe, ABOBA_CHARACTER_DEEP_MALE);
    check(s == ABOBA_OK, "set_character DeepMale OK");
    check(aboba_pipeline_get_character(pipe) == ABOBA_CHARACTER_DEEP_MALE,
          "get_character returns DeepMale");
    s = aboba_pipeline_set_character(pipe, (aboba_character)999);
    check(s == ABOBA_ERR_OUT_OF_RANGE, "invalid character rejected");

    /* ============================================================ */
    printf("\n=== Autotune ===\n");
    s = aboba_pipeline_set_autotune_enabled(pipe, 1);
    check(s == ABOBA_OK, "autotune enable OK");
    check(aboba_pipeline_get_autotune_enabled(pipe) == 1, "autotune is on");
    s = aboba_pipeline_set_autotune_scale(pipe, ABOBA_SCALE_MAJOR, 0);
    check(s == ABOBA_OK, "set_autotune_scale OK");
    s = aboba_pipeline_set_autotune_strength(pipe, 0.7f);
    check(s == ABOBA_OK, "set_autotune_strength OK");

    /* ============================================================ */
    printf("\n=== Reverb ===\n");
    s = aboba_pipeline_set_reverb_enabled(pipe, 1);
    check(s == ABOBA_OK, "reverb enable OK");
    check(aboba_pipeline_get_reverb_enabled(pipe) == 1, "reverb is on");
    s = aboba_pipeline_set_reverb_room_size(pipe, 0.5f);
    check(s == ABOBA_OK, "reverb room_size OK");

    /* ============================================================ */
    printf("\n=== Watchdog stats ===\n");
    {
        const size_t N = 256;
        float* in  = (float*)calloc(N, sizeof(float));
        float* out = (float*)calloc(N, sizeof(float));
        for (int i = 0; i < 100; ++i) {
            aboba_pipeline_process(pipe, in, out, N);
        }
        aboba_pipeline_stats st;
        s = aboba_pipeline_get_stats(pipe, &st);
        check(s == ABOBA_OK, "get_stats OK");
        printf("    total=%llu bypassed=%llu last_us=%.1f p99=%.1f bypass_now=%d\n",
            (unsigned long long)st.total_blocks,
            (unsigned long long)st.bypassed_blocks,
            (double)st.last_block_us,
            (double)st.p99_block_us,
            st.currently_bypassed);
        check(st.total_blocks >= 100, "100 blocks counted");
        free(in); free(out);
    }

    /* ============================================================ */
    printf("\n=== Budget watchdog (artificially low budget) ===\n");
    {
        /* Force budget to 1 us so we definitely overrun */
        s = aboba_pipeline_set_max_block_us(pipe, 1);
        check(s == ABOBA_OK, "set_max_block_us OK");
        s = aboba_pipeline_set_budget_policy(pipe, ABOBA_BUDGET_POLICY_BYPASS);
        check(s == ABOBA_OK, "set_budget_policy OK");

        const size_t N = 256;
        float* in  = (float*)calloc(N, sizeof(float));
        float* out = (float*)calloc(N, sizeof(float));
        fill_sine(in, N, 440.0f, 48000.0f, 0.2f);
        /* Reset state so we have a clean baseline */
        aboba_pipeline_reset(pipe);
        for (int i = 0; i < 30; ++i) {
            aboba_pipeline_process(pipe, in, out, N);
        }
        aboba_pipeline_stats st;
        aboba_pipeline_get_stats(pipe, &st);
        printf("    bypassed=%llu / total=%llu\n",
            (unsigned long long)st.bypassed_blocks,
            (unsigned long long)st.total_blocks);
        check(st.bypassed_blocks > 0, "watchdog triggered bypass");
        check(is_finite_bounded(out, N, 2.0f),
              "output finite even in bypass (passthrough)");
        free(in); free(out);

        /* Restore reasonable budget */
        aboba_pipeline_set_max_block_us(pipe, 5000);
        aboba_pipeline_reset(pipe);
    }

    /* ============================================================ */
    printf("\n=== Low-latency pipeline ===\n");
    {
        aboba_pipeline* lpipe = NULL;
        s = aboba_pipeline_create_lowlatency(be, 48000.0, &lpipe);
        check(s == ABOBA_OK, "lowlatency create OK");
        check(lpipe != NULL, "lpipe non-null");

        size_t l_latency = aboba_pipeline_latency_samples(lpipe);
        printf("    lowlatency latency: %zu samples (%.2f ms)\n",
            l_latency, (double)l_latency / 48.0);
        check(l_latency <= 256, "lowlatency latency <= 256 samples");

        const size_t N = 256;
        float* in  = (float*)calloc(N, sizeof(float));
        float* out = (float*)calloc(N, sizeof(float));
        fill_sine(in, N, 220.0f, 48000.0f, 0.3f);

        s = aboba_pipeline_set_pitch_semitones(lpipe, 2.0f);
        check(s == ABOBA_OK, "lowlatency pitch shift OK");

        for (int i = 0; i < 50; ++i) {
            s = aboba_pipeline_process(lpipe, in, out, N);
            if (s != ABOBA_OK) break;
        }
        check(s == ABOBA_OK, "50 blocks processed without error");
        check(is_finite_bounded(out, N, 2.0f), "lowlatency output finite");

        /* Autotune should be rejected (NOT_IMPLEMENTED in lowlatency) */
        s = aboba_pipeline_set_autotune_enabled(lpipe, 1);
        check(s == ABOBA_ERR_NOT_IMPLEMENTED,
              "autotune not available in lowlatency");

        aboba_pipeline_destroy(lpipe);
        free(in); free(out);
    }

    /* ============================================================ */
    printf("\n=== Config from TOML ===\n");
    {
        const char* toml =
            "name = \"c-test\"\n"
            "[pipeline]\n"
            "profile = \"balanced\"\n"
            "[character]\n"
            "preset = \"warm-male\"\n"
            "[autotune]\n"
            "enabled = true\n"
            "scale = \"major\"\n"
            "root = \"C\"\n"
            "strength = 0.5\n";

        aboba_config* cfg = NULL;
        s = aboba_config_parse_string(toml, &cfg);
        check(s == ABOBA_OK, "parse_string OK");
        check(cfg != NULL, "config non-null");

        aboba_pipeline* p2 = NULL;
        s = aboba_pipeline_create_from_config(be, cfg, &p2);
        check(s == ABOBA_OK, "create_from_config OK");
        check(p2 != NULL, "p2 non-null");
        if (p2) {
            check(aboba_pipeline_get_character(p2) == ABOBA_CHARACTER_WARM_MALE,
                  "character was applied from config");
            check(aboba_pipeline_get_autotune_enabled(p2) == 1,
                  "autotune applied from config");
            aboba_pipeline_destroy(p2);
        }
        aboba_config_destroy(cfg);
    }

    /* ============================================================ */
    printf("\n=== Bad TOML returns parse error ===\n");
    {
        aboba_config* cfg = NULL;
        s = aboba_config_parse_string("[autotune]\nenabled = yes\n", &cfg);
        check(s == ABOBA_ERR_PARSE, "bad TOML returns ERR_PARSE");
        check(cfg == NULL, "config not allocated on error");
        const char* err = aboba_config_last_error();
        check(err != NULL && strlen(err) > 0, "last_error has message");
        int line = aboba_config_last_error_line();
        printf("    error: '%s' at line %d\n", err ? err : "NULL", line);
        check(line > 0, "last_error_line is positive");
    }

    /* ============================================================ */
    printf("\n=== Missing config file ===\n");
    {
        aboba_config* cfg = NULL;
        s = aboba_config_load_file("/this/should/not/exist.toml", &cfg);
        check(s == ABOBA_ERR_FILE_IO, "missing file returns ERR_FILE_IO");
        check(cfg == NULL, "config not allocated on I/O error");
    }

    /* ============================================================ */
    printf("\n=== Null tolerance ===\n");
    aboba_pipeline_destroy(NULL);  /* must not crash */
    aboba_backend_destroy(NULL);
    aboba_config_destroy(NULL);
    check(1, "destroy(NULL) doesn't crash");

    /* ============================================================ */
    printf("\n=== Clean teardown ===\n");
    aboba_pipeline_destroy(pipe);
    aboba_backend_destroy(be);
    check(1, "clean teardown completes");

    printf("\n========================================\n");
    if (g_fail == 0) {
        printf("Total: %d/%d passed \xE2\x9C\x93\n", g_total, g_total);
        return 0;
    } else {
        printf("Total: %d/%d passed - %d FAILED \xE2\x9C\x97\n",
            g_total - g_fail, g_total, g_fail);
        return 1;
    }
}
