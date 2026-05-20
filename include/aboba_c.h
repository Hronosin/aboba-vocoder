/* SPDX-License-Identifier: GPL-3.0-or-later */
/*
 * aboba_c.h — Stable C ABI for the Aboba Vocoder framework.
 *
 * This header exposes the framework via a C interface that is:
 *   * Compilable in C99 and any C++ version
 *   * ABI-stable across minor versions (we promise not to change struct
 *     layouts or signatures within a major version)
 *   * Free of C++ types (no std::string, no exceptions across the
 *     boundary, no templates)
 *   * Safe to call from C, C++, C#, Rust, Zig, or any FFI-capable language
 *
 * Versioning:
 *   ABOBA_C_API_VERSION_MAJOR — bumped on breaking changes
 *   ABOBA_C_API_VERSION_MINOR — bumped on backward-compatible additions
 *   aboba_runtime_abi_version()   returns the linked library's version
 *
 * If your application requires a specific major version, check this at
 * startup:
 *     if (aboba_runtime_abi_version_major() != ABOBA_C_API_VERSION_MAJOR) {
 *         abort();  // ABI mismatch
 *     }
 *
 * Error handling:
 *   Functions return aboba_status (an int). 0 = success. Non-zero values
 *   are defined below. Use aboba_status_message(s) for a human-readable
 *   string. C exceptions are NEVER propagated out of this layer.
 *
 * Handles:
 *   All types in this header are opaque pointers. Treat them like FILE* —
 *   you create them with aboba_X_create*(), use them through C functions,
 *   and destroy them with aboba_X_destroy(). After destroy, the pointer
 *   is invalid; do not use it.
 *
 * Threading:
 *   Each handle is single-threaded (one thread at a time). Different
 *   handles can be used from different threads concurrently. Read-only
 *   query functions (aboba_*_get_*) are safe to call from any thread.
 *
 * Memory:
 *   All allocations happen inside the library; the caller never passes
 *   memory ownership across the ABI. Audio buffers are caller-owned and
 *   their lifetime is the duration of the call.
 *
 * Real-time / game-engine usage:
 *   * Use aboba_pipeline_create_lowlatency() for hard <2ms-per-block
 *     processing. See aboba_lowlatency.h for details on the quality tradeoff.
 *   * Use aboba_pipeline_set_max_block_us() to set a hard processing budget
 *     beyond which the pipeline will bypass to passthrough rather than
 *     glitch.
 *   * Never block, allocate, or call back into the library from inside a
 *     game audio callback unless you've configured the pipeline for it.
 */
#ifndef ABOBA_C_H
#define ABOBA_C_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* =============================================================
 * Versioning
 * ============================================================= */

#define ABOBA_C_API_VERSION_MAJOR  1
#define ABOBA_C_API_VERSION_MINOR  0

/* Visibility / linkage. On Windows we need __declspec(dllexport) when
 * building the library and __declspec(dllimport) when consuming it.
 * Define ABOBA_C_BUILDING when compiling the library itself.
 */
#if defined(_WIN32) || defined(__CYGWIN__)
#  ifdef ABOBA_C_BUILDING
#    define ABOBA_API __declspec(dllexport)
#  else
#    define ABOBA_API __declspec(dllimport)
#  endif
#elif defined(__GNUC__) || defined(__clang__)
#  define ABOBA_API __attribute__((visibility("default")))
#else
#  define ABOBA_API
#endif

ABOBA_API int aboba_runtime_abi_version_major(void);
ABOBA_API int aboba_runtime_abi_version_minor(void);
ABOBA_API const char* aboba_runtime_version_string(void); /* "0.9.0" etc */

/* =============================================================
 * Status codes / error handling
 * ============================================================= */

typedef int aboba_status;
#define ABOBA_OK                       0
#define ABOBA_ERR_INVALID_ARG         -1
#define ABOBA_ERR_NULL_POINTER        -2
#define ABOBA_ERR_BUFFER_SIZE         -3
#define ABOBA_ERR_NO_BACKEND          -4
#define ABOBA_ERR_INTERNAL            -5  /* unexpected C++ exception */
#define ABOBA_ERR_NOT_IMPLEMENTED     -6
#define ABOBA_ERR_OUT_OF_RANGE        -7
#define ABOBA_ERR_FILE_IO             -8
#define ABOBA_ERR_PARSE               -9

/* Human-readable message for the given status. Returns a static
 * NUL-terminated string; the caller must NOT free it. */
ABOBA_API const char* aboba_status_message(aboba_status s);

/* =============================================================
 * Enum mirrors of the C++ enums. Values are stable across versions.
 * ============================================================= */

typedef enum {
    ABOBA_PROFILE_QUALITY     = 0,
    ABOBA_PROFILE_BALANCED    = 1,
    ABOBA_PROFILE_PERFORMANCE = 2
} aboba_quality_profile;

typedef enum {
    ABOBA_CHARACTER_NEUTRAL         = 0,
    ABOBA_CHARACTER_DEEP_MALE       = 1,
    ABOBA_CHARACTER_WARM_MALE       = 2,
    ABOBA_CHARACTER_CHESTY_MALE     = 3,
    ABOBA_CHARACTER_YOUNG_FEMALE    = 4,
    ABOBA_CHARACTER_ANIME_GIRL      = 5,
    ABOBA_CHARACTER_CHIPMUNK        = 6,
    ABOBA_CHARACTER_GIANT           = 7,
    ABOBA_CHARACTER_DEMON           = 8,
    ABOBA_CHARACTER_ROBOT           = 9,
    ABOBA_CHARACTER_RADIO_HOST      = 10,
    ABOBA_CHARACTER_WHISPER         = 11,
    ABOBA_CHARACTER_HELIUM          = 12,
    ABOBA_CHARACTER_CARTOON_VILLAIN = 13,
    ABOBA_CHARACTER_COUNT
} aboba_character;

typedef enum {
    ABOBA_SCALE_CHROMATIC        = 0,
    ABOBA_SCALE_MAJOR            = 1,
    ABOBA_SCALE_MINOR            = 2,
    ABOBA_SCALE_HARMONIC_MINOR   = 3,
    ABOBA_SCALE_PENTATONIC_MAJOR = 4,
    ABOBA_SCALE_PENTATONIC_MINOR = 5,
    ABOBA_SCALE_BLUES            = 6,
    ABOBA_SCALE_WHOLE_TONE       = 7,
    ABOBA_SCALE_CUSTOM           = 8
} aboba_scale;

typedef enum {
    ABOBA_BACKEND_TYPE_CPU = 0,
    ABOBA_BACKEND_TYPE_HIP = 1
} aboba_backend_type;

/* =============================================================
 * Opaque handle types
 * ============================================================= */

typedef struct aboba_backend_s   aboba_backend;
typedef struct aboba_pipeline_s  aboba_pipeline;
typedef struct aboba_config_s    aboba_config;

/* =============================================================
 * Backend
 * ============================================================= */

/* Create the best available backend (HIP if AMD GPU, else CPU).
 * On success, *out_backend is set and ABOBA_OK is returned. */
ABOBA_API aboba_status aboba_backend_create_best(aboba_backend** out_backend);

/* Create a specific backend type. Returns ABOBA_ERR_NO_BACKEND if the
 * requested type isn't available on this build. */
ABOBA_API aboba_status aboba_backend_create(aboba_backend_type type,
                                            aboba_backend** out_backend);

/* Destroy. Safe to call with NULL. */
ABOBA_API void aboba_backend_destroy(aboba_backend* b);

/* Read-only queries. The returned string points into static storage and
 * is valid for the lifetime of the backend. */
ABOBA_API const char* aboba_backend_name(const aboba_backend* b);
ABOBA_API aboba_backend_type aboba_backend_type_get(const aboba_backend* b);

/* =============================================================
 * VoicePipeline — the main processing object
 * ============================================================= */

/* Create a pipeline with the given sample rate, profile, and backend.
 * The backend must outlive the pipeline. */
ABOBA_API aboba_status aboba_pipeline_create(
    aboba_backend* backend,
    double sample_rate,
    aboba_quality_profile profile,
    aboba_pipeline** out_pipeline);

/* Create a low-latency pipeline. See aboba_lowlatency.h. This skips the
 * formant vocoder (~42ms latency) in favor of a sample-by-sample chain
 * that fits in ~2ms per block. Quality is reduced — see docs. */
ABOBA_API aboba_status aboba_pipeline_create_lowlatency(
    aboba_backend* backend,
    double sample_rate,
    aboba_pipeline** out_pipeline);

/* Destroy. Safe with NULL. */
ABOBA_API void aboba_pipeline_destroy(aboba_pipeline* p);

/* ----- Audio processing ------------------------------------------------
 *
 * input/output:   pointers to float arrays of n_samples elements
 * n_samples:      number of audio samples (NOT bytes)
 *
 * It is safe (and common) to use input == output for in-place processing.
 *
 * The pipeline operates on mono float audio. For multi-channel input,
 * pre-mix on the caller side and post-process per-channel after if
 * needed.
 *
 * Real-time:
 *   The processing has internal latency (~42ms for normal pipeline, ~2ms
 *   for lowlatency). The pipeline buffers internally; callers see only the
 *   block-level call cost.
 *
 *   If the call exceeds aboba_pipeline_set_max_block_us(), the pipeline
 *   will (depending on policy) bypass to passthrough for THIS block. The
 *   audio still flows; no NaN, no clicks.
 */
ABOBA_API aboba_status aboba_pipeline_process(
    aboba_pipeline* p,
    const float* input,
    float* output,
    size_t n_samples);

/* Reset internal state (buffers, envelopes, etc). Call when the audio
 * stream changes, e.g. user starts a new recording. */
ABOBA_API aboba_status aboba_pipeline_reset(aboba_pipeline* p);

/* ----- Pitch / formant / character ------------------------------------- */

ABOBA_API aboba_status aboba_pipeline_set_pitch_semitones(
    aboba_pipeline* p, float semitones);

ABOBA_API aboba_status aboba_pipeline_set_formant_semitones(
    aboba_pipeline* p, float semitones);

ABOBA_API aboba_status aboba_pipeline_set_character(
    aboba_pipeline* p, aboba_character c);

/* Returns ABOBA_CHARACTER_COUNT on error. */
ABOBA_API aboba_character aboba_pipeline_get_character(const aboba_pipeline* p);

/* ----- Autotune --------------------------------------------------------- */

ABOBA_API aboba_status aboba_pipeline_set_autotune_enabled(
    aboba_pipeline* p, int enabled);   /* 0 or 1 */

ABOBA_API int aboba_pipeline_get_autotune_enabled(const aboba_pipeline* p);

ABOBA_API aboba_status aboba_pipeline_set_autotune_scale(
    aboba_pipeline* p, aboba_scale scale, int root_semitones);

ABOBA_API aboba_status aboba_pipeline_set_autotune_strength(
    aboba_pipeline* p, float strength);    /* 0..1 */

ABOBA_API aboba_status aboba_pipeline_set_autotune_glide_ms(
    aboba_pipeline* p, float glide_ms);

/* ----- Reverb ---------------------------------------------------------- */

ABOBA_API aboba_status aboba_pipeline_set_reverb_enabled(
    aboba_pipeline* p, int enabled);

ABOBA_API int aboba_pipeline_get_reverb_enabled(const aboba_pipeline* p);

ABOBA_API aboba_status aboba_pipeline_set_reverb_room_size(
    aboba_pipeline* p, float room_size);

ABOBA_API aboba_status aboba_pipeline_set_reverb_damping(
    aboba_pipeline* p, float damping);

ABOBA_API aboba_status aboba_pipeline_set_reverb_wet(
    aboba_pipeline* p, float wet);

/* ----- Stability / watchdog -------------------------------------------- */

/* Set the per-block processing budget in microseconds. If a process()
 * call would exceed this, the watchdog policy kicks in (see below).
 *
 * Default: 5000 (5ms). For game engines, set lower (1500-2000 us is a
 * typical hard real-time budget at small block sizes). 0 = unlimited.
 *
 * NOTE: this is a WATCHDOG, not a deadline. The current block still
 * completes; the next block(s) may be auto-bypassed if the trend is bad. */
ABOBA_API aboba_status aboba_pipeline_set_max_block_us(
    aboba_pipeline* p, int max_us);

typedef enum {
    /* When budget exceeded: emit a single dropped-block log entry but
     * continue processing fully. The default. */
    ABOBA_BUDGET_POLICY_LOG   = 0,
    /* When budget exceeded: switch to passthrough (copy input to output)
     * for subsequent calls until N consecutive blocks fit the budget. */
    ABOBA_BUDGET_POLICY_BYPASS = 1
} aboba_budget_policy;

ABOBA_API aboba_status aboba_pipeline_set_budget_policy(
    aboba_pipeline* p, aboba_budget_policy policy);

/* Counters available to game-engine instrumentation. */
typedef struct {
    uint64_t total_blocks;
    uint64_t bypassed_blocks;       /* triggered by budget watchdog */
    uint64_t exception_recoveries;  /* caught C++ exceptions */
    float    last_block_us;         /* most recent processing time */
    float    p99_block_us;          /* rolling p99 estimate */
    int      currently_bypassed;    /* 1 = pipeline is in bypass mode */
} aboba_pipeline_stats;

ABOBA_API aboba_status aboba_pipeline_get_stats(
    const aboba_pipeline* p, aboba_pipeline_stats* out_stats);

/* ----- Latency reporting ----------------------------------------------- */

/* How many samples of buffering the pipeline introduces. Game engines
 * use this to compensate for A/V sync. */
ABOBA_API size_t aboba_pipeline_latency_samples(const aboba_pipeline* p);

/* =============================================================
 * Voice configs (TOML)
 *
 * Configs are POD-on-the-Python-side, opaque on the C side. Use the
 * accessors to inspect / mutate.
 * ============================================================= */

ABOBA_API aboba_status aboba_config_load_file(
    const char* path, aboba_config** out_config);
ABOBA_API aboba_status aboba_config_parse_string(
    const char* toml_text, aboba_config** out_config);

ABOBA_API void aboba_config_destroy(aboba_config* c);

/* If the loader failed, this returns the error message (otherwise NULL).
 * The pointer is valid until aboba_config_destroy or another
 * aboba_config_load_file/parse_string call from the same thread. */
ABOBA_API const char* aboba_config_last_error(void);
ABOBA_API int aboba_config_last_error_line(void);

/* Construct a pipeline from a config (encapsulates create + apply). */
ABOBA_API aboba_status aboba_pipeline_create_from_config(
    aboba_backend* backend,
    const aboba_config* cfg,
    aboba_pipeline** out_pipeline);

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif  /* ABOBA_C_H */
