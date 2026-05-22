// SPDX-License-Identifier: GPL-3.0-or-later
//
// aboba_c.cpp — C ABI bridge implementation.
//
// Rules:
//   * Every entry point catches ALL C++ exceptions. We translate to a
//     status code. NO exception ever crosses the C boundary.
//   * Handles are opaque pointers to C++ objects. We never expose their
//     internal layout.
//   * Null-pointer arguments return ABOBA_ERR_NULL_POINTER, not segfault.
//   * We treat the user as an adversary: validate everything.
//
#define ABOBA_C_BUILDING
#include "aboba_c.h"

#include "aboba/backend.hpp"
#include "aboba/lowlatency.hpp"
#include "aboba/paranoia.hpp"
#include "aboba/pipeline.hpp"
#include "aboba/voice_character.hpp"
#include "aboba/voice_config.hpp"

#include <atomic>
#include <chrono>
#include <cmath>
#include <cstring>
#include <memory>
#include <new>
#include <string>

// We can't expose C++ types in the ABI but we still need to package state
// into the opaque handles. Each handle is a struct with the underlying
// C++ objects plus any state needed for the watchdog/budget machinery.

struct aboba_backend_s {
    std::unique_ptr<aboba::Backend> backend;
};

struct aboba_pipeline_s {
    // EXACTLY ONE of `normal` and `lowlatency` is set.
    std::unique_ptr<aboba::VoicePipeline>      normal;
    std::unique_ptr<aboba::LowLatencyPipeline> lowlatency;

    // Watchdog state, used for the normal pipeline path (lowlatency has
    // its own internal watchdog).
    int max_block_us  = 5000;
    aboba_budget_policy budget_policy = ABOBA_BUDGET_POLICY_LOG;
    std::atomic<bool> currently_bypassed {false};
    int  recovery_count = 0;
    static constexpr int kRecoveryBlocksNeeded = 8;

    // PARANOIA: pre-allocated probe buffer used by the bypass-recovery
    // path. Allocating on the hot audio path is a DoS / glitch vector;
    // we size this lazily but only ONCE per pipeline, never during
    // process(). Access only from the calling thread (single-threaded
    // per-handle, per the ABI contract).
    std::vector<float> probe_buffer;

    // Stats
    std::atomic<std::uint64_t> total_blocks       {0};
    std::atomic<std::uint64_t> bypassed_blocks    {0};
    std::atomic<std::uint64_t> exception_recovers {0};
    std::atomic<std::uint64_t> rejected_blocks    {0};  // input validation
    std::atomic<float>         last_us            {0.0f};
    std::atomic<float>         p99_us             {0.0f};
};

struct aboba_config_s {
    aboba::VoiceConfig cfg;
};

// Thread-local error message for the loader. Cleared/overwritten on every
// load/parse call so the user can query the most recent error.
namespace {
thread_local std::string g_last_config_error;
thread_local int         g_last_config_error_line = 0;
}  // namespace

// =========================================================================
// Helpers
// =========================================================================

namespace {

const char* k_status_messages[] = {
    "OK",
    "Invalid argument",
    "Null pointer",
    "Buffer size mismatch",
    "No backend available",
    "Internal C++ exception",
    "Not implemented",
    "Value out of range",
    "File I/O error",
    "Parse error",
};

// Negate status to use as index into the table
inline const char* msg_for_status(aboba_status s) {
    if (s == ABOBA_OK) return k_status_messages[0];
    const int idx = -s;
    if (idx < 1 || idx >= static_cast<int>(sizeof(k_status_messages) /
                                            sizeof(k_status_messages[0]))) {
        return "Unknown error";
    }
    return k_status_messages[idx];
}

template <typename F>
aboba_status guard(F&& f) {
    try {
        return f();
    } catch (const std::invalid_argument&) {
        return ABOBA_ERR_INVALID_ARG;
    } catch (const std::out_of_range&) {
        return ABOBA_ERR_OUT_OF_RANGE;
    } catch (const std::bad_alloc&) {
        return ABOBA_ERR_INTERNAL;
    } catch (const std::exception&) {
        return ABOBA_ERR_INTERNAL;
    } catch (...) {
        return ABOBA_ERR_INTERNAL;
    }
}

aboba::QualityProfile to_cpp_profile(aboba_quality_profile p) {
    switch (p) {
        case ABOBA_PROFILE_QUALITY:     return aboba::QualityProfile::Quality;
        case ABOBA_PROFILE_BALANCED:    return aboba::QualityProfile::Balanced;
        case ABOBA_PROFILE_PERFORMANCE: return aboba::QualityProfile::Performance;
    }
    return aboba::QualityProfile::Balanced;
}

aboba::VoiceCharacter to_cpp_character(aboba_character c) {
    const int idx = static_cast<int>(c);
    if (idx < 0 || idx >= aboba::character_count()) {
        return aboba::VoiceCharacter::Neutral;
    }
    return static_cast<aboba::VoiceCharacter>(idx);
}

aboba_character from_cpp_character(aboba::VoiceCharacter c) {
    return static_cast<aboba_character>(static_cast<int>(c));
}

aboba::MusicalScale to_cpp_scale(aboba_scale s) {
    switch (s) {
        case ABOBA_SCALE_CHROMATIC:        return aboba::MusicalScale::Chromatic;
        case ABOBA_SCALE_MAJOR:            return aboba::MusicalScale::Major;
        case ABOBA_SCALE_MINOR:            return aboba::MusicalScale::Minor;
        case ABOBA_SCALE_HARMONIC_MINOR:   return aboba::MusicalScale::HarmonicMinor;
        case ABOBA_SCALE_PENTATONIC_MAJOR: return aboba::MusicalScale::PentatonicMajor;
        case ABOBA_SCALE_PENTATONIC_MINOR: return aboba::MusicalScale::PentatonicMinor;
        case ABOBA_SCALE_BLUES:            return aboba::MusicalScale::Blues;
        case ABOBA_SCALE_WHOLE_TONE:       return aboba::MusicalScale::WholeTone;
        case ABOBA_SCALE_CUSTOM:           return aboba::MusicalScale::Custom;
    }
    return aboba::MusicalScale::Chromatic;
}

}  // namespace

// =========================================================================
// Versioning
// =========================================================================

extern "C" {

int aboba_runtime_abi_version_major(void) { return ABOBA_C_API_VERSION_MAJOR; }
int aboba_runtime_abi_version_minor(void) { return ABOBA_C_API_VERSION_MINOR; }

const char* aboba_runtime_version_string(void) {
#ifdef ABOBA_VERSION_STRING
    return ABOBA_VERSION_STRING;
#else
    return "0.0.0";
#endif
}

const char* aboba_status_message(aboba_status s) {
    return msg_for_status(s);
}

// =========================================================================
// Backend
// =========================================================================

aboba_status aboba_backend_create_best(aboba_backend** out_backend) {
    if (!out_backend) return ABOBA_ERR_NULL_POINTER;
    *out_backend = nullptr;
    return guard([&]() -> aboba_status {
        auto cpp_be = aboba::create_best_backend();
        if (!cpp_be) return ABOBA_ERR_NO_BACKEND;
        auto h = new aboba_backend_s;
        h->backend = std::move(cpp_be);
        *out_backend = h;
        return ABOBA_OK;
    });
}

aboba_status aboba_backend_create(aboba_backend_type type,
                                   aboba_backend** out_backend) {
    if (!out_backend) return ABOBA_ERR_NULL_POINTER;
    *out_backend = nullptr;
    return guard([&]() -> aboba_status {
        aboba::BackendType cpp_type =
            (type == ABOBA_BACKEND_TYPE_HIP) ? aboba::BackendType::HIP
                                              : aboba::BackendType::CPU;
        auto cpp_be = aboba::create_backend(cpp_type);
        if (!cpp_be) return ABOBA_ERR_NO_BACKEND;
        auto h = new aboba_backend_s;
        h->backend = std::move(cpp_be);
        *out_backend = h;
        return ABOBA_OK;
    });
}

void aboba_backend_destroy(aboba_backend* b) {
    if (!b) return;
    delete b;
}

const char* aboba_backend_name(const aboba_backend* b) {
    if (!b || !b->backend) return "";
    return b->backend->name();
}

aboba_backend_type aboba_backend_type_get(const aboba_backend* b) {
    if (!b || !b->backend) return ABOBA_BACKEND_TYPE_CPU;
    return (b->backend->type() == aboba::BackendType::HIP)
        ? ABOBA_BACKEND_TYPE_HIP : ABOBA_BACKEND_TYPE_CPU;
}

// =========================================================================
// Pipeline lifecycle
// =========================================================================

aboba_status aboba_pipeline_create(aboba_backend* backend,
                                    double sample_rate,
                                    aboba_quality_profile profile,
                                    aboba_pipeline** out_pipeline) {
    if (!backend || !backend->backend) return ABOBA_ERR_NULL_POINTER;
    if (!out_pipeline) return ABOBA_ERR_NULL_POINTER;
    *out_pipeline = nullptr;
    // PARANOIA: reject NaN/Inf/out-of-range sample rates. The pipeline
    // would crash deep inside the FFT if given garbage here.
    if (!::aboba::paranoia::valid_sample_rate_strict(sample_rate)) {
        return ABOBA_ERR_INVALID_ARG;
    }
    return guard([&]() -> aboba_status {
        aboba::VoicePipelineConfig pc;
        pc.sample_rate = sample_rate;
        pc.profile     = to_cpp_profile(profile);
        auto h = new aboba_pipeline_s;
        h->normal = std::make_unique<aboba::VoicePipeline>(pc, backend->backend.get());
        *out_pipeline = h;
        return ABOBA_OK;
    });
}

aboba_status aboba_pipeline_create_lowlatency(aboba_backend* backend,
                                               double sample_rate,
                                               aboba_pipeline** out_pipeline) {
    if (!backend || !backend->backend) return ABOBA_ERR_NULL_POINTER;
    if (!out_pipeline) return ABOBA_ERR_NULL_POINTER;
    *out_pipeline = nullptr;
    if (!::aboba::paranoia::valid_sample_rate_strict(sample_rate)) {
        return ABOBA_ERR_INVALID_ARG;
    }
    return guard([&]() -> aboba_status {
        aboba::LowLatencyConfig lc;
        lc.sample_rate = sample_rate;
        lc.max_block_us = 2000;  // hard 2ms default
        auto h = new aboba_pipeline_s;
        h->lowlatency = std::make_unique<aboba::LowLatencyPipeline>(
            lc, backend->backend.get());
        h->max_block_us = 2000;
        h->budget_policy = ABOBA_BUDGET_POLICY_BYPASS;
        *out_pipeline = h;
        return ABOBA_OK;
    });
}

void aboba_pipeline_destroy(aboba_pipeline* p) {
    if (!p) return;
    delete p;
}

// =========================================================================
// Pipeline processing
// =========================================================================

// Hard cap on per-call block size. Anything bigger than this is almost
// certainly a logic bug (e.g. someone fed in bytes instead of sample count).
// 1 MiB samples = 21.8 seconds at 48 kHz — well above any realtime block.
static constexpr std::size_t kMaxBlockSamples = 1024 * 1024;

aboba_status aboba_pipeline_process(aboba_pipeline* p,
                                     const float* input,
                                     float* output,
                                     size_t n_samples) {
    // PARANOIA LAYER 1: pointer and size validation. ALL of these must
    // pass before we touch any buffer. We use the namespace-qualified
    // helpers so the inlined fast path optimizes well.
    namespace pn = ::aboba::paranoia;

    if (pn::any_null(p))               return ABOBA_ERR_NULL_POINTER;
    if (pn::any_null(input, output))   return ABOBA_ERR_NULL_POINTER;
    if (n_samples == 0)                return ABOBA_OK;
    if (pn::reject_huge_block(n_samples)) return ABOBA_ERR_BUFFER_SIZE;

    // PARANOIA LAYER 2: aliasing check. in == out is fine (in-place);
    // partial overlap would smear samples and corrupt processing.
    if (pn::unsafe_partial_overlap(input, output, n_samples)) {
        p->rejected_blocks.fetch_add(1, std::memory_order_relaxed);
        return ABOBA_ERR_INVALID_ARG;
    }

    return guard([&]() -> aboba_status {
        p->total_blocks.fetch_add(1, std::memory_order_relaxed);

        // PARANOIA LAYER 3: input NaN/Inf scan. If the caller hands us
        // garbage, we don't propagate it through internal IIR filters
        // (which would poison the pipeline state for many subsequent
        // blocks). Emit silence and count as a rejection.
        if (ABOBA_UNLIKELY(!pn::block_is_finite(input, n_samples))) {
            p->exception_recovers.fetch_add(1, std::memory_order_relaxed);
            std::memset(output, 0, n_samples * sizeof(float));
            return ABOBA_OK;
        }

        // PARANOIA LAYER 4: RAII output sanitizer. Even if every internal
        // stage misbehaves, the output buffer is scrubbed of NaN/Inf and
        // clamped on scope exit.
        pn::ScopedOutputSanitizer scrub_guard(output, n_samples, 4.0f);

        // ---- LowLatency path -----------------------------------------
        if (p->lowlatency) {
            try {
                p->lowlatency->process(input, output, n_samples);
            } catch (...) {
                p->exception_recovers.fetch_add(1, std::memory_order_relaxed);
                pn::emergency_passthrough(input, output, n_samples);
                return ABOBA_OK;
            }
            auto s = p->lowlatency->stats();
            p->last_us.store(s.last_block_us);
            p->p99_us.store(s.p99_block_us);
            p->currently_bypassed.store(s.currently_bypassed);
            if (s.currently_bypassed) {
                p->bypassed_blocks.fetch_add(1, std::memory_order_relaxed);
            }
            // LAYER 5: hard limiter — speaker safety, last line.
            pn::hard_limit_block(output, n_samples, 1.0f);
            return ABOBA_OK;
        }

        if (!p->normal) return ABOBA_ERR_NULL_POINTER;

        // ---- Normal pipeline: bypass-mode probe path -----------------
        if (p->currently_bypassed.load(std::memory_order_relaxed)) {
            // Default action: passthrough.
            std::memcpy(output, input, n_samples * sizeof(float));
            p->bypassed_blocks.fetch_add(1, std::memory_order_relaxed);

            // Probe: try full processing into pre-allocated buffer to
            // measure current cost. NO ALLOCATION on the hot path —
            // we lazily grow the probe buffer to fit `n_samples`, but
            // only when bypassed (which is by definition not the hot
            // path). The probe itself is exception-guarded.
            try {
                if (p->probe_buffer.size() < n_samples) {
                    // Resize defensively. If this throws bad_alloc we
                    // just stay in bypass (caught below).
                    p->probe_buffer.resize(n_samples);
                }
                const auto t0 = std::chrono::steady_clock::now();
                p->normal->process(input, p->probe_buffer.data(), n_samples);
                const auto t1 = std::chrono::steady_clock::now();
                const auto us = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
                p->last_us.store(static_cast<float>(us));
                if (p->max_block_us > 0 && us <= p->max_block_us) {
                    if (++p->recovery_count >= aboba_pipeline_s::kRecoveryBlocksNeeded) {
                        p->currently_bypassed.store(false);
                        p->recovery_count = 0;
                        // Adopt probe result for THIS block since we
                        // computed it in budget anyway.
                        std::memcpy(output, p->probe_buffer.data(),
                                     n_samples * sizeof(float));
                    }
                } else {
                    p->recovery_count = 0;
                }
            } catch (const std::bad_alloc&) {
                // OOM during probe buffer resize. Stay in bypass; we'll
                // try again on the next block when there might be more
                // memory available.
                p->exception_recovers.fetch_add(1, std::memory_order_relaxed);
                p->recovery_count = 0;
            } catch (...) {
                p->exception_recovers.fetch_add(1, std::memory_order_relaxed);
                p->recovery_count = 0;
            }
            // Scrub before returning (output may have probe data)
            return ABOBA_OK;
        }

        // ---- Normal pipeline: standard processing path ---------------
        const auto t0 = std::chrono::steady_clock::now();
        try {
            p->normal->process(input, output, n_samples);
        } catch (...) {
            // Any C++ exception during processing -> passthrough.
            p->exception_recovers.fetch_add(1, std::memory_order_relaxed);
            pn::emergency_passthrough(input, output, n_samples);
            return ABOBA_OK;
        }
        const auto t1 = std::chrono::steady_clock::now();
        const auto us = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
        const float us_f = static_cast<float>(us);
        p->last_us.store(us_f);
        {
            float prev = p->p99_us.load(std::memory_order_relaxed);
            const float decay = 0.995f;
            const float new_p99 = (us_f > prev) ? us_f : (prev * decay);
            p->p99_us.store(new_p99);
        }
        if (p->max_block_us > 0 && us > p->max_block_us) {
            p->bypassed_blocks.fetch_add(1, std::memory_order_relaxed);
            if (p->budget_policy == ABOBA_BUDGET_POLICY_BYPASS) {
                p->currently_bypassed.store(true);
                p->recovery_count = 0;
            }
        }
        // LAYER 5: hard limiter — speaker safety, even if the pipeline's
        // own internal limiter is somehow misconfigured.
        pn::hard_limit_block(output, n_samples, 1.0f);
        return ABOBA_OK;
    });
}

aboba_status aboba_pipeline_reset(aboba_pipeline* p) {
    if (!p) return ABOBA_ERR_NULL_POINTER;
    return guard([&]() -> aboba_status {
        if (p->normal)     p->normal->reset();
        if (p->lowlatency) p->lowlatency->reset();
        p->currently_bypassed.store(false);
        p->recovery_count = 0;
        return ABOBA_OK;
    });
}

// =========================================================================
// Pipeline setters (pitch, formant, character)
// =========================================================================

aboba_status aboba_pipeline_set_pitch_semitones(aboba_pipeline* p, float st) {
    if (!p) return ABOBA_ERR_NULL_POINTER;
    // Clamp at ABI boundary. Insane values (NaN, +/-1e30) would propagate
    // and could cause subsequent FFT calls to produce inf/nan output.
    st = ::aboba::paranoia::clamp_to(st, -48.0f, 48.0f);
    return guard([&]() -> aboba_status {
        if (p->normal)     p->normal->set_pitch_semitones(st);
        if (p->lowlatency) p->lowlatency->set_pitch_semitones(st);
        return ABOBA_OK;
    });
}

aboba_status aboba_pipeline_set_formant_semitones(aboba_pipeline* p, float st) {
    if (!p) return ABOBA_ERR_NULL_POINTER;
    st = ::aboba::paranoia::clamp_to(st, -24.0f, 24.0f);
    return guard([&]() -> aboba_status {
        if (p->normal) {
            p->normal->set_formant_semitones(st);
            return ABOBA_OK;
        }
        return ABOBA_OK;
    });
}

aboba_status aboba_pipeline_set_character(aboba_pipeline* p, aboba_character c) {
    if (!p) return ABOBA_ERR_NULL_POINTER;
    if (static_cast<int>(c) < 0 ||
        static_cast<int>(c) >= aboba::character_count())
        return ABOBA_ERR_OUT_OF_RANGE;
    return guard([&]() -> aboba_status {
        if (p->normal)     p->normal->set_character(to_cpp_character(c));
        if (p->lowlatency) {
            // For lowlatency, apply just the pitch component of the
            // character. Formant is ignored (no preservation stage).
            const auto params = aboba::character_params(to_cpp_character(c));
            p->lowlatency->set_pitch_semitones(params.pitch_semitones);
        }
        return ABOBA_OK;
    });
}

aboba_character aboba_pipeline_get_character(const aboba_pipeline* p) {
    if (!p) return ABOBA_CHARACTER_COUNT;
    try {
        if (p->normal) return from_cpp_character(p->normal->current_character());
    } catch (...) {}
    return ABOBA_CHARACTER_NEUTRAL;
}

// =========================================================================
// Autotune
// =========================================================================

aboba_status aboba_pipeline_set_autotune_enabled(aboba_pipeline* p, int enabled) {
    if (!p) return ABOBA_ERR_NULL_POINTER;
    if (!p->normal) return ABOBA_ERR_NOT_IMPLEMENTED;  // not in lowlatency
    return guard([&]() -> aboba_status {
        p->normal->set_autotune_enabled(enabled != 0);
        return ABOBA_OK;
    });
}

int aboba_pipeline_get_autotune_enabled(const aboba_pipeline* p) {
    if (!p || !p->normal) return 0;
    try { return p->normal->autotune_enabled() ? 1 : 0; } catch (...) { return 0; }
}

aboba_status aboba_pipeline_set_autotune_scale(aboba_pipeline* p,
                                                aboba_scale s, int root) {
    if (!p) return ABOBA_ERR_NULL_POINTER;
    if (!p->normal) return ABOBA_ERR_NOT_IMPLEMENTED;
    return guard([&]() -> aboba_status {
        p->normal->set_autotune_scale(to_cpp_scale(s), root);
        return ABOBA_OK;
    });
}

aboba_status aboba_pipeline_set_autotune_strength(aboba_pipeline* p, float s) {
    if (!p) return ABOBA_ERR_NULL_POINTER;
    if (!p->normal) return ABOBA_ERR_NOT_IMPLEMENTED;
    // PARANOIA: strength is a 0..1 mix coefficient
    s = ::aboba::paranoia::clamp01(s);
    return guard([&]() -> aboba_status {
        p->normal->set_autotune_strength(s);
        return ABOBA_OK;
    });
}

aboba_status aboba_pipeline_set_autotune_glide_ms(aboba_pipeline* p, float ms) {
    if (!p) return ABOBA_ERR_NULL_POINTER;
    if (!p->normal) return ABOBA_ERR_NOT_IMPLEMENTED;
    // PARANOIA: glide in [0, 1000] ms. Negative would cause unstable
    // exponential; huge would freeze pitch contour.
    ms = ::aboba::paranoia::clamp_to(ms, 0.0f, 1000.0f);
    return guard([&]() -> aboba_status {
        p->normal->set_autotune_glide_ms(ms);
        return ABOBA_OK;
    });
}

// =========================================================================
// Reverb
// =========================================================================

aboba_status aboba_pipeline_set_reverb_enabled(aboba_pipeline* p, int e) {
    if (!p) return ABOBA_ERR_NULL_POINTER;
    if (!p->normal) return ABOBA_ERR_NOT_IMPLEMENTED;
    return guard([&]() -> aboba_status {
        p->normal->set_reverb_enabled(e != 0);
        return ABOBA_OK;
    });
}

int aboba_pipeline_get_reverb_enabled(const aboba_pipeline* p) {
    if (!p || !p->normal) return 0;
    try { return p->normal->reverb_enabled() ? 1 : 0; } catch (...) { return 0; }
}

aboba_status aboba_pipeline_set_reverb_room_size(aboba_pipeline* p, float v) {
    if (!p) return ABOBA_ERR_NULL_POINTER;
    if (!p->normal) return ABOBA_ERR_NOT_IMPLEMENTED;
    // PARANOIA: room_size must be in [0, 1]. NaN/Inf would propagate to
    // the comb filter feedback coefficient and produce unbounded output.
    v = ::aboba::paranoia::clamp01(v);
    return guard([&]() -> aboba_status {
        p->normal->set_reverb_room_size(v); return ABOBA_OK;
    });
}

aboba_status aboba_pipeline_set_reverb_damping(aboba_pipeline* p, float v) {
    if (!p) return ABOBA_ERR_NULL_POINTER;
    if (!p->normal) return ABOBA_ERR_NOT_IMPLEMENTED;
    v = ::aboba::paranoia::clamp01(v);
    return guard([&]() -> aboba_status {
        p->normal->set_reverb_damping(v); return ABOBA_OK;
    });
}

aboba_status aboba_pipeline_set_reverb_wet(aboba_pipeline* p, float v) {
    if (!p) return ABOBA_ERR_NULL_POINTER;
    if (!p->normal) return ABOBA_ERR_NOT_IMPLEMENTED;
    v = ::aboba::paranoia::clamp01(v);
    return guard([&]() -> aboba_status {
        p->normal->set_reverb_wet(v); return ABOBA_OK;
    });
}

// =========================================================================
// Watchdog / stats
// =========================================================================

aboba_status aboba_pipeline_set_max_block_us(aboba_pipeline* p, int us) {
    if (!p) return ABOBA_ERR_NULL_POINTER;
    if (us < 0) return ABOBA_ERR_INVALID_ARG;
    return guard([&]() -> aboba_status {
        p->max_block_us = us;
        if (p->lowlatency) p->lowlatency->set_max_block_us(us);
        return ABOBA_OK;
    });
}

aboba_status aboba_pipeline_set_budget_policy(aboba_pipeline* p,
                                               aboba_budget_policy policy) {
    if (!p) return ABOBA_ERR_NULL_POINTER;
    return guard([&]() -> aboba_status {
        p->budget_policy = policy;
        if (p->lowlatency) {
            p->lowlatency->set_bypass_policy(
                policy == ABOBA_BUDGET_POLICY_BYPASS
                    ? aboba::LowLatencyPipeline::BypassPolicy::Bypass
                    : aboba::LowLatencyPipeline::BypassPolicy::Log);
        }
        if (policy == ABOBA_BUDGET_POLICY_LOG) {
            p->currently_bypassed.store(false);
        }
        return ABOBA_OK;
    });
}

aboba_status aboba_pipeline_get_stats(const aboba_pipeline* p,
                                       aboba_pipeline_stats* out_stats) {
    if (!p || !out_stats) return ABOBA_ERR_NULL_POINTER;
    return guard([&]() -> aboba_status {
        out_stats->total_blocks         = p->total_blocks.load();
        out_stats->bypassed_blocks      = p->bypassed_blocks.load();
        out_stats->exception_recoveries = p->exception_recovers.load();
        out_stats->last_block_us        = p->last_us.load();
        out_stats->p99_block_us         = p->p99_us.load();
        out_stats->currently_bypassed   = p->currently_bypassed.load() ? 1 : 0;
        return ABOBA_OK;
    });
}

size_t aboba_pipeline_latency_samples(const aboba_pipeline* p) {
    if (!p) return 0;
    try {
        if (p->normal)     return p->normal->latency_samples();
        if (p->lowlatency) return p->lowlatency->latency_samples();
    } catch (...) {}
    return 0;
}

// =========================================================================
// Voice config (TOML)
// =========================================================================

aboba_status aboba_config_load_file(const char* path, aboba_config** out_config) {
    if (!path || !out_config) return ABOBA_ERR_NULL_POINTER;
    *out_config = nullptr;
    // PARANOIA: bound the path length. A multi-megabyte path is either
    // attack or bug; refuse before passing to fopen().
    if (::aboba::paranoia::reject_string(path)) return ABOBA_ERR_INVALID_ARG;
    return guard([&]() -> aboba_status {
        auto r = aboba::load_voice_config(path);
        if (!r.ok()) {
            g_last_config_error      = r.error;
            g_last_config_error_line = r.line_number;
            // Distinguish I/O vs parse errors
            if (r.error.find("cannot open") != std::string::npos) {
                return ABOBA_ERR_FILE_IO;
            }
            return ABOBA_ERR_PARSE;
        }
        g_last_config_error.clear();
        g_last_config_error_line = 0;
        auto h = new aboba_config_s;
        h->cfg = std::move(*r.value);
        *out_config = h;
        return ABOBA_OK;
    });
}

aboba_status aboba_config_parse_string(const char* toml_text,
                                        aboba_config** out_config) {
    if (!toml_text || !out_config) return ABOBA_ERR_NULL_POINTER;
    *out_config = nullptr;
    // PARANOIA: bound TOML body to 16 MiB. Anything bigger is a memory
    // exhaustion attempt; we refuse before parsing.
    if (::aboba::paranoia::reject_string(toml_text, 16u * 1024u * 1024u))
        return ABOBA_ERR_INVALID_ARG;
    return guard([&]() -> aboba_status {
        auto r = aboba::parse_voice_config(toml_text);
        if (!r.ok()) {
            g_last_config_error      = r.error;
            g_last_config_error_line = r.line_number;
            return ABOBA_ERR_PARSE;
        }
        g_last_config_error.clear();
        g_last_config_error_line = 0;
        auto h = new aboba_config_s;
        h->cfg = std::move(*r.value);
        *out_config = h;
        return ABOBA_OK;
    });
}

void aboba_config_destroy(aboba_config* c) {
    if (!c) return;
    delete c;
}

const char* aboba_config_last_error(void) {
    return g_last_config_error.empty() ? nullptr : g_last_config_error.c_str();
}

int aboba_config_last_error_line(void) {
    return g_last_config_error_line;
}

aboba_status aboba_pipeline_create_from_config(aboba_backend* backend,
                                                const aboba_config* cfg,
                                                aboba_pipeline** out_pipeline) {
    if (!backend || !backend->backend || !cfg) return ABOBA_ERR_NULL_POINTER;
    if (!out_pipeline) return ABOBA_ERR_NULL_POINTER;
    *out_pipeline = nullptr;
    return guard([&]() -> aboba_status {
        auto pc = cfg->cfg.to_pipeline_config();
        auto h = new aboba_pipeline_s;
        h->normal = std::make_unique<aboba::VoicePipeline>(pc, backend->backend.get());
        cfg->cfg.apply_runtime(*h->normal);
        *out_pipeline = h;
        return ABOBA_OK;
    });
}

}  // extern "C"
