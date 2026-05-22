// SPDX-License-Identifier: GPL-3.0-or-later
//
// paranoia.hpp — centralized defensive utilities used throughout Aboba.
//
// Philosophy:
//   Every audio pipeline has multiple "danger zones" — places where bad
//   input, numerical instability, or a misbehaving caller can corrupt
//   output, leak memory, or crash the process. We add defenses at EVERY
//   layer, not just the outer one. This is deliberate redundancy: if a
//   bug bypasses the C ABI's null check, the pipeline's bounds check
//   still catches it; if that fails, the DSP block's NaN sanitizer
//   catches the propagated noise; if that fails, the limiter clamps
//   output to [-1, 1].
//
// What this header provides:
//   * Compile-time hardening macros (ABOBA_LIKELY, ABOBA_UNLIKELY)
//   * Runtime assertions that DON'T crash in release (ABOBA_CHECK)
//   * Input validation helpers (validate_buffer, validate_sample_rate)
//   * NaN/Inf sanitizers (one sample, one block, one buffer-pair)
//   * Realtime allocation detector (debug builds only)
//   * Stack-corruption canary checks for critical paths
//   * Bound clamping for float parameters
//
// All functions are header-only / inline-friendly. Zero overhead in
// release builds for the common case.
#pragma once

#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>

// =============================================================
// Branch prediction hints
// =============================================================
#if defined(__GNUC__) || defined(__clang__)
#  define ABOBA_LIKELY(x)   __builtin_expect(!!(x), 1)
#  define ABOBA_UNLIKELY(x) __builtin_expect(!!(x), 0)
#  define ABOBA_HOT         __attribute__((hot))
#  define ABOBA_COLD        __attribute__((cold))
#  define ABOBA_ALWAYS_INLINE inline __attribute__((always_inline))
#  define ABOBA_NOINLINE    __attribute__((noinline))
#else
#  define ABOBA_LIKELY(x)   (x)
#  define ABOBA_UNLIKELY(x) (x)
#  define ABOBA_HOT
#  define ABOBA_COLD
#  define ABOBA_ALWAYS_INLINE inline
#  define ABOBA_NOINLINE
#endif

namespace aboba {
namespace paranoia {

// =============================================================
// Runtime checks
// =============================================================
//
// ABOBA_CHECK(cond, msg)         — if !cond, throw with msg (cold path).
// ABOBA_CHECK_NOTHROW(cond, msg) — if !cond, log and continue (debug only).
// ABOBA_REQUIRE(cond)            — fatal in debug, throws in release.
//
// We DELIBERATELY throw runtime errors rather than calling std::abort,
// because the C ABI layer catches them and translates to status codes.
// Aborting the process is too hostile for a library used inside game
// engines and DAWs.

[[noreturn]] ABOBA_COLD ABOBA_NOINLINE
inline void check_failed(const char* msg, const char* file, int line) {
    std::string s = "aboba check failed at ";
    s += file; s += ":"; s += std::to_string(line); s += ": "; s += msg;
    throw std::runtime_error(s);
}

#define ABOBA_CHECK(cond, msg)                                          \
    do {                                                                \
        if (ABOBA_UNLIKELY(!(cond))) {                                  \
            ::aboba::paranoia::check_failed(msg, __FILE__, __LINE__);   \
        }                                                               \
    } while (0)

#define ABOBA_CHECK_RANGE(value, lo, hi, name)                          \
    do {                                                                \
        if (ABOBA_UNLIKELY((value) < (lo) || (value) > (hi))) {         \
            ::aboba::paranoia::check_failed(                            \
                name " out of range", __FILE__, __LINE__);              \
        }                                                               \
    } while (0)

// =============================================================
// Pointer / buffer validation
// =============================================================

ABOBA_ALWAYS_INLINE void validate_buffer(const void* ptr, std::size_t n,
                                          const char* what) {
    if (ABOBA_UNLIKELY(n == 0)) return;  // empty buffer is fine
    ABOBA_CHECK(ptr != nullptr, what);
}

ABOBA_ALWAYS_INLINE void validate_size(std::size_t n, std::size_t max_n,
                                        const char* what) {
    ABOBA_CHECK(n <= max_n, what);
}

ABOBA_ALWAYS_INLINE void validate_sample_rate(double sr) {
    // We support 8 kHz .. 384 kHz. Below this is unreasonable; above is
    // also unreasonable for voice.
    ABOBA_CHECK(sr >= 8000.0 && sr <= 384000.0, "sample_rate out of range");
}

// =============================================================
// NaN / Inf sanitization
// =============================================================
//
// Real-time audio HATES non-finite values. A single NaN propagates
// through every IIR filter and ruins the rest of the session. We catch
// them at every block boundary.

ABOBA_ALWAYS_INLINE float sanitize_sample(float x) noexcept {
    // The ternary is the fastest portable check on x86_64; std::isfinite
    // is also fast in optimized builds but produces uglier asm.
    return (x == x && x > -std::numeric_limits<float>::infinity()
                   && x <  std::numeric_limits<float>::infinity()) ? x : 0.0f;
}

// Returns the number of NaN/Inf values found and replaced.
ABOBA_HOT
inline std::size_t sanitize_block(float* buf, std::size_t n) noexcept {
    std::size_t count = 0;
    for (std::size_t i = 0; i < n; ++i) {
        const float v = buf[i];
        if (ABOBA_UNLIKELY(!(v == v && v > -std::numeric_limits<float>::infinity()
                                    && v <  std::numeric_limits<float>::infinity()))) {
            buf[i] = 0.0f;
            ++count;
        }
    }
    return count;
}

// Validate input WITHOUT modifying; returns false if any non-finite.
ABOBA_HOT
inline bool block_is_finite(const float* buf, std::size_t n) noexcept {
    for (std::size_t i = 0; i < n; ++i) {
        const float v = buf[i];
        if (ABOBA_UNLIKELY(!(v == v && v > -std::numeric_limits<float>::infinity()
                                    && v <  std::numeric_limits<float>::infinity()))) {
            return false;
        }
    }
    return true;
}

// =============================================================
// Output limiter — last-line-of-defense clamp.
// =============================================================
//
// Even if a NaN was introduced and missed, even if a parameter swept
// past its range, even if a buffer was over-amplified — this guarantees
// the speaker doesn't blow up.

ABOBA_ALWAYS_INLINE float hard_limit(float x, float ceiling = 1.0f) noexcept {
    // First sanitize, then clamp.
    x = sanitize_sample(x);
    if (x >  ceiling) return  ceiling;
    if (x < -ceiling) return -ceiling;
    return x;
}

ABOBA_HOT
inline std::size_t hard_limit_block(float* buf, std::size_t n,
                                     float ceiling = 1.0f) noexcept {
    std::size_t clipped = 0;
    for (std::size_t i = 0; i < n; ++i) {
        const float v = sanitize_sample(buf[i]);
        if (v >  ceiling) { buf[i] =  ceiling; ++clipped; }
        else if (v < -ceiling) { buf[i] = -ceiling; ++clipped; }
        else { buf[i] = v; }
    }
    return clipped;
}

// =============================================================
// Parameter clamping (for setter functions)
// =============================================================

ABOBA_ALWAYS_INLINE float clamp_to(float x, float lo, float hi) noexcept {
    if (!(x == x)) return (lo + hi) * 0.5f;  // NaN -> midpoint
    if (x < lo) return lo;
    if (x > hi) return hi;
    return x;
}

// =============================================================
// Realtime allocation detector (DEBUG ONLY)
// =============================================================
//
// In debug builds, code paths that promise to be alloc-free can register
// a scope where any operator new throws. This is a development aid; in
// release builds it's a no-op.

#ifndef NDEBUG
class NoAllocScope {
public:
    NoAllocScope() noexcept  { active().fetch_add(1, std::memory_order_relaxed); }
    ~NoAllocScope() noexcept { active().fetch_sub(1, std::memory_order_relaxed); }
    NoAllocScope(const NoAllocScope&) = delete;
    NoAllocScope& operator=(const NoAllocScope&) = delete;
    static bool is_active() noexcept {
        return active().load(std::memory_order_relaxed) > 0;
    }
private:
    static std::atomic<int>& active() noexcept {
        static std::atomic<int> s{0};
        return s;
    }
};
#else
class NoAllocScope {
public:
    NoAllocScope() noexcept = default;
    static bool is_active() noexcept { return false; }
};
#endif

// =============================================================
// Stack canary for critical paths.
// =============================================================
//
// Used in functions where corrupted stack/locals would be a security
// issue (e.g. when processing untrusted TOML configs). Compiler-level
// -fstack-protector-strong is the real defense; this is belt-and-suspenders.

class StackCanary {
public:
    static constexpr std::uint64_t kMagic = 0xDEADBEEFCAFEBABE;
    StackCanary() noexcept : canary_(kMagic) {}
    ~StackCanary() noexcept(false) {
        if (ABOBA_UNLIKELY(canary_ != kMagic)) {
            // Stack corruption detected. Don't try to do anything clever;
            // just terminate — this almost certainly means we're in a
            // very bad state.
            std::fprintf(stderr,
                "FATAL: aboba stack canary corrupted (got %llx, expected %llx)\n",
                static_cast<unsigned long long>(canary_),
                static_cast<unsigned long long>(kMagic));
            std::abort();
        }
    }
private:
    volatile std::uint64_t canary_;
};

// =============================================================
// Audio statistics for monitoring / self-check
// =============================================================

struct AudioBlockStats {
    float   peak_abs        = 0.0f;
    float   rms             = 0.0f;
    std::size_t nan_count    = 0;
    std::size_t inf_count    = 0;
    std::size_t clipped_count = 0;  // |x| >= 1.0
    std::size_t denormal_count = 0; // |x| < 1e-30
};

ABOBA_HOT
inline AudioBlockStats analyze_block(const float* buf, std::size_t n) noexcept {
    AudioBlockStats s;
    if (n == 0) return s;
    double ss = 0.0;
    float peak = 0.0f;
    for (std::size_t i = 0; i < n; ++i) {
        const float v = buf[i];
        if (v != v) { ++s.nan_count; continue; }
        if (v > std::numeric_limits<float>::max() ||
            v < -std::numeric_limits<float>::max()) { ++s.inf_count; continue; }
        const float a = (v < 0.0f) ? -v : v;
        if (a >= 1.0f) ++s.clipped_count;
        if (a > 0.0f && a < 1e-30f) ++s.denormal_count;
        if (a > peak) peak = a;
        ss += static_cast<double>(v) * v;
    }
    s.peak_abs = peak;
    s.rms = static_cast<float>(std::sqrt(ss / static_cast<double>(n)));
    return s;
}

// ============================================================
// HOSTILE-INPUT DEFENSE (added in paranoid-hardening pass)
// ============================================================
//
// These helpers guard against adversarial / malformed input from external
// callers (C ABI, VST3 host, game engine, Python). They are designed for
// fast-path use: zero allocations, branchless where possible.
//
// PHILOSOPHY: in a real-time audio path, the cost of returning silence
// for one block is microscopic. The cost of UB or a crash is the whole
// host going down. So when in doubt, refuse-and-passthrough.

// (Uses ABOBA_ALWAYS_INLINE, ABOBA_LIKELY, ABOBA_UNLIKELY defined above.)

// --- Resource ceilings -------------------------------------------------
// Refuse to honor pathologically large requests. These are HUGE for
// legitimate use (16M samples = 6 minutes @ 48 kHz; 1M-point FFT) while
// still bounding allocation requests to prevent OOM via maliciously large
// `n_samples` arguments from FFI callers.
constexpr std::size_t kMaxBlockSamplesHard = 16u * 1024u * 1024u;
constexpr std::size_t kMaxFftSize          = 1u << 20;
constexpr std::size_t kMaxChannels         = 256;
constexpr std::size_t kMaxBatch            = 16u * 1024u;
constexpr double      kMinSampleRate       = 4000.0;
constexpr double      kMaxSampleRate       = 384000.0;
constexpr int         kMaxSemitones        = 60;
constexpr std::size_t kMaxConfigStringLen  = 4096;

ABOBA_ALWAYS_INLINE bool reject_huge_block(std::size_t n) noexcept {
    return ABOBA_UNLIKELY(n > kMaxBlockSamplesHard);
}
ABOBA_ALWAYS_INLINE bool reject_huge_fft(std::size_t n) noexcept {
    return ABOBA_UNLIKELY(n > kMaxFftSize || n < 2);
}
ABOBA_ALWAYS_INLINE bool reject_huge_batch(std::size_t b) noexcept {
    return ABOBA_UNLIKELY(b > kMaxBatch);
}
ABOBA_ALWAYS_INLINE bool valid_sample_rate_strict(double sr) noexcept {
    return std::isfinite(sr) && sr >= kMinSampleRate && sr <= kMaxSampleRate;
}

// Multiply two size_t values; return true on overflow OR if the result
// exceeds our block ceiling. Use before computing total buffer sizes.
ABOBA_ALWAYS_INLINE bool mul_overflows_block(std::size_t a, std::size_t b) noexcept {
    if (a == 0 || b == 0) return false;
    return ABOBA_UNLIKELY(a > kMaxBlockSamplesHard / b);
}

// --- Null-pointer fast checks ----------------------------------------
ABOBA_ALWAYS_INLINE bool any_null(const void* a) noexcept {
    return ABOBA_UNLIKELY(a == nullptr);
}
ABOBA_ALWAYS_INLINE bool any_null(const void* a, const void* b) noexcept {
    return ABOBA_UNLIKELY(a == nullptr || b == nullptr);
}
ABOBA_ALWAYS_INLINE bool any_null(const void* a, const void* b,
                                   const void* c) noexcept {
    return ABOBA_UNLIKELY(a == nullptr || b == nullptr || c == nullptr);
}

// --- Aliasing detection ----------------------------------------------
//
// in_place: in == out is OK (most DSP supports it).
// partial_overlap: out points INTO the middle of in (or vice versa). This
// would corrupt processing. We detect and refuse.
ABOBA_ALWAYS_INLINE bool unsafe_partial_overlap(const float* a, const float* b,
                                                 std::size_t n) noexcept {
    if (a == b) return false;
    if (n == 0) return false;
    const std::uintptr_t pa = reinterpret_cast<std::uintptr_t>(a);
    const std::uintptr_t pb = reinterpret_cast<std::uintptr_t>(b);
    const std::uintptr_t end_a = pa + n * sizeof(float);
    const std::uintptr_t end_b = pb + n * sizeof(float);
    return (pa < end_b) && (pb < end_a);
}

// --- Shorthand clamps -------------------------------------------------
// `clamp_to(x, lo, hi)` already exists above and handles NaN. These are
// just commonly-used shortcuts.
ABOBA_ALWAYS_INLINE float clamp_semitones(float st) noexcept {
    return clamp_to(st, -static_cast<float>(kMaxSemitones),
                         static_cast<float>(kMaxSemitones));
}
ABOBA_ALWAYS_INLINE float clamp01(float x) noexcept {
    return clamp_to(x, 0.0f, 1.0f);
}

// --- Emergency fallback ----------------------------------------------
// When we detect bad input on a real-time path, the safest action is to
// passthrough. If even that's impossible (out == null), do nothing.
inline void emergency_passthrough(const float* in, float* out,
                                  std::size_t n) noexcept {
    if (!out || n == 0) return;
    if (!in) {
        std::memset(out, 0, n * sizeof(float));
        return;
    }
    if (in == out) return;
    std::memcpy(out, in, n * sizeof(float));
}

// --- String validation -----------------------------------------------
// Reject null, empty, or unreasonably long strings before storing them.
// Returns true if string is BAD (should be rejected).
ABOBA_ALWAYS_INLINE bool reject_string(const char* s,
                                        std::size_t max_len = kMaxConfigStringLen) noexcept {
    if (!s) return true;
    if (s[0] == '\0') return true;
    // Bound-limited strnlen
    for (std::size_t i = 0; i <= max_len; ++i) {
        if (s[i] == '\0') return false;  // valid
    }
    return true;  // exceeded max_len without finding terminator
}

// --- RAII output sanitizer -------------------------------------------
// Wrap a DSP block that may produce NaN/Inf. On scope exit, scrubs the
// output buffer. Use at the BOUNDARY of any externally-visible function.
//
//     void process(float* out, std::size_t n) {
//         paranoia::ScopedOutputSanitizer guard(out, n);
//         do_potentially_unsafe_dsp(out, n);
//     }   // guard's destructor scrubs NaN/Inf from `out`
struct ScopedOutputSanitizer {
    float*      buf;
    std::size_t n;
    float       bound;

    ScopedOutputSanitizer(float* b, std::size_t sz, float bd = 4.0f) noexcept
        : buf(b), n(sz), bound(bd) {}

    ~ScopedOutputSanitizer() {
        if (!buf || n == 0) return;
        for (std::size_t i = 0; i < n; ++i) {
            float v = buf[i];
            if (!std::isfinite(v))      v = 0.0f;
            else if (v >  bound)        v =  bound;
            else if (v < -bound)        v = -bound;
            buf[i] = v;
        }
    }

    ScopedOutputSanitizer(const ScopedOutputSanitizer&)            = delete;
    ScopedOutputSanitizer& operator=(const ScopedOutputSanitizer&) = delete;
};

}  // namespace paranoia
}  // namespace aboba
