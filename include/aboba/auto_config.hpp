// SPDX-License-Identifier: GPL-3.0-or-later
//
// Runtime auto-tuning: translate detected hardware (sysinfo + platform)
// into knobs the rest of the framework actually uses.
//
// The pattern is intentionally minimal. We do NOT auto-tune everything —
// only the few values where wrong defaults waste real cycles or latency:
//
//   * FFTW planner effort
//       Big workstation -> MEASURE (plan once, fly forever).
//       Anything small  -> ESTIMATE (plans in microseconds, ~5% slower runtime).
//   * FFTW thread count
//       Real-time audio chunks are TINY (256-2048 samples). Splitting one
//       small FFT across cores is almost always net-negative because of
//       thread sync overhead. The right answer is "1" for the audio thread
//       and let multi-channel parallelism happen at a higher level.
//       We expose this anyway because larger offline batches benefit.
//   * Suggested STFT FFT/hop sizes per quality profile
//       These are constants for now but live here so users have one place
//       to look.
//   * Suggested PortAudio block size
//       Below 128 frames: very low latency but very high callback rate.
//       Above 1024: noticeable lag. We pick something the hardware can
//       service comfortably.
//
// NONE of these are mandatory. If you build a vocoder directly with hard-
// coded values, that still works. AutoTuning is a hint, not a contract.
//
// Thread-safety: detect_tuning() reads cached sysinfo and is safe to call
// from any thread. The returned struct is plain data.
#pragma once

#include "quality.hpp"

#include <cstddef>
#include <cstdint>
#include <string>

namespace aboba {

// FFTW planner effort. Names match FFTW's own enum but we don't include
// FFTW headers here.
enum class FftwPlannerEffort : std::uint8_t {
    Estimate    = 0,   // microseconds to plan, ~5% slower runtime
    Measure     = 1,   // ~1 second to plan, fast runtime
    Patient     = 2,   // many seconds to plan, slightly faster runtime
    Exhaustive  = 3,   // do not use unless you enjoy waiting
};

inline const char* effort_name(FftwPlannerEffort e) {
    switch (e) {
        case FftwPlannerEffort::Estimate:   return "estimate";
        case FftwPlannerEffort::Measure:    return "measure";
        case FftwPlannerEffort::Patient:    return "patient";
        case FftwPlannerEffort::Exhaustive: return "exhaustive";
    }
    return "?";
}

struct AbobaTuning {
    // FFTW
    FftwPlannerEffort fftw_effort   = FftwPlannerEffort::Estimate;
    int               fftw_threads  = 1;  // see note above; usually 1

    // Suggested STFT parameters per profile.
    // FFT size MUST be a power of two between 64 and 2^20.
    // Hop size MUST be > 0 and <= fft_size; for Hann window, fft_size/4 is
    // ideal (perfect COLA² reconstruction).
    std::size_t fft_size_quality      = 4096;
    std::size_t fft_size_balanced     = 2048;
    std::size_t fft_size_performance  = 1024;
    std::size_t hop_divisor           = 4;     // hop = fft_size / hop_divisor

    // PortAudio block size (frames per callback). 0 = let the host decide.
    std::size_t pa_frames_per_buffer  = 256;

    // Suggested top-level worker pool for offline batch processing.
    // Real-time path should NOT spawn threads; use the audio callback.
    int batch_worker_threads = 1;

    // Buffer alignment hint, in bytes, for SIMD-friendly access.
    std::size_t suggested_alignment = 32;

    // Human-readable explanation of the choices above. One line per knob.
    // Useful in --info dumps. Not consumed programmatically.
    std::string rationale;
};

// Inspect the host and return a tuning that should perform well.
// The result is cached after the first call. Safe to call repeatedly.
const AbobaTuning& detect_tuning();

// Override the cached tuning (useful for tests and for hosts that want a
// specific shape regardless of detected hardware). Pass an empty rationale
// and we'll prepend "[user override] " for clarity.
void set_tuning_override(AbobaTuning t);

// Reset to auto-detected values. Mostly for tests.
void clear_tuning_override();

// Return the (fft_size, hop_size) pair for a given profile per current
// tuning. Convenience accessor.
struct StftSizes { std::size_t fft_size; std::size_t hop_size; };
StftSizes suggested_stft_sizes(QualityProfile p);

}  // namespace aboba
