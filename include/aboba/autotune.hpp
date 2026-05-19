// SPDX-License-Identifier: GPL-3.0-or-later
//
// PitchCorrector — autotune-style pitch quantization.
//
// Listens to incoming audio via YIN, detects F0, snaps it to the nearest
// note in a configured scale, and exposes a pitch ratio that the vocoder
// should apply to push the actual F0 onto the target.
//
// Two knobs control "how aggressive":
//   * strength ∈ [0,1]   1 = full T-Pain snap, 0 = pass-through
//   * smoothing ∈ [0,1]  1 = hold last ratio forever (smooth glide),
//                         0 = instant snap (per-block updates)
//
// Latency: one YIN window worth of input is needed before any correction
// kicks in. At 48 kHz / f0_min=60 Hz that's ~33 ms. After that, correction
// is essentially per-block (~5 ms at 256-sample blocks).
//
// NOT a vocal-melody composer. This snaps to scale notes only. For
// expressive auto-tune work you'd want to additionally control vibrato,
// inflection, and intentional un-snap zones — out of scope here.
//
// Paranoia:
//   * All buffers pre-allocated. Lock-free, allocation-free on the
//     audio path.
//   * NaN/Inf inputs sanitized at YIN entry.
//   * Pitch ratio clamped to a sane range (±2 octaves).
//   * Silence + unvoiced -> ratio stays at 1.0.
#pragma once

#include "yin.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace aboba {

enum class ScaleMode : std::uint8_t {
    Chromatic = 0,   // all 12 semitones (mild correction; just snap drift)
    Major     = 1,
    NaturalMinor = 2,
    HarmonicMinor = 3,
    Dorian    = 4,
    Mixolydian = 5,
    PentatonicMajor = 6,
    PentatonicMinor = 7,
    Blues     = 8,
    WholeTone = 9,
    Custom    = 10,  // use ScaleConfig::custom[12]
};

struct ScaleConfig {
    ScaleMode mode = ScaleMode::Chromatic;
    int  root_midi = 60;          // C4 (root note)
    // If mode == Custom, this bit-set determines allowed semitones
    // relative to root (index 0 = root, 11 = b7). Indices outside [0,11]
    // ignored.
    bool custom_semitones[12] = {true, false, true, false, true, true,
                                  false, true, false, true, false, true};
};

struct AutotuneConfig {
    double sample_rate = 48000.0;
    float  f0_min_hz   = 60.0f;
    float  f0_max_hz   = 1200.0f;
    ScaleConfig scale  = {};
    float strength     = 1.0f;     // [0,1]
    float smoothing    = 0.5f;     // [0,1]; higher = smoother glide
    // Aperiodicity threshold above which we treat the frame as unvoiced
    // and do not apply correction. Default matches YIN's default.
    float voicing_threshold = 0.20f;
    // How often (in samples) to re-run YIN. Default = hop ≈ 256.
    std::size_t hop_samples = 256;
};

struct AutotuneStats {
    float last_detected_f0_hz   = 0.0f;
    float last_target_f0_hz     = 0.0f;
    int   last_target_midi      = -1;
    float last_aperiodicity     = 1.0f;
    bool  last_voiced           = false;
    float current_ratio         = 1.0f;
    std::uint64_t frames_total  = 0;
    std::uint64_t frames_voiced = 0;
};

class PitchCorrector {
public:
    explicit PitchCorrector(AutotuneConfig cfg);

    // Feed audio in. Internal ring buffer accumulates and YIN runs every
    // `hop_samples`. Returns the *current* pitch ratio to apply (the one
    // computed from the most recent YIN run; 1.0 if not yet computed or
    // unvoiced).
    float process_block(const float* in, std::size_t n) noexcept;

    // Just read the ratio without feeding new data.
    float current_ratio() const noexcept { return current_ratio_; }

    void reset();

    void set_scale(const ScaleConfig& s) noexcept;
    void set_strength (float s) noexcept;
    void set_smoothing(float s) noexcept;
    void set_voicing_threshold(float v) noexcept;

    AutotuneStats stats() const noexcept;

private:
    // Run YIN once on the accumulated buffer; update internal state.
    void run_detection() noexcept;

    // Snap an F0 to the nearest scale note. Returns target F0 in Hz.
    float snap_to_scale(float f0_hz) const noexcept;

    AutotuneConfig cfg_;
    std::unique_ptr<YinDetector> yin_;

    // Ring buffer for YIN input — sized to YIN's needed window + tau_max
    std::vector<float> buf_;
    std::size_t        buf_capacity_ = 0;
    std::size_t        buf_fill_     = 0;
    std::size_t        samples_since_yin_ = 0;

    float current_ratio_ = 1.0f;
    float target_ratio_  = 1.0f;    // before smoothing

    // Stats
    float last_f0_hz_     = 0.0f;
    float last_target_hz_ = 0.0f;
    int   last_target_midi_ = -1;
    float last_aper_      = 1.0f;
    bool  last_voiced_    = false;
    std::uint64_t cnt_total_  = 0;
    std::uint64_t cnt_voiced_ = 0;
};

// Helpers — exposed so tests can verify scale mathematics.
int  midi_for_hz(float hz) noexcept;
float hz_for_midi(int midi) noexcept;
bool  semitone_in_scale(int semitone_from_root, const ScaleConfig& s) noexcept;

}  // namespace aboba
