// SPDX-License-Identifier: GPL-3.0-or-later
//
// VoicePipeline: full chain from raw mic input to processed output.
//
//   input
//     |
//     v
//   [DC blocker]            <- remove offset/drift, always on
//     |
//     v
//   [High-pass]             <- remove rumble below 80 Hz, optional
//     |
//     v
//   [Noise gate]            <- silence breath/room noise, optional
//     |
//     v
//   [Formant vocoder]       <- the main thing
//     |
//     v
//   [De-esser]              <- tame sibilance, optional
//     |
//     v
//   [Soft limiter]          <- last line of defense against clipping
//     |
//     v
//   output
//
// All effects can be toggled. The pipeline holds pre-allocated scratch
// buffers for inter-stage data so we never allocate in process().
//
// Three quality profiles are exposed (see quality.hpp). Choosing a profile
// adjusts BOTH the vocoder internals AND which effects are enabled by
// default.
#pragma once

#include "backend.hpp"
#include "dsp_blocks.hpp"
#include "formant_vocoder.hpp"
#include "noise_reduction.hpp"
#include "pitch_corrector.hpp"
#include "quality.hpp"
#include "reverb.hpp"
#include "voice_character.hpp"

#include <cstddef>
#include <memory>
#include <vector>

namespace aboba {

struct VoicePipelineConfig {
    double sample_rate    = 48000.0;
    std::size_t fft_size  = 2048;
    std::size_t hop_size  = 512;

    QualityProfile profile = QualityProfile::Balanced;

    // Per-effect enable. When unset, defaults follow the profile:
    //   Quality:     all on
    //   Balanced:    all on except de-esser
    //   Performance: only DC blocker + limiter
    bool enable_dc_blocker     = true;
    bool enable_highpass       = true;
    bool enable_noise_reducer  = true;
    bool enable_gate           = true;
    bool enable_agc            = true;
    bool enable_vocoder        = true;
    bool enable_de_esser       = true;
    bool enable_reverb         = false;   // off by default (taste-dependent)
    bool enable_autotune       = false;   // off by default (creative effect)
    bool enable_limiter        = true;
    // Use LookaheadLimiter (sample-accurate ceiling) instead of SoftLimiter.
    // Adds ~2ms latency. Default off — turn on for broadcast/streaming.
    bool use_lookahead_limiter = false;

    // High-pass cutoff
    float highpass_cutoff_hz = 80.0f;

    // F0 range (passed to vocoder)
    float f0_min_hz = 60.0f;
    float f0_max_hz = 1200.0f;
};

class VoicePipeline {
public:
    VoicePipeline(VoicePipelineConfig cfg, Backend* backend);

    void process(const float* input, float* output, std::size_t n_samples);

    void set_pitch_semitones(float st);

    // Independent formant shift (see formant_vocoder.hpp for details).
    void set_formant_semitones(float st);

    // Reverb controls. set_reverb_enabled() also toggles the pipeline flag.
    void set_reverb_enabled(bool on) noexcept;
    void set_reverb_room_size(float v) noexcept;
    void set_reverb_damping(float v) noexcept;
    void set_reverb_wet(float v) noexcept;
    bool reverb_enabled() const noexcept { return cfg_.enable_reverb; }

    // ----- Autotune / pitch correction -----
    // The pitch corrector analyses input, decides on a correction in
    // semitones, and adds it to the manually-set pitch shift. So:
    //   final_pitch = manual_pitch_setting + autotune_correction
    // To use as pure autotune, leave manual pitch at 0.
    void set_autotune_enabled(bool on) noexcept;
    bool autotune_enabled() const noexcept { return cfg_.enable_autotune; }
    void set_autotune_scale(MusicalScale s, int root_semis = 0) noexcept;
    void set_autotune_custom_scale(std::uint16_t mask, int root_semis = 0) noexcept;
    void set_autotune_strength(float s) noexcept;   // 0..1
    void set_autotune_glide_ms(float ms) noexcept;
    PitchCorrectorStats autotune_stats() const noexcept;

    // Apply a preset combination of pitch + formant + (future) EQ/effects.
    // Other pipeline flags (gate, limiter) are NOT modified — they stay
    // however the user configured them.
    void set_character(VoiceCharacter c);

    // Current character (set explicitly via set_character, else Neutral).
    VoiceCharacter current_character() const noexcept { return current_character_; }

    void set_profile(QualityProfile p);

    void reset();

    // Latency in samples is dominated by the vocoder
    std::size_t latency_samples() const noexcept { return vocoder_->latency_samples(); }

    // Diagnostics
    FormantVocoderStats vocoder_stats() const noexcept { return vocoder_->stats(); }
    NoiseReductionStats noise_stats() const noexcept;
    bool gate_open() const noexcept { return gate_.is_open(); }

    // Calibrate noise profile from a known-quiet buffer.
    // No-op if noise reducer is disabled.
    void learn_noise_profile(const float* silence, std::size_t n);

private:
    void apply_profile_defaults();

    VoicePipelineConfig cfg_;
    Backend*            backend_;

    DcBlocker        dc_;
    OnePoleHighPass  hp_;
    std::unique_ptr<NoiseReducer> noise_reducer_;
    NoiseGate        gate_;
    AutoGain         agc_;
    std::unique_ptr<FormantVocoder> vocoder_;
    DeEsser          de_;
    std::unique_ptr<Reverb> reverb_;
    std::unique_ptr<PitchCorrector> autotune_;
    float                   manual_pitch_st_ = 0.0f;
    SoftLimiter      limiter_;
    LookaheadLimiter lookahead_;

    VoiceCharacter   current_character_ = VoiceCharacter::Neutral;

    // Inter-stage scratch — sized to a "reasonable" maximum block.
    std::vector<float> stage_a_;
    std::vector<float> stage_b_;
};

}  // namespace aboba
