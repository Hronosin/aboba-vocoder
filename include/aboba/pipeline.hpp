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
#include "quality.hpp"

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
    bool enable_dc_blocker = true;
    bool enable_highpass   = true;
    bool enable_gate       = true;
    bool enable_vocoder    = true;
    bool enable_de_esser   = true;
    bool enable_limiter    = true;

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
    void set_profile(QualityProfile p);

    void reset();

    // Latency in samples is dominated by the vocoder
    std::size_t latency_samples() const noexcept { return vocoder_->latency_samples(); }

    // Diagnostics
    FormantVocoderStats vocoder_stats() const noexcept { return vocoder_->stats(); }
    bool gate_open() const noexcept { return gate_.is_open(); }

private:
    void apply_profile_defaults();

    VoicePipelineConfig cfg_;
    Backend*            backend_;

    DcBlocker        dc_;
    OnePoleHighPass  hp_;
    NoiseGate        gate_;
    std::unique_ptr<FormantVocoder> vocoder_;
    DeEsser          de_;
    SoftLimiter      limiter_;

    // Inter-stage scratch — sized to a "reasonable" maximum block.
    std::vector<float> stage_a_;
    std::vector<float> stage_b_;
};

}  // namespace aboba
