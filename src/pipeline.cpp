// SPDX-License-Identifier: GPL-3.0-or-later
#include "aboba/pipeline.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>

namespace aboba {

namespace {
constexpr std::size_t kScratchInitial = 4096;
}

VoicePipeline::VoicePipeline(VoicePipelineConfig cfg, Backend* backend)
    : cfg_(cfg), backend_(backend) {
    if (!backend_) throw std::invalid_argument("VoicePipeline: null backend");
    if (!(cfg_.sample_rate > 0.0)) {
        throw std::invalid_argument("VoicePipeline: invalid sample_rate");
    }

    apply_profile_defaults();

    // Configure each stage
    dc_.configure(cfg_.sample_rate, 20.0f);
    hp_.configure(cfg_.sample_rate, cfg_.highpass_cutoff_hz);
    gate_.configure(cfg_.sample_rate, -45.0f, -55.0f, 0.005f, 0.150f);
    agc_.configure(cfg_.sample_rate, -16.0f, 24.0f, -12.0f, 50.0f, 1000.0f, -55.0f);
    de_.configure(cfg_.sample_rate, 5500.0f, -22.0f, 6.0f);
    limiter_.configure(cfg_.sample_rate, 0.95f, 0.050f);
    lookahead_.configure(cfg_.sample_rate, -1.0f, 2.0f, 50.0f);

    FormantVocoderConfig vc;
    vc.fft_size    = cfg_.fft_size;
    vc.hop_size    = cfg_.hop_size;
    vc.sample_rate = cfg_.sample_rate;
    vc.profile     = cfg_.profile;
    vc.use_voicing_gate = (cfg_.profile != QualityProfile::Performance);
    vc.f0_min_hz = cfg_.f0_min_hz;
    vc.f0_max_hz = cfg_.f0_max_hz;
    vocoder_ = std::make_unique<FormantVocoder>(vc, backend_);

    // Noise reducer uses its own (smaller) FFT for less latency
    NoiseReductionConfig nc;
    nc.sample_rate = cfg_.sample_rate;
    nc.fft_size    = 1024;
    nc.hop_size    = 256;
    // Tune aggressiveness by profile
    switch (cfg_.profile) {
        case QualityProfile::Quality:
            nc.oversubtraction = 1.5f;
            nc.spectral_floor  = 0.07f;
            nc.gain_smoothing  = 0.6f;
            break;
        case QualityProfile::Balanced:
            nc.oversubtraction = 1.5f;
            nc.spectral_floor  = 0.05f;
            nc.gain_smoothing  = 0.5f;
            break;
        case QualityProfile::Performance:
            nc.oversubtraction = 1.3f;
            nc.spectral_floor  = 0.10f;
            nc.gain_smoothing  = 0.4f;
            break;
    }
    noise_reducer_ = std::make_unique<NoiseReducer>(nc, backend_);

    ReverbConfig rc;
    rc.sample_rate = cfg_.sample_rate;
    rc.room_size   = 0.6f;
    rc.damping     = 0.3f;
    rc.wet         = 0.25f;
    reverb_ = std::make_unique<Reverb>(rc);

    // Pitch corrector — always constructed but only RUN if
    // cfg_.enable_autotune is true.
    PitchCorrectorConfig pcc;
    pcc.sample_rate = cfg_.sample_rate;
    pcc.f0_min_hz   = cfg_.f0_min_hz;
    pcc.f0_max_hz   = cfg_.f0_max_hz;
    pcc.scale       = MusicalScale::Chromatic;
    pcc.root_semis  = 0;
    pcc.strength    = 1.0f;
    pcc.glide_ms    = 30.0f;
    autotune_ = std::make_unique<PitchCorrector>(pcc);

    stage_a_.assign(kScratchInitial, 0.0f);
    stage_b_.assign(kScratchInitial, 0.0f);
}

void VoicePipeline::apply_profile_defaults() {
    // The user can still override individual flags after construction,
    // but the defaults track the profile.
    switch (cfg_.profile) {
        case QualityProfile::Quality:
            cfg_.use_lookahead_limiter = true;   // broadcast-grade ceiling
            break;
        case QualityProfile::Balanced:
            cfg_.enable_de_esser = false;
            break;
        case QualityProfile::Performance:
            cfg_.enable_highpass      = false;
            cfg_.enable_noise_reducer = false;
            cfg_.enable_gate          = false;
            cfg_.enable_agc           = false;
            cfg_.enable_de_esser      = false;
            // DC blocker + limiter always on
            break;
    }
}

void VoicePipeline::set_pitch_semitones(float st) {
    if (!std::isfinite(st)) st = 0.0f;
    manual_pitch_st_ = st;
    // If autotune is off, vocoder gets the manual setting directly. If on,
    // process() will sum manual + correction every block.
    if (!cfg_.enable_autotune) {
        vocoder_->set_pitch_semitones(st);
    }
}

void VoicePipeline::set_formant_semitones(float st) {
    vocoder_->set_formant_semitones(st);
}

void VoicePipeline::set_reverb_enabled(bool on) noexcept {
    cfg_.enable_reverb = on;
}
void VoicePipeline::set_reverb_room_size(float v) noexcept {
    reverb_->set_room_size(v);
}
void VoicePipeline::set_reverb_damping(float v) noexcept {
    reverb_->set_damping(v);
}
void VoicePipeline::set_reverb_wet(float v) noexcept {
    reverb_->set_wet(v);
}

void VoicePipeline::set_autotune_enabled(bool on) noexcept {
    cfg_.enable_autotune = on;
    if (!on) {
        // Restore manual pitch (no correction)
        vocoder_->set_pitch_semitones(manual_pitch_st_);
    }
}
void VoicePipeline::set_autotune_scale(MusicalScale s, int root_semis) noexcept {
    autotune_->set_scale(s, root_semis);
}
void VoicePipeline::set_autotune_custom_scale(std::uint16_t mask, int root_semis) noexcept {
    autotune_->set_custom_scale(mask, root_semis);
}
void VoicePipeline::set_autotune_strength(float s) noexcept {
    autotune_->set_strength(s);
}
void VoicePipeline::set_autotune_glide_ms(float ms) noexcept {
    autotune_->set_glide_ms(ms);
}
PitchCorrectorStats VoicePipeline::autotune_stats() const noexcept {
    return autotune_->stats();
}

void VoicePipeline::set_character(VoiceCharacter c) {
    const auto p = character_params(c);
    vocoder_->set_pitch_semitones(p.pitch_semitones);
    vocoder_->set_formant_semitones(p.formant_semitones);
    current_character_ = c;
    // eq_tilt_db, presence_boost, roughness are reserved for future stages
    // (saturator + EQ block). They're carried in the struct so the
    // contract is stable even before those stages exist.
}

void VoicePipeline::set_profile(QualityProfile p) {
    cfg_.profile = p;
    apply_profile_defaults();
    vocoder_->set_profile(p);
}

void VoicePipeline::reset() {
    dc_.reset();
    hp_.reset();
    noise_reducer_->reset();
    gate_.reset();
    agc_.reset();
    de_.reset();
    reverb_->reset();
    limiter_.reset();
    lookahead_.reset();
    vocoder_->reset();
    if (autotune_) autotune_->reset();
    manual_pitch_st_ = 0.0f;
    current_character_ = VoiceCharacter::Neutral;
}

NoiseReductionStats VoicePipeline::noise_stats() const noexcept {
    return noise_reducer_->stats();
}

void VoicePipeline::learn_noise_profile(const float* silence, std::size_t n) {
    if (cfg_.enable_noise_reducer && noise_reducer_) {
        noise_reducer_->learn_noise_profile(silence, n);
    }
}

void VoicePipeline::process(const float* input, float* output, std::size_t n) {
    if (n == 0) return;
    if (!input || !output) {
        throw std::invalid_argument("VoicePipeline::process: null buffer");
    }

    // Make sure scratch is big enough. This can only grow (one-time alloc).
    if (stage_a_.size() < n) stage_a_.resize(n);
    if (stage_b_.size() < n) stage_b_.resize(n);

    // Toggle between two scratch buffers. `cur` is the current input view,
    // `next` is where to write next stage's output. After writing, swap.
    float* a = stage_a_.data();
    float* b = stage_b_.data();
    const float* cur = input;
    float*       next = a;
    auto advance = [&]() {
        cur  = next;
        next = (next == a) ? b : a;
    };

    if (cfg_.enable_dc_blocker)    { dc_.process_block(cur, next, n);             advance(); }
    if (cfg_.enable_highpass)      { hp_.process_block(cur, next, n);             advance(); }
    if (cfg_.enable_noise_reducer) { noise_reducer_->process(cur, next, n);       advance(); }
    if (cfg_.enable_gate)          { gate_.process_block(cur, next, n);           advance(); }
    if (cfg_.enable_agc)           { agc_.process_block(cur, next, n);            advance(); }

    // Autotune analysis happens HERE — on the cleaned, gated, AGC'd signal
    // but BEFORE the vocoder. The corrector itself does not modify samples;
    // it computes a correction in semitones which we add to the manual
    // pitch setting and push to the vocoder for this block.
    if (cfg_.enable_autotune && cfg_.enable_vocoder && autotune_) {
        const float correction = autotune_->analyze(cur, n);
        float total = manual_pitch_st_ + correction;
        if (!std::isfinite(total)) total = manual_pitch_st_;
        vocoder_->set_pitch_semitones(total);
    }

    if (cfg_.enable_vocoder)       { vocoder_->process(cur, next, n);             advance(); }
    if (cfg_.enable_de_esser)      { de_.process_block(cur, next, n);             advance(); }
    if (cfg_.enable_reverb)        { reverb_->process_block(cur, next, n);        advance(); }
    if (cfg_.enable_limiter) {
        if (cfg_.use_lookahead_limiter) {
            lookahead_.process_block(cur, next, n);
        } else {
            limiter_.process_block(cur, next, n);
        }
        advance();
    }

    // Final stage result is in `cur`. Copy to output.
    // (If no stages ran, cur still points at input — also valid.)
    std::memcpy(output, cur, n * sizeof(float));
}

}  // namespace aboba
