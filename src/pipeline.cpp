// SPDX-License-Identifier: GPL-3.0-or-later
#include "aboba/pipeline.hpp"

#include <algorithm>
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
    de_.configure(cfg_.sample_rate, 5500.0f, -22.0f, 6.0f);
    limiter_.configure(cfg_.sample_rate, 0.95f, 0.050f);

    FormantVocoderConfig vc;
    vc.fft_size    = cfg_.fft_size;
    vc.hop_size    = cfg_.hop_size;
    vc.sample_rate = cfg_.sample_rate;
    vc.profile     = cfg_.profile;
    vc.use_voicing_gate = (cfg_.profile != QualityProfile::Performance);
    vc.f0_min_hz = cfg_.f0_min_hz;
    vc.f0_max_hz = cfg_.f0_max_hz;
    vocoder_ = std::make_unique<FormantVocoder>(vc, backend_);

    stage_a_.assign(kScratchInitial, 0.0f);
    stage_b_.assign(kScratchInitial, 0.0f);
}

void VoicePipeline::apply_profile_defaults() {
    // The user can still override individual flags after construction,
    // but the defaults track the profile.
    switch (cfg_.profile) {
        case QualityProfile::Quality:
            // Leave whatever the user set; defaults are all-on.
            break;
        case QualityProfile::Balanced:
            cfg_.enable_de_esser = false;  // skip de-esser by default
            break;
        case QualityProfile::Performance:
            cfg_.enable_highpass = false;
            cfg_.enable_gate     = false;
            cfg_.enable_de_esser = false;
            // DC blocker + limiter always on (cheap and important)
            break;
    }
}

void VoicePipeline::set_pitch_semitones(float st) {
    vocoder_->set_pitch_semitones(st);
}

void VoicePipeline::set_profile(QualityProfile p) {
    cfg_.profile = p;
    apply_profile_defaults();
    vocoder_->set_profile(p);
}

void VoicePipeline::reset() {
    dc_.reset();
    hp_.reset();
    gate_.reset();
    de_.reset();
    limiter_.reset();
    vocoder_->reset();
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

    if (cfg_.enable_dc_blocker) { dc_.process_block(cur, next, n);    advance(); }
    if (cfg_.enable_highpass)   { hp_.process_block(cur, next, n);    advance(); }
    if (cfg_.enable_gate)       { gate_.process_block(cur, next, n);  advance(); }
    if (cfg_.enable_vocoder)    { vocoder_->process(cur, next, n);    advance(); }
    if (cfg_.enable_de_esser)   { de_.process_block(cur, next, n);    advance(); }
    if (cfg_.enable_limiter)    { limiter_.process_block(cur, next, n); advance(); }

    // Final stage result is in `cur`. Copy to output.
    // (If no stages ran, cur still points at input — also valid.)
    std::memcpy(output, cur, n * sizeof(float));
}

}  // namespace aboba
