// SPDX-License-Identifier: GPL-3.0-or-later
//
// AbobaVST3Processor.cpp
//
// Audio processor for the Aboba VST3 plugin. Implements IAudioProcessor.
//
// Architecture:
//   * One VoicePipeline (Aboba's C++ object) per processor instance
//   * All parameters mapped from VST3 parameter IDs to pipeline setters
//   * Process callback feeds the host's float buffers into the pipeline
//   * Sample rate / block size set in setupProcessing
//   * Reset on stop or sample rate change
//
// Threading:
//   process() runs on the VST3 audio thread. We perform NO allocations
//   inside process(). The pipeline is set up in setupProcessing() (called
//   on the main thread).
//
// Building:
//   Requires the official Steinberg VST3 SDK at $VST3_SDK_PATH. The
//   CMakeLists in vst3_wrapper/ pulls in the SDK's helper functions
//   (SingleComponentEffect) and links Aboba's static library.
//
// Layout:
//   - input  bus: 1 stereo channel (we downmix to mono internally)
//   - output bus: 1 stereo channel (we duplicate mono out across L/R)
//
// This is a deliberately simple wrapper; a production plugin would expose
// MPE, MIDI controls, and program changes. For now we keep the surface
// minimal and the audio path obvious.
#include "AbobaVST3Processor.h"

#include "public.sdk/source/vst/vstaudioeffect.h"
#include "pluginterfaces/base/ibstream.h"
#include "pluginterfaces/vst/ivstparameterchanges.h"
#include "pluginterfaces/vst/ivstprocesscontext.h"

#include "aboba/backend.hpp"
#include "aboba/pipeline.hpp"
#include "aboba/voice_character.hpp"

namespace aboba_vst3 {

// Parameter IDs (must be stable across versions for project compatibility).
enum ParamId : Steinberg::Vst::ParamID {
    kParamPitchSemitones    = 1,
    kParamFormantSemitones  = 2,
    kParamCharacter         = 3,
    kParamReverbEnabled     = 4,
    kParamReverbWet         = 5,
    kParamAutotuneEnabled   = 6,
    kParamAutotuneScale     = 7,
    kParamAutotuneStrength  = 8,
    // The "bypass" parameter is special — VST3 hosts use it for the
    // built-in bypass button. We honor it as a hint to passthrough.
    kParamBypass            = 9,
};

// Helper: clamp a normalized 0..1 to [0..1] safely
static inline float clamp01(float v) {
    if (v < 0.0f) return 0.0f;
    if (v > 1.0f) return 1.0f;
    return v;
}

Processor::Processor() {
    setControllerClass(kAbobaControllerUID);
}

Processor::~Processor() {
    // pipeline_ destruction is automatic; nothing to do here.
}

Steinberg::tresult PLUGIN_API Processor::initialize(Steinberg::FUnknown* context) {
    auto r = AudioEffect::initialize(context);
    if (r != Steinberg::kResultOk) return r;
    addAudioInput(STR16("Voice In"),  Steinberg::Vst::SpeakerArr::kStereo);
    addAudioOutput(STR16("Voice Out"), Steinberg::Vst::SpeakerArr::kStereo);
    return Steinberg::kResultOk;
}

Steinberg::tresult PLUGIN_API Processor::terminate() {
    return AudioEffect::terminate();
}

Steinberg::tresult PLUGIN_API Processor::setActive(Steinberg::TBool state) {
    if (state) {
        // Construct backend + pipeline. We build a fresh pipeline whenever
        // the plugin is activated; this matches VST3 lifecycle.
        try {
            backend_ = aboba::create_best_backend();
            aboba::VoicePipelineConfig cfg;
            cfg.sample_rate = sample_rate_;
            cfg.profile     = aboba::QualityProfile::Balanced;
            pipeline_ = std::make_unique<aboba::VoicePipeline>(cfg, backend_.get());
            // Apply cached parameter values
            apply_all_params();
        } catch (...) {
            // If construction fails, fall through to passthrough mode in process()
            pipeline_.reset();
            backend_.reset();
        }
    } else {
        pipeline_.reset();
        backend_.reset();
    }
    return AudioEffect::setActive(state);
}

Steinberg::tresult PLUGIN_API Processor::setupProcessing(
        Steinberg::Vst::ProcessSetup& setup) {
    sample_rate_ = setup.sampleRate;
    max_block_   = setup.maxSamplesPerBlock;
    // Pre-size scratch buffers so process() doesn't allocate
    mono_in_.resize (static_cast<std::size_t>(max_block_));
    mono_out_.resize(static_cast<std::size_t>(max_block_));
    return AudioEffect::setupProcessing(setup);
}

void Processor::apply_all_params() {
    if (!pipeline_) return;
    pipeline_->set_pitch_semitones(pitch_semitones_);
    pipeline_->set_formant_semitones(formant_semitones_);
    if (character_idx_ >= 0 &&
        character_idx_ < aboba::character_count()) {
        pipeline_->set_character(
            static_cast<aboba::VoiceCharacter>(character_idx_));
    }
    pipeline_->set_reverb_enabled(reverb_enabled_);
    pipeline_->set_reverb_wet(reverb_wet_);
    pipeline_->set_autotune_enabled(autotune_enabled_);
    if (autotune_scale_idx_ >= 0) {
        pipeline_->set_autotune_scale(
            static_cast<aboba::MusicalScale>(autotune_scale_idx_), 0);
    }
    pipeline_->set_autotune_strength(autotune_strength_);
}

void Processor::handle_param_change(Steinberg::Vst::ParamID id, double value) {
    // VST3 parameter values are always normalized [0..1]. Map to our ranges.
    switch (id) {
        case kParamPitchSemitones:
            // Map 0..1 -> -24..+24 semitones
            pitch_semitones_ = static_cast<float>((value - 0.5) * 48.0);
            if (pipeline_) pipeline_->set_pitch_semitones(pitch_semitones_);
            break;
        case kParamFormantSemitones:
            formant_semitones_ = static_cast<float>((value - 0.5) * 24.0);
            if (pipeline_) pipeline_->set_formant_semitones(formant_semitones_);
            break;
        case kParamCharacter:
            // Map 0..1 -> 0..character_count-1
            character_idx_ = static_cast<int>(
                value * static_cast<double>(aboba::character_count() - 1) + 0.5);
            if (pipeline_) {
                pipeline_->set_character(
                    static_cast<aboba::VoiceCharacter>(character_idx_));
            }
            break;
        case kParamReverbEnabled:
            reverb_enabled_ = value >= 0.5;
            if (pipeline_) pipeline_->set_reverb_enabled(reverb_enabled_);
            break;
        case kParamReverbWet:
            reverb_wet_ = clamp01(static_cast<float>(value));
            if (pipeline_) pipeline_->set_reverb_wet(reverb_wet_);
            break;
        case kParamAutotuneEnabled:
            autotune_enabled_ = value >= 0.5;
            if (pipeline_) pipeline_->set_autotune_enabled(autotune_enabled_);
            break;
        case kParamAutotuneScale:
            autotune_scale_idx_ = static_cast<int>(value * 7.0 + 0.5);  // 8 scales
            if (pipeline_) {
                pipeline_->set_autotune_scale(
                    static_cast<aboba::MusicalScale>(autotune_scale_idx_), 0);
            }
            break;
        case kParamAutotuneStrength:
            autotune_strength_ = clamp01(static_cast<float>(value));
            if (pipeline_) pipeline_->set_autotune_strength(autotune_strength_);
            break;
        case kParamBypass:
            bypass_ = value >= 0.5;
            break;
        default:
            break;
    }
}

Steinberg::tresult PLUGIN_API Processor::process(Steinberg::Vst::ProcessData& data) {
    // 1. Drain parameter changes
    if (data.inputParameterChanges) {
        const Steinberg::int32 num = data.inputParameterChanges->getParameterCount();
        for (Steinberg::int32 i = 0; i < num; ++i) {
            auto* q = data.inputParameterChanges->getParameterData(i);
            if (!q) continue;
            const auto id = q->getParameterId();
            const Steinberg::int32 cnt = q->getPointCount();
            if (cnt == 0) continue;
            // Take the LAST point in the block (simplification — a proper
            // implementation would slope-interpolate between points)
            Steinberg::int32 sampleOffset = 0;
            Steinberg::Vst::ParamValue value = 0;
            if (q->getPoint(cnt - 1, sampleOffset, value) == Steinberg::kResultOk) {
                handle_param_change(id, value);
            }
        }
    }

    // 2. Audio. Bypass either explicitly or if pipeline construction failed.
    if (data.numSamples == 0) return Steinberg::kResultOk;
    if (!pipeline_ || bypass_) {
        // Passthrough: copy each channel directly
        if (data.numInputs > 0 && data.numOutputs > 0 &&
            data.inputs[0].numChannels > 0 && data.outputs[0].numChannels > 0) {
            const auto chan_in  = data.inputs[0].numChannels;
            const auto chan_out = data.outputs[0].numChannels;
            const auto n = data.numSamples;
            for (Steinberg::int32 c = 0; c < chan_out; ++c) {
                const float* src = (c < chan_in)
                    ? data.inputs[0].channelBuffers32[c]
                    : data.inputs[0].channelBuffers32[0];
                float* dst = data.outputs[0].channelBuffers32[c];
                std::memcpy(dst, src, static_cast<std::size_t>(n) * sizeof(float));
            }
        }
        return Steinberg::kResultOk;
    }

    // 3. Downmix stereo -> mono, process, expand back to stereo.
    const std::size_t n = static_cast<std::size_t>(data.numSamples);
    if (mono_in_.size() < n) mono_in_.resize(n);   // shouldn't grow normally
    if (mono_out_.size() < n) mono_out_.resize(n);
    const float* L = data.inputs[0].channelBuffers32[0];
    const float* R = (data.inputs[0].numChannels > 1)
        ? data.inputs[0].channelBuffers32[1] : L;
    for (std::size_t i = 0; i < n; ++i) {
        mono_in_[i] = 0.5f * (L[i] + R[i]);
    }
    try {
        pipeline_->process(mono_in_.data(), mono_out_.data(), n);
    } catch (...) {
        // Catastrophic: still emit audio so we don't crash the host.
        std::memcpy(mono_out_.data(), mono_in_.data(), n * sizeof(float));
    }
    // Distribute mono to all output channels
    for (Steinberg::int32 c = 0; c < data.outputs[0].numChannels; ++c) {
        float* dst = data.outputs[0].channelBuffers32[c];
        std::memcpy(dst, mono_out_.data(), n * sizeof(float));
    }
    return Steinberg::kResultOk;
}

Steinberg::tresult PLUGIN_API Processor::setState(Steinberg::IBStream* state) {
    // Read parameter values from project file
    Steinberg::IBStreamer s(state, kLittleEndian);
    float v;
    if (s.readFloat(v)) pitch_semitones_   = v;
    if (s.readFloat(v)) formant_semitones_ = v;
    Steinberg::int32 i;
    if (s.readInt32(i)) character_idx_     = i;
    Steinberg::uint8 b;
    if (s.readUInt8(b)) reverb_enabled_    = (b != 0);
    if (s.readFloat(v)) reverb_wet_        = v;
    if (s.readUInt8(b)) autotune_enabled_  = (b != 0);
    if (s.readInt32(i)) autotune_scale_idx_ = i;
    if (s.readFloat(v)) autotune_strength_  = v;
    if (s.readUInt8(b)) bypass_             = (b != 0);
    apply_all_params();
    return Steinberg::kResultOk;
}

Steinberg::tresult PLUGIN_API Processor::getState(Steinberg::IBStream* state) {
    Steinberg::IBStreamer s(state, kLittleEndian);
    s.writeFloat(pitch_semitones_);
    s.writeFloat(formant_semitones_);
    s.writeInt32(character_idx_);
    s.writeUInt8(reverb_enabled_  ? 1 : 0);
    s.writeFloat(reverb_wet_);
    s.writeUInt8(autotune_enabled_ ? 1 : 0);
    s.writeInt32(autotune_scale_idx_);
    s.writeFloat(autotune_strength_);
    s.writeUInt8(bypass_ ? 1 : 0);
    return Steinberg::kResultOk;
}

}  // namespace aboba_vst3
