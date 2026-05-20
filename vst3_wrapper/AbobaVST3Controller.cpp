// SPDX-License-Identifier: GPL-3.0-or-later
//
// AbobaVST3Controller.cpp
//
// IEditController implementation. This is the part the DAW uses to:
//   * Display parameter names / units in the host UI
//   * Convert normalized values to display strings
//   * Handle automation lanes
//
// Crucially, the controller does NOT touch audio. It only manages
// parameter metadata. The processor (in a separate object, possibly even
// in a separate process) does the audio.
//
// We use the default generic GUI (the host renders a simple slider for
// each parameter). A custom GUI would require VSTGUI or a similar
// toolkit; out of scope here.
#include "AbobaVST3Controller.h"

#include "pluginterfaces/base/ustring.h"
#include "pluginterfaces/vst/ivstparameterchanges.h"
#include "public.sdk/source/vst/vstparameters.h"

#include "aboba/voice_character.hpp"
#include "aboba/musical_scale.hpp"

#include <cstdio>
#include <cstring>

namespace aboba_vst3 {

// Same enum as in the processor — keep in sync.
enum ParamId : Steinberg::Vst::ParamID {
    kParamPitchSemitones    = 1,
    kParamFormantSemitones  = 2,
    kParamCharacter         = 3,
    kParamReverbEnabled     = 4,
    kParamReverbWet         = 5,
    kParamAutotuneEnabled   = 6,
    kParamAutotuneScale     = 7,
    kParamAutotuneStrength  = 8,
    kParamBypass            = 9,
};

Steinberg::tresult PLUGIN_API Controller::initialize(Steinberg::FUnknown* context) {
    auto r = EditController::initialize(context);
    if (r != Steinberg::kResultOk) return r;

    // ---- Pitch: -24..+24 semitones, default 0 ----
    parameters.addParameter(STR16("Pitch"), STR16("st"),
        0, 0.5, Steinberg::Vst::ParameterInfo::kCanAutomate,
        kParamPitchSemitones);

    // ---- Formant: -12..+12 semitones, default 0 ----
    parameters.addParameter(STR16("Formant"), STR16("st"),
        0, 0.5, Steinberg::Vst::ParameterInfo::kCanAutomate,
        kParamFormantSemitones);

    // ---- Character: discrete list ----
    {
        auto* p = new Steinberg::Vst::StringListParameter(
            STR16("Character"), kParamCharacter);
        for (int i = 0; i < aboba::character_count(); ++i) {
            auto cid = aboba::character_id(
                static_cast<aboba::VoiceCharacter>(i));
            Steinberg::Vst::String128 s{};
            // UTF-8 -> UTF-16 (cheaply, for ASCII names)
            for (std::size_t j = 0; j < cid.size() && j < 127; ++j) {
                s[j] = static_cast<char16_t>(cid[j]);
            }
            p->appendString(s);
        }
        parameters.addParameter(p);
    }

    // ---- Reverb on/off ----
    parameters.addParameter(STR16("Reverb"), nullptr,
        1, 0.0, Steinberg::Vst::ParameterInfo::kCanAutomate,
        kParamReverbEnabled);

    // ---- Reverb wet 0..1 ----
    parameters.addParameter(STR16("Reverb Wet"), STR16("%"),
        0, 0.2, Steinberg::Vst::ParameterInfo::kCanAutomate,
        kParamReverbWet);

    // ---- Autotune on/off ----
    parameters.addParameter(STR16("Autotune"), nullptr,
        1, 0.0, Steinberg::Vst::ParameterInfo::kCanAutomate,
        kParamAutotuneEnabled);

    // ---- Autotune scale ----
    {
        auto* p = new Steinberg::Vst::StringListParameter(
            STR16("Scale"), kParamAutotuneScale);
        for (int i = 0; i < 8; ++i) {
            auto name = aboba::scale_name(
                static_cast<aboba::MusicalScale>(i));
            Steinberg::Vst::String128 s{};
            for (std::size_t j = 0; j < std::strlen(name) && j < 127; ++j) {
                s[j] = static_cast<char16_t>(name[j]);
            }
            p->appendString(s);
        }
        parameters.addParameter(p);
    }

    // ---- Autotune strength 0..1 ----
    parameters.addParameter(STR16("Tune Strength"), STR16("%"),
        0, 1.0, Steinberg::Vst::ParameterInfo::kCanAutomate,
        kParamAutotuneStrength);

    // ---- Bypass (special flag) ----
    parameters.addParameter(STR16("Bypass"), nullptr,
        1, 0.0,
        Steinberg::Vst::ParameterInfo::kCanAutomate |
        Steinberg::Vst::ParameterInfo::kIsBypass,
        kParamBypass);

    return Steinberg::kResultOk;
}

Steinberg::tresult PLUGIN_API Controller::setComponentState(Steinberg::IBStream* state) {
    if (!state) return Steinberg::kResultFalse;
    Steinberg::IBStreamer s(state, kLittleEndian);
    // Read in the same order the processor writes
    float vf; Steinberg::int32 vi; Steinberg::uint8 vb;
    if (s.readFloat(vf)) setParamNormalized(kParamPitchSemitones,   (vf + 24.0f) / 48.0);
    if (s.readFloat(vf)) setParamNormalized(kParamFormantSemitones, (vf + 12.0f) / 24.0);
    if (s.readInt32(vi)) setParamNormalized(kParamCharacter,
        static_cast<double>(vi) / static_cast<double>(aboba::character_count() - 1));
    if (s.readUInt8(vb)) setParamNormalized(kParamReverbEnabled,    vb ? 1.0 : 0.0);
    if (s.readFloat(vf)) setParamNormalized(kParamReverbWet,        vf);
    if (s.readUInt8(vb)) setParamNormalized(kParamAutotuneEnabled,  vb ? 1.0 : 0.0);
    if (s.readInt32(vi)) setParamNormalized(kParamAutotuneScale,    vi / 7.0);
    if (s.readFloat(vf)) setParamNormalized(kParamAutotuneStrength, vf);
    if (s.readUInt8(vb)) setParamNormalized(kParamBypass,           vb ? 1.0 : 0.0);
    return Steinberg::kResultOk;
}

}  // namespace aboba_vst3
