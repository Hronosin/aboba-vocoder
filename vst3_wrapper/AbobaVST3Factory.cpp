// SPDX-License-Identifier: GPL-3.0-or-later
//
// AbobaVST3Factory.cpp — VST3 module entry points.
//
// Every VST3 plugin must export `GetPluginFactory()` returning an
// IPluginFactory. The SDK provides BEGIN_FACTORY_DEF and DEF_CLASS2
// macros to make this declarative.
//
// Vendor / URL / email are shown in the DAW's plugin info dialog. We
// register one component (the processor) and one controller, paired by
// their UIDs.

#include "AbobaVST3Processor.h"
#include "AbobaVST3Controller.h"

#include "public.sdk/source/main/pluginfactory.h"
#include "pluginterfaces/vst/ivstcomponent.h"

#define ABOBA_VENDOR     "Aboba Audio"
#define ABOBA_URL        "https://github.com/aboba/vocoder"
#define ABOBA_EMAIL      "issues@example.invalid"
#define ABOBA_VERSION    "0.1.0"

BEGIN_FACTORY_DEF(ABOBA_VENDOR, ABOBA_URL, "mailto:" ABOBA_EMAIL)

    DEF_CLASS2(INLINE_UID_FROM_FUID(aboba_vst3::kAbobaProcessorUID),
               PClassInfo::kManyInstances,
               kVstAudioEffectClass,
               "Aboba Vocoder",
               Steinberg::Vst::kDistributable,
               Steinberg::Vst::PlugType::kFxModulation,
               ABOBA_VERSION,
               kVstVersionString,
               aboba_vst3::Processor::createInstance)

    DEF_CLASS2(INLINE_UID_FROM_FUID(aboba_vst3::kAbobaControllerUID),
               PClassInfo::kManyInstances,
               kVstComponentControllerClass,
               "Aboba Vocoder Controller",
               0,
               "",
               ABOBA_VERSION,
               kVstVersionString,
               aboba_vst3::Controller::createInstance)

END_FACTORY
