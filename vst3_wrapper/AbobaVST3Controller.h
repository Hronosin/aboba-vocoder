// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once

#include "public.sdk/source/vst/vsteditcontroller.h"

namespace aboba_vst3 {

class Controller : public Steinberg::Vst::EditController {
public:
    Controller() = default;
    ~Controller() override = default;

    static Steinberg::FUnknown* createInstance(void*) {
        return static_cast<Steinberg::Vst::IEditController*>(new Controller());
    }

    Steinberg::tresult PLUGIN_API initialize(Steinberg::FUnknown* context) override;
    Steinberg::tresult PLUGIN_API setComponentState(Steinberg::IBStream* state) override;
};

}  // namespace aboba_vst3
