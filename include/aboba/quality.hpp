// SPDX-License-Identifier: GPL-3.0-or-later
//
// Quality profiles for the audio pipeline.
//
// Each effect interprets the profile in a way that makes sense for it:
//   * Quality:     full algorithms, larger FFT, formant preservation on,
//                  cepstral envelope smoothing at high order.
//   * Balanced:    formant preservation on with lighter envelope,
//                  standard FFT size, faster F0 detection.
//   * Performance: formant preservation off (classic bin-shift), smaller
//                  FFT, simplest F0 method.
//
// The vocoder pipeline reads this at construction time. Some effects also
// support runtime profile changes via set_profile(); see each header.
#pragma once

#include <cstdint>

namespace aboba {

enum class QualityProfile : std::uint8_t {
    Quality     = 0,
    Balanced    = 1,
    Performance = 2,
};

inline const char* profile_name(QualityProfile p) {
    switch (p) {
        case QualityProfile::Quality:     return "quality";
        case QualityProfile::Balanced:    return "balanced";
        case QualityProfile::Performance: return "performance";
    }
    return "?";
}

}  // namespace aboba
