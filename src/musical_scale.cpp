// SPDX-License-Identifier: GPL-3.0-or-later
#include "aboba/musical_scale.hpp"

#include <cmath>
#include <cstdint>

namespace aboba {

namespace {

// Standard scale masks, root at bit 0. To get scale "in key X", rotate
// these by X semitones (handled in snap_to_scale).
//                                  bit11 bit10 ... bit1 bit0
constexpr std::uint16_t kChromatic        = 0b111111111111;  // 0xFFF
constexpr std::uint16_t kMajor            = 0b101010110101;  // I II III IV V VI VII
constexpr std::uint16_t kMinor            = 0b010110101101;  // I II b3 IV V b6 b7
constexpr std::uint16_t kHarmonicMinor    = 0b100110101101;  // raised 7th
constexpr std::uint16_t kPentMajor        = 0b001010010101;  // 1 2 3 5 6
constexpr std::uint16_t kPentMinor        = 0b010010101001;  // 1 b3 4 5 b7
constexpr std::uint16_t kBlues            = 0b010011101001;  // 1 b3 4 b5 5 b7
constexpr std::uint16_t kWholeTone        = 0b010101010101;  // 1 2 3 #4 #5 #6

}  // namespace

std::uint16_t scale_mask(MusicalScale s) noexcept {
    switch (s) {
        case MusicalScale::Chromatic:       return kChromatic;
        case MusicalScale::Major:           return kMajor;
        case MusicalScale::Minor:           return kMinor;
        case MusicalScale::HarmonicMinor:   return kHarmonicMinor;
        case MusicalScale::PentatonicMajor: return kPentMajor;
        case MusicalScale::PentatonicMinor: return kPentMinor;
        case MusicalScale::Blues:           return kBlues;
        case MusicalScale::WholeTone:       return kWholeTone;
        case MusicalScale::Custom:          return kChromatic;  // sentinel
        case MusicalScale::Count: ;
    }
    return kChromatic;
}

const char* scale_name(MusicalScale s) noexcept {
    switch (s) {
        case MusicalScale::Chromatic:       return "chromatic";
        case MusicalScale::Major:           return "major";
        case MusicalScale::Minor:           return "minor";
        case MusicalScale::HarmonicMinor:   return "harmonic-minor";
        case MusicalScale::PentatonicMajor: return "pentatonic-major";
        case MusicalScale::PentatonicMinor: return "pentatonic-minor";
        case MusicalScale::Blues:           return "blues";
        case MusicalScale::WholeTone:       return "whole-tone";
        case MusicalScale::Custom:          return "custom";
        case MusicalScale::Count: ;
    }
    return "?";
}

float hz_to_midi(float hz) noexcept {
    if (!(hz > 0.0f) || !std::isfinite(hz)) return 0.0f;
    return 69.0f + 12.0f * std::log2(hz / 440.0f);
}

float midi_to_hz(float midi) noexcept {
    if (!std::isfinite(midi)) return 0.0f;
    return 440.0f * std::pow(2.0f, (midi - 69.0f) / 12.0f);
}

float snap_to_scale(float midi_note, std::uint16_t scale_bits,
                    int root_semitones) noexcept {
    if (!std::isfinite(midi_note)) return midi_note;
    if (scale_bits == 0) return midi_note;  // nothing in scale — passthrough

    // Normalize root to [0, 12)
    int root = root_semitones % 12;
    if (root < 0) root += 12;

    // Find the integer "candidate" note within this octave that is closest
    // to midi_note and is in the scale. We scan ±12 semitones around the
    // nearest integer.
    const float center = std::round(midi_note);
    float best_note = midi_note;
    float best_dist = std::numeric_limits<float>::infinity();
    for (int delta = -12; delta <= 12; ++delta) {
        const float candidate = center + static_cast<float>(delta);
        const int rel = static_cast<int>(candidate) - root;
        int bit = ((rel % 12) + 12) % 12;
        if ((scale_bits >> bit) & 1) {
            const float dist = std::fabs(midi_note - candidate);
            if (dist < best_dist) {
                best_dist = dist;
                best_note = candidate;
            }
        }
    }
    return best_note;
}

}  // namespace aboba
