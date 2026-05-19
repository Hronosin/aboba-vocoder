// SPDX-License-Identifier: GPL-3.0-or-later
//
// Musical scales for pitch correction.
//
// A scale is a set of semitone offsets from the root that are allowed.
// We use a 12-bit mask (1 bit per semitone in an octave): bit i set means
// note i is in the scale. Snap = find the nearest set bit relative to the
// root, wrap-aware.
//
// Adding a custom scale is one line in scale_mask(). Common scales are
// predefined for convenience.
#pragma once

#include <cstdint>

namespace aboba {

enum class MusicalScale : std::uint8_t {
    Chromatic        = 0,   // all 12 notes
    Major            = 1,   // Ionian: W W H W W W H
    Minor            = 2,   // Natural minor: W H W W H W W
    HarmonicMinor    = 3,
    PentatonicMajor  = 4,
    PentatonicMinor  = 5,
    Blues            = 6,
    WholeTone        = 7,
    Custom           = 8,   // bitmask provided externally
    Count
};

// 12-bit mask of allowed scale degrees (bit 0 = root, bit 1 = root+1, ...,
// bit 11 = root+11). For Chromatic this is 0xFFF; for Major it's
// 0b101010110101 = 0xAB5.
std::uint16_t scale_mask(MusicalScale s) noexcept;

const char* scale_name(MusicalScale s) noexcept;

// Snap a MIDI note number to the nearest scale member.
//   midi_note      : input note, fractional allowed (e.g., 60.4)
//   scale_bits     : 12-bit mask (e.g., from scale_mask())
//   root_semitones : 0..11 — what is the root of the scale (0=C, 9=A)
//
// Returns the snapped MIDI note as float. Wrap-around: if no scale member
// in the same octave is closer than the one in the next/prev octave, we
// pick the nearest absolute.
float snap_to_scale(float midi_note,
                    std::uint16_t scale_bits,
                    int root_semitones) noexcept;

// Helpers for converting between Hz and MIDI semitones (A4 = 440 = MIDI 69).
float hz_to_midi(float hz) noexcept;
float midi_to_hz(float midi) noexcept;

}  // namespace aboba
