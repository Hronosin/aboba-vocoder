// SPDX-License-Identifier: GPL-3.0-or-later
//
// Voice character presets.
//
// A "character" is a curated combination of:
//   * pitch_semitones        — fundamental shift
//   * formant_semitones      — independent formant (timbre) shift
//   * eq_tilt_db             — broadband tilt (negative = darker)
//   * presence_boost         — 2-4 kHz lift for intelligibility
//   * roughness              — small amount of saturation for grit
//   * description            — what it sounds like
//
// Apply via VoicePipeline::set_character() (see pipeline.hpp). The
// underlying pitch and formant control still works after a character is
// applied — use it as a starting point and fine-tune.
//
// Naming convention: short kebab-style identifiers ("deep-male", "anime"),
// so they can later be loaded from a TOML config without scope conflicts.
//
// All characters are deterministic and stateless: identical input ->
// identical output (no random modulation). If you want variation, layer
// the audio effects (reverb, chorus) on top.
#pragma once

#include <cstddef>
#include <cstdint>

namespace aboba {

enum class VoiceCharacter : std::uint8_t {
    Neutral         = 0,    // identity — pass-through with full quality
    DeepMale        = 1,    // narrator / bass
    WarmMale        = 2,    // friendly adult male
    ChestyMale      = 3,    // theatrical / very large body
    YoungFemale     = 4,    // adult female, lifted
    AnimeGirl       = 5,    // cartoonish high voice
    Chipmunk        = 6,    // extreme high, comedic
    Giant           = 7,    // very low, large body
    Demon           = 8,    // low + grit
    Robot           = 9,    // monotone-friendly preset (pitch correction
                            // off, but pipeline-wise: no pitch change,
                            // some saturation)
    RadioHost       = 10,   // broadcast-style presence + compression hint
    Whisper         = 11,   // unchanged pitch, slightly lifted formants,
                            // de-essed
    HeliumBalloon   = 12,   // very high comedic
    CartoonVillain  = 13,   // low + dramatic
    Count
};

// Parameters that VoicePipeline::set_character() applies. Other pipeline
// settings (limiter, gate) are not modified.
struct VoiceCharacterParams {
    float pitch_semitones    = 0.0f;
    float formant_semitones  = 0.0f;
    float eq_tilt_db         = 0.0f;   // not yet implemented as a stage;
                                        // reserved for future EQ block
    bool  presence_boost     = false;   // ditto
    float roughness          = 0.0f;   // 0..1, hint for future saturator
    const char* id           = "";
    const char* description  = "";
};

// Look up the parameters for a character. Always returns a valid struct;
// unknown values map to Neutral.
VoiceCharacterParams character_params(VoiceCharacter c) noexcept;

// Resolve by string id (case-insensitive). Returns VoiceCharacter::Count
// (sentinel) if not found.
VoiceCharacter character_from_id(const char* id) noexcept;

// String name of a character. Stable across versions — safe for configs.
const char* character_id(VoiceCharacter c) noexcept;

// Number of defined characters (= static_cast<int>(VoiceCharacter::Count)).
inline constexpr int character_count() {
    return static_cast<int>(VoiceCharacter::Count);
}

}  // namespace aboba
