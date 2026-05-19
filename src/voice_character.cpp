// SPDX-License-Identifier: GPL-3.0-or-later
#include "aboba/voice_character.hpp"

#include <cstring>
#include <cctype>

namespace aboba {

namespace {

// Single source of truth: the preset table. Order MUST match the enum.
constexpr VoiceCharacterParams kCharacters[] = {
    // pitch  formant  tilt_db  presence  rough  id              description
    {  0.0f,   0.0f,    0.0f,    false,   0.0f,  "neutral",
        "Identity — full-quality pass-through." },
    { -3.0f,  -2.0f,    0.0f,    true,    0.05f, "deep-male",
        "Bass narrator — chest voice, broadcast-style presence." },
    { -1.0f,  -1.0f,    0.0f,    false,   0.0f,  "warm-male",
        "Friendly adult male — subtle lowering with body." },
    { -5.0f,  -3.0f,    0.0f,    true,    0.10f, "chesty-male",
        "Theatrical large-body male — like a TV villain monologue." },
    { +2.0f,  +2.0f,    1.5f,    false,   0.0f,  "young-female",
        "Adult female with slight lift, brighter timbre." },
    { +6.0f,  +4.0f,    2.0f,    false,   0.0f,  "anime",
        "Cartoonish high voice — formants stretched up." },
    {+12.0f,  +6.0f,    0.0f,    false,   0.0f,  "chipmunk",
        "Extreme high, comedic chipmunk effect." },
    { -7.0f,  -5.0f,    -2.0f,   false,   0.0f,  "giant",
        "Very low fundamental and large body — slow-talker effect." },
    {-10.0f,  -4.0f,    -3.0f,   false,   0.30f, "demon",
        "Low pitch + heavy roughness (saturator hint)." },
    {  0.0f,   0.0f,    0.0f,    false,   0.15f, "robot",
        "Pitch unchanged, light saturation. Pair with autotune for "
        "classic robot effect." },
    {  0.0f,  -0.5f,    0.0f,    true,    0.05f, "radio-host",
        "Broadcast: lightly chesty, presence boost, light compression hint." },
    {  0.0f,  +0.5f,    -1.5f,   false,   0.0f,  "whisper",
        "Slightly brighter formants, darkened tilt — leave gate aggressive." },
    { +9.0f,  +5.0f,    1.0f,    false,   0.0f,  "helium",
        "Cartoonish very-high voice." },
    { -3.0f,  -2.0f,    -1.0f,   false,   0.20f, "cartoon-villain",
        "Low + dramatic, mild roughness." },
};

static_assert(sizeof(kCharacters) / sizeof(kCharacters[0])
              == static_cast<std::size_t>(VoiceCharacter::Count),
              "kCharacters table size must match VoiceCharacter enum");

inline char to_lower(char c) noexcept {
    return (c >= 'A' && c <= 'Z') ? static_cast<char>(c + 32) : c;
}

bool iequals(const char* a, const char* b) noexcept {
    if (!a || !b) return false;
    while (*a && *b) {
        if (to_lower(*a) != to_lower(*b)) return false;
        ++a; ++b;
    }
    return *a == *b;
}

}  // namespace

VoiceCharacterParams character_params(VoiceCharacter c) noexcept {
    const auto idx = static_cast<std::size_t>(c);
    if (idx >= static_cast<std::size_t>(VoiceCharacter::Count)) {
        return kCharacters[0];  // Neutral
    }
    return kCharacters[idx];
}

VoiceCharacter character_from_id(const char* id) noexcept {
    if (!id) return VoiceCharacter::Count;
    for (int i = 0; i < character_count(); ++i) {
        if (iequals(id, kCharacters[i].id)) {
            return static_cast<VoiceCharacter>(i);
        }
    }
    return VoiceCharacter::Count;
}

const char* character_id(VoiceCharacter c) noexcept {
    const auto idx = static_cast<std::size_t>(c);
    if (idx >= static_cast<std::size_t>(VoiceCharacter::Count)) return "";
    return kCharacters[idx].id;
}

}  // namespace aboba
