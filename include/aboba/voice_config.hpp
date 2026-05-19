// SPDX-License-Identifier: GPL-3.0-or-later
//
// Voice configuration files (TOML format).
//
// Format (all fields optional, sensible defaults applied):
//
//   # voice.toml — example config
//   name = "my-streamer-voice"
//   description = "Punchy male voice for streaming"
//
//   [pipeline]
//   profile = "balanced"          # "quality" | "balanced" | "performance"
//   sample_rate = 48000
//   fft_size = 2048
//   hop_size = 512
//
//   [pitch]
//   semitones = -2.0              # manual pitch shift
//   formant_semitones = -1.0      # independent formant shift
//
//   [character]
//   preset = "warm-male"          # or omit to use pitch/formant above
//
//   [autotune]
//   enabled = true
//   scale = "major"               # or "minor", "blues", "chromatic", ...
//   root = "C"                    # or 0..11 numeric
//   strength = 0.7
//   glide_ms = 35.0
//
//   [effects]
//   noise_gate = true
//   highpass = true
//   highpass_cutoff_hz = 80.0
//   noise_reducer = true
//   agc = true
//   de_esser = false
//   reverb = false                # or true with [reverb] section below
//
//   [reverb]
//   room_size = 0.5
//   damping = 0.3
//   wet = 0.2
//
//   [limiter]
//   lookahead = true              # use sample-accurate limiter
//
// Loading:
//   auto cfg = aboba::load_voice_config("/path/to/voice.toml");
//   if (!cfg) {
//       std::cerr << cfg.error() << "\n";
//       return 1;
//   }
//   auto pipe = std::make_unique<VoicePipeline>(cfg->to_pipeline_config(), backend);
//   cfg->apply_runtime(*pipe);
//
// We implement a small TOML subset by hand (no external dependency). The
// parser handles:
//   * key = value
//   * sections [name]
//   * strings (double-quoted, with \n \t \" escape support)
//   * integers and floats
//   * booleans (true/false)
//   * comments (# to end-of-line)
//
// Tables, arrays, dotted keys, multi-line strings, datetimes — NOT
// supported. Anything we don't understand is an error (we'd rather fail
// loud than silently misconfigure). If you need full TOML, swap in
// tomlplusplus and replace voice_config_loader.cpp.
#pragma once

#include "pipeline.hpp"
#include "voice_character.hpp"
#include "musical_scale.hpp"
#include "quality.hpp"

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace aboba {

struct VoiceConfig {
    std::string name        = "unnamed";
    std::string description;

    // Pipeline
    QualityProfile profile  = QualityProfile::Balanced;
    double      sample_rate = 48000.0;
    std::size_t fft_size    = 2048;
    std::size_t hop_size    = 512;

    // Pitch / formant
    float pitch_semitones    = 0.0f;
    float formant_semitones  = 0.0f;
    bool  has_character      = false;
    VoiceCharacter character = VoiceCharacter::Neutral;

    // Autotune
    bool          autotune_enabled  = false;
    MusicalScale  autotune_scale    = MusicalScale::Chromatic;
    int           autotune_root     = 0;
    float         autotune_strength = 1.0f;
    float         autotune_glide_ms = 30.0f;

    // Effects toggles
    bool noise_gate    = true;
    bool highpass      = true;
    float highpass_cutoff_hz = 80.0f;
    bool noise_reducer = true;
    bool agc           = true;
    bool de_esser      = true;
    bool reverb        = false;
    bool lookahead     = false;

    // Reverb
    float reverb_room_size = 0.5f;
    float reverb_damping   = 0.3f;
    float reverb_wet       = 0.2f;

    // Build a VoicePipelineConfig from this. Fields that don't fit into
    // the static config (autotune, reverb params, character) are applied
    // separately via apply_runtime() after the pipeline is constructed.
    VoicePipelineConfig to_pipeline_config() const;

    // Push character/autotune/reverb settings into an already-constructed
    // pipeline.
    void apply_runtime(VoicePipeline& pipe) const;
};

// Loader result. On success, .value is populated and .ok() is true.
// On failure, .error contains a human-readable diagnostic.
struct VoiceConfigResult {
    std::unique_ptr<VoiceConfig> value;
    std::string error;
    int line_number = 0;          // 1-based, where parse failed

    bool ok() const noexcept { return value != nullptr; }
    explicit operator bool() const noexcept { return ok(); }
    VoiceConfig*       operator->()       noexcept { return value.get(); }
    const VoiceConfig* operator->() const noexcept { return value.get(); }
    VoiceConfig&       operator*()        noexcept { return *value; }
    const VoiceConfig& operator*()  const noexcept { return *value; }
};

// Load and parse a config file. Returns a non-ok result on any I/O or
// parse error.
VoiceConfigResult load_voice_config(const std::string& path);

// Same, but from an in-memory string (useful for tests / hot-reload).
VoiceConfigResult parse_voice_config(const std::string& toml_text,
                                     const std::string& source_label = "<memory>");

// Serialize a VoiceConfig back to TOML. Useful for save-as.
std::string serialize_voice_config(const VoiceConfig& cfg);

}  // namespace aboba
