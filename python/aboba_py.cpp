// SPDX-License-Identifier: GPL-3.0-or-later
//
// Python bindings for Aboba Vocoder (pybind11).
//
// Design choices:
//   * The library uses raw float* + size pairs everywhere. Python uses
//     numpy arrays. We bridge by accepting numpy.ndarray[float32, C_CONTIG]
//     and returning newly-allocated numpy arrays.
//   * We DO NOT release the GIL for short calls (overhead would dominate).
//     For long offline-batch calls (process_block on a multi-minute file),
//     we DO release the GIL.
//   * Backend is created on the C++ side and held shared. Python sees an
//     opaque `Backend` object — no need to know it's CPU vs HIP.
//   * VoicePipeline owns its backend reference. We keep the backend alive
//     by tying it to the pipeline's lifetime via a Python attribute.
//
// Usage from Python:
//
//   import numpy as np
//   import aboba
//
//   backend = aboba.create_backend()
//   pipe = aboba.VoicePipeline(sample_rate=48000, backend=backend)
//   pipe.set_character(aboba.VoiceCharacter.AnimeGirl)
//
//   # 1 second of mock audio
//   x = (0.3 * np.sin(2 * np.pi * 220 * np.arange(48000) / 48000)).astype(np.float32)
//   y = pipe.process(x)
//   print("In RMS:", np.sqrt(np.mean(x**2)), "Out RMS:", np.sqrt(np.mean(y**2)))
//
//   # Load from TOML
//   cfg = aboba.load_voice_config("/path/to/voice.toml")
//   pipe2 = aboba.VoicePipeline.from_config(cfg, backend)

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "aboba/backend.hpp"
#include "aboba/dsp_blocks.hpp"
#include "aboba/formant_vocoder.hpp"
#include "aboba/musical_scale.hpp"
#include "aboba/noise_reduction.hpp"
#include "aboba/pipeline.hpp"
#include "aboba/pitch_corrector.hpp"
#include "aboba/quality.hpp"
#include "aboba/voice_character.hpp"
#include "aboba/voice_config.hpp"
#include "aboba/yin.hpp"

#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace py = pybind11;
using namespace aboba;

namespace {

using FloatArray = py::array_t<float, py::array::c_style | py::array::forcecast>;

// Get a flat 1D view over a numpy array. Throws if not 1D.
const float* as_flat(const FloatArray& arr, std::size_t& n_out) {
    if (arr.ndim() != 1) {
        throw std::invalid_argument(
            "expected 1D numpy array (got ndim=" +
            std::to_string(arr.ndim()) + ")");
    }
    n_out = static_cast<std::size_t>(arr.shape(0));
    return arr.data();
}

// Allocate a new numpy array of the same shape (1D length N) as the input.
FloatArray make_output(std::size_t n) {
    return FloatArray(static_cast<py::ssize_t>(n));
}

// Decide whether to release the GIL based on workload size. Audio is
// realtime-tiny but offline-batch can be enormous.
inline bool should_release_gil(std::size_t n) noexcept {
    // ~10 ms at 48 kHz. Below this, GIL release is more overhead than win.
    return n > 4096;
}

}  // namespace

PYBIND11_MODULE(aboba, m) {
    m.doc() = "Aboba Vocoder — AMD-flavored real-time voice framework (Python bindings)";

    // ============================================================
    // Enums
    // ============================================================
    py::enum_<QualityProfile>(m, "QualityProfile")
        .value("Quality",     QualityProfile::Quality)
        .value("Balanced",    QualityProfile::Balanced)
        .value("Performance", QualityProfile::Performance)
        .export_values();

    py::enum_<VoiceCharacter> vc_enum(m, "VoiceCharacter");
    for (int i = 0; i < character_count(); ++i) {
        auto ch = static_cast<VoiceCharacter>(i);
        // The enum name in Python uses the C++ identifier (DeepMale etc),
        // and we additionally provide a `.id` attribute via static method
        // below. The actual `id` string is in character_id(c).
        const char* name = "";
        switch (ch) {
            case VoiceCharacter::Neutral:        name = "Neutral";        break;
            case VoiceCharacter::DeepMale:       name = "DeepMale";       break;
            case VoiceCharacter::WarmMale:       name = "WarmMale";       break;
            case VoiceCharacter::ChestyMale:     name = "ChestyMale";     break;
            case VoiceCharacter::YoungFemale:    name = "YoungFemale";    break;
            case VoiceCharacter::AnimeGirl:      name = "AnimeGirl";      break;
            case VoiceCharacter::Chipmunk:       name = "Chipmunk";       break;
            case VoiceCharacter::Giant:          name = "Giant";          break;
            case VoiceCharacter::Demon:          name = "Demon";          break;
            case VoiceCharacter::Robot:          name = "Robot";          break;
            case VoiceCharacter::RadioHost:      name = "RadioHost";      break;
            case VoiceCharacter::Whisper:        name = "Whisper";        break;
            case VoiceCharacter::HeliumBalloon:  name = "HeliumBalloon";  break;
            case VoiceCharacter::CartoonVillain: name = "CartoonVillain"; break;
            case VoiceCharacter::Count: continue;
        }
        vc_enum.value(name, ch);
    }
    vc_enum.export_values();

    m.def("character_id", &character_id,
          "Stable string id of a character (use in TOML configs).",
          py::arg("character"));
    m.def("character_from_id", &character_from_id,
          "Look up a character by its string id (case-insensitive).",
          py::arg("id"));

    py::enum_<MusicalScale>(m, "MusicalScale")
        .value("Chromatic",       MusicalScale::Chromatic)
        .value("Major",           MusicalScale::Major)
        .value("Minor",           MusicalScale::Minor)
        .value("HarmonicMinor",   MusicalScale::HarmonicMinor)
        .value("PentatonicMajor", MusicalScale::PentatonicMajor)
        .value("PentatonicMinor", MusicalScale::PentatonicMinor)
        .value("Blues",           MusicalScale::Blues)
        .value("WholeTone",       MusicalScale::WholeTone)
        .value("Custom",          MusicalScale::Custom)
        .export_values();

    m.def("scale_name", &scale_name, py::arg("scale"));
    m.def("hz_to_midi", &hz_to_midi, py::arg("hz"));
    m.def("midi_to_hz", &midi_to_hz, py::arg("midi"));
    m.def("snap_to_scale",
          [](float midi, MusicalScale s, int root) {
              return snap_to_scale(midi, scale_mask(s), root);
          },
          py::arg("midi_note"), py::arg("scale"), py::arg("root_semitones") = 0,
          "Snap a MIDI note to the nearest scale member.");

    // ============================================================
    // Backend
    // ============================================================
    py::class_<Backend, std::shared_ptr<Backend>>(m, "Backend")
        .def("name", &Backend::name,
             "Human-readable backend identifier (e.g. 'CPU (FFTW3)').");

    m.def("create_backend",
          []() -> std::shared_ptr<Backend> {
              return std::shared_ptr<Backend>(create_best_backend().release());
          },
          "Create the best available backend (HIP if AMD GPU, else CPU).");

    // ============================================================
    // YIN F0 detector — useful standalone for research / debugging
    // ============================================================
    py::class_<YinResult>(m, "YinResult")
        .def_readonly("f0_hz",         &YinResult::f0_hz)
        .def_readonly("aperiodicity",  &YinResult::aperiodicity)
        .def_readonly("is_voiced",     &YinResult::is_voiced)
        .def("__repr__", [](const YinResult& r) {
            return "<YinResult f0=" + std::to_string(r.f0_hz) +
                   " aper=" + std::to_string(r.aperiodicity) +
                   " voiced=" + (r.is_voiced ? "True" : "False") + ">";
        });

    py::class_<YinDetector>(m, "YinDetector")
        .def(py::init([](double sample_rate, float f0_min, float f0_max,
                         float threshold) {
                 YinConfig c;
                 c.sample_rate = sample_rate;
                 c.f0_min_hz   = f0_min;
                 c.f0_max_hz   = f0_max;
                 c.threshold   = threshold;
                 return std::make_unique<YinDetector>(c);
             }),
             py::arg("sample_rate")  = 48000.0,
             py::arg("f0_min_hz")    = 60.0f,
             py::arg("f0_max_hz")    = 1500.0f,
             py::arg("threshold")    = 0.15f)
        .def("detect",
             [](YinDetector& self, FloatArray a) {
                 std::size_t n;
                 const float* p = as_flat(a, n);
                 return self.detect(p, n);
             },
             py::arg("samples"),
             "Detect F0 from a 1D numpy array of float32 samples.")
        .def_property_readonly("window_size", &YinDetector::window_size);

    // ============================================================
    // FormantVocoder — exposed so users can build a custom pipeline
    // ============================================================
    py::class_<FormantVocoderStats>(m, "FormantVocoderStats")
        .def_readonly("frames_total",       &FormantVocoderStats::frames_total)
        .def_readonly("frames_voiced",      &FormantVocoderStats::frames_voiced)
        .def_readonly("frames_unvoiced",    &FormantVocoderStats::frames_unvoiced)
        .def_readonly("frames_silent",      &FormantVocoderStats::frames_silent)
        .def_readonly("frames_degenerate",  &FormantVocoderStats::frames_degenerate)
        .def_readonly("last_f0_hz",         &FormantVocoderStats::last_f0_hz)
        .def_readonly("last_aperiodicity",  &FormantVocoderStats::last_aperiodicity);

    py::class_<FormantVocoder>(m, "FormantVocoder")
        .def(py::init([](std::shared_ptr<Backend> backend,
                          std::size_t fft_size, std::size_t hop_size,
                          double sample_rate, QualityProfile profile) {
                 FormantVocoderConfig c;
                 c.fft_size = fft_size;
                 c.hop_size = hop_size;
                 c.sample_rate = sample_rate;
                 c.profile = profile;
                 return std::make_unique<FormantVocoder>(c, backend.get());
             }),
             py::arg("backend"),
             py::arg("fft_size") = 2048, py::arg("hop_size") = 512,
             py::arg("sample_rate") = 48000.0,
             py::arg("profile") = QualityProfile::Balanced,
             // Keep the backend alive as long as the vocoder lives
             py::keep_alive<1, 2>())
        .def("set_pitch_semitones",   &FormantVocoder::set_pitch_semitones)
        .def("set_formant_semitones", &FormantVocoder::set_formant_semitones)
        .def("set_profile",           &FormantVocoder::set_profile)
        .def("reset",                 &FormantVocoder::reset)
        .def("stats",                 &FormantVocoder::stats)
        .def_property_readonly("latency_samples", &FormantVocoder::latency_samples)
        .def("process",
             [](FormantVocoder& self, FloatArray input) {
                 std::size_t n;
                 const float* p = as_flat(input, n);
                 auto out = make_output(n);
                 float* op = out.mutable_data();
                 if (should_release_gil(n)) {
                     py::gil_scoped_release release;
                     self.process(p, op, n);
                 } else {
                     self.process(p, op, n);
                 }
                 return out;
             },
             py::arg("samples"),
             "Process a 1D numpy array and return a new array of the same length.");

    // ============================================================
    // PitchCorrector
    // ============================================================
    py::class_<PitchCorrectorStats>(m, "PitchCorrectorStats")
        .def_readonly("analyses_total",       &PitchCorrectorStats::analyses_total)
        .def_readonly("analyses_voiced",      &PitchCorrectorStats::analyses_voiced)
        .def_readonly("analyses_unvoiced",    &PitchCorrectorStats::analyses_unvoiced)
        .def_readonly("last_input_f0_hz",     &PitchCorrectorStats::last_input_f0_hz)
        .def_readonly("last_target_f0_hz",    &PitchCorrectorStats::last_target_f0_hz)
        .def_readonly("last_correction_st",   &PitchCorrectorStats::last_correction_st)
        .def_readonly("smoothed_correction_st", &PitchCorrectorStats::smoothed_correction_st);

    py::class_<PitchCorrector>(m, "PitchCorrector")
        .def(py::init([](double sample_rate, MusicalScale scale, int root,
                         float strength, float glide_ms) {
                 PitchCorrectorConfig c;
                 c.sample_rate = sample_rate;
                 c.scale       = scale;
                 c.root_semis  = root;
                 c.strength    = strength;
                 c.glide_ms    = glide_ms;
                 return std::make_unique<PitchCorrector>(c);
             }),
             py::arg("sample_rate") = 48000.0,
             py::arg("scale")    = MusicalScale::Chromatic,
             py::arg("root")     = 0,
             py::arg("strength") = 1.0f,
             py::arg("glide_ms") = 30.0f)
        .def("analyze",
             [](PitchCorrector& self, FloatArray a) {
                 std::size_t n;
                 const float* p = as_flat(a, n);
                 return self.analyze(p, n);
             },
             py::arg("samples"),
             "Analyze a block. Returns the smoothed correction in semitones.")
        .def("set_scale",
             [](PitchCorrector& self, MusicalScale s, int root) {
                 self.set_scale(s, root);
             },
             py::arg("scale"), py::arg("root") = 0)
        .def("set_strength", &PitchCorrector::set_strength)
        .def("set_glide_ms", &PitchCorrector::set_glide_ms)
        .def("reset",        &PitchCorrector::reset)
        .def("stats",        &PitchCorrector::stats);

    // ============================================================
    // VoicePipeline — the main entry point for typical use
    // ============================================================
    py::class_<VoicePipeline>(m, "VoicePipeline")
        .def(py::init([](std::shared_ptr<Backend> backend,
                          double sample_rate, QualityProfile profile) {
                 VoicePipelineConfig c;
                 c.sample_rate = sample_rate;
                 c.profile     = profile;
                 return std::make_unique<VoicePipeline>(c, backend.get());
             }),
             py::arg("backend"),
             py::arg("sample_rate") = 48000.0,
             py::arg("profile")     = QualityProfile::Balanced,
             py::keep_alive<1, 2>())
        // Pitch / formant / character
        .def("set_pitch_semitones",   &VoicePipeline::set_pitch_semitones)
        .def("set_formant_semitones", &VoicePipeline::set_formant_semitones)
        .def("set_character",         &VoicePipeline::set_character)
        .def_property_readonly("current_character", &VoicePipeline::current_character)
        .def("set_profile",           &VoicePipeline::set_profile)
        // Reverb
        .def("set_reverb_enabled",    &VoicePipeline::set_reverb_enabled)
        .def("set_reverb_room_size",  &VoicePipeline::set_reverb_room_size)
        .def("set_reverb_damping",    &VoicePipeline::set_reverb_damping)
        .def("set_reverb_wet",        &VoicePipeline::set_reverb_wet)
        .def_property_readonly("reverb_enabled", &VoicePipeline::reverb_enabled)
        // Autotune
        .def("set_autotune_enabled",  &VoicePipeline::set_autotune_enabled)
        .def("set_autotune_scale",    &VoicePipeline::set_autotune_scale,
             py::arg("scale"), py::arg("root") = 0)
        .def("set_autotune_strength", &VoicePipeline::set_autotune_strength)
        .def("set_autotune_glide_ms", &VoicePipeline::set_autotune_glide_ms)
        .def_property_readonly("autotune_enabled", &VoicePipeline::autotune_enabled)
        .def("autotune_stats",        &VoicePipeline::autotune_stats)
        // Lifecycle
        .def("reset",                 &VoicePipeline::reset)
        .def_property_readonly("latency_samples", &VoicePipeline::latency_samples)
        // Diagnostics
        .def("vocoder_stats",         &VoicePipeline::vocoder_stats)
        .def("noise_stats",           &VoicePipeline::noise_stats)
        .def("gate_open",             &VoicePipeline::gate_open)
        // Noise profile
        .def("learn_noise_profile",
             [](VoicePipeline& self, FloatArray a) {
                 std::size_t n;
                 const float* p = as_flat(a, n);
                 self.learn_noise_profile(p, n);
             },
             py::arg("silence"))
        // Main processing
        .def("process",
             [](VoicePipeline& self, FloatArray input) {
                 std::size_t n;
                 const float* p = as_flat(input, n);
                 auto out = make_output(n);
                 float* op = out.mutable_data();
                 if (should_release_gil(n)) {
                     py::gil_scoped_release release;
                     self.process(p, op, n);
                 } else {
                     self.process(p, op, n);
                 }
                 return out;
             },
             py::arg("samples"),
             "Process a 1D float32 numpy array. Returns the processed array.")
        .def("process_into",
             [](VoicePipeline& self, FloatArray input, FloatArray output) {
                 std::size_t n_in, n_out;
                 const float* p_in  = as_flat(input,  n_in);
                 float*       p_out = output.mutable_data();
                 n_out = static_cast<std::size_t>(output.shape(0));
                 if (n_in != n_out) {
                     throw std::invalid_argument(
                         "input and output arrays must be the same length");
                 }
                 if (output.ndim() != 1) {
                     throw std::invalid_argument("output must be 1D");
                 }
                 if (should_release_gil(n_in)) {
                     py::gil_scoped_release release;
                     self.process(p_in, p_out, n_in);
                 } else {
                     self.process(p_in, p_out, n_in);
                 }
             },
             py::arg("samples"), py::arg("out"),
             "In-place variant — writes into a preallocated array.");

    // Convenience: build from a VoiceConfig
    m.def("pipeline_from_config",
          [](const VoiceConfig& cfg, std::shared_ptr<Backend> backend) {
              auto pipe_cfg = cfg.to_pipeline_config();
              auto p = std::make_unique<VoicePipeline>(pipe_cfg, backend.get());
              cfg.apply_runtime(*p);
              return p;
          },
          py::arg("config"), py::arg("backend"),
          py::keep_alive<0, 2>(),
          "Construct a VoicePipeline from a parsed VoiceConfig.");

    // ============================================================
    // Voice config / TOML
    // ============================================================
    py::class_<VoiceConfig>(m, "VoiceConfig")
        .def(py::init<>())
        .def_readwrite("name",                &VoiceConfig::name)
        .def_readwrite("description",         &VoiceConfig::description)
        .def_readwrite("profile",             &VoiceConfig::profile)
        .def_readwrite("sample_rate",         &VoiceConfig::sample_rate)
        .def_readwrite("fft_size",            &VoiceConfig::fft_size)
        .def_readwrite("hop_size",            &VoiceConfig::hop_size)
        .def_readwrite("pitch_semitones",     &VoiceConfig::pitch_semitones)
        .def_readwrite("formant_semitones",   &VoiceConfig::formant_semitones)
        .def_readwrite("has_character",       &VoiceConfig::has_character)
        .def_readwrite("character",           &VoiceConfig::character)
        .def_readwrite("autotune_enabled",    &VoiceConfig::autotune_enabled)
        .def_readwrite("autotune_scale",      &VoiceConfig::autotune_scale)
        .def_readwrite("autotune_root",       &VoiceConfig::autotune_root)
        .def_readwrite("autotune_strength",   &VoiceConfig::autotune_strength)
        .def_readwrite("autotune_glide_ms",   &VoiceConfig::autotune_glide_ms)
        .def_readwrite("noise_gate",          &VoiceConfig::noise_gate)
        .def_readwrite("highpass",            &VoiceConfig::highpass)
        .def_readwrite("highpass_cutoff_hz",  &VoiceConfig::highpass_cutoff_hz)
        .def_readwrite("noise_reducer",       &VoiceConfig::noise_reducer)
        .def_readwrite("agc",                 &VoiceConfig::agc)
        .def_readwrite("de_esser",            &VoiceConfig::de_esser)
        .def_readwrite("reverb",              &VoiceConfig::reverb)
        .def_readwrite("lookahead",           &VoiceConfig::lookahead)
        .def_readwrite("reverb_room_size",    &VoiceConfig::reverb_room_size)
        .def_readwrite("reverb_damping",      &VoiceConfig::reverb_damping)
        .def_readwrite("reverb_wet",          &VoiceConfig::reverb_wet)
        .def("serialize",
             [](const VoiceConfig& c) { return serialize_voice_config(c); },
             "Return the config as a TOML string.");

    m.def("load_voice_config",
          [](const std::string& path) {
              auto r = load_voice_config(path);
              if (!r.ok()) {
                  throw std::runtime_error(
                      "load_voice_config('" + path + "'): " + r.error +
                      " (line " + std::to_string(r.line_number) + ")");
              }
              return std::move(*r.value);
          },
          py::arg("path"),
          "Load and parse a TOML voice config from a file path.");

    m.def("parse_voice_config",
          [](const std::string& toml_text) {
              auto r = parse_voice_config(toml_text);
              if (!r.ok()) {
                  throw std::runtime_error(
                      "parse_voice_config: " + r.error +
                      " (line " + std::to_string(r.line_number) + ")");
              }
              return std::move(*r.value);
          },
          py::arg("toml_text"),
          "Parse a TOML voice config from an in-memory string.");

    // ============================================================
    // Module-level metadata
    // ============================================================
    m.attr("__version__") = "0.1.0";
}
