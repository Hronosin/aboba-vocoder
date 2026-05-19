// SPDX-License-Identifier: GPL-3.0-or-later
//
// Formant-preserving streaming phase vocoder.
//
// Why this exists:
//   The plain StreamingPhaseVocoder does bin-shift: every spectral bin moves
//   by pitch_ratio. This shifts BOTH the harmonics AND the spectral envelope
//   (the formants). Formants are what make a voice sound like a human voice;
//   shifting them is what makes pitch-shifted speech sound like helium or
//   demonic.
//
// Algorithm:
//   1. Standard analysis: STFT -> (magn, phase) per bin.
//   2. Estimate the spectral envelope E(k) from log|S(k)| via cepstral
//      liftering (low-quefrency cepstrum reconstructed back to spectrum).
//   3. Flatten: S'(k) = S(k) / E(k). Now S' has only the harmonic structure.
//   4. Pitch-shift S' via bin-shift (frequency-domain remapping). This moves
//      harmonics without touching formants.
//   5. Re-apply the ORIGINAL envelope: S''(k) = S'_shifted(k) * E(k).
//   6. iSTFT with phase propagation.
//
// Quality knobs:
//   * cepstral_order: how many cepstral coefficients to keep. Higher =
//     finer envelope (more formants captured) but risks capturing the
//     pitch itself. Typical 24-48 for voice.
//
// Voicing gate (from YinDetector):
//   * On unvoiced frames (consonants, sibilance, noise) we either bypass
//     formant preservation (pass through plain bin-shift) or skip pitch
//     shift entirely. This avoids artifacts on "s" / "sh" / "f" which
//     don't have meaningful harmonic structure to preserve.
//
// All paranoia from StreamingPhaseVocoder applies plus:
//   - Envelope is bounded above and below to avoid division explosions
//   - Cepstral order is clamped to [4, fft_size/4]
//   - Envelope min/max ratios checked; if envelope is degenerate
//     (silence frame slipping past silence-detector), fall back to
//     bin-shift only.
#pragma once

#include "backend.hpp"
#include "quality.hpp"
#include "yin.hpp"

#include <complex>
#include <cstddef>
#include <memory>
#include <vector>

namespace aboba {

struct FormantVocoderConfig {
    std::size_t fft_size       = 2048;
    std::size_t hop_size       = 512;
    double      sample_rate    = 48000.0;
    QualityProfile profile     = QualityProfile::Balanced;

    // Override cepstral order. 0 = derived from profile.
    int cepstral_order_override = 0;

    // Set false to disable F0-based voicing gate.
    bool use_voicing_gate = true;

    // F0 range for YIN (used by voicing gate). Defaults work for adult voices.
    float f0_min_hz = 60.0f;
    float f0_max_hz = 1200.0f;
};

struct FormantVocoderStats {
    std::uint64_t frames_total       = 0;
    std::uint64_t frames_voiced      = 0;
    std::uint64_t frames_unvoiced    = 0;
    std::uint64_t frames_silent      = 0;
    std::uint64_t frames_degenerate  = 0;  // envelope rejected
    float         last_f0_hz         = 0.0f;
    float         last_aperiodicity  = 1.0f;
};

class FormantVocoder {
public:
    FormantVocoder(FormantVocoderConfig cfg, Backend* backend);

    void process(const float* input, float* output, std::size_t n_samples);

    void set_pitch_semitones(float st);

    // Independent formant shift, in semitones.
    //   +N semitones -> formants stretched up the spectrum (smaller/younger
    //                   voice, "anime girl" direction)
    //   -N semitones -> formants compressed down (larger/older voice,
    //                   "giant" direction)
    //   0 (default)  -> formants preserved (anti-helium, classic operation)
    //
    // Combined with pitch shift this is the main creative knob: pitch
    // controls perceived gender/age via fundamental, formants control
    // perceived body-size/character. They are orthogonal.
    void set_formant_semitones(float st);

    void set_profile(QualityProfile p);

    void reset();

    std::size_t latency_samples() const noexcept { return cfg_.fft_size; }
    FormantVocoderStats stats() const noexcept;

private:
    void process_one_frame();

    // Estimate spectral envelope from current spec_ into envelope_.
    // Returns true on success, false if envelope is degenerate.
    bool estimate_envelope();

    void apply_pitch_shift_with_envelope();
    void apply_pitch_shift_no_envelope();

    FormantVocoderConfig cfg_;
    Backend*    backend_ = nullptr;
    float       pitch_ratio_   = 1.0f;
    float       formant_ratio_ = 1.0f;
    int         cepstral_order_ = 24;

    std::size_t n_bins_   = 0;
    float       ola_norm_ = 1.0f;

    // I/O buffers
    std::vector<float> window_;
    std::vector<float> in_buf_;
    std::size_t        in_fill_ = 0;
    std::vector<float> out_buf_;
    std::size_t        out_write_ = 0;
    std::size_t        out_read_  = 0;

    // STFT scratch
    std::vector<float>                frame_;
    std::vector<std::complex<float>>  spec_;

    // Analysis-side
    std::vector<float> ana_magn_;
    std::vector<float> ana_freq_;     // in bin units
    std::vector<float> envelope_;     // magnitude envelope
    std::vector<float> log_mag_;      // log magnitude per bin
    std::vector<float> cepstrum_;     // real cepstrum (size fft_size)
    std::vector<std::complex<float>>  cepstrum_spec_;  // size n_bins
    std::vector<float> last_phase_;
    std::vector<float> sum_phase_;

    // Synthesis-side
    std::vector<float> syn_magn_;
    std::vector<float> syn_freq_;

    // YIN voicing gate
    std::unique_ptr<YinDetector> yin_;
    std::vector<float>           yin_buf_;
    bool                         last_voiced_ = false;
    float                        last_f0_     = 0.0f;
    float                        last_aper_   = 1.0f;

    // Stats
    std::uint64_t cnt_total_     = 0;
    std::uint64_t cnt_voiced_    = 0;
    std::uint64_t cnt_unvoiced_  = 0;
    std::uint64_t cnt_silent_    = 0;
    std::uint64_t cnt_degenerate_ = 0;
};

}  // namespace aboba
