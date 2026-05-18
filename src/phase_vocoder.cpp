// SPDX-License-Identifier: GPL-3.0-or-later
#include "aboba/phase_vocoder.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

namespace aboba {

namespace {
constexpr float kPi = 3.14159265358979323846f;
constexpr float kTwoPi = 2.0f * kPi;

// Wrap phase to [-pi, pi]
inline float wrap_phase(float x) {
    x = std::fmod(x + kPi, kTwoPi);
    if (x < 0.0f) x += kTwoPi;
    return x - kPi;
}

// Simple linear resampler. Good enough for now; replace with a sinc-based
// resampler later for better quality at large pitch shifts.
std::size_t linear_resample(const float* in, std::size_t n_in,
                            float factor,
                            std::vector<float>& out) {
    if (n_in == 0) { out.clear(); return 0; }
    const std::size_t n_out = static_cast<std::size_t>(
        std::floor(static_cast<float>(n_in) / factor));
    out.resize(n_out);
    for (std::size_t i = 0; i < n_out; ++i) {
        const float src = static_cast<float>(i) * factor;
        const std::size_t i0 = static_cast<std::size_t>(src);
        const std::size_t i1 = std::min(i0 + 1, n_in - 1);
        const float frac = src - static_cast<float>(i0);
        out[i] = in[i0] * (1.0f - frac) + in[i1] * frac;
    }
    return n_out;
}

}  // namespace

PhaseVocoder::PhaseVocoder(std::size_t fft_size,
                           std::size_t hop_size,
                           Backend* backend)
    : stft_(fft_size, hop_size, WindowType::Hann, backend),
      backend_(backend) {
    const std::size_t bins = stft_.n_bins();
    last_phase_.assign(bins, 0.0f);
    sum_phase_.assign(bins, 0.0f);
    omega_.resize(bins);
    for (std::size_t k = 0; k < bins; ++k) {
        // Expected phase advance per analysis hop for bin k
        omega_[k] = kTwoPi * static_cast<float>(k)
                  * static_cast<float>(hop_size) / static_cast<float>(fft_size);
    }
}

void PhaseVocoder::propagate_phases(const std::complex<float>* in_spec,
                                    std::size_t n_frames,
                                    std::size_t analysis_hop,
                                    std::size_t synthesis_hop,
                                    std::complex<float>* out_spec) {
    const std::size_t bins = stft_.n_bins();
    const std::size_t fft_size = stft_.fft_size();

    // Reset phase tracking
    std::fill(last_phase_.begin(), last_phase_.end(), 0.0f);
    std::fill(sum_phase_.begin(), sum_phase_.end(), 0.0f);

    // Per-bin expected analysis-hop phase advance
    std::vector<float> omega_a(bins);
    for (std::size_t k = 0; k < bins; ++k) {
        omega_a[k] = kTwoPi * static_cast<float>(k)
                   * static_cast<float>(analysis_hop)
                   / static_cast<float>(fft_size);
    }
    const float hop_ratio = static_cast<float>(synthesis_hop) /
                            static_cast<float>(analysis_hop);

    for (std::size_t f = 0; f < n_frames; ++f) {
        const std::complex<float>* in_frame = in_spec + f * bins;
        std::complex<float>* out_frame = out_spec + f * bins;

        for (std::size_t k = 0; k < bins; ++k) {
            const float mag = std::abs(in_frame[k]);
            const float phase = std::arg(in_frame[k]);

            // Phase deviation from expected
            float delta = phase - last_phase_[k] - omega_a[k];
            delta = wrap_phase(delta);

            // True frequency for this bin in this frame
            const float true_freq = omega_a[k] + delta;

            // Accumulate synthesis phase scaled by synthesis/analysis hop ratio
            sum_phase_[k] += true_freq * hop_ratio;
            sum_phase_[k] = wrap_phase(sum_phase_[k]);

            last_phase_[k] = phase;

            out_frame[k] = std::polar(mag, sum_phase_[k]);
        }
    }
}

std::size_t PhaseVocoder::time_stretch(const float* input,
                                       std::size_t n_samples,
                                       float factor,
                                       std::vector<float>& output) {
    if (factor <= 0.0f) {
        output.clear();
        return 0;
    }

    const std::size_t analysis_hop = stft_.hop_size();
    // We keep the analysis hop fixed and change the synthesis hop.
    // factor > 1 means longer output => synthesis_hop > analysis_hop.
    const std::size_t synthesis_hop = std::max<std::size_t>(
        1, static_cast<std::size_t>(
            std::round(static_cast<float>(analysis_hop) * factor)));

    // Forward STFT
    const std::size_t n_frames = stft_.num_frames(n_samples);
    if (n_frames == 0) { output.clear(); return 0; }
    const std::size_t bins = stft_.n_bins();

    std::vector<std::complex<float>> spec(n_frames * bins);
    stft_.forward(input, n_samples, spec.data());

    // Phase propagation
    std::vector<std::complex<float>> stretched(n_frames * bins);
    propagate_phases(spec.data(), n_frames,
                     analysis_hop, synthesis_hop,
                     stretched.data());

    // iSTFT with the synthesis hop. We construct a temporary STFT instance
    // with the new hop. This is a bit wasteful (rebuilds window etc.) — a
    // future optimization is to factor out window/OLA from STFT.
    STFT synth_stft(stft_.fft_size(), synthesis_hop,
                    WindowType::Hann, backend_);
    const std::size_t out_len = (n_frames - 1) * synthesis_hop + stft_.fft_size();
    output.assign(out_len, 0.0f);
    synth_stft.inverse(stretched.data(), n_frames, output.data());

    return output.size();
}

std::size_t PhaseVocoder::pitch_shift(const float* input,
                                      std::size_t n_samples,
                                      float semitones,
                                      std::vector<float>& output) {
    // Pitch ratio: 2^(semitones/12). >1 for up, <1 for down.
    const float ratio = std::pow(2.0f, semitones / 12.0f);

    // Scheme:
    //   1) Time-stretch by `ratio` (so the signal becomes longer when pitching up).
    //   2) Resample by `ratio` (decimation), which compresses time back to ~original
    //      and multiplies all frequencies by `ratio`.
    // Net: same duration, pitch * ratio. The two ops must be in the SAME direction
    // so they cancel in length and compose in pitch.
    std::vector<float> stretched;
    time_stretch(input, n_samples, ratio, stretched);

    return linear_resample(stretched.data(), stretched.size(),
                           ratio, output);
}

}  // namespace aboba
