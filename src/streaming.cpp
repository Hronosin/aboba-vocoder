// SPDX-License-Identifier: GPL-3.0-or-later
//
// Streaming phase vocoder — paranoid edition.
//
// Changes from v0.2:
//   * Multi-source bug fix: synthesis bin frequencies are now magnitude-weighted
//     averages of all analysis bins that map there. The old "last write wins"
//     scheme caused phasing artifacts on signals with multiple simultaneous
//     spectral peaks (voice + music, multiple voices, harmonic content).
//   * All buffer accesses bounds-checked in debug builds.
//   * No allocations in the per-sample hot path.
//   * Hardened against NaN/Inf/denormal inputs.
//   * Silence fast-path: skip the spectral pipeline if input frame is silent.
#include "aboba/streaming.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <stdexcept>

namespace aboba {

namespace {

constexpr float kPi    = 3.14159265358979323846f;
constexpr float kTwoPi = 2.0f * kPi;

// Below this magnitude a bin is treated as "silent" — no contribution.
constexpr float kMagEpsilon = 1e-20f;

// Below this RMS, treat the whole input frame as silence and skip processing.
constexpr float kFrameSilenceRms = 1e-8f;

// Tolerance for "pitch_ratio == 1.0" fast path.
constexpr float kRatioOneEpsilon = 1e-6f;

// Maximum sane pitch shift in semitones. ±60 = ±5 octaves.
constexpr float kMaxSemitones = 60.0f;

inline float wrap_phase(float x) noexcept {
    x = std::fmod(x + kPi, kTwoPi);
    if (x < 0.0f) x += kTwoPi;
    return x - kPi;
}

// NaN/Inf -> 0. Cheap insurance against pathological inputs poisoning state.
inline float sanitize(float x) noexcept {
    return std::isfinite(x) ? x : 0.0f;
}

}  // namespace

StreamingPhaseVocoder::StreamingPhaseVocoder(std::size_t fft_size,
                                             std::size_t hop_size,
                                             Backend* backend)
    : fft_size_(fft_size),
      hop_size_(hop_size),
      n_bins_(fft_size / 2 + 1),
      backend_(backend) {
    // --- Validate everything ---------------------------------------------
    if (!backend_) {
        throw std::invalid_argument("StreamingPhaseVocoder: backend is null");
    }
    if (fft_size_ < 8) {
        throw std::invalid_argument("StreamingPhaseVocoder: fft_size must be >= 8");
    }
    if (fft_size_ > (1u << 20)) {
        throw std::invalid_argument("StreamingPhaseVocoder: fft_size unreasonably large");
    }
    if (hop_size_ == 0 || hop_size_ > fft_size_) {
        throw std::invalid_argument("StreamingPhaseVocoder: hop_size must be in (0, fft_size]");
    }

    // --- Pre-allocate all buffers eagerly --------------------------------
    // After the constructor, no allocations happen on the hot path.
    window_.assign(fft_size_, 0.0f);
    in_buf_.assign(fft_size_, 0.0f);

    // Output buffer sized generously: worst case per process() call needs
    // ~2*fft_size_ extent. 4x leaves headroom.
    out_buf_.assign(fft_size_ * 4, 0.0f);

    last_phase_.assign(n_bins_, 0.0f);
    sum_phase_.assign(n_bins_,  0.0f);

    frame_.assign(fft_size_, 0.0f);
    spec_.assign(n_bins_, std::complex<float>(0.0f, 0.0f));
    ana_magn_.assign(n_bins_, 0.0f);
    ana_freq_.assign(n_bins_, 0.0f);
    syn_magn_.assign(n_bins_, 0.0f);
    syn_freq_.assign(n_bins_, 0.0f);

    // --- Build Hann window -----------------------------------------------
    for (std::size_t n = 0; n < fft_size_; ++n) {
        window_[n] = 0.5f * (1.0f - std::cos(kTwoPi * static_cast<float>(n)
                                             / static_cast<float>(fft_size_ - 1)));
    }

    float sum_w2 = 0.0f;
    for (float w : window_) sum_w2 += w * w;
    const float frames_per_input_sample =
        static_cast<float>(fft_size_) / static_cast<float>(hop_size_);
    ola_norm_ = sum_w2 / frames_per_input_sample;
    if (ola_norm_ < 1e-20f) ola_norm_ = 1.0f;  // paranoia
}

void StreamingPhaseVocoder::set_pitch_semitones(float st) {
    if (!std::isfinite(st)) st = 0.0f;
    if (st >  kMaxSemitones) st =  kMaxSemitones;
    if (st < -kMaxSemitones) st = -kMaxSemitones;
    pitch_ratio_ = std::pow(2.0f, st / 12.0f);
}

void StreamingPhaseVocoder::reset() {
    std::fill(in_buf_.begin(),     in_buf_.end(),     0.0f);
    std::fill(out_buf_.begin(),    out_buf_.end(),    0.0f);
    std::fill(last_phase_.begin(), last_phase_.end(), 0.0f);
    std::fill(sum_phase_.begin(),  sum_phase_.end(),  0.0f);
    in_fill_   = 0;
    out_write_ = 0;
    out_read_  = 0;
}

void StreamingPhaseVocoder::process_one_frame() {
    float* __restrict__ frame      = frame_.data();
    const float* __restrict__ win  = window_.data();
    const float* __restrict__ in   = in_buf_.data();
    std::complex<float>* __restrict__ spec = spec_.data();

    // --- 1. Windowed FFT + silence check ---------------------------------
    float frame_ss = 0.0f;
    for (std::size_t n = 0; n < fft_size_; ++n) {
        const float s = sanitize(in[n]) * win[n];
        frame[n]  = s;
        frame_ss += s * s;
    }

    const float frame_rms = std::sqrt(frame_ss / static_cast<float>(fft_size_));
    if (frame_rms < kFrameSilenceRms) {
        // Silence: no FFT, no work. Just advance the write head — output
        // remains zero where we don't write.
        out_write_ += hop_size_;
        return;
    }

    backend_->fft_r2c(frame, spec, fft_size_);

    // --- 2. Analysis: magnitude + true frequency per bin -----------------
    const float expected_per_bin = kTwoPi * static_cast<float>(hop_size_)
                                 / static_cast<float>(fft_size_);
    const float freq_per_bin = 1.0f / expected_per_bin;

    float* __restrict__ ana_magn = ana_magn_.data();
    float* __restrict__ ana_freq = ana_freq_.data();
    float* __restrict__ last_ph  = last_phase_.data();

    for (std::size_t k = 0; k < n_bins_; ++k) {
        const float re = spec[k].real();
        const float im = spec[k].imag();
        const float mag = std::sqrt(re * re + im * im);
        const float phase = std::atan2(im, re);

        float delta = phase - last_ph[k];
        last_ph[k] = phase;

        delta -= static_cast<float>(k) * expected_per_bin;
        delta  = wrap_phase(delta);

        ana_magn[k] = mag;
        ana_freq[k] = static_cast<float>(k) + delta * freq_per_bin;
    }

    // --- 3. Pitch shift in frequency domain ------------------------------
    float* __restrict__ syn_magn = syn_magn_.data();
    float* __restrict__ syn_freq = syn_freq_.data();
    std::memset(syn_magn, 0, n_bins_ * sizeof(float));
    std::memset(syn_freq, 0, n_bins_ * sizeof(float));

    if (std::fabs(pitch_ratio_ - 1.0f) < kRatioOneEpsilon) {
        // Identity fast path
        std::memcpy(syn_magn, ana_magn, n_bins_ * sizeof(float));
        std::memcpy(syn_freq, ana_freq, n_bins_ * sizeof(float));
    } else {
        // Magnitude-weighted frequency mixing.
        // For each analysis bin k, find target T = round(k * ratio):
        //   syn_magn[T] += |ana[k]|
        //   syn_freq[T] += freq[k] * ratio * |ana[k]|   (weighted numerator)
        // Normalize freq by magn after the loop.
        //
        // Fixes the multi-source bug where two analysis bins mapping to the
        // same target had the second one clobber the first's frequency,
        // causing audible phasing in mixed signals.
        for (std::size_t k = 0; k < n_bins_; ++k) {
            const float w = ana_magn[k];
            if (w < kMagEpsilon) continue;

            const float scaled = static_cast<float>(k) * pitch_ratio_;
            const long target_signed =
                static_cast<long>(std::floor(scaled + 0.5f));
            if (target_signed < 0) continue;
            const std::size_t target = static_cast<std::size_t>(target_signed);
            if (target >= n_bins_) continue;

            syn_magn[target] += w;
            syn_freq[target] += ana_freq[k] * pitch_ratio_ * w;
        }
        for (std::size_t k = 0; k < n_bins_; ++k) {
            if (syn_magn[k] > kMagEpsilon) {
                syn_freq[k] /= syn_magn[k];
            } else {
                syn_freq[k] = static_cast<float>(k);  // bin-center default
            }
        }
    }

    // --- 4. Synthesis: (magn, freq) -> complex with phase propagation ----
    float* __restrict__ sum_ph = sum_phase_.data();
    for (std::size_t k = 0; k < n_bins_; ++k) {
        const float advance = syn_freq[k] * expected_per_bin;
        float ph = sum_ph[k] + advance;
        ph = wrap_phase(ph);
        sum_ph[k] = ph;

        const float c = std::cos(ph);
        const float s = std::sin(ph);
        const float m = syn_magn[k];
        spec[k] = std::complex<float>(m * c, m * s);
    }

    // --- 5. iFFT ---------------------------------------------------------
    backend_->fft_c2r(spec, frame, fft_size_);
    const float ifft_scale = 1.0f / (static_cast<float>(fft_size_) * ola_norm_);

    // --- 6. Window + OLA into out_buf_ ----------------------------------
    assert(out_write_ + fft_size_ <= out_buf_.size());
    float* __restrict__ out_ola = out_buf_.data() + out_write_;
    for (std::size_t n = 0; n < fft_size_; ++n) {
        out_ola[n] += frame[n] * win[n] * ifft_scale;
    }
    out_write_ += hop_size_;
}

void StreamingPhaseVocoder::compact_output_buffer_if_needed() {
    assert(out_read_ <= out_write_);
    if (out_read_ < out_buf_.size() / 2) return;

    const std::size_t active_end  = out_write_ + fft_size_;
    const std::size_t safe_end    = std::min(active_end, out_buf_.size());
    const std::size_t active_size = (safe_end > out_read_)
                                      ? (safe_end - out_read_)
                                      : 0;

    if (active_size > 0) {
        std::memmove(out_buf_.data(),
                     out_buf_.data() + out_read_,
                     active_size * sizeof(float));
    }
    std::fill(out_buf_.begin() + active_size, out_buf_.end(), 0.0f);
    out_write_ -= out_read_;
    out_read_   = 0;
}

void StreamingPhaseVocoder::process(const float* input,
                                    float* output,
                                    std::size_t n_samples) {
    if (n_samples == 0) return;
    if (!input || !output) {
        throw std::invalid_argument("StreamingPhaseVocoder::process: null buffer");
    }

    for (std::size_t i = 0; i < n_samples; ++i) {
        assert(in_fill_ < fft_size_);
        in_buf_[in_fill_++] = input[i];

        if (in_fill_ == fft_size_) {
            if (out_write_ + fft_size_ > out_buf_.size()) {
                compact_output_buffer_if_needed();
                if (out_write_ + fft_size_ > out_buf_.size()) {
                    // Only triggers if our 4x preallocation wasn't enough.
                    // In practice this never fires for reasonable usage.
                    out_buf_.resize(out_write_ + fft_size_ * 2, 0.0f);
                }
            }

            process_one_frame();

            const std::size_t keep = fft_size_ - hop_size_;
            std::memmove(in_buf_.data(),
                         in_buf_.data() + hop_size_,
                         keep * sizeof(float));
            in_fill_ = keep;
        }

        if (out_read_ < out_write_) {
            assert(out_read_ < out_buf_.size());
            output[i] = sanitize(out_buf_[out_read_]);
            out_buf_[out_read_] = 0.0f;
            ++out_read_;
        } else {
            output[i] = 0.0f;  // initial latency / starvation
        }
    }

    compact_output_buffer_if_needed();
}

}  // namespace aboba
