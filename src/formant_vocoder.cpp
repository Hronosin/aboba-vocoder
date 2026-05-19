// SPDX-License-Identifier: GPL-3.0-or-later
#include "aboba/formant_vocoder.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <stdexcept>

namespace aboba {

namespace {

constexpr float kPi    = 3.14159265358979323846f;
constexpr float kTwoPi = 2.0f * kPi;

// Envelope safety bounds. Magnitudes much smaller than this are treated as
// "missing" and replaced by a small constant to avoid divide-by-zero.
constexpr float kEnvelopeFloor = 1e-8f;
constexpr float kEnvelopeCeil  = 1e6f;

// Below this RMS, frame is silent — skip all expensive analysis.
constexpr float kFrameSilenceRms = 1e-7f;

// Pitch ratio clamp (same as StreamingPhaseVocoder).
constexpr float kMaxSemitones = 60.0f;

inline float wrap_phase(float x) noexcept {
    x = std::fmod(x + kPi, kTwoPi);
    if (x < 0.0f) x += kTwoPi;
    return x - kPi;
}

inline float sanitize(float x) noexcept {
    return std::isfinite(x) ? x : 0.0f;
}

// Cepstral order to keep, picked by profile. Higher = sharper envelope.
int cepstral_order_for(QualityProfile p, std::size_t fft_size) {
    int o = 24;
    switch (p) {
        case QualityProfile::Quality:     o = 48; break;
        case QualityProfile::Balanced:    o = 32; break;
        case QualityProfile::Performance: o = 24; break;
    }
    // Cap at fft_size/4 to avoid leaking pitch into envelope.
    const int hard_cap = static_cast<int>(fft_size / 4);
    if (o > hard_cap) o = hard_cap;
    if (o < 4) o = 4;
    return o;
}

}  // namespace

FormantVocoder::FormantVocoder(FormantVocoderConfig cfg, Backend* backend)
    : cfg_(cfg), backend_(backend) {
    if (!backend_) {
        throw std::invalid_argument("FormantVocoder: backend is null");
    }
    if (cfg_.fft_size < 64) {
        throw std::invalid_argument("FormantVocoder: fft_size too small (>= 64)");
    }
    if (cfg_.fft_size > (1u << 20)) {
        throw std::invalid_argument("FormantVocoder: fft_size unreasonably large");
    }
    if (cfg_.hop_size == 0 || cfg_.hop_size > cfg_.fft_size) {
        throw std::invalid_argument("FormantVocoder: hop_size out of range");
    }
    if (!(cfg_.sample_rate > 0.0)) {
        throw std::invalid_argument("FormantVocoder: sample_rate must be > 0");
    }

    n_bins_ = cfg_.fft_size / 2 + 1;

    // Cepstral order
    cepstral_order_ = (cfg_.cepstral_order_override > 0)
        ? cfg_.cepstral_order_override
        : cepstral_order_for(cfg_.profile, cfg_.fft_size);
    if (cepstral_order_ < 4) cepstral_order_ = 4;
    const int hard_cap = static_cast<int>(cfg_.fft_size / 4);
    if (cepstral_order_ > hard_cap) cepstral_order_ = hard_cap;

    // Pre-allocate everything
    window_.assign(cfg_.fft_size, 0.0f);
    in_buf_.assign(cfg_.fft_size, 0.0f);
    out_buf_.assign(cfg_.fft_size * 4, 0.0f);

    frame_.assign(cfg_.fft_size, 0.0f);
    spec_.assign(n_bins_, std::complex<float>(0.0f, 0.0f));
    cepstrum_.assign(cfg_.fft_size, 0.0f);
    cepstrum_spec_.assign(n_bins_, std::complex<float>(0.0f, 0.0f));
    log_mag_.assign(n_bins_, 0.0f);
    envelope_.assign(n_bins_, 1.0f);

    ana_magn_.assign(n_bins_, 0.0f);
    ana_freq_.assign(n_bins_, 0.0f);
    syn_magn_.assign(n_bins_, 0.0f);
    syn_freq_.assign(n_bins_, 0.0f);

    last_phase_.assign(n_bins_, 0.0f);
    sum_phase_.assign(n_bins_,  0.0f);

    // Hann window + OLA norm
    for (std::size_t n = 0; n < cfg_.fft_size; ++n) {
        window_[n] = 0.5f * (1.0f - std::cos(kTwoPi * static_cast<float>(n)
                                             / static_cast<float>(cfg_.fft_size - 1)));
    }
    float sum_w2 = 0.0f;
    for (float w : window_) sum_w2 += w * w;
    // COLA² gain for windowed OLA reconstruction: sum_w² / hop. See
    // identical derivation in StreamingPhaseVocoder.
    ola_norm_ = sum_w2 / static_cast<float>(cfg_.hop_size);
    if (ola_norm_ < 1e-20f) ola_norm_ = 1.0f;

    // YIN
    if (cfg_.use_voicing_gate) {
        YinConfig yc;
        yc.sample_rate = cfg_.sample_rate;
        yc.f0_min_hz   = cfg_.f0_min_hz;
        yc.f0_max_hz   = cfg_.f0_max_hz;
        yin_ = std::make_unique<YinDetector>(yc);
        yin_buf_.assign(yin_->window_size() + yin_->tau_max(), 0.0f);
    }
}

void FormantVocoder::set_pitch_semitones(float st) {
    if (!std::isfinite(st)) st = 0.0f;
    if (st >  kMaxSemitones) st =  kMaxSemitones;
    if (st < -kMaxSemitones) st = -kMaxSemitones;
    pitch_ratio_ = std::pow(2.0f, st / 12.0f);
}

void FormantVocoder::set_formant_semitones(float st) {
    // Use a tighter clamp than pitch — extreme formant shifts (beyond ~24
    // semitones, i.e. 4x) make the envelope alias against the FFT bin grid.
    constexpr float kFormantMax = 24.0f;
    if (!std::isfinite(st)) st = 0.0f;
    if (st >  kFormantMax) st =  kFormantMax;
    if (st < -kFormantMax) st = -kFormantMax;
    formant_ratio_ = std::pow(2.0f, st / 12.0f);
}

void FormantVocoder::set_profile(QualityProfile p) {
    cfg_.profile = p;
    cepstral_order_ = (cfg_.cepstral_order_override > 0)
        ? cfg_.cepstral_order_override
        : cepstral_order_for(p, cfg_.fft_size);
    if (cepstral_order_ < 4) cepstral_order_ = 4;
    const int hard_cap = static_cast<int>(cfg_.fft_size / 4);
    if (cepstral_order_ > hard_cap) cepstral_order_ = hard_cap;
}

void FormantVocoder::reset() {
    std::fill(in_buf_.begin(),     in_buf_.end(),     0.0f);
    std::fill(out_buf_.begin(),    out_buf_.end(),    0.0f);
    std::fill(last_phase_.begin(), last_phase_.end(), 0.0f);
    std::fill(sum_phase_.begin(),  sum_phase_.end(),  0.0f);
    in_fill_   = 0;
    out_write_ = 0;
    out_read_  = 0;
    last_voiced_ = false;
    last_f0_     = 0.0f;
    last_aper_   = 1.0f;
    cnt_total_ = cnt_voiced_ = cnt_unvoiced_ =
        cnt_silent_ = cnt_degenerate_ = 0;
}

FormantVocoderStats FormantVocoder::stats() const noexcept {
    FormantVocoderStats s;
    s.frames_total      = cnt_total_;
    s.frames_voiced     = cnt_voiced_;
    s.frames_unvoiced   = cnt_unvoiced_;
    s.frames_silent     = cnt_silent_;
    s.frames_degenerate = cnt_degenerate_;
    s.last_f0_hz        = last_f0_;
    s.last_aperiodicity = last_aper_;
    return s;
}

bool FormantVocoder::estimate_envelope() {
    // Approach: smooth log|S(k)| with a box filter, then exponentiate.
    //
    // This is simpler and far more robust than cepstral liftering. The
    // "cepstral order" knob is reinterpreted as a smoothing window width:
    //   smoothing_bins = max(2, n_bins / order)
    //
    // Cepstral envelope is more precise on stationary voice but very
    // sensitive to numerical normalization in the FFT round-trip. For our
    // use case (real-time voice changer, not vocal synthesis) the smoothed
    // approach is plenty good and predictable.

    // Build log magnitude with floor
    for (std::size_t k = 0; k < n_bins_; ++k) {
        float m = ana_magn_[k];
        if (m < kEnvelopeFloor) m = kEnvelopeFloor;
        const float lm = std::log(m);
        log_mag_[k] = std::isfinite(lm) ? lm : std::log(kEnvelopeFloor);
    }

    // Smoothing window width. Higher cepstral_order_ -> narrower window
    // (sharper formants). For voice we want ~50-200 Hz worth of bins.
    // At 48 kHz, fft=2048 -> bin width ~23 Hz. 50 Hz ~ 2 bins, 200 Hz ~ 9 bins.
    std::size_t win = n_bins_ / static_cast<std::size_t>(cepstral_order_);
    if (win < 3)  win = 3;
    if (win > n_bins_ / 4) win = n_bins_ / 4;
    if (win < 2)  win = 2;

    // Centered box smoothing. We use cepstrum_ as scratch — repurposing it
    // saves another buffer. (Same lifetime, same fft_size storage, just
    // borrowed name.)
    // Compute running average over [k-win, k+win].
    const std::size_t half = win;
    float sum = 0.0f;
    std::size_t count = 0;
    // Initial window: [0, half]
    for (std::size_t k = 0; k <= half && k < n_bins_; ++k) {
        sum += log_mag_[k];
        ++count;
    }
    cepstrum_[0] = sum / static_cast<float>(count);

    for (std::size_t k = 1; k < n_bins_; ++k) {
        // Add right edge if in range
        const std::size_t r = k + half;
        if (r < n_bins_) {
            sum += log_mag_[r];
            ++count;
        }
        // Remove left edge if it's now out of window
        if (k > half) {
            const std::size_t l = k - half - 1;
            sum -= log_mag_[l];
            --count;
        }
        cepstrum_[k] = sum / static_cast<float>(std::max<std::size_t>(count, 1));
    }

    // Exponentiate. Compute raw max first for the relative floor.
    float env_max_raw = 0.0f;
    for (std::size_t k = 0; k < n_bins_; ++k) {
        float v = cepstrum_[k];
        if (!std::isfinite(v)) v = 0.0f;
        if (v > 30.0f)  v = 30.0f;
        if (v < -30.0f) v = -30.0f;
        const float e = std::exp(v);
        envelope_[k] = e;
        if (e > env_max_raw) env_max_raw = e;
    }

    // Apply RELATIVE floor: cap flat-spectrum amplification at 1000x (60 dB).
    const float floor_rel = env_max_raw / 1000.0f;
    const float floor_abs = std::max(floor_rel, kEnvelopeFloor);

    float env_min = floor_abs;
    float env_max = 0.0f;
    for (std::size_t k = 0; k < n_bins_; ++k) {
        float e = envelope_[k];
        if (e < floor_abs) e = floor_abs;
        if (e > kEnvelopeCeil) e = kEnvelopeCeil;
        envelope_[k] = e;
        if (e < env_min) env_min = e;
        if (e > env_max) env_max = e;
    }

    // Reject if no structure
    const float ratio = env_max / env_min;
    if (!std::isfinite(ratio) || ratio < 1.5f) {
        return false;
    }
    return true;
}

void FormantVocoder::apply_pitch_shift_no_envelope() {
    // Magnitude-weighted bin shift, identical math to StreamingPhaseVocoder.
    std::memset(syn_magn_.data(), 0, n_bins_ * sizeof(float));
    std::memset(syn_freq_.data(), 0, n_bins_ * sizeof(float));

    if (std::fabs(pitch_ratio_ - 1.0f) < 1e-6f) {
        std::memcpy(syn_magn_.data(), ana_magn_.data(),
                    n_bins_ * sizeof(float));
        std::memcpy(syn_freq_.data(), ana_freq_.data(),
                    n_bins_ * sizeof(float));
        return;
    }

    for (std::size_t k = 0; k < n_bins_; ++k) {
        const float w = ana_magn_[k];
        if (w < 1e-20f) continue;
        const float scaled = static_cast<float>(k) * pitch_ratio_;
        const long ts = static_cast<long>(std::floor(scaled + 0.5f));
        if (ts < 0) continue;
        const std::size_t target = static_cast<std::size_t>(ts);
        if (target >= n_bins_) continue;
        syn_magn_[target] += w;
        syn_freq_[target] += ana_freq_[k] * pitch_ratio_ * w;
    }
    for (std::size_t k = 0; k < n_bins_; ++k) {
        if (syn_magn_[k] > 1e-20f) {
            syn_freq_[k] /= syn_magn_[k];
        } else {
            syn_freq_[k] = static_cast<float>(k);
        }
    }
}

void FormantVocoder::apply_pitch_shift_with_envelope() {
    // 1. Flatten: divide each magnitude by envelope to get harmonic-only spec
    std::vector<float>& flat = ana_magn_;  // overwrite in place — saves memory
    for (std::size_t k = 0; k < n_bins_; ++k) {
        const float e = envelope_[k];
        flat[k] = ana_magn_[k] / e;  // safe: e was floored
    }

    // 2. Bin-shift the flattened harmonics
    std::memset(syn_magn_.data(), 0, n_bins_ * sizeof(float));
    std::memset(syn_freq_.data(), 0, n_bins_ * sizeof(float));

    if (std::fabs(pitch_ratio_ - 1.0f) < 1e-6f) {
        std::memcpy(syn_magn_.data(), flat.data(),
                    n_bins_ * sizeof(float));
        std::memcpy(syn_freq_.data(), ana_freq_.data(),
                    n_bins_ * sizeof(float));
    } else {
        for (std::size_t k = 0; k < n_bins_; ++k) {
            const float w = flat[k];
            if (w < 1e-20f) continue;
            const float scaled = static_cast<float>(k) * pitch_ratio_;
            const long ts = static_cast<long>(std::floor(scaled + 0.5f));
            if (ts < 0) continue;
            const std::size_t target = static_cast<std::size_t>(ts);
            if (target >= n_bins_) continue;
            syn_magn_[target] += w;
            syn_freq_[target] += ana_freq_[k] * pitch_ratio_ * w;
        }
        for (std::size_t k = 0; k < n_bins_; ++k) {
            if (syn_magn_[k] > 1e-20f) {
                syn_freq_[k] /= syn_magn_[k];
            } else {
                syn_freq_[k] = static_cast<float>(k);
            }
        }
    }

    // 3. Re-apply the envelope, scaled by formant_ratio_.
    //
    // For formant_ratio_ != 1.0 we sample the original envelope at a
    // scaled index: out_bin k uses envelope at index k / formant_ratio_.
    //   formant_ratio_ > 1  -> sample at lower indices -> envelope contour
    //                          stretches upward in frequency -> formants
    //                          appear higher (smaller / younger voice)
    //   formant_ratio_ < 1  -> sample at higher indices -> envelope
    //                          contour compresses downward in frequency
    //                          -> formants appear lower (bigger / older
    //                          voice)
    //
    // Linear interpolation between adjacent envelope bins keeps the result
    // smooth and avoids stairstep artifacts.
    if (std::fabs(formant_ratio_ - 1.0f) < 1e-6f) {
        for (std::size_t k = 0; k < n_bins_; ++k) {
            syn_magn_[k] *= envelope_[k];
        }
    } else {
        const float inv = 1.0f / formant_ratio_;
        const float max_idx = static_cast<float>(n_bins_ - 1);
        for (std::size_t k = 0; k < n_bins_; ++k) {
            float src = static_cast<float>(k) * inv;
            // Clamp to envelope domain. Above the highest bin we'd be
            // extrapolating into nothing — pin to the last value (rolls off
            // naturally because the high bins have small envelope values).
            if (src < 0.0f) src = 0.0f;
            if (src > max_idx) src = max_idx;
            const std::size_t i0 = static_cast<std::size_t>(src);
            const std::size_t i1 = std::min<std::size_t>(i0 + 1, n_bins_ - 1);
            const float t = src - static_cast<float>(i0);
            const float e = envelope_[i0] * (1.0f - t) + envelope_[i1] * t;
            syn_magn_[k] *= e;
        }
    }
}

void FormantVocoder::process_one_frame() {
    ++cnt_total_;

    // --- 1. Windowed FFT + silence check ---
    float* __restrict__ frame      = frame_.data();
    const float* __restrict__ win  = window_.data();
    const float* __restrict__ in   = in_buf_.data();

    float frame_ss = 0.0f;
    for (std::size_t n = 0; n < cfg_.fft_size; ++n) {
        const float s = sanitize(in[n]) * win[n];
        frame[n] = s;
        frame_ss += s * s;
    }
    const float frame_rms = std::sqrt(frame_ss / static_cast<float>(cfg_.fft_size));
    if (frame_rms < kFrameSilenceRms) {
        ++cnt_silent_;
        out_write_ += cfg_.hop_size;
        return;
    }

    backend_->fft_r2c(frame, spec_.data(), cfg_.fft_size);

    // --- 2. Analyze: magnitudes + true frequencies ---
    const float expected_per_bin = kTwoPi * static_cast<float>(cfg_.hop_size)
                                 / static_cast<float>(cfg_.fft_size);
    const float freq_per_bin = 1.0f / expected_per_bin;

    for (std::size_t k = 0; k < n_bins_; ++k) {
        const float re = spec_[k].real();
        const float im = spec_[k].imag();
        const float mag = std::sqrt(re * re + im * im);
        const float ph  = std::atan2(im, re);
        float delta = ph - last_phase_[k];
        last_phase_[k] = ph;
        delta -= static_cast<float>(k) * expected_per_bin;
        delta  = wrap_phase(delta);
        ana_magn_[k] = mag;
        ana_freq_[k] = static_cast<float>(k) + delta * freq_per_bin;
    }

    // --- 3. Voicing detection (if gate enabled) ---
    bool use_envelope = (cfg_.profile != QualityProfile::Performance);
    bool voiced       = true;

    if (cfg_.use_voicing_gate && yin_) {
        // YIN needs ~(window + tau_max) samples from the current input buffer.
        // We have cfg_.fft_size samples in in_buf_. If that's not enough,
        // skip the gate and assume voiced.
        const std::size_t needed = yin_->window_size() + yin_->tau_max();
        if (cfg_.fft_size >= needed) {
            // Run on the un-windowed input (YIN works on raw audio)
            auto r = yin_->detect(in_buf_.data(), needed);
            last_f0_ = r.f0_hz;
            last_aper_ = r.aperiodicity;
            voiced = r.is_voiced;
        }
    }

    if (voiced) ++cnt_voiced_; else ++cnt_unvoiced_;

    // --- 4. Envelope estimation + pitch shift ---
    bool envelope_ok = false;
    if (use_envelope && voiced) {
        envelope_ok = estimate_envelope();
        if (!envelope_ok) ++cnt_degenerate_;
    }

    if (envelope_ok) {
        apply_pitch_shift_with_envelope();
    } else {
        apply_pitch_shift_no_envelope();
    }
    last_voiced_ = voiced;

    // --- 5. Synthesize complex spectrum with phase propagation ---
    for (std::size_t k = 0; k < n_bins_; ++k) {
        const float advance = syn_freq_[k] * expected_per_bin;
        float ph = sum_phase_[k] + advance;
        ph = wrap_phase(ph);
        sum_phase_[k] = ph;
        const float c = std::cos(ph);
        const float s = std::sin(ph);
        const float m = syn_magn_[k];
        spec_[k] = std::complex<float>(m * c, m * s);
    }

    // --- 6. iFFT + OLA ---
    backend_->fft_c2r(spec_.data(), frame, cfg_.fft_size);
    const float ifft_scale = 1.0f / (static_cast<float>(cfg_.fft_size) * ola_norm_);

    assert(out_write_ + cfg_.fft_size <= out_buf_.size());
    float* __restrict__ out_ola = out_buf_.data() + out_write_;
    for (std::size_t n = 0; n < cfg_.fft_size; ++n) {
        out_ola[n] += frame[n] * win[n] * ifft_scale;
    }
    out_write_ += cfg_.hop_size;
}

void FormantVocoder::process(const float* input, float* output, std::size_t n_samples) {
    if (n_samples == 0) return;
    if (!input || !output) {
        throw std::invalid_argument("FormantVocoder::process: null buffer");
    }

    for (std::size_t i = 0; i < n_samples; ++i) {
        in_buf_[in_fill_++] = input[i];

        if (in_fill_ == cfg_.fft_size) {
            if (out_write_ + cfg_.fft_size > out_buf_.size()) {
                // Compact: shift active region [out_read_, end) to front.
                // CLAMP source range to buffer size to avoid UB memmove.
                const std::size_t active_end =
                    std::min(out_write_ + cfg_.fft_size, out_buf_.size());
                const std::size_t active =
                    (active_end > out_read_) ? (active_end - out_read_) : 0;
                if (active > 0 && out_read_ > 0) {
                    std::memmove(out_buf_.data(),
                                 out_buf_.data() + out_read_,
                                 active * sizeof(float));
                    std::fill(out_buf_.begin() + static_cast<std::ptrdiff_t>(active), out_buf_.end(), 0.0f);
                    out_write_ -= out_read_;
                    out_read_   = 0;
                }
                if (out_write_ + cfg_.fft_size > out_buf_.size()) {
                    out_buf_.resize(out_write_ + cfg_.fft_size * 2, 0.0f);
                }
            }

            process_one_frame();

            const std::size_t keep = cfg_.fft_size - cfg_.hop_size;
            std::memmove(in_buf_.data(),
                         in_buf_.data() + cfg_.hop_size,
                         keep * sizeof(float));
            in_fill_ = keep;
        }

        if (out_read_ < out_write_) {
            output[i] = sanitize(out_buf_[out_read_]);
            out_buf_[out_read_] = 0.0f;
            ++out_read_;
        } else {
            output[i] = 0.0f;
        }
    }

    // Periodic compaction
    if (out_read_ > out_buf_.size() / 2) {
        const std::size_t active_end = std::min(out_write_ + cfg_.fft_size,
                                                out_buf_.size());
        const std::size_t active = (active_end > out_read_)
            ? (active_end - out_read_) : 0;
        if (active > 0) {
            std::memmove(out_buf_.data(),
                         out_buf_.data() + out_read_,
                         active * sizeof(float));
        }
        std::fill(out_buf_.begin() + static_cast<std::ptrdiff_t>(active), out_buf_.end(), 0.0f);
        out_write_ -= out_read_;
        out_read_   = 0;
    }
}

}  // namespace aboba
