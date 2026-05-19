// SPDX-License-Identifier: GPL-3.0-or-later
#include "aboba/autotune.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>

namespace aboba {

namespace {

// Scale interval tables — relative semitones from root, sorted.
// (Always including 0 = root.)
constexpr int kMajor[]            = {0, 2, 4, 5, 7, 9, 11};
constexpr int kNaturalMinor[]     = {0, 2, 3, 5, 7, 8, 10};
constexpr int kHarmonicMinor[]    = {0, 2, 3, 5, 7, 8, 11};
constexpr int kDorian[]           = {0, 2, 3, 5, 7, 9, 10};
constexpr int kMixolydian[]       = {0, 2, 4, 5, 7, 9, 10};
constexpr int kPentaMajor[]       = {0, 2, 4, 7, 9};
constexpr int kPentaMinor[]       = {0, 3, 5, 7, 10};
constexpr int kBlues[]            = {0, 3, 5, 6, 7, 10};
constexpr int kWholeTone[]        = {0, 2, 4, 6, 8, 10};

template <std::size_t N>
inline bool in_table(int semitone_mod, const int (&table)[N]) noexcept {
    for (std::size_t i = 0; i < N; ++i) {
        if (table[i] == semitone_mod) return true;
    }
    return false;
}

inline float clamp01(float v) noexcept {
    if (!std::isfinite(v)) return 0.0f;
    if (v < 0.0f) return 0.0f;
    if (v > 1.0f) return 1.0f;
    return v;
}

}  // namespace

int midi_for_hz(float hz) noexcept {
    if (!(hz > 0.0f) || !std::isfinite(hz)) return -1;
    const float midi_f = 69.0f + 12.0f * std::log2(hz / 440.0f);
    return static_cast<int>(std::lround(midi_f));
}

float hz_for_midi(int midi) noexcept {
    return 440.0f * std::pow(2.0f, static_cast<float>(midi - 69) / 12.0f);
}

bool semitone_in_scale(int semitone_from_root, const ScaleConfig& s) noexcept {
    int s12 = semitone_from_root % 12;
    if (s12 < 0) s12 += 12;
    switch (s.mode) {
        case ScaleMode::Chromatic:        return true;
        case ScaleMode::Major:            return in_table(s12, kMajor);
        case ScaleMode::NaturalMinor:     return in_table(s12, kNaturalMinor);
        case ScaleMode::HarmonicMinor:    return in_table(s12, kHarmonicMinor);
        case ScaleMode::Dorian:           return in_table(s12, kDorian);
        case ScaleMode::Mixolydian:       return in_table(s12, kMixolydian);
        case ScaleMode::PentatonicMajor:  return in_table(s12, kPentaMajor);
        case ScaleMode::PentatonicMinor:  return in_table(s12, kPentaMinor);
        case ScaleMode::Blues:            return in_table(s12, kBlues);
        case ScaleMode::WholeTone:        return in_table(s12, kWholeTone);
        case ScaleMode::Custom:
            return s.custom_semitones[static_cast<std::size_t>(s12)];
    }
    return true;
}

PitchCorrector::PitchCorrector(AutotuneConfig cfg) : cfg_(cfg) {
    if (!(cfg_.sample_rate > 0.0)) {
        throw std::invalid_argument("PitchCorrector: invalid sample_rate");
    }
    if (cfg_.f0_min_hz <= 0.0f || cfg_.f0_max_hz <= cfg_.f0_min_hz) {
        throw std::invalid_argument("PitchCorrector: invalid f0 range");
    }
    cfg_.strength  = clamp01(cfg_.strength);
    cfg_.smoothing = clamp01(cfg_.smoothing);
    if (cfg_.voicing_threshold < 0.01f) cfg_.voicing_threshold = 0.01f;
    if (cfg_.voicing_threshold > 1.0f)  cfg_.voicing_threshold = 1.0f;
    if (cfg_.hop_samples < 32)   cfg_.hop_samples = 32;
    if (cfg_.hop_samples > 4096) cfg_.hop_samples = 4096;

    YinConfig yc;
    yc.sample_rate = cfg_.sample_rate;
    yc.f0_min_hz   = cfg_.f0_min_hz;
    yc.f0_max_hz   = cfg_.f0_max_hz;
    yc.threshold   = cfg_.voicing_threshold;
    yin_ = std::make_unique<YinDetector>(yc);

    buf_capacity_ = yin_->window_size() + yin_->tau_max();
    buf_.assign(buf_capacity_, 0.0f);
    buf_fill_ = 0;
    samples_since_yin_ = 0;
}

void PitchCorrector::reset() {
    std::fill(buf_.begin(), buf_.end(), 0.0f);
    buf_fill_ = 0;
    samples_since_yin_ = 0;
    current_ratio_   = 1.0f;
    target_ratio_    = 1.0f;
    last_f0_hz_      = 0.0f;
    last_target_hz_  = 0.0f;
    last_target_midi_ = -1;
    last_aper_       = 1.0f;
    last_voiced_     = false;
    cnt_total_       = 0;
    cnt_voiced_      = 0;
}

void PitchCorrector::set_scale(const ScaleConfig& s) noexcept {
    cfg_.scale = s;
}

void PitchCorrector::set_strength(float s) noexcept {
    cfg_.strength = clamp01(s);
}

void PitchCorrector::set_smoothing(float s) noexcept {
    cfg_.smoothing = clamp01(s);
}

void PitchCorrector::set_voicing_threshold(float v) noexcept {
    if (!std::isfinite(v) || v < 0.01f) v = 0.01f;
    if (v > 1.0f) v = 1.0f;
    cfg_.voicing_threshold = v;
}

AutotuneStats PitchCorrector::stats() const noexcept {
    AutotuneStats s;
    s.last_detected_f0_hz = last_f0_hz_;
    s.last_target_f0_hz   = last_target_hz_;
    s.last_target_midi    = last_target_midi_;
    s.last_aperiodicity   = last_aper_;
    s.last_voiced         = last_voiced_;
    s.current_ratio       = current_ratio_;
    s.frames_total        = cnt_total_;
    s.frames_voiced       = cnt_voiced_;
    return s;
}

float PitchCorrector::snap_to_scale(float f0_hz) const noexcept {
    if (!(f0_hz > 0.0f) || !std::isfinite(f0_hz)) return f0_hz;

    const int midi_now = midi_for_hz(f0_hz);
    if (midi_now < 0) return f0_hz;

    // Find the nearest scale-valid MIDI by searching outward
    int best = midi_now;
    bool found = false;
    for (int d = 0; d < 12 && !found; ++d) {
        for (int sign : {0, +1, -1}) {
            if (sign == 0 && d != 0) continue;
            const int cand = midi_now + sign * d;
            const int rel  = cand - cfg_.scale.root_midi;
            if (semitone_in_scale(rel, cfg_.scale)) {
                best = cand;
                found = true;
                break;
            }
        }
    }
    return hz_for_midi(best);
}

void PitchCorrector::run_detection() noexcept {
    // YIN requires the full buffer
    if (buf_fill_ < buf_capacity_) return;

    auto r = yin_->detect(buf_.data(), buf_capacity_);
    ++cnt_total_;
    last_f0_hz_    = r.f0_hz;
    last_aper_     = r.aperiodicity;
    last_voiced_   = r.is_voiced;
    if (!r.is_voiced || r.f0_hz <= 0.0f) {
        // Drift target back toward 1.0 (no correction) over time
        target_ratio_ = 1.0f;
        last_target_hz_ = 0.0f;
        last_target_midi_ = -1;
        return;
    }
    ++cnt_voiced_;

    const float snapped = snap_to_scale(r.f0_hz);
    last_target_hz_ = snapped;
    last_target_midi_ = midi_for_hz(snapped);

    // Pitch ratio that takes detected -> snapped
    float ratio = snapped / r.f0_hz;
    // Strength interpolation: 1 -> full snap, 0 -> no snap
    ratio = 1.0f + cfg_.strength * (ratio - 1.0f);

    // Clamp to ±2 octaves
    if (ratio < 0.25f) ratio = 0.25f;
    if (ratio > 4.0f)  ratio = 4.0f;
    if (!std::isfinite(ratio)) ratio = 1.0f;
    target_ratio_ = ratio;
}

float PitchCorrector::process_block(const float* in, std::size_t n) noexcept {
    if (!in || n == 0) return current_ratio_;

    // Shift accumulated buffer to make room (drop oldest, append newest)
    for (std::size_t i = 0; i < n; ++i) {
        const float s = std::isfinite(in[i]) ? in[i] : 0.0f;
        if (buf_fill_ < buf_capacity_) {
            buf_[buf_fill_++] = s;
        } else {
            // shift left by 1, append at end
            std::memmove(buf_.data(),
                         buf_.data() + 1,
                         (buf_capacity_ - 1) * sizeof(float));
            buf_[buf_capacity_ - 1] = s;
        }
        ++samples_since_yin_;
    }

    if (samples_since_yin_ >= cfg_.hop_samples) {
        run_detection();
        samples_since_yin_ = 0;
    }

    // Smooth current toward target
    const float s = cfg_.smoothing;
    current_ratio_ = s * current_ratio_ + (1.0f - s) * target_ratio_;
    if (!std::isfinite(current_ratio_)) current_ratio_ = 1.0f;
    return current_ratio_;
}

}  // namespace aboba
