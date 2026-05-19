// SPDX-License-Identifier: GPL-3.0-or-later
#include "aboba/reverb.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace aboba {

namespace {

inline float sanitize(float x) noexcept {
    return std::isfinite(x) ? x : 0.0f;
}

// Reference delays from the original Freeverb (44.1 kHz, in samples).
// We scale these to the actual sample rate at configure time.
constexpr int kCombsRef[8]    = {1116, 1188, 1277, 1356, 1422, 1491, 1557, 1617};
constexpr int kAllpassRef[4]  = { 225,  556,  441,  341};
constexpr double kRefSampleRate = 44100.0;

// Stability cap on feedback. Above this things ring forever.
constexpr float kMaxFeedback = 0.98f;

inline float clamp01(float v) noexcept {
    if (!std::isfinite(v)) return 0.0f;
    if (v < 0.0f) return 0.0f;
    if (v > 1.0f) return 1.0f;
    return v;
}

}  // namespace

Reverb::Reverb(ReverbConfig cfg) : cfg_(cfg) {
    if (!(cfg_.sample_rate > 0.0)) {
        throw std::invalid_argument("Reverb: invalid sample_rate");
    }

    // Predelay
    float pre_ms = cfg_.predelay_ms;
    if (!std::isfinite(pre_ms) || pre_ms < 0.0f) pre_ms = 0.0f;
    if (pre_ms > 200.0f) pre_ms = 200.0f;
    std::size_t pre_samples = static_cast<std::size_t>(
        std::ceil(static_cast<double>(pre_ms) * 0.001 * cfg_.sample_rate));
    if (pre_samples < 1) pre_samples = 1;
    pre_.assign(pre_samples, 0.0f);
    pre_idx_ = 0;

    // Configure combs and allpasses with scaled delays
    const double scale = cfg_.sample_rate / kRefSampleRate;
    for (int i = 0; i < 8; ++i) {
        std::size_t d = static_cast<std::size_t>(
            std::ceil(static_cast<double>(kCombsRef[i]) * scale));
        if (d < 8) d = 8;
        combs_[i].configure(d);
    }
    for (int i = 0; i < 4; ++i) {
        std::size_t d = static_cast<std::size_t>(
            std::ceil(static_cast<double>(kAllpassRef[i]) * scale));
        if (d < 8) d = 8;
        allpasses_[i].configure(d);
        allpasses_[i].fb_gain = 0.5f;  // classic Freeverb value
    }

    set_room_size(cfg_.room_size);
    set_damping  (cfg_.damping);
    set_wet      (cfg_.wet);
}

void Reverb::set_room_size(float v) noexcept {
    v = clamp01(v);
    // Map [0..1] to [0.7..0.97] — a useful range; pure 0 sounds dead
    float fb = 0.7f + 0.27f * v;
    if (fb > kMaxFeedback) fb = kMaxFeedback;
    room_size_ = v;
    for (auto& c : combs_) c.fb_gain = fb;
}

void Reverb::set_damping(float v) noexcept {
    v = clamp01(v);
    damping_ = v;
    for (auto& c : combs_) c.damp = v;
}

void Reverb::set_wet(float v) noexcept {
    wet_ = clamp01(v);
}

void Reverb::reset() noexcept {
    std::fill(pre_.begin(), pre_.end(), 0.0f);
    pre_idx_ = 0;
    for (auto& c : combs_)     c.reset();
    for (auto& a : allpasses_) a.reset();
}

float Reverb::process_one(float x) noexcept {
    x = sanitize(x);

    // Predelay
    const float delayed = pre_[pre_idx_];
    pre_[pre_idx_] = x;
    ++pre_idx_;
    if (pre_idx_ >= pre_.size()) pre_idx_ = 0;

    // Parallel combs: sum the outputs
    float wet_sum = 0.0f;
    // Scale comb input down so the parallel sum doesn't clip
    constexpr float kInputScale = 0.015f;
    const float comb_in = delayed * kInputScale;
    for (auto& c : combs_) wet_sum += c.process(comb_in);

    // Series allpass filters for diffusion
    for (auto& a : allpasses_) wet_sum = a.process(wet_sum);

    // Mix dry + wet
    const float dry = 1.0f - wet_;
    float y = dry * x + wet_ * wet_sum;
    // Last-resort sanitize in case of pathological feedback
    if (!std::isfinite(y)) y = 0.0f;
    return y;
}

void Reverb::process_block(const float* in, float* out, std::size_t n) noexcept {
    for (std::size_t i = 0; i < n; ++i) out[i] = process_one(in[i]);
}

}  // namespace aboba
