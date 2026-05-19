// SPDX-License-Identifier: GPL-3.0-or-later
//
// Reverb — Schroeder/Moorer "Freeverb"-style algorithmic reverb.
//
// Topology:
//   input → predelay → 8 parallel comb filters (each with LP damping in
//          feedback loop) → sum → 4 series allpass filters → wet output
//   final = dry * input + wet_gain * wet_output
//
// Why algorithmic, not convolution:
//   * No IR files or initial convolution latency setup
//   * Real-time tunable room_size / damping / wet without re-loading
//   * Lower CPU footprint, predictable budget
//   * For voice (which is what this framework cares about) algorithmic
//     reverbs sound great and are what most streamers/podcasters use
//
// Parameters:
//   * room_size  ∈ [0,1] : comb feedback gain. Larger = longer tail.
//   * damping    ∈ [0,1] : high-freq absorption in feedback loop.
//                          Larger = darker tail (more like fabric room).
//   * wet        ∈ [0,1] : mix gain for the wet signal. 0 = dry only.
//   * predelay_ms        : ms before the first reflection arrives.
//
// Paranoia:
//   * All buffers pre-allocated. No allocations on the audio path.
//   * NaN/Inf input -> 0 sanitized.
//   * room_size internally clamped <0.98 to guarantee stability.
//   * Output bounded — no internal feedback can grow without bound.
//
// Latency:
//   * predelay_samples (the user-set predelay) plus implicit comb delays
//     but those are the reverb tail, not a "latency" in the perceptual
//     sense. The dry path is sample-accurate.
#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace aboba {

struct ReverbConfig {
    double sample_rate     = 48000.0;
    float  room_size       = 0.6f;
    float  damping         = 0.3f;
    float  wet             = 0.25f;
    float  predelay_ms     = 10.0f;
    // Stereo width is reserved for a future stereo version. The current
    // implementation is mono.
};

class Reverb {
public:
    explicit Reverb(ReverbConfig cfg);

    // Per-sample processing.
    float process_one(float x) noexcept;

    void process_block(const float* in, float* out, std::size_t n) noexcept;

    void reset() noexcept;

    // Real-time-safe parameter updates. Internally clamped.
    void set_room_size(float v) noexcept;
    void set_damping  (float v) noexcept;
    void set_wet      (float v) noexcept;

    float room_size() const noexcept { return room_size_; }
    float damping()   const noexcept { return damping_; }
    float wet()       const noexcept { return wet_; }

    std::size_t latency_samples() const noexcept { return 0; }

private:
    // Comb filter with LP damping in the feedback loop.
    struct CombLP {
        std::vector<float> buf;
        std::size_t        idx     = 0;
        float              fb_gain = 0.5f;   // room_size
        float              damp    = 0.3f;
        float              lp_z    = 0.0f;   // one-sample state for the LP

        void configure(std::size_t delay_samples) {
            buf.assign(delay_samples, 0.0f);
            idx = 0;
            lp_z = 0.0f;
        }
        inline float process(float x) noexcept {
            float out = buf[idx];
            // One-pole LP: lp_z = damp * lp_z + (1 - damp) * out
            lp_z = damp * lp_z + (1.0f - damp) * out;
            // Feedback (LP-shaped output back into the buffer)
            buf[idx] = x + lp_z * fb_gain;
            ++idx;
            if (idx >= buf.size()) idx = 0;
            return out;
        }
        void reset() {
            std::fill(buf.begin(), buf.end(), 0.0f);
            idx = 0;
            lp_z = 0.0f;
        }
    };

    // Allpass filter for diffusion.
    struct Allpass {
        std::vector<float> buf;
        std::size_t        idx     = 0;
        float              fb_gain = 0.5f;

        void configure(std::size_t delay_samples) {
            buf.assign(delay_samples, 0.0f);
            idx = 0;
        }
        inline float process(float x) noexcept {
            const float buf_out = buf[idx];
            const float out     = -x + buf_out;
            buf[idx] = x + buf_out * fb_gain;
            ++idx;
            if (idx >= buf.size()) idx = 0;
            return out;
        }
        void reset() {
            std::fill(buf.begin(), buf.end(), 0.0f);
            idx = 0;
        }
    };

    // Predelay ring
    std::vector<float> pre_;
    std::size_t        pre_idx_ = 0;

    // 8 parallel comb filters + 4 series allpass filters — standard
    // Schroeder Verb topology.
    CombLP  combs_[8];
    Allpass allpasses_[4];

    float room_size_ = 0.6f;
    float damping_   = 0.3f;
    float wet_       = 0.25f;

    ReverbConfig cfg_;
};

}  // namespace aboba
