// SPDX-License-Identifier: GPL-3.0-or-later
//
// Streaming phase vocoder for real-time use.
//
// Unlike the offline PhaseVocoder (which uses time-stretch + resample), this
// uses frequency-domain bin shifting (Bernsee's classic algorithm). Input and
// output have the SAME sample count — no resampling, suitable for streaming
// audio callbacks.
//
// Latency = fft_size samples (one frame of look-ahead). At 48 kHz with
// fft_size=2048, that's ~43 ms — usable for voice changing in OBS / mic apps.
// Reduce fft_size for lower latency at the cost of frequency resolution.
#pragma once

#include "backend.hpp"
#include <complex>
#include <vector>

namespace aboba {

class StreamingPhaseVocoder {
public:
    StreamingPhaseVocoder(std::size_t fft_size,
                          std::size_t hop_size,
                          Backend* backend);

    // Process `n_samples` samples. Input and output buffers may overlap.
    // Output is delayed by latency_samples() compared to input.
    void process(const float* input, float* output, std::size_t n_samples);

    void set_pitch_ratio(float ratio)       { pitch_ratio_ = ratio; }
    void set_pitch_semitones(float st);

    float pitch_ratio()  const { return pitch_ratio_; }
    std::size_t latency_samples() const { return fft_size_; }

    // Reset all internal state. Call between unrelated streams.
    void reset();

private:
    void process_one_frame();
    void compact_output_buffer_if_needed();

    std::size_t fft_size_;
    std::size_t hop_size_;
    std::size_t n_bins_;
    Backend*    backend_;
    float       pitch_ratio_ = 1.0f;

    // Window (used for both analysis and synthesis)
    std::vector<float> window_;
    float ola_norm_;  // sum of window^2 over hops (COLA constant)

    // Input accumulator: collects fft_size samples, processes, shifts left by hop.
    std::vector<float> in_buf_;
    std::size_t in_fill_ = 0;

    // Output accumulator: OLA buffer. Linear, compacted periodically.
    std::vector<float> out_buf_;
    std::size_t out_write_ = 0;  // write head (where next iSTFT frame OLAs to)
    std::size_t out_read_  = 0;  // read head (next sample to output)

    // Per-bin phase state
    std::vector<float> last_phase_;
    std::vector<float> sum_phase_;

    // Per-frame scratch (analysis -> shift -> synthesis)
    std::vector<float>                frame_;
    std::vector<std::complex<float>>  spec_;
    std::vector<float>                ana_magn_;
    std::vector<float>                ana_freq_;   // true freq, in "bins" units
    std::vector<float>                syn_magn_;
    std::vector<float>                syn_freq_;
};

}  // namespace aboba
