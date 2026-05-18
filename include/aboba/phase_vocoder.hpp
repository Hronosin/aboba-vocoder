// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once

#include "stft.hpp"
#include <complex>
#include <vector>

namespace aboba {

// Classic phase vocoder with phase propagation.
// Supports:
//   - Time stretching (preserves pitch, changes duration)
//   - Pitch shifting  (preserves duration via stretch + resample)
//
// Algorithm reference: Laroche & Dolson, "Improved Phase Vocoder
// Time-Scale Modification of Audio" (1999). We use identity phase
// locking for transient preservation.
class PhaseVocoder {
public:
    PhaseVocoder(std::size_t fft_size,
                 std::size_t hop_size,
                 Backend* backend);

    // Time-stretch by `factor` (>1 = longer, <1 = shorter).
    // Returns number of output samples.
    std::size_t time_stretch(const float* input,
                             std::size_t n_samples,
                             float factor,
                             std::vector<float>& output);

    // Pitch-shift by `semitones`. Duration is preserved.
    // Implementation: time-stretch by 2^(-semitones/12), then resample.
    std::size_t pitch_shift(const float* input,
                            std::size_t n_samples,
                            float semitones,
                            std::vector<float>& output);

private:
    // Core: propagate phases across frames given new synthesis hop.
    void propagate_phases(const std::complex<float>* in_spec,
                          std::size_t n_frames,
                          std::size_t analysis_hop,
                          std::size_t synthesis_hop,
                          std::complex<float>* out_spec);

    STFT stft_;
    Backend* backend_;

    // Phase tracking state (per bin)
    std::vector<float> last_phase_;
    std::vector<float> sum_phase_;
    std::vector<float> omega_;      // bin center frequencies
};

}  // namespace aboba
