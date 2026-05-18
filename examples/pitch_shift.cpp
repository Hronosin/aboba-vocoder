// SPDX-License-Identifier: GPL-3.0-or-later
// Tiny example: load a raw PCM file, pitch-shift it, write back.
//
// Usage:
//   aboba_pitch_shift <input.raw> <output.raw> <semitones>
//
// Raw format: 32-bit float, mono, any sample rate (we don't need it for the
// algorithm itself — only when actually playing back).
#include "aboba/backend.hpp"
#include "aboba/phase_vocoder.hpp"

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <vector>

int main(int argc, char** argv) {
    if (argc != 4) {
        std::fprintf(stderr,
            "Usage: %s <input.raw> <output.raw> <semitones>\n"
            "  raw = float32 mono PCM\n", argv[0]);
        return 1;
    }

    const char* in_path  = argv[1];
    const char* out_path = argv[2];
    const float semitones = static_cast<float>(std::atof(argv[3]));

    // Load
    std::ifstream in(in_path, std::ios::binary | std::ios::ate);
    if (!in) { std::perror("open input"); return 1; }
    const std::streamsize bytes = in.tellg();
    in.seekg(0);
    std::vector<float> input(bytes / sizeof(float));
    in.read(reinterpret_cast<char*>(input.data()), bytes);

    std::fprintf(stderr, "[aboba] loaded %zu samples\n", input.size());

    // Process
    auto backend = aboba::create_best_backend();
    aboba::PhaseVocoder pv(/*fft_size=*/2048, /*hop_size=*/512, backend.get());

    std::vector<float> output;
    pv.pitch_shift(input.data(), input.size(), semitones, output);

    std::fprintf(stderr, "[aboba] pitch-shifted by %.2f semitones -> %zu samples\n",
                 semitones, output.size());

    // Save
    std::ofstream out(out_path, std::ios::binary);
    if (!out) { std::perror("open output"); return 1; }
    out.write(reinterpret_cast<const char*>(output.data()),
              output.size() * sizeof(float));

    return 0;
}
