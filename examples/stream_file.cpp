// SPDX-License-Identifier: GPL-3.0-or-later
//
// Streaming pitch-shift on a raw file. Same algorithm path as the realtime
// version, but driven by a file in fixed-size chunks. Useful for CI / smoke
// tests where no audio hardware is available.
#include "aboba/backend.hpp"
#include "aboba/streaming.hpp"

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <vector>

int main(int argc, char** argv) {
    if (argc < 4 || argc > 5) {
        std::fprintf(stderr,
            "Usage: %s <input.raw> <output.raw> <semitones> [block_size=256]\n"
            "  raw = float32 mono PCM\n", argv[0]);
        return 1;
    }
    const char* in_path  = argv[1];
    const char* out_path = argv[2];
    const float semitones = static_cast<float>(std::atof(argv[3]));
    const std::size_t block = (argc == 5) ? std::atoi(argv[4]) : 256;

    std::ifstream in(in_path, std::ios::binary | std::ios::ate);
    if (!in) { std::perror("open input"); return 1; }
    const std::streamsize bytes = in.tellg();
    in.seekg(0);
    std::vector<float> input(bytes / sizeof(float));
    in.read(reinterpret_cast<char*>(input.data()), bytes);

    auto backend = aboba::create_best_backend();
    aboba::StreamingPhaseVocoder vocoder(2048, 512, backend.get());
    vocoder.set_pitch_semitones(semitones);

    std::vector<float> output(input.size(), 0.0f);
    std::vector<float> chunk_in(block), chunk_out(block);

    for (std::size_t i = 0; i + block <= input.size(); i += block) {
        std::copy(input.begin() + i, input.begin() + i + block, chunk_in.begin());
        vocoder.process(chunk_in.data(), chunk_out.data(), block);
        std::copy(chunk_out.begin(), chunk_out.end(), output.begin() + i);
    }

    std::ofstream out(out_path, std::ios::binary);
    if (!out) { std::perror("open output"); return 1; }
    out.write(reinterpret_cast<const char*>(output.data()),
              output.size() * sizeof(float));

    std::fprintf(stderr,
        "[aboba] streamed %zu samples in %zu-sample blocks, "
        "pitch=%+.2f st, latency=%zu samples\n",
        input.size(), block, semitones, vocoder.latency_samples());
    return 0;
}
