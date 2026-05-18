// SPDX-License-Identifier: GPL-3.0-or-later
//
// Real-time voice changer: mic -> pitch shift -> output device.
// Pick the output device to be your virtual cable for OBS/Discord routing.
#include "aboba/backend.hpp"
#include "aboba/realtime.hpp"
#include "aboba/streaming.hpp"

#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>

namespace {
std::atomic<bool> g_stop{false};
void on_sigint(int) { g_stop = true; }
}

static void print_devices() {
    auto ins  = aboba::RealtimeEngine::list_input_devices();
    auto outs = aboba::RealtimeEngine::list_output_devices();
    std::fprintf(stderr, "\n=== Input devices ===\n");
    for (const auto& d : ins) {
        std::fprintf(stderr, "  [%d] %s (ch=%d, sr=%.0f)\n",
                     d.index, d.name.c_str(),
                     d.max_input_channels, d.default_sample_rate);
    }
    std::fprintf(stderr, "\n=== Output devices ===\n");
    for (const auto& d : outs) {
        std::fprintf(stderr, "  [%d] %s (ch=%d, sr=%.0f)\n",
                     d.index, d.name.c_str(),
                     d.max_output_channels, d.default_sample_rate);
    }
    std::fprintf(stderr, "\nUse --in N --out M --semitones X\n");
}

int main(int argc, char** argv) {
    int   in_dev   = -1;
    int   out_dev  = -1;
    int   sr       = 48000;
    int   block    = 256;
    float semitones = 0.0f;
    bool  list     = false;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--list")              list = true;
        else if (a == "--in"        && i+1 < argc) in_dev  = std::atoi(argv[++i]);
        else if (a == "--out"       && i+1 < argc) out_dev = std::atoi(argv[++i]);
        else if (a == "--sr"        && i+1 < argc) sr      = std::atoi(argv[++i]);
        else if (a == "--block"     && i+1 < argc) block   = std::atoi(argv[++i]);
        else if (a == "--semitones" && i+1 < argc) semitones = std::atof(argv[++i]);
        else if (a == "-h" || a == "--help") {
            std::fprintf(stderr,
                "Usage: %s [--list] [--in N] [--out M] [--sr 48000] "
                "[--block 256] [--semitones 0]\n",
                argv[0]);
            return 0;
        }
    }

    if (list) { print_devices(); return 0; }

    auto backend = aboba::create_best_backend();
    aboba::StreamingPhaseVocoder vocoder(/*fft=*/2048, /*hop=*/512, backend.get());
    vocoder.set_pitch_semitones(semitones);

    aboba::RealtimeEngine engine;
    engine.start(in_dev, out_dev, static_cast<double>(sr),
                 static_cast<std::size_t>(block),
                 [&](const float* in, float* out, std::size_t n) {
                     vocoder.process(in, out, n);
                 });

    std::fprintf(stderr,
        "[aboba] running. pitch=%+.2f semitones, sr=%d, block=%d frames, "
        "latency~=%zu samples (%.1f ms). Ctrl-C to stop.\n",
        semitones, sr, block, vocoder.latency_samples(),
        1000.0 * vocoder.latency_samples() / sr);

    std::signal(SIGINT, on_sigint);
    while (!g_stop) std::this_thread::sleep_for(std::chrono::milliseconds(100));

    engine.stop();
    std::fprintf(stderr, "[aboba] stopped.\n");
    return 0;
}
