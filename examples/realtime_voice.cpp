// SPDX-License-Identifier: GPL-3.0-or-later
//
// Real-time voice changer with self-protection.
//
//   1. Run a self-check before opening the stream. Refuses to start if the
//      processor can't reliably meet its deadline.
//   2. Wrap the processor in a HealthMonitor that auto-bypasses on jitter
//      spikes instead of dragging down the whole audio stack.
//   3. Print live stats every second so you can see what's happening.
#include "aboba/backend.hpp"
#include "aboba/health.hpp"
#include "aboba/realtime.hpp"
#include "aboba/self_check.hpp"
#include "aboba/streaming.hpp"

#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
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
    int   in_dev    = -1;
    int   out_dev   = -1;
    int   sr        = 48000;
    int   block     = 256;
    float semitones = 0.0f;
    bool  list      = false;
    bool  skip_self_check = false;
    bool  force_start = false;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if      (a == "--list") list = true;
        else if (a == "--in"        && i+1 < argc) in_dev  = std::atoi(argv[++i]);
        else if (a == "--out"       && i+1 < argc) out_dev = std::atoi(argv[++i]);
        else if (a == "--sr"        && i+1 < argc) sr      = std::atoi(argv[++i]);
        else if (a == "--block"     && i+1 < argc) block   = std::atoi(argv[++i]);
        else if (a == "--semitones" && i+1 < argc)
            semitones = static_cast<float>(std::atof(argv[++i]));
        else if (a == "--no-self-check") skip_self_check = true;
        else if (a == "--force")         force_start    = true;
        else if (a == "-h" || a == "--help") {
            std::fprintf(stderr,
                "Usage: %s [options]\n"
                "  --list                 list audio devices and exit\n"
                "  --in N                 input device index (-1 = default)\n"
                "  --out M                output device index (-1 = default)\n"
                "  --sr 48000             sample rate\n"
                "  --block 256            audio callback block size\n"
                "  --semitones 0          pitch shift in semitones\n"
                "  --no-self-check        skip self-check (dangerous)\n"
                "  --force                start stream even if self-check fails\n",
                argv[0]);
            return 0;
        }
    }

    if (list) { print_devices(); return 0; }

    auto backend = aboba::create_best_backend();
    aboba::StreamingPhaseVocoder vocoder(/*fft=*/2048, /*hop=*/512,
                                         backend.get());
    vocoder.set_pitch_semitones(semitones);

    // ----- Self-check ------------------------------------------------
    if (!skip_self_check) {
        aboba::SelfCheckConfig cfg;
        cfg.block_size  = static_cast<std::size_t>(block);
        cfg.sample_rate = static_cast<double>(sr);
        cfg.iterations  = 300;

        std::fprintf(stderr, "[aboba] running self-check...\n");
        auto result = aboba::run_self_check(
            [&](const float* in, float* out, std::size_t n) {
                vocoder.process(in, out, n);
            }, cfg);
        aboba::print_self_check(result);

        if (!result.safe_to_run && !force_start) {
            std::fprintf(stderr,
                "\n[aboba] REFUSING to start: %s\n"
                "  This protects your audio system from glitches.\n"
                "  Try a larger --block, or pass --force to override.\n",
                result.reason);
            return 2;
        }
        vocoder.reset();  // clear state from self-check warm-up
    }

    // ----- HealthMonitor --------------------------------------------
    aboba::HealthConfig hcfg;
    hcfg.budget_us = static_cast<std::uint32_t>(
        static_cast<double>(block) / sr * 1e6 * 0.5);  // 50% of budget
    hcfg.hard_ceiling_us = static_cast<std::uint32_t>(
        static_cast<double>(block) / sr * 1e6 * 2.0);  // 2x = catastrophic

    aboba::HealthMonitor monitor(
        [&](const float* in, float* out, std::size_t n) {
            vocoder.process(in, out, n);
        }, hcfg);

    // ----- Open audio stream ----------------------------------------
    aboba::RealtimeEngine engine;
    try {
        engine.start(in_dev, out_dev, static_cast<double>(sr),
                     static_cast<std::size_t>(block),
                     [&](const float* in, float* out, std::size_t n) {
                         monitor.process(in, out, n);
                     });
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[aboba] failed to start stream: %s\n", e.what());
        return 1;
    }

    std::fprintf(stderr,
        "[aboba] running. pitch=%+.2f st, sr=%d, block=%d, "
        "latency~=%zu samples (%.1f ms)\n"
        "[aboba] press Ctrl-C to stop. Stats every 1s below.\n\n",
        static_cast<double>(semitones), sr, block,
        vocoder.latency_samples(),
        1000.0 * static_cast<double>(vocoder.latency_samples()) / sr);

    std::signal(SIGINT, on_sigint);

    auto last_report = std::chrono::steady_clock::now();
    while (!g_stop) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        const auto now = std::chrono::steady_clock::now();
        if (now - last_report >= std::chrono::seconds(1)) {
            last_report = now;
            const auto s = monitor.snapshot();
            std::fprintf(stderr,
                "\r[aboba] calls=%llu over=%llu bypass=%s "
                "last=%uus max=%uus budget=%uus trips=%llu  ",
                static_cast<unsigned long long>(s.calls),
                static_cast<unsigned long long>(s.over_budget),
                s.bypass_active ? "ON " : "off",
                s.last_us, s.max_us, s.budget_us,
                static_cast<unsigned long long>(s.auto_bypass_trips));
            std::fflush(stderr);
        }
    }

    engine.stop();
    const auto s = monitor.snapshot();
    std::fprintf(stderr,
        "\n[aboba] stopped.\n"
        "  final stats: %llu calls, %llu over-budget, %llu auto-trips, "
        "%llu bypassed, %llu exceptions\n",
        static_cast<unsigned long long>(s.calls),
        static_cast<unsigned long long>(s.over_budget),
        static_cast<unsigned long long>(s.auto_bypass_trips),
        static_cast<unsigned long long>(s.bypassed),
        static_cast<unsigned long long>(s.exceptions));
    return 0;
}
