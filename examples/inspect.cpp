// SPDX-License-Identifier: GPL-3.0-or-later
//
// aboba_inspect — debug CLI tool.
//
// Usage:
//   aboba_inspect --sysinfo                       Print system report and exit
//   aboba_inspect --benchmark [--profile=PROFILE] Quick built-in benchmark
//   aboba_inspect file.wav [--pitch=N] [--profile=PROFILE] [--dump-stages]
//
// Profiles: quality | balanced | performance
//
// Modes
//   --sysinfo:       what was detected at compile + runtime, mismatches
//   --benchmark:     processes synthetic audio, reports per-stage timings
//   file processing: reads .wav, runs the pipeline, writes _out.wav and a
//                    per-stage report. With --dump-stages, also writes the
//                    output of each stage to stage_N.wav for offline review.
#include "aboba/debug.hpp"
#include "aboba/dsp_blocks.hpp"
#include "aboba/formant_vocoder.hpp"
#include "aboba/pipeline.hpp"
#include "aboba/sysinfo.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace {

constexpr float kPi = 3.14159265358979323846f;

aboba::QualityProfile parse_profile(const std::string& s) {
    if (s == "quality")     return aboba::QualityProfile::Quality;
    if (s == "balanced")    return aboba::QualityProfile::Balanced;
    if (s == "performance") return aboba::QualityProfile::Performance;
    std::fprintf(stderr, "Unknown profile '%s' — using balanced\n", s.c_str());
    return aboba::QualityProfile::Balanced;
}

// ---- minimal WAV I/O for 32-bit float and 16-bit PCM mono ---------------

bool read_wav(const std::string& path, std::vector<float>& out, double& sr) {
    std::FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) { std::fprintf(stderr, "Cannot open '%s'\n", path.c_str()); return false; }
    char hdr[44] = {0};
    if (std::fread(hdr, 1, 44, f) != 44) { std::fclose(f); return false; }
    if (std::memcmp(hdr, "RIFF", 4) != 0 || std::memcmp(hdr+8, "WAVE", 4) != 0) {
        std::fprintf(stderr, "Not a WAV file\n"); std::fclose(f); return false;
    }
    std::uint16_t fmt, ch, bps; std::uint32_t srate;
    std::memcpy(&fmt,   hdr + 20, 2);
    std::memcpy(&ch,    hdr + 22, 2);
    std::memcpy(&srate, hdr + 24, 4);
    std::memcpy(&bps,   hdr + 34, 2);
    sr = static_cast<double>(srate);

    // Walk chunks until 'data'
    std::uint32_t data_size = 0;
    std::fseek(f, 12, SEEK_SET);
    for (int i = 0; i < 16; ++i) {
        char tag[5] = {0}; std::uint32_t sz;
        if (std::fread(tag, 1, 4, f) != 4) break;
        if (std::fread(&sz, 4, 1, f) != 1) break;
        if (std::memcmp(tag, "data", 4) == 0) { data_size = sz; break; }
        std::fseek(f, static_cast<long>(sz), SEEK_CUR);
    }
    if (data_size == 0) { std::fclose(f); return false; }

    // Read samples
    const std::size_t bytes_per_sample = bps / 8;
    const std::size_t n_samples = data_size / (bytes_per_sample * ch);
    out.resize(n_samples);
    if (fmt == 0x0003 && bps == 32) {
        std::vector<float> raw(n_samples * ch);
        std::fread(raw.data(), sizeof(float), raw.size(), f);
        // Downmix to mono
        for (std::size_t i = 0; i < n_samples; ++i) {
            float s = 0;
            for (int c = 0; c < ch; ++c) s += raw[i*ch + c];
            out[i] = s / ch;
        }
    } else if (fmt == 0x0001 && bps == 16) {
        std::vector<std::int16_t> raw(n_samples * ch);
        std::fread(raw.data(), sizeof(std::int16_t), raw.size(), f);
        for (std::size_t i = 0; i < n_samples; ++i) {
            int sum = 0;
            for (int c = 0; c < ch; ++c) sum += raw[i*ch + c];
            out[i] = static_cast<float>(sum) / static_cast<float>(ch) / 32768.0f;
        }
    } else {
        std::fprintf(stderr, "Unsupported WAV format: fmt=0x%x, bps=%d\n", fmt, bps);
        std::fclose(f);
        return false;
    }
    std::fclose(f);
    return true;
}

void cmd_sysinfo() {
    const auto& r = aboba::collect_system_report();
    aboba::print_system_report(r, std::cout);
    std::cout << "Short form: " << aboba::short_system_string(r) << "\n";
}

void cmd_benchmark(aboba::QualityProfile prof) {
    using namespace aboba;
    auto backend = create_best_backend();
    const int sr = 48000;
    const int N  = sr * 5;  // 5 seconds

    std::vector<float> in(N), out(N);
    // Voice-like signal
    const float harmonics[] = {1.0f, 0.6f, 0.4f, 0.25f, 0.15f, 0.1f};
    for (int h = 0; h < 6; ++h) {
        const float hz = 180.0f * (h + 1);
        for (int i = 0; i < N; ++i) {
            in[i] += 0.15f * harmonics[h] * std::sin(2.0f*kPi*hz*i/sr);
        }
    }

    VoicePipelineConfig pc;
    pc.sample_rate = sr;
    pc.profile     = prof;
    VoicePipeline pipe(pc, backend.get());
    pipe.set_pitch_semitones(3.0f);

    PerfProbe       p_total(std::string("pipeline-") + profile_name(prof));
    SignalInspector inspector;

    // Warm-up: prime the FFT planner and OLA
    pipe.process(in.data(), out.data(), 256);
    inspector.reset();
    p_total.reset();

    // Process in callback-sized chunks
    const std::size_t block = 256;
    std::size_t off = 256;
    while (off + block <= static_cast<std::size_t>(N)) {
        PerfProbe::Scope s(p_total);
        pipe.process(in.data() + off, out.data() + off, block);
        inspector.inspect(out.data() + off, block);
        off += block;
    }

    std::cout << "\n=== Benchmark: " << profile_name(prof) << " profile ===\n";
    std::cout << "Sample rate: " << sr << " Hz, block: " << block
              << ", duration: " << (N - 256.0)/sr << " sec\n";
    p_total.print(std::cout);
    inspector.print(std::cout);

    // Realtime budget = (block / sr) seconds. Compare against mean and p99.
    const double budget_us = static_cast<double>(block) / sr * 1e6;
    auto st = p_total.stats();
    std::cout << "Budget per block: " << budget_us << " us\n";
    std::cout << "Mean usage: " << (st.mean_us / budget_us * 100.0) << "% of budget\n";
    std::cout << "p99  usage: " << (st.p99_us  / budget_us * 100.0) << "% of budget\n";
    if (st.p99_us > budget_us) {
        std::cout << "  WARNING: p99 latency exceeds realtime budget. "
                     "Consider a lower-cost profile or larger block size.\n";
    }
}

bool write_wav_float(const std::string& path, const std::vector<float>& s, double sr) {
    aboba::FrameDumper d(path, sr, 1);
    if (!d.good()) return false;
    d.write(s.data(), s.size());
    return true;
}

void cmd_process_file(const std::string& path, aboba::QualityProfile prof,
                      float pitch_st, bool dump_stages) {
    using namespace aboba;
    std::vector<float> in;
    double sr = 48000.0;
    if (!read_wav(path, in, sr)) {
        std::fprintf(stderr, "Failed to read '%s'\n", path.c_str());
        return;
    }
    std::cout << "Loaded " << in.size() << " samples at "
              << sr << " Hz (" << in.size()/sr << " sec)\n";

    auto backend = create_best_backend();

    // We instantiate each effect separately so we can probe and dump between
    // stages. (Calling VoicePipeline once would be tidier but hides the
    // internals from the inspector.)
    DcBlocker dc;       dc.configure(sr);
    OnePoleHighPass hp; hp.configure(sr, 80.0f);
    NoiseGate gate;     gate.configure(sr);
    FormantVocoderConfig fc; fc.sample_rate = sr; fc.profile = prof;
    fc.use_voicing_gate = (prof != QualityProfile::Performance);
    FormantVocoder voc(fc, backend.get());
    voc.set_pitch_semitones(pitch_st);
    DeEsser de;         de.configure(sr);
    SoftLimiter lim;    lim.configure(sr);

    std::vector<float> a(in.size()), b(in.size());

    PerfProbe       p_dc("dc"),  p_hp("hp"),   p_gate("gate"),
                    p_voc("voc"),p_de("de"),   p_lim("limiter");
    SignalInspector i_in,        i_dc,         i_hp,
                    i_gate,      i_voc,        i_de,        i_lim;

    i_in.inspect(in.data(), in.size());

    {
        PerfProbe::Scope s(p_dc);
        dc.process_block(in.data(), a.data(), in.size());
    }
    i_dc.inspect(a.data(), a.size());

    {
        PerfProbe::Scope s(p_hp);
        hp.process_block(a.data(), b.data(), b.size());
    }
    i_hp.inspect(b.data(), b.size());

    {
        PerfProbe::Scope s(p_gate);
        gate.process_block(b.data(), a.data(), a.size());
    }
    i_gate.inspect(a.data(), a.size());

    {
        PerfProbe::Scope s(p_voc);
        voc.process(a.data(), b.data(), b.size());
    }
    i_voc.inspect(b.data(), b.size());

    {
        PerfProbe::Scope s(p_de);
        de.process_block(b.data(), a.data(), a.size());
    }
    i_de.inspect(a.data(), a.size());

    {
        PerfProbe::Scope s(p_lim);
        lim.process_block(a.data(), b.data(), b.size());
    }
    i_lim.inspect(b.data(), b.size());

    // b now holds the final processed signal
    const std::string out_path = path.substr(0, path.find_last_of('.')) + "_out.wav";
    write_wav_float(out_path, b, sr);
    std::cout << "Wrote output: " << out_path << "\n";

    if (dump_stages) {
        write_wav_float(path + ".stage1_dc.wav",      a /* stale; only for shape */, sr);
        // Re-run to dump each stage cleanly (above buffers get reused)
        std::vector<float> tmp(in.size());
        dc.reset();   dc.process_block(in.data(),  tmp.data(), tmp.size());
        write_wav_float(path + ".stage1_dc.wav", tmp, sr);
        hp.reset();   hp.process_block(tmp.data(), a.data(),   a.size());
        write_wav_float(path + ".stage2_hp.wav", a, sr);
        gate.reset(); gate.process_block(a.data(), tmp.data(), tmp.size());
        write_wav_float(path + ".stage3_gate.wav", tmp, sr);
        voc.reset();  voc.process(tmp.data(), a.data(), a.size());
        write_wav_float(path + ".stage4_vocoder.wav", a, sr);
        de.reset();   de.process_block(a.data(), tmp.data(), tmp.size());
        write_wav_float(path + ".stage5_deesser.wav", tmp, sr);
        lim.reset();  lim.process_block(tmp.data(), a.data(), a.size());
        write_wav_float(path + ".stage6_limiter.wav", a, sr);
        std::cout << "Wrote per-stage dumps next to input.\n";
    }

    std::cout << "\n=== Per-stage timing ===\n";
    p_dc.print(std::cout);
    p_hp.print(std::cout);
    p_gate.print(std::cout);
    p_voc.print(std::cout);
    p_de.print(std::cout);
    p_lim.print(std::cout);

    std::cout << "\n=== Per-stage signal stats ===\n";
    std::cout << "input    : "; i_in.print(std::cout);
    std::cout << "after dc : "; i_dc.print(std::cout);
    std::cout << "after hp : "; i_hp.print(std::cout);
    std::cout << "after gate: "; i_gate.print(std::cout);
    std::cout << "after voc: "; i_voc.print(std::cout);
    std::cout << "after de : "; i_de.print(std::cout);
    std::cout << "after lim: "; i_lim.print(std::cout);

    auto vstats = voc.stats();
    std::cout << "\n=== Vocoder internals ===\n";
    std::printf("frames: total=%llu voiced=%llu unvoiced=%llu silent=%llu degenerate=%llu\n",
                (unsigned long long)vstats.frames_total,
                (unsigned long long)vstats.frames_voiced,
                (unsigned long long)vstats.frames_unvoiced,
                (unsigned long long)vstats.frames_silent,
                (unsigned long long)vstats.frames_degenerate);
    std::printf("last F0: %.1f Hz, aperiodicity: %.3f\n",
                (double)vstats.last_f0_hz, (double)vstats.last_aperiodicity);
}

void print_usage(const char* argv0) {
    std::cout
        << "aboba_inspect — debug / profiling CLI for the Aboba framework.\n\n"
        << "Usage:\n"
        << "  " << argv0 << " --sysinfo\n"
        << "  " << argv0 << " --benchmark [--profile=quality|balanced|performance]\n"
        << "  " << argv0 << " file.wav [--pitch=SEMI] [--profile=NAME] [--dump-stages]\n\n"
        << "Examples:\n"
        << "  " << argv0 << " --sysinfo                  # show detected arch + features\n"
        << "  " << argv0 << " --benchmark                # quick perf check\n"
        << "  " << argv0 << " voice.wav --pitch=5        # shift up 5 semitones\n"
        << "  " << argv0 << " voice.wav --dump-stages    # write each stage to disk\n";
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) { print_usage(argv[0]); return 1; }

    std::string file;
    bool sysinfo = false, benchmark = false, dump_stages = false;
    float pitch = 3.0f;
    aboba::QualityProfile profile = aboba::QualityProfile::Balanced;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--sysinfo")           sysinfo = true;
        else if (a == "--benchmark")    benchmark = true;
        else if (a == "--dump-stages")  dump_stages = true;
        else if (a.rfind("--pitch=", 0) == 0)
            pitch = std::strtof(a.c_str() + 8, nullptr);
        else if (a.rfind("--profile=", 0) == 0)
            profile = parse_profile(a.substr(10));
        else if (a == "--help" || a == "-h") { print_usage(argv[0]); return 0; }
        else if (a[0] == '-') {
            std::fprintf(stderr, "Unknown option: %s\n", a.c_str());
            return 1;
        }
        else file = a;
    }

    if (sysinfo)   { cmd_sysinfo(); return 0; }
    if (benchmark) { cmd_benchmark(profile); return 0; }
    if (!file.empty()) { cmd_process_file(file, profile, pitch, dump_stages); return 0; }

    print_usage(argv[0]);
    return 1;
}
