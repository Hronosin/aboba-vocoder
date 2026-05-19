// SPDX-License-Identifier: GPL-3.0-or-later
//
// Tests for sysinfo + debug tools.
//
// Verifies:
//   * SystemReport is well-formed (sane core counts, OS detected, etc.)
//   * PerfProbe accuracy and concurrent recording
//   * SignalInspector counts the right pathologies
//   * FrameDumper produces a readable WAV
//   * AsciiMeter doesn't crash and produces non-empty output
//
// All tests are CPU-only and synthetic; no audio device, no GPU needed.
#include "aboba/debug.hpp"
#include "aboba/sysinfo.hpp"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <thread>
#include <vector>

namespace {

int g_failures = 0;
int g_total    = 0;
void check(bool ok, const char* what) {
    ++g_total;
    if (ok) std::printf("  PASS  %s\n", what);
    else    { std::printf("  FAIL  %s\n", what); ++g_failures; }
}

}  // namespace

int main() {
    using namespace aboba;

    std::printf("=== SystemReport ===\n");
    {
        const auto& r = collect_system_report();
        check(!r.os_name.empty(),  "OS name detected");
        check(!r.cpu_arch.empty(), "CPU architecture detected");
        check(r.cpu_logical_cores > 0, "logical core count > 0");
        check(r.cpu_physical_cores > 0, "physical core count > 0");
        check(r.cpu_physical_cores <= r.cpu_logical_cores,
              "physical <= logical cores");
        check(!r.compiler_name.empty(), "compiler name detected");
        check(!r.compiler_version.empty(), "compiler version detected");

        // Print for visibility
        print_system_report(r, std::cout);
        std::printf("Short form: %s\n", short_system_string(r).c_str());

        // Sanity: calling collect_system_report() twice returns the same data
        const auto& r2 = collect_system_report();
        check(r.os_name == r2.os_name && r.cpu_logical_cores == r2.cpu_logical_cores,
              "SystemReport is cached / stable across calls");
    }

    std::printf("\n=== PerfProbe: basic timing ===\n");
    {
        PerfProbe p("test");
        for (int i = 0; i < 100; ++i) {
            PerfProbe::Scope s(p);
            // tiny busy-wait
            volatile int sum = 0;
            for (int j = 0; j < 1000; ++j) sum += j;
            (void)sum;
        }
        auto st = p.stats();
        check(st.count == 100, "PerfProbe records all 100 scopes");
        check(st.mean_us > 0.0, "PerfProbe mean > 0");
        check(st.max_us >= st.min_us, "PerfProbe max >= min");
        check(st.p99_us >= st.p50_us, "PerfProbe p99 >= p50");
        std::printf("    ");
        p.print(std::cout);
    }

    std::printf("\n=== PerfProbe: reset clears state ===\n");
    {
        PerfProbe p("test");
        for (int i = 0; i < 50; ++i) {
            PerfProbe::Scope s(p);
        }
        p.reset();
        auto st = p.stats();
        check(st.count == 0, "PerfProbe::reset() clears count");
        check(st.mean_us == 0.0, "PerfProbe::reset() clears mean");
    }

    std::printf("\n=== PerfProbe: concurrent recording is safe ===\n");
    {
        PerfProbe p("concurrent");
        const int n_threads = 4;
        const int per_thread = 1000;
        std::vector<std::thread> threads;
        for (int t = 0; t < n_threads; ++t) {
            threads.emplace_back([&p, per_thread]() {
                for (int i = 0; i < per_thread; ++i) {
                    p.record(static_cast<double>(i+1) * 1e-6);
                }
            });
        }
        for (auto& th : threads) th.join();
        auto st = p.stats();
        check(st.count == static_cast<std::uint64_t>(n_threads * per_thread),
              "concurrent records all counted");
    }

    std::printf("\n=== PerfProbe: ignores NaN / negative ===\n");
    {
        PerfProbe p("test");
        p.record(std::numeric_limits<double>::quiet_NaN());
        p.record(std::numeric_limits<double>::infinity());
        p.record(-1.0);
        auto st = p.stats();
        check(st.count == 0, "rejects NaN, Inf, and negative");
    }

    std::printf("\n=== SignalInspector: clean signal ===\n");
    {
        SignalInspector si;
        std::vector<float> sig(1024);
        for (std::size_t i = 0; i < sig.size(); ++i)
            sig[i] = 0.3f * std::sin(static_cast<float>(i) * 0.1f);
        si.inspect(sig.data(), sig.size());
        auto s = si.stats();
        check(s.samples_total == 1024,  "samples counted");
        check(s.nan_count == 0,         "no NaN");
        check(s.inf_count == 0,         "no Inf");
        check(s.clip_count == 0,        "no clipping");
        check(s.silence_blocks == 0,    "block not classified as silent");
        check(s.peak_abs > 0.0f,        "peak detected");
        std::printf("    ");
        si.print(std::cout);
    }

    std::printf("\n=== SignalInspector: catches pathologies ===\n");
    {
        SignalInspector si;
        std::vector<float> bad(100);
        for (auto& v : bad) v = 0.5f;
        bad[10] = std::numeric_limits<float>::quiet_NaN();
        bad[20] = std::numeric_limits<float>::infinity();
        bad[30] = 5.0f;   // clip
        bad[40] = -3.0f;  // clip
        si.inspect(bad.data(), bad.size());
        auto s = si.stats();
        check(s.nan_count == 1,  "counts NaN");
        check(s.inf_count == 1,  "counts Inf");
        check(s.clip_count == 2, "counts clipping");
    }

    std::printf("\n=== SignalInspector: silence detection ===\n");
    {
        SignalInspector si;
        std::vector<float> quiet(512, 0.0f);
        si.inspect(quiet.data(), quiet.size());
        auto s = si.stats();
        check(s.silence_blocks == 1, "silent block detected");
    }

    std::printf("\n=== SignalInspector: thread-safe ===\n");
    {
        SignalInspector si;
        std::vector<float> sig(4096, 0.1f);
        std::vector<std::thread> threads;
        for (int t = 0; t < 4; ++t) {
            threads.emplace_back([&]() {
                for (int i = 0; i < 100; ++i) si.inspect(sig.data(), sig.size());
            });
        }
        for (auto& th : threads) th.join();
        auto s = si.stats();
        check(s.samples_total == 4u * 100u * 4096u,
              "concurrent inspect counts all samples");
    }

    std::printf("\n=== FrameDumper: writes valid WAV ===\n");
    {
        const std::string path = "/tmp/aboba_dump_test.wav";
        {
            FrameDumper d(path, 48000.0, 1);
            check(d.good(), "FrameDumper opens file");
            std::vector<float> sig(48000);
            for (std::size_t i = 0; i < sig.size(); ++i)
                sig[i] = 0.5f * std::sin(static_cast<float>(i) * 0.05f);
            d.write(sig.data(), sig.size());
            // d goes out of scope -> finalizes
        }

        std::ifstream f(path, std::ios::binary);
        check(f.is_open(), "WAV file exists");
        char hdr[44]; f.read(hdr, 44);
        check(std::memcmp(hdr, "RIFF", 4) == 0, "RIFF magic");
        check(std::memcmp(hdr+8, "WAVE", 4) == 0, "WAVE magic");
        std::uint32_t data_size; std::memcpy(&data_size, hdr+40, 4);
        check(data_size == 48000u * 4u, "data size matches sample count");
    }

    std::printf("\n=== FrameDumper: sanitizes NaN/Inf to zero ===\n");
    {
        const std::string path = "/tmp/aboba_dump_nan.wav";
        {
            FrameDumper d(path, 48000.0);
            std::vector<float> bad(100);
            for (auto& v : bad) v = 0.5f;
            bad[10] = std::numeric_limits<float>::quiet_NaN();
            bad[20] = std::numeric_limits<float>::infinity();
            d.write(bad.data(), bad.size());
        }
        std::ifstream f(path, std::ios::binary);
        f.seekg(44);
        std::vector<float> data(100);
        f.read(reinterpret_cast<char*>(data.data()), 400);
        check(std::isfinite(data[10]),                "NaN replaced with finite");
        check(data[10] == 0.0f,                       "NaN -> 0");
        check(std::isfinite(data[20]),                "Inf replaced with finite");
        check(data[20] == 0.0f,                       "Inf -> 0");
        check(data[5] == 0.5f,                        "normal samples passed through");
    }

    std::printf("\n=== AsciiMeter: renders something ===\n");
    {
        AsciiMeter m("test", 20);
        std::vector<float> sig(1024);
        for (std::size_t i = 0; i < sig.size(); ++i)
            sig[i] = 0.3f * std::sin(static_cast<float>(i) * 0.1f);
        m.update(sig.data(), sig.size());

        std::ostringstream os;
        m.render(os, false);
        const std::string out = os.str();
        check(!out.empty(),                "output is non-empty");
        check(out.find("|") != std::string::npos, "contains bar delimiter");
        check(out.find("rms=") != std::string::npos, "contains rms label");
        check(out.find("peak=") != std::string::npos, "contains peak label");
        std::printf("    sample output: %s", out.c_str());
    }

    std::printf("\n========================================\n");
    std::printf("Total: %d/%d passed", g_total - g_failures, g_total);
    if (g_failures == 0) { std::printf(" \u2713\n"); return 0; }
    else { std::printf(" - %d FAILED \u2717\n", g_failures); return 1; }
}
