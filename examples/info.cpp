// SPDX-License-Identifier: GPL-3.0-or-later
//
// aboba_info — print platform/GPU detection results.
//
//   ./aboba_info          # real detection
//   ./aboba_info --demo   # print every possible banner scenario (no detection)
//
// Useful for verifying your machine's classification, and for seeing the
// trolling you would inflict on a green-card-having visitor.
#include "aboba/platform.hpp"

#include <cstdio>
#include <cstring>
#include <string>

static void demo_banner(const char* label, aboba::PlatformInfo info,
                        const char* backend) {
    std::fprintf(stderr, "=== %s ===\n", label);
    aboba::print_platform_banner(info, backend);
}

int main(int argc, char** argv) {
    bool demo = false;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--demo") == 0) demo = true;
        else if (std::strcmp(argv[i], "-h") == 0 || std::strcmp(argv[i], "--help") == 0) {
            std::printf("Usage: %s [--demo]\n"
                        "  (no args)  show real platform detection\n"
                        "  --demo     simulate every banner scenario\n", argv[0]);
            return 0;
        }
    }

    if (!demo) {
        const auto info = aboba::detect_platform();
        // Pretend a CPU backend was selected, just for display.
        aboba::print_platform_banner(info, "CPU (FFTW3)");
        return 0;
    }

    // --- Demo mode: hand-craft every scenario ----------------------------
    using namespace aboba;

    {
        PlatformInfo p;
        p.os = OS::Linux; p.os_name = "Linux"; p.arch = "x86_64";
        p.gpus.push_back({GpuVendor::AMD,    "Radeon RX 7900 XTX", "1002:744c"});
        demo_banner("Linux + AMD only (the dream)", p, "HIP/rocFFT [gfx1100] (SAM)");
    }
    {
        PlatformInfo p;
        p.os = OS::Linux; p.os_name = "Linux"; p.arch = "x86_64";
        p.gpus.push_back({GpuVendor::NVIDIA, "GeForce RTX 4090", "10de:2684"});
        demo_banner("Linux + NVIDIA only (the trolling)", p, "CPU (FFTW3)");
    }
    {
        PlatformInfo p;
        p.os = OS::Linux; p.os_name = "Linux"; p.arch = "x86_64";
        p.gpus.push_back({GpuVendor::AMD,    "Radeon RX 7900 XTX", "1002:744c"});
        p.gpus.push_back({GpuVendor::NVIDIA, "GeForce RTX 4090",   "10de:2684"});
        demo_banner("Linux + mixed (somehow)", p, "HIP/rocFFT [gfx1100] (SAM)");
    }
    {
        PlatformInfo p;
        p.os = OS::Linux; p.os_name = "Linux"; p.arch = "x86_64";
        p.gpus.push_back({GpuVendor::Intel,  "UHD Graphics 770", "8086:4680"});
        demo_banner("Linux + integrated Intel", p, "CPU (FFTW3)");
    }
    {
        PlatformInfo p;
        p.os = OS::Linux; p.os_name = "Linux"; p.arch = "x86_64";
        demo_banner("Linux, no GPU (CI server vibe)", p, "CPU (FFTW3)");
    }
    {
        PlatformInfo p;
        p.os = OS::FreeBSD; p.os_name = "FreeBSD"; p.arch = "x86_64";
        p.gpus.push_back({GpuVendor::AMD, "Radeon RX 6800", "1002:73bf"});
        demo_banner("FreeBSD + AMD (peak free software)", p, "HIP/rocFFT");
    }
    {
        PlatformInfo p;
        p.os = OS::OpenBSD; p.os_name = "OpenBSD"; p.arch = "aarch64";
        demo_banner("OpenBSD aarch64 (unhinged)", p, "CPU (FFTW3)");
    }
    {
        PlatformInfo p;
        p.os = OS::MacOS; p.os_name = "macOS"; p.arch = "aarch64";
        p.gpus.push_back({GpuVendor::Other, "Apple M3 Max", ""});
        demo_banner("macOS Apple Silicon", p, "CPU (FFTW3)");
    }
    {
        PlatformInfo p;
        p.os = OS::Windows; p.os_name = "Windows"; p.arch = "x86_64";
        p.gpus.push_back({GpuVendor::NVIDIA, "(NVIDIA driver present)", ""});
        demo_banner("Windows + NVIDIA", p, "CPU (FFTW3)");
    }

    return 0;
}
