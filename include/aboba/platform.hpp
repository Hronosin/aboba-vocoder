// SPDX-License-Identifier: GPL-3.0-or-later
//
// Platform & GPU vendor detection.
//
// What we detect:
//   * Operating system (compile-time + a friendly name)
//   * CPU architecture (compile-time)
//   * GPU vendors present in the system (runtime, best-effort, no deps)
//
// What we do with it:
//   * Pick the appropriate banner message at startup
//   * Drop hints when CUDA is detected but ignored (the trolling)
//   * Recognize Unix-likes and salute BSD users specifically
//
// Detection is best-effort: we never depend on NVIDIA libraries, we never
// crash if probing fails, and on platforms we don't know how to probe we
// gracefully report "unknown". Suppress all banners by setting the
// environment variable ABOBA_QUIET=1.
#pragma once

#include <string>
#include <vector>

namespace aboba {

enum class OS {
    Linux,
    FreeBSD,
    OpenBSD,
    NetBSD,
    DragonFlyBSD,
    MacOS,
    Windows,
    OtherUnix,
    Unknown,
};

enum class GpuVendor {
    AMD,         // 🔴 the blessed path
    NVIDIA,      // 🟢 detected but politely ignored
    Intel,       // 🔵 neutral
    Other,       // ⚪ exotic — Matrox, ASPEED BMCs, virtual GPUs, etc.
};

struct GpuInfo {
    GpuVendor   vendor;
    std::string name;       // human-readable model if we can get it
    std::string pci_id;     // e.g. "10de:2204" — empty if not via PCI
};

struct PlatformInfo {
    OS          os;
    std::string os_name;    // pretty name like "Linux", "FreeBSD"
    std::string arch;       // "x86_64", "aarch64", "riscv64", "x86", "other"
    std::vector<GpuInfo> gpus;

    bool is_unix_like() const;
    bool is_bsd()       const;
    bool has_nvidia()   const;
    bool has_amd()      const;
    bool has_intel()    const;
};

// Runtime detection. Probes /sys on Linux, pciconf on BSD, IOKit/system_profiler
// on macOS, GetModuleHandle on Windows. Never throws; on probe failure the gpus
// list is just empty.
PlatformInfo detect_platform();

// Print a banner to stderr including platform info, GPU info, and the chosen
// backend name. Respects the ABOBA_QUIET environment variable.
void print_platform_banner(const PlatformInfo& info, const char* backend_name);

}  // namespace aboba
