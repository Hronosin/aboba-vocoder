// SPDX-License-Identifier: GPL-3.0-or-later
//
// Runtime system information report.
//
// Purpose: lets framework users (and Aboba itself) answer:
//   * "What was this binary built for?" — compile-time arch, features
//   * "What is the runtime CPU actually capable of?" — runtime CPUID
//   * "Will the binary run cleanly here?" — mismatch detection
//
// The single function `collect_system_report()` returns a `SystemReport`
// struct with everything detected. `print_system_report()` formats it for
// humans. Both are safe to call any number of times; results are cached on
// first call.
//
// This complements `platform.hpp` (which prints user-facing GPU/OS banners).
// `sysinfo.hpp` is the developer-facing diagnostic view.
#pragma once

#include <cstdint>
#include <ostream>
#include <string>
#include <vector>

namespace aboba {

struct SystemReport {
    // Operating system / architecture
    std::string os_name;          // "Linux", "Darwin", "Windows"
    std::string cpu_arch;         // "x86_64", "aarch64", ...
    std::string cpu_vendor;       // "GenuineIntel", "AuthenticAMD", "Apple", ...
    std::string cpu_model;        // raw model string when available

    int cpu_logical_cores  = 0;
    int cpu_physical_cores = 0;   // 0 if unknown; treat same as logical

    std::uint64_t total_memory_bytes = 0;  // 0 if unknown

    // Runtime CPU feature support (from CPUID or equivalent)
    bool rt_has_sse4_2 = false;
    bool rt_has_avx    = false;
    bool rt_has_avx2   = false;
    bool rt_has_fma    = false;
    bool rt_has_avx512f = false;
    bool rt_has_neon   = false;

    // Compile-time features (what the binary was actually compiled with)
    bool ct_has_avx2    = false;
    bool ct_has_avx512f = false;
    bool ct_has_fma     = false;
    bool ct_has_neon    = false;

    // Compiler & build info
    std::string compiler_name;     // "gcc", "clang", "msvc"
    std::string compiler_version;  // "12.3.0"
    std::string build_type;        // "Release", "Debug" — best effort
    std::string aboba_version;     // taken from CMake project version macro

    // Backend availability (compiled in)
    bool has_cpu_backend = true;   // always true
    bool has_hip_backend = false;
    bool has_realtime    = false;  // PortAudio engine compiled in?

    // Warnings detected by the compatibility check
    std::vector<std::string> compatibility_warnings;
};

// Collect a fresh report. First call may take a few microseconds; subsequent
// calls return a cached copy.
const SystemReport& collect_system_report();

// Pretty-print to a stream. Uses 80-column layout, no ANSI by default.
void print_system_report(const SystemReport& r, std::ostream& os);

// Returns a one-line summary, useful for crash logs and error reports.
//   "Linux x86_64 / AMD Ryzen / 16 cores / AVX2+FMA / gcc 12 / Release"
std::string short_system_string(const SystemReport& r);

}  // namespace aboba
