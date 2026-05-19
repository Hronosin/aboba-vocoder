// SPDX-License-Identifier: GPL-3.0-or-later
#include "aboba/sysinfo.hpp"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <mutex>
#include <sstream>
#include <thread>

#if defined(__linux__)
    #include <sys/sysinfo.h>
    #include <unistd.h>
#elif defined(__APPLE__)
    #include <sys/sysctl.h>
    #include <sys/types.h>
#elif defined(_WIN32)
    #include <windows.h>
#endif

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
    #define ABOBA_ARCH_X86 1
    #if defined(__GNUC__) || defined(__clang__)
        #include <cpuid.h>
    #elif defined(_MSC_VER)
        #include <intrin.h>
    #endif
#endif

namespace aboba {

namespace {

// Cache the report; first call fills, subsequent calls just return.
std::mutex g_report_mutex;
bool       g_report_valid = false;
SystemReport g_report;

#if defined(ABOBA_ARCH_X86)
// Pack EAX/EBX/ECX/EDX from CPUID into 12-char vendor string
std::string cpu_vendor_x86() {
    unsigned int eax = 0, ebx = 0, ecx = 0, edx = 0;
    #if defined(__GNUC__) || defined(__clang__)
        if (!__get_cpuid(0, &eax, &ebx, &ecx, &edx)) return {};
    #elif defined(_MSC_VER)
        int regs[4]; __cpuid(regs, 0);
        eax = (unsigned)regs[0]; ebx = (unsigned)regs[1];
        ecx = (unsigned)regs[2]; edx = (unsigned)regs[3];
    #else
        return {};
    #endif
    char v[13] = {0};
    std::memcpy(v + 0, &ebx, 4);
    std::memcpy(v + 4, &edx, 4);
    std::memcpy(v + 8, &ecx, 4);
    return std::string(v);
}

std::string cpu_model_x86() {
    char brand[49] = {0};
    unsigned int eax, ebx, ecx, edx;
    auto get = [&](unsigned int leaf, char* dst) {
    #if defined(__GNUC__) || defined(__clang__)
        if (!__get_cpuid(leaf, &eax, &ebx, &ecx, &edx)) {
            eax = ebx = ecx = edx = 0;
        }
    #elif defined(_MSC_VER)
        int regs[4]; __cpuid(regs, (int)leaf);
        eax=(unsigned)regs[0]; ebx=(unsigned)regs[1];
        ecx=(unsigned)regs[2]; edx=(unsigned)regs[3];
    #else
        eax = ebx = ecx = edx = 0;
    #endif
        std::memcpy(dst + 0,  &eax, 4);
        std::memcpy(dst + 4,  &ebx, 4);
        std::memcpy(dst + 8,  &ecx, 4);
        std::memcpy(dst + 12, &edx, 4);
    };
    get(0x80000002u, brand);
    get(0x80000003u, brand + 16);
    get(0x80000004u, brand + 32);

    // Strip leading/trailing whitespace
    std::string s(brand);
    auto a = s.find_first_not_of(" \t");
    auto b = s.find_last_not_of(" \t\0");
    if (a == std::string::npos) return {};
    return s.substr(a, b - a + 1);
}
#endif  // ABOBA_ARCH_X86

void detect_cpu_features(SystemReport& r) {
#if defined(__GNUC__) || defined(__clang__)
    // __builtin_cpu_supports is available on x86 GCC/Clang.
    #if defined(ABOBA_ARCH_X86)
        __builtin_cpu_init();
        r.rt_has_sse4_2  = __builtin_cpu_supports("sse4.2");
        r.rt_has_avx     = __builtin_cpu_supports("avx");
        r.rt_has_avx2    = __builtin_cpu_supports("avx2");
        r.rt_has_fma     = __builtin_cpu_supports("fma");
        r.rt_has_avx512f = __builtin_cpu_supports("avx512f");
    #elif defined(__aarch64__)
        r.rt_has_neon = true;  // mandatory on aarch64
    #endif
#endif
}

void detect_compiletime_features(SystemReport& r) {
#if defined(__AVX2__)
    r.ct_has_avx2 = true;
#endif
#if defined(__AVX512F__)
    r.ct_has_avx512f = true;
#endif
#if defined(__FMA__)
    r.ct_has_fma = true;
#endif
#if defined(__ARM_NEON) || defined(__aarch64__)
    r.ct_has_neon = true;
#endif
}

void detect_cores(SystemReport& r) {
    r.cpu_logical_cores  = static_cast<int>(std::thread::hardware_concurrency());
    r.cpu_physical_cores = r.cpu_logical_cores;

#if defined(__linux__)
    // Try to count unique core ids in /proc/cpuinfo
    std::ifstream f("/proc/cpuinfo");
    std::string line;
    std::vector<std::string> core_ids;
    while (std::getline(f, line)) {
        if (line.rfind("core id", 0) == 0) {
            auto p = line.find(':');
            if (p != std::string::npos) {
                std::string v = line.substr(p + 1);
                // strip ws
                auto a = v.find_first_not_of(" \t");
                auto b = v.find_last_not_of(" \t\r");
                if (a != std::string::npos) core_ids.push_back(v.substr(a, b - a + 1));
            }
        }
    }
    // Dedup
    std::vector<std::string> uniq;
    for (auto& s : core_ids) {
        if (std::find(uniq.begin(), uniq.end(), s) == uniq.end()) uniq.push_back(s);
    }
    if (!uniq.empty()) r.cpu_physical_cores = static_cast<int>(uniq.size());
#elif defined(__APPLE__)
    int v = 0;
    std::size_t sz = sizeof(v);
    if (sysctlbyname("hw.physicalcpu", &v, &sz, nullptr, 0) == 0 && v > 0) {
        r.cpu_physical_cores = v;
    }
#endif
}

void detect_memory(SystemReport& r) {
#if defined(__linux__)
    struct sysinfo si{};
    if (sysinfo(&si) == 0) {
        r.total_memory_bytes = static_cast<std::uint64_t>(si.totalram)
                             * static_cast<std::uint64_t>(si.mem_unit);
    }
#elif defined(__APPLE__)
    int64_t v = 0;
    std::size_t sz = sizeof(v);
    if (sysctlbyname("hw.memsize", &v, &sz, nullptr, 0) == 0 && v > 0) {
        r.total_memory_bytes = static_cast<std::uint64_t>(v);
    }
#elif defined(_WIN32)
    MEMORYSTATUSEX m{};
    m.dwLength = sizeof(m);
    if (GlobalMemoryStatusEx(&m)) {
        r.total_memory_bytes = m.ullTotalPhys;
    }
#endif
}

void detect_os(SystemReport& r) {
#if defined(__linux__)
    r.os_name = "Linux";
#elif defined(__APPLE__)
    r.os_name = "Darwin";
#elif defined(_WIN32)
    r.os_name = "Windows";
#elif defined(__FreeBSD__)
    r.os_name = "FreeBSD";
#elif defined(__OpenBSD__)
    r.os_name = "OpenBSD";
#else
    r.os_name = "Unknown";
#endif

#if defined(__x86_64__) || defined(_M_X64)
    r.cpu_arch = "x86_64";
#elif defined(__i386__) || defined(_M_IX86)
    r.cpu_arch = "x86";
#elif defined(__aarch64__) || defined(_M_ARM64)
    r.cpu_arch = "aarch64";
#elif defined(__arm__) || defined(_M_ARM)
    r.cpu_arch = "arm";
#else
    r.cpu_arch = "unknown";
#endif
}

void detect_compiler(SystemReport& r) {
#if defined(__clang__)
    r.compiler_name = "clang";
    char buf[32];
    std::snprintf(buf, sizeof(buf), "%d.%d.%d",
                  __clang_major__, __clang_minor__, __clang_patchlevel__);
    r.compiler_version = buf;
#elif defined(__GNUC__)
    r.compiler_name = "gcc";
    char buf[32];
    std::snprintf(buf, sizeof(buf), "%d.%d.%d",
                  __GNUC__, __GNUC_MINOR__, __GNUC_PATCHLEVEL__);
    r.compiler_version = buf;
#elif defined(_MSC_VER)
    r.compiler_name = "msvc";
    char buf[16];
    std::snprintf(buf, sizeof(buf), "%d", _MSC_VER);
    r.compiler_version = buf;
#else
    r.compiler_name = "unknown";
#endif

#if defined(NDEBUG)
    r.build_type = "Release";
#else
    r.build_type = "Debug";
#endif
}

void check_compatibility(SystemReport& r) {
    // Binary built with feature X but runtime CPU doesn't have it -> SIGILL
    // waiting to happen. Loud warning.
    auto warn = [&](const char* feat) {
        std::string m = "Binary was compiled with ";
        m += feat;
        m += " but this CPU does not support it. Expect SIGILL.";
        r.compatibility_warnings.push_back(std::move(m));
    };
    if (r.ct_has_avx2    && !r.rt_has_avx2)    warn("AVX2");
    if (r.ct_has_avx512f && !r.rt_has_avx512f) warn("AVX-512F");
    if (r.ct_has_fma     && !r.rt_has_fma)     warn("FMA");
    if (r.ct_has_neon    && !r.rt_has_neon
        && r.cpu_arch == "aarch64") warn("NEON");

    // Binary built without features the host has -> performance left on
    // the table. Informational, not a warning per se.
    if (!r.ct_has_avx2 && r.rt_has_avx2) {
        r.compatibility_warnings.push_back(
            "Host has AVX2 but binary was not built with it. "
            "Rebuild with -march=native (ABOBA_AUTO_ARCH=ON) for better perf.");
    }
}

}  // namespace

const SystemReport& collect_system_report() {
    std::lock_guard<std::mutex> lk(g_report_mutex);
    if (g_report_valid) return g_report;

    SystemReport r;
    detect_os(r);
#if defined(ABOBA_ARCH_X86)
    r.cpu_vendor = cpu_vendor_x86();
    r.cpu_model  = cpu_model_x86();
#elif defined(__APPLE__) && defined(__aarch64__)
    r.cpu_vendor = "Apple";
    char buf[128] = {0}; std::size_t sz = sizeof(buf);
    if (sysctlbyname("machdep.cpu.brand_string", buf, &sz, nullptr, 0) == 0) {
        r.cpu_model = buf;
    }
#endif

    detect_cpu_features(r);
    detect_compiletime_features(r);
    detect_cores(r);
    detect_memory(r);
    detect_compiler(r);

    // Aboba version (set by CMake)
#ifdef ABOBA_VERSION_STRING
    r.aboba_version = ABOBA_VERSION_STRING;
#endif

#ifdef ABOBA_HAS_HIP
    r.has_hip_backend = true;
#endif
#ifdef ABOBA_HAS_REALTIME
    r.has_realtime = true;
#endif

    check_compatibility(r);

    g_report = std::move(r);
    g_report_valid = true;
    return g_report;
}

void print_system_report(const SystemReport& r, std::ostream& os) {
    auto pad = [](const std::string& s, std::size_t n) {
        return s.size() >= n ? s : s + std::string(n - s.size(), ' ');
    };

    os << "\n=== Aboba system report ===\n";
    if (!r.aboba_version.empty())
        os << "Aboba    : " << r.aboba_version << "\n";
    os << "OS       : " << r.os_name << " (" << r.cpu_arch << ")\n";
    os << "CPU      : ";
    if (!r.cpu_model.empty())  os << r.cpu_model << "\n";
    else if (!r.cpu_vendor.empty()) os << r.cpu_vendor << "\n";
    else                       os << "(unknown)\n";

    if (!r.cpu_vendor.empty() && !r.cpu_model.empty())
        os << "Vendor   : " << r.cpu_vendor << "\n";

    os << "Cores    : " << r.cpu_physical_cores << " physical / "
       << r.cpu_logical_cores  << " logical\n";

    if (r.total_memory_bytes > 0) {
        const double gib = static_cast<double>(r.total_memory_bytes)
                         / (1024.0 * 1024.0 * 1024.0);
        os << "Memory   : ";
        char buf[64]; std::snprintf(buf, sizeof(buf), "%.1f GiB\n", gib);
        os << buf;
    }

    // Features: print runtime, mark which were also compiled in
    auto feat = [&](const char* name, bool rt, bool ct) {
        if (!rt && !ct) return std::string{};
        std::string s = name;
        if (rt && ct) s += "(rt+ct)";
        else if (rt)  s += "(rt)";
        else          s += "(ct-only)";
        return s + " ";
    };
    std::string feats;
    feats += feat("SSE4.2 ",  r.rt_has_sse4_2,  false);
    feats += feat("AVX ",     r.rt_has_avx,     false);
    feats += feat("AVX2 ",    r.rt_has_avx2,    r.ct_has_avx2);
    feats += feat("FMA ",     r.rt_has_fma,     r.ct_has_fma);
    feats += feat("AVX-512F ",r.rt_has_avx512f, r.ct_has_avx512f);
    feats += feat("NEON ",    r.rt_has_neon,    r.ct_has_neon);
    if (feats.empty()) feats = "(none detected)";
    os << "Features : " << feats << "\n";
    os << "  Key    : (rt)=runtime support, (ct)=compiled in, (rt+ct)=both\n";

    os << "Compiler : " << r.compiler_name << " " << r.compiler_version
       << "  [" << r.build_type << "]\n";

    os << "Backends : CPU";
    if (r.has_hip_backend) os << " + HIP";
    if (r.has_realtime)    os << " + PortAudio realtime";
    os << "\n";

    if (!r.compatibility_warnings.empty()) {
        os << "\nWarnings :\n";
        for (const auto& w : r.compatibility_warnings) {
            os << "  ! " << w << "\n";
        }
    }
    os << "============================\n\n";
    (void)pad;
}

std::string short_system_string(const SystemReport& r) {
    std::ostringstream s;
    s << r.os_name << " " << r.cpu_arch;
    if (!r.cpu_vendor.empty()) s << " / " << r.cpu_vendor;
    s << " / " << r.cpu_logical_cores << " cores / ";
    if (r.rt_has_avx512f) s << "AVX-512";
    else if (r.rt_has_avx2 && r.rt_has_fma) s << "AVX2+FMA";
    else if (r.rt_has_avx2) s << "AVX2";
    else if (r.rt_has_neon) s << "NEON";
    else s << "scalar";
    s << " / " << r.compiler_name << " " << r.compiler_version
      << " / " << r.build_type;
    return s.str();
}

}  // namespace aboba
