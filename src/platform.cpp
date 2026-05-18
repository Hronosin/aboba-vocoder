// SPDX-License-Identifier: GPL-3.0-or-later
//
// Platform & GPU detection — implementation.
//
// All probing is best-effort and never throws. Failed probes leave fields empty
// rather than reporting wrong info.
#include "aboba/platform.hpp"

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

// --- OS detection (compile-time) ------------------------------------------
#if defined(__linux__)
  #define ABOBA_OS_LINUX 1
#elif defined(__FreeBSD__)
  #define ABOBA_OS_FREEBSD 1
#elif defined(__OpenBSD__)
  #define ABOBA_OS_OPENBSD 1
#elif defined(__NetBSD__)
  #define ABOBA_OS_NETBSD 1
#elif defined(__DragonFly__)
  #define ABOBA_OS_DRAGONFLY 1
#elif defined(__APPLE__) && defined(__MACH__)
  #define ABOBA_OS_MACOS 1
#elif defined(_WIN32) || defined(_WIN64)
  #define ABOBA_OS_WINDOWS 1
#elif defined(__unix__)
  #define ABOBA_OS_OTHERUNIX 1
#else
  #define ABOBA_OS_UNKNOWN 1
#endif

// Unix-like family marker (for syscall availability)
#if defined(ABOBA_OS_LINUX)    || defined(ABOBA_OS_FREEBSD) ||              \
    defined(ABOBA_OS_OPENBSD)  || defined(ABOBA_OS_NETBSD)  ||              \
    defined(ABOBA_OS_DRAGONFLY)|| defined(ABOBA_OS_MACOS)   ||              \
    defined(ABOBA_OS_OTHERUNIX)
  #define ABOBA_UNIX_LIKE 1
#endif

#ifdef ABOBA_OS_LINUX
  #include <dirent.h>
#endif

#ifdef ABOBA_UNIX_LIKE
  #include <unistd.h>
  #include <sys/stat.h>
#endif

#ifdef ABOBA_OS_WINDOWS
  // Minimal Windows.h
  #ifndef WIN32_LEAN_AND_MEAN
    #define WIN32_LEAN_AND_MEAN
  #endif
  #include <windows.h>
#endif

namespace aboba {

namespace {

constexpr unsigned kVendorNvidia = 0x10de;
constexpr unsigned kVendorAmd    = 0x1002;
constexpr unsigned kVendorAti    = 0x1002;  // AMD bought ATI; same ID
constexpr unsigned kVendorIntel  = 0x8086;

GpuVendor classify_vendor(unsigned vendor_id) {
    switch (vendor_id) {
        case kVendorNvidia: return GpuVendor::NVIDIA;
        case kVendorAmd:    return GpuVendor::AMD;
        case kVendorIntel:  return GpuVendor::Intel;
        default:            return GpuVendor::Other;
    }
}

std::string trim(const std::string& s) {
    auto b = s.begin();
    while (b != s.end() && std::isspace(static_cast<unsigned char>(*b))) ++b;
    auto e = s.end();
    while (e != b && std::isspace(static_cast<unsigned char>(*(e - 1)))) --e;
    return std::string(b, e);
}

// --- OS / arch resolution ------------------------------------------------
OS compile_time_os() {
#if defined(ABOBA_OS_LINUX)
    return OS::Linux;
#elif defined(ABOBA_OS_FREEBSD)
    return OS::FreeBSD;
#elif defined(ABOBA_OS_OPENBSD)
    return OS::OpenBSD;
#elif defined(ABOBA_OS_NETBSD)
    return OS::NetBSD;
#elif defined(ABOBA_OS_DRAGONFLY)
    return OS::DragonFlyBSD;
#elif defined(ABOBA_OS_MACOS)
    return OS::MacOS;
#elif defined(ABOBA_OS_WINDOWS)
    return OS::Windows;
#elif defined(ABOBA_OS_OTHERUNIX)
    return OS::OtherUnix;
#else
    return OS::Unknown;
#endif
}

std::string os_pretty_name(OS os) {
    switch (os) {
        case OS::Linux:         return "Linux";
        case OS::FreeBSD:       return "FreeBSD";
        case OS::OpenBSD:       return "OpenBSD";
        case OS::NetBSD:        return "NetBSD";
        case OS::DragonFlyBSD:  return "DragonFly BSD";
        case OS::MacOS:         return "macOS";
        case OS::Windows:       return "Windows";
        case OS::OtherUnix:     return "Unix";
        case OS::Unknown:       return "Unknown";
    }
    return "Unknown";
}

std::string detect_arch() {
#if defined(__x86_64__) || defined(_M_X64)
    return "x86_64";
#elif defined(__aarch64__) || defined(_M_ARM64)
    return "aarch64";
#elif defined(__riscv) && (__riscv_xlen == 64)
    return "riscv64";
#elif defined(__powerpc64__) || defined(__ppc64__)
    return "ppc64";
#elif defined(__i386__) || defined(_M_IX86)
    return "x86";
#elif defined(__arm__) || defined(_M_ARM)
    return "arm";
#else
    return "other";
#endif
}

// --- Linux GPU detection via /sys/bus/pci -------------------------------
#ifdef ABOBA_OS_LINUX

// Read first line of a small sysfs/procfs file. Returns empty on failure.
std::string read_first_line(const std::string& path) {
    std::ifstream f(path);
    if (!f.good()) return {};
    std::string s;
    std::getline(f, s);
    return s;
}

// Parse "0x10de\n" -> 0x10de. Returns 0 on failure.
unsigned parse_hex_id(const std::string& s) {
    if (s.empty()) return 0;
    try {
        return static_cast<unsigned>(std::stoul(s, nullptr, 16));
    } catch (...) {
        return 0;
    }
}

// Read a class code like "0x030000" — the high byte is the class.
// 0x03 = display controller (covers VGA, 3D, other display).
bool is_display_class(const std::string& class_str) {
    const unsigned cls = parse_hex_id(class_str);
    return ((cls >> 16) & 0xFF) == 0x03;
}

std::vector<GpuInfo> detect_gpus_linux() {
    std::vector<GpuInfo> out;
    const char* pci_path = "/sys/bus/pci/devices";
    DIR* dir = opendir(pci_path);
    if (!dir) return out;

    while (struct dirent* ent = readdir(dir)) {
        if (ent->d_name[0] == '.') continue;
        const std::string base = std::string(pci_path) + "/" + ent->d_name;

        // Filter: must be a display controller
        const std::string cls = trim(read_first_line(base + "/class"));
        if (!is_display_class(cls)) continue;

        // Read vendor & device IDs
        const unsigned vendor = parse_hex_id(trim(read_first_line(base + "/vendor")));
        const unsigned device = parse_hex_id(trim(read_first_line(base + "/device")));
        if (!vendor) continue;

        GpuInfo info;
        info.vendor = classify_vendor(vendor);
        char idbuf[16];
        std::snprintf(idbuf, sizeof(idbuf), "%04x:%04x", vendor, device);
        info.pci_id = idbuf;
        // We don't bundle a pci.ids database. Friendly name stays empty.
        out.push_back(std::move(info));
    }
    closedir(dir);
    return out;
}

#endif  // ABOBA_OS_LINUX

// --- BSD GPU detection via popen("pciconf") -----------------------------
#if defined(ABOBA_OS_FREEBSD) || defined(ABOBA_OS_DRAGONFLY)

std::vector<GpuInfo> detect_gpus_freebsd() {
    std::vector<GpuInfo> out;
    // pciconf -l prints lines like:
    //   vgapci0@pci0:1:0:0:    class=0x030000 ... chip=0x22041002 ...
    // chip = (device << 16) | vendor.
    FILE* p = popen("pciconf -l 2>/dev/null", "r");
    if (!p) return out;
    char line[1024];
    while (std::fgets(line, sizeof(line), p)) {
        // Must be a display class to count.
        if (!std::strstr(line, "class=0x03")) continue;
        const char* chip = std::strstr(line, "chip=0x");
        if (!chip) continue;
        chip += 7;  // skip "chip=0x"
        unsigned long u = std::strtoul(chip, nullptr, 16);
        const unsigned vendor = static_cast<unsigned>(u & 0xFFFFu);
        const unsigned device = static_cast<unsigned>((u >> 16) & 0xFFFFu);
        if (!vendor) continue;

        GpuInfo info;
        info.vendor = classify_vendor(vendor);
        char idbuf[16];
        std::snprintf(idbuf, sizeof(idbuf), "%04x:%04x", vendor, device);
        info.pci_id = idbuf;
        out.push_back(std::move(info));
    }
    pclose(p);
    return out;
}

#endif  // FreeBSD / DragonFly

#if defined(ABOBA_OS_OPENBSD) || defined(ABOBA_OS_NETBSD)

std::vector<GpuInfo> detect_gpus_openbsd() {
    // OpenBSD's pcidump requires root. Try lspci-style detection via
    // dmesg.boot which is world-readable.
    // dmesg lines like:  "vga1 at pci1 dev 0 function 0 "AMD ..." rev 0x00"
    // This is fragile but better than nothing.
    std::vector<GpuInfo> out;
    FILE* p = std::fopen("/var/run/dmesg.boot", "r");
    if (!p) return out;
    char line[1024];
    while (std::fgets(line, sizeof(line), p)) {
        const char* q = std::strstr(line, "vga");
        if (!q) continue;
        GpuInfo info;
        if      (std::strstr(line, "NVIDIA") || std::strstr(line, "nVidia"))
            info.vendor = GpuVendor::NVIDIA;
        else if (std::strstr(line, "AMD")    || std::strstr(line, "ATI")   ||
                 std::strstr(line, "Radeon"))
            info.vendor = GpuVendor::AMD;
        else if (std::strstr(line, "Intel"))
            info.vendor = GpuVendor::Intel;
        else
            info.vendor = GpuVendor::Other;
        out.push_back(std::move(info));
    }
    std::fclose(p);
    return out;
}

#endif  // OpenBSD / NetBSD

// --- macOS GPU detection via system_profiler ----------------------------
#ifdef ABOBA_OS_MACOS

std::vector<GpuInfo> detect_gpus_macos() {
    std::vector<GpuInfo> out;
    // system_profiler is reliable on macOS but slow (~0.5–2s). Limit to a
    // single call at startup.
    FILE* p = popen("system_profiler SPDisplaysDataType 2>/dev/null", "r");
    if (!p) return out;
    char line[1024];
    while (std::fgets(line, sizeof(line), p)) {
        const char* m = std::strstr(line, "Chipset Model:");
        if (!m) continue;
        m += std::strlen("Chipset Model:");
        std::string name = trim(m);
        GpuInfo info;
        info.name = name;
        // Keyword classification
        if      (name.find("NVIDIA") != std::string::npos ||
                 name.find("GeForce")!= std::string::npos)
            info.vendor = GpuVendor::NVIDIA;
        else if (name.find("AMD")    != std::string::npos ||
                 name.find("Radeon") != std::string::npos)
            info.vendor = GpuVendor::AMD;
        else if (name.find("Intel")  != std::string::npos)
            info.vendor = GpuVendor::Intel;
        else if (name.find("Apple")  != std::string::npos)
            // Apple Silicon GPU. Treat as Other (we won't run HIP on it anyway).
            info.vendor = GpuVendor::Other;
        else
            info.vendor = GpuVendor::Other;
        out.push_back(std::move(info));
    }
    pclose(p);
    return out;
}

#endif  // macOS

// --- Windows GPU detection via module presence --------------------------
#ifdef ABOBA_OS_WINDOWS

// We don't link CUDA, ever. But we can ask the OS if NVIDIA's user-mode
// driver is loadable. GetModuleHandle doesn't load — it only checks if a
// module is already mapped. So we use LoadLibraryEx with LOAD_LIBRARY_AS_
// DATAFILE which doesn't run DllMain or initialize anything, then free.
bool dll_present(const char* name) {
    HMODULE h = LoadLibraryExA(name, nullptr, LOAD_LIBRARY_AS_DATAFILE);
    if (!h) return false;
    FreeLibrary(h);
    return true;
}

std::vector<GpuInfo> detect_gpus_windows() {
    std::vector<GpuInfo> out;
    if (dll_present("nvcuda.dll")) {
        GpuInfo i; i.vendor = GpuVendor::NVIDIA; i.name = "(NVIDIA driver present)";
        out.push_back(i);
    }
    if (dll_present("amdhip64.dll") || dll_present("amd_ags_x64.dll")) {
        GpuInfo i; i.vendor = GpuVendor::AMD; i.name = "(AMD driver present)";
        out.push_back(i);
    }
    // Intel detection on Windows is messy without WMI; skip.
    return out;
}

#endif  // Windows

}  // namespace

// --- PlatformInfo methods ------------------------------------------------

bool PlatformInfo::is_unix_like() const {
    switch (os) {
        case OS::Linux: case OS::FreeBSD: case OS::OpenBSD:
        case OS::NetBSD: case OS::DragonFlyBSD: case OS::MacOS:
        case OS::OtherUnix:
            return true;
        default:
            return false;
    }
}

bool PlatformInfo::is_bsd() const {
    return os == OS::FreeBSD || os == OS::OpenBSD ||
           os == OS::NetBSD  || os == OS::DragonFlyBSD;
}

bool PlatformInfo::has_nvidia() const {
    for (const auto& g : gpus) if (g.vendor == GpuVendor::NVIDIA) return true;
    return false;
}

bool PlatformInfo::has_amd() const {
    for (const auto& g : gpus) if (g.vendor == GpuVendor::AMD) return true;
    return false;
}

bool PlatformInfo::has_intel() const {
    for (const auto& g : gpus) if (g.vendor == GpuVendor::Intel) return true;
    return false;
}

// --- detect_platform() ---------------------------------------------------

PlatformInfo detect_platform() {
    PlatformInfo info;
    info.os      = compile_time_os();
    info.os_name = os_pretty_name(info.os);
    info.arch    = detect_arch();

#if defined(ABOBA_OS_LINUX)
    info.gpus = detect_gpus_linux();
#elif defined(ABOBA_OS_FREEBSD) || defined(ABOBA_OS_DRAGONFLY)
    info.gpus = detect_gpus_freebsd();
#elif defined(ABOBA_OS_OPENBSD) || defined(ABOBA_OS_NETBSD)
    info.gpus = detect_gpus_openbsd();
#elif defined(ABOBA_OS_MACOS)
    info.gpus = detect_gpus_macos();
#elif defined(ABOBA_OS_WINDOWS)
    info.gpus = detect_gpus_windows();
#endif

    return info;
}

// --- print_platform_banner() --------------------------------------------

namespace {

bool quiet_mode() {
    const char* q = std::getenv("ABOBA_QUIET");
    return q && q[0] != '\0' && q[0] != '0';
}

const char* vendor_label(GpuVendor v) {
    switch (v) {
        case GpuVendor::AMD:    return "AMD";
        case GpuVendor::NVIDIA: return "NVIDIA";
        case GpuVendor::Intel:  return "Intel";
        case GpuVendor::Other:  return "Other";
    }
    return "?";
}

const char* vendor_note(GpuVendor v) {
    switch (v) {
        case GpuVendor::AMD:    return "BLESSED \u2713";
        case GpuVendor::NVIDIA: return "politely ignored, see manifesto";
        case GpuVendor::Intel:  return "neutral observer";
        case GpuVendor::Other:  return "exotic guest";
    }
    return "";
}

// Pick a contextual closing line based on what we found.
const char* aboba_blessing(const PlatformInfo& info) {
    if (info.is_bsd()) {
        return "BSD detected. You have our automatic respect.";
    }
    if (info.os == OS::MacOS) {
        // macOS users are off the trolling chain because they can't get either
        // CUDA or ROCm. Just be nice.
        for (const auto& g : info.gpus) {
            if (g.name.find("Apple") != std::string::npos) {
                return "Apple Silicon. The CPU path is the only path.";
            }
        }
        return "macOS user. We trust your aesthetics.";
    }
    if (info.has_amd() && !info.has_nvidia()) {
        return "Pure AMD system detected. We see you, comrade.";
    }
    if (info.has_amd() && info.has_nvidia()) {
        return "Mixed system. Aboba will use AMD. NVIDIA stays on the bench.";
    }
    if (info.has_nvidia() && !info.has_amd()) {
        return "Your GPU has been excused from this session. CPU it is.";
    }
    if (info.has_intel() && !info.has_amd() && !info.has_nvidia()) {
        return "Integrated Intel graphics. The everyman's path.";
    }
    if (info.gpus.empty()) {
        return "No discrete GPU detected. The way nature intended.";
    }
    return "Welcome.";
}

}  // namespace

void print_platform_banner(const PlatformInfo& info, const char* backend_name) {
    if (quiet_mode()) return;

    std::fprintf(stderr, "\n");
    std::fprintf(stderr, "  \xF0\x9F\x94\xB4 Aboba Vocoder\n");
    std::fprintf(stderr, "    Platform : %s %s%s\n",
                 info.os_name.c_str(),
                 info.arch.c_str(),
                 info.is_bsd() ? " (BSD family \u2014 respect)" : "");

    if (info.gpus.empty()) {
        std::fprintf(stderr, "    GPUs     : none detected\n");
    } else {
        for (std::size_t i = 0; i < info.gpus.size(); ++i) {
            const auto& g = info.gpus[i];
            std::fprintf(stderr, "    GPU [%zu]  : %s%s%s  -- %s\n",
                         i,
                         vendor_label(g.vendor),
                         g.pci_id.empty() ? "" : "  ",
                         g.pci_id.c_str(),
                         vendor_note(g.vendor));
        }
    }
    std::fprintf(stderr, "    Backend  : %s\n", backend_name ? backend_name : "?");
    std::fprintf(stderr, "    > %s\n\n", aboba_blessing(info));
}

}  // namespace aboba
