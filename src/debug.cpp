// SPDX-License-Identifier: GPL-3.0-or-later
#include "aboba/debug.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <ostream>
#include <stdexcept>

namespace aboba {

namespace {

// ---- helpers for atomic-float compare-exchange tricks --------------------

inline std::uint32_t float_bits(float x) noexcept {
    std::uint32_t u;
    std::memcpy(&u, &x, sizeof(u));
    return u;
}
inline float bits_float(std::uint32_t u) noexcept {
    float x;
    std::memcpy(&x, &u, sizeof(x));
    return x;
}
inline std::uint64_t double_bits(double x) noexcept {
    std::uint64_t u;
    std::memcpy(&u, &x, sizeof(u));
    return u;
}
inline double bits_double(std::uint64_t u) noexcept {
    double x;
    std::memcpy(&x, &u, sizeof(x));
    return x;
}

// Atomic max/min for float via CAS loop.
void atomic_min_f(std::atomic<std::uint32_t>& slot, float v) noexcept {
    if (!std::isfinite(v)) return;
    std::uint32_t cur = slot.load(std::memory_order_relaxed);
    while (v < bits_float(cur)) {
        const std::uint32_t want = float_bits(v);
        if (slot.compare_exchange_weak(cur, want,
                std::memory_order_relaxed,
                std::memory_order_relaxed)) return;
    }
}
void atomic_max_f(std::atomic<std::uint32_t>& slot, float v) noexcept {
    if (!std::isfinite(v)) return;
    std::uint32_t cur = slot.load(std::memory_order_relaxed);
    while (v > bits_float(cur)) {
        const std::uint32_t want = float_bits(v);
        if (slot.compare_exchange_weak(cur, want,
                std::memory_order_relaxed,
                std::memory_order_relaxed)) return;
    }
}
void atomic_add_d(std::atomic<std::uint64_t>& slot, double v) noexcept {
    if (!std::isfinite(v)) return;
    std::uint64_t cur = slot.load(std::memory_order_relaxed);
    for (;;) {
        const double newv = bits_double(cur) + v;
        const std::uint64_t want = double_bits(newv);
        if (slot.compare_exchange_weak(cur, want,
                std::memory_order_relaxed,
                std::memory_order_relaxed)) return;
    }
}
void atomic_min_u64(std::atomic<std::uint64_t>& slot, std::uint64_t v) noexcept {
    std::uint64_t cur = slot.load(std::memory_order_relaxed);
    while (v < cur) {
        if (slot.compare_exchange_weak(cur, v,
                std::memory_order_relaxed,
                std::memory_order_relaxed)) return;
    }
}
void atomic_max_u64(std::atomic<std::uint64_t>& slot, std::uint64_t v) noexcept {
    std::uint64_t cur = slot.load(std::memory_order_relaxed);
    while (v > cur) {
        if (slot.compare_exchange_weak(cur, v,
                std::memory_order_relaxed,
                std::memory_order_relaxed)) return;
    }
}

// Quarter-decade log bucket: returns 0..63.
inline std::size_t bucket_for_ns(std::uint64_t ns) noexcept {
    if (ns < 1000) return 0;  // < 1 µs floor
    // log2(ns/1000) * 4 → quarter-decade index
    const double l = std::log2(static_cast<double>(ns) / 1000.0);
    long i = static_cast<long>(l * 4.0);
    if (i < 0)  i = 0;
    if (i > 63) i = 63;
    return static_cast<std::size_t>(i);
}

inline double bucket_center_us(std::size_t i) noexcept {
    // Center of bucket i in microseconds
    return std::pow(2.0, (static_cast<double>(i) + 0.5) / 4.0);
}

}  // namespace

// ======================================================================
// PerfProbe
// ======================================================================
PerfProbe::PerfProbe(std::string name) : name_(std::move(name)) {
    reset();
}

void PerfProbe::reset() noexcept {
    for (auto& b : buckets_) b.store(0, std::memory_order_relaxed);
    count_.store(0, std::memory_order_relaxed);
    total_ns_.store(0, std::memory_order_relaxed);
    min_ns_.store(std::numeric_limits<std::uint64_t>::max(),
                  std::memory_order_relaxed);
    max_ns_.store(0, std::memory_order_relaxed);
}

void PerfProbe::record(double seconds) noexcept {
    if (!std::isfinite(seconds) || seconds < 0.0) return;
    const auto ns = static_cast<std::uint64_t>(seconds * 1e9);
    const std::size_t b = bucket_for_ns(ns);
    buckets_[b].fetch_add(1, std::memory_order_relaxed);
    count_.fetch_add(1, std::memory_order_relaxed);
    total_ns_.fetch_add(ns, std::memory_order_relaxed);
    atomic_min_u64(min_ns_, ns);
    atomic_max_u64(max_ns_, ns);
}

PerfProbe::Stats PerfProbe::stats() const noexcept {
    Stats s;
    s.count = count_.load(std::memory_order_relaxed);
    if (s.count == 0) return s;

    const std::uint64_t total_ns = total_ns_.load(std::memory_order_relaxed);
    s.total_us = static_cast<double>(total_ns) / 1000.0;
    s.mean_us  = s.total_us / static_cast<double>(s.count);

    const std::uint64_t min_ns = min_ns_.load(std::memory_order_relaxed);
    s.min_us = (min_ns == std::numeric_limits<std::uint64_t>::max())
        ? 0.0 : (static_cast<double>(min_ns) / 1000.0);
    s.max_us = static_cast<double>(max_ns_.load(std::memory_order_relaxed)) / 1000.0;

    // Percentiles: walk buckets from low to high, accumulate counts.
    std::uint64_t hist[kBuckets];
    std::uint64_t total = 0;
    for (std::size_t i = 0; i < kBuckets; ++i) {
        hist[i] = buckets_[i].load(std::memory_order_relaxed);
        total  += hist[i];
    }
    if (total == 0) return s;

    auto pct = [&](double p) -> double {
        const std::uint64_t target = static_cast<std::uint64_t>(
            std::ceil(p * static_cast<double>(total)));
        std::uint64_t cum = 0;
        for (std::size_t i = 0; i < kBuckets; ++i) {
            cum += hist[i];
            if (cum >= target) return bucket_center_us(i);
        }
        return bucket_center_us(kBuckets - 1);
    };
    s.p50_us = pct(0.50);
    s.p95_us = pct(0.95);
    s.p99_us = pct(0.99);
    return s;
}

void PerfProbe::print(std::ostream& os) const {
    auto s = stats();
    os << "[perf] " << name_ << ": ";
    if (s.count == 0) { os << "(no samples)\n"; return; }
    char buf[256];
    std::snprintf(buf, sizeof(buf),
        "n=%llu  min=%.1f  mean=%.1f  p50=%.1f  p95=%.1f  p99=%.1f  max=%.1f  total=%.1f ms (us unless noted)\n",
        static_cast<unsigned long long>(s.count),
        s.min_us, s.mean_us, s.p50_us, s.p95_us, s.p99_us, s.max_us,
        s.total_us / 1000.0);
    os << buf;
}

// ======================================================================
// SignalInspector
// ======================================================================
SignalInspector::SignalInspector() {
    reset();
}

void SignalInspector::reset() noexcept {
    samples_.store(0, std::memory_order_relaxed);
    nan_count_.store(0, std::memory_order_relaxed);
    inf_count_.store(0, std::memory_order_relaxed);
    clip_count_.store(0, std::memory_order_relaxed);
    silence_blocks_.store(0, std::memory_order_relaxed);
    blocks_total_.store(0, std::memory_order_relaxed);
    min_bits_.store(float_bits(std::numeric_limits<float>::infinity()),
                    std::memory_order_relaxed);
    max_bits_.store(float_bits(-std::numeric_limits<float>::infinity()),
                    std::memory_order_relaxed);
    peak_bits_.store(float_bits(0.0f), std::memory_order_relaxed);
    sum_abs_bits_.store(double_bits(0.0), std::memory_order_relaxed);
    sum_sq_bits_.store (double_bits(0.0), std::memory_order_relaxed);
}

void SignalInspector::inspect(const float* x, std::size_t n) noexcept {
    if (!x || n == 0) return;

    std::uint64_t nans = 0, infs = 0, clips = 0;
    float local_min = std::numeric_limits<float>::infinity();
    float local_max = -std::numeric_limits<float>::infinity();
    float local_peak = 0.0f;
    double local_sum_abs = 0.0, local_sum_sq = 0.0;
    bool   block_silent = true;

    for (std::size_t i = 0; i < n; ++i) {
        const float v = x[i];
        if (std::isnan(v)) { ++nans; continue; }
        if (std::isinf(v)) { ++infs; continue; }
        const float a = std::fabs(v);
        if (a > clip_threshold_) ++clips;
        if (v < local_min) local_min = v;
        if (v > local_max) local_max = v;
        if (a > local_peak) local_peak = a;
        if (a > silence_threshold_) block_silent = false;
        local_sum_abs += static_cast<double>(a);
        local_sum_sq  += static_cast<double>(v) * static_cast<double>(v);
    }

    samples_.fetch_add(n, std::memory_order_relaxed);
    blocks_total_.fetch_add(1, std::memory_order_relaxed);
    if (nans)  nan_count_.fetch_add(nans,  std::memory_order_relaxed);
    if (infs)  inf_count_.fetch_add(infs,  std::memory_order_relaxed);
    if (clips) clip_count_.fetch_add(clips,std::memory_order_relaxed);
    if (block_silent) silence_blocks_.fetch_add(1, std::memory_order_relaxed);

    atomic_min_f(min_bits_,  local_min);
    atomic_max_f(max_bits_,  local_max);
    atomic_max_f(peak_bits_, local_peak);
    atomic_add_d(sum_abs_bits_, local_sum_abs);
    atomic_add_d(sum_sq_bits_,  local_sum_sq);
}

SignalInspector::Stats SignalInspector::stats() const noexcept {
    Stats s;
    s.samples_total  = samples_.load(std::memory_order_relaxed);
    s.nan_count      = nan_count_.load(std::memory_order_relaxed);
    s.inf_count      = inf_count_.load(std::memory_order_relaxed);
    s.clip_count     = clip_count_.load(std::memory_order_relaxed);
    s.silence_blocks = silence_blocks_.load(std::memory_order_relaxed);
    s.blocks_total   = blocks_total_.load(std::memory_order_relaxed);
    s.min       = bits_float(min_bits_.load(std::memory_order_relaxed));
    s.max       = bits_float(max_bits_.load(std::memory_order_relaxed));
    s.peak_abs  = bits_float(peak_bits_.load(std::memory_order_relaxed));
    s.sum_abs   = bits_double(sum_abs_bits_.load(std::memory_order_relaxed));
    s.sum_sq    = bits_double(sum_sq_bits_.load(std::memory_order_relaxed));
    return s;
}

void SignalInspector::print(std::ostream& os) const {
    auto s = stats();
    char buf[256];
    if (s.samples_total == 0) {
        os << "[signal] (no samples)\n";
        return;
    }
    const double mean_abs = s.sum_abs / static_cast<double>(s.samples_total);
    const double rms      = std::sqrt(s.sum_sq /
                                      static_cast<double>(s.samples_total));
    std::snprintf(buf, sizeof(buf),
        "[signal] samples=%llu blocks=%llu  min=%.4f max=%.4f peak=%.4f  "
        "mean|x|=%.4f rms=%.4f  nan=%llu inf=%llu clip=%llu silent=%llu/%llu\n",
        static_cast<unsigned long long>(s.samples_total),
        static_cast<unsigned long long>(s.blocks_total),
        static_cast<double>(s.min), static_cast<double>(s.max),
        static_cast<double>(s.peak_abs),
        mean_abs, rms,
        static_cast<unsigned long long>(s.nan_count),
        static_cast<unsigned long long>(s.inf_count),
        static_cast<unsigned long long>(s.clip_count),
        static_cast<unsigned long long>(s.silence_blocks),
        static_cast<unsigned long long>(s.blocks_total));
    os << buf;
}

// ======================================================================
// FrameDumper
// ======================================================================
FrameDumper::FrameDumper(const std::string& path, double sample_rate, int channels)
    : fp_(std::fopen(path.c_str(), "wb"))
    , sample_rate_(sample_rate)
    , channels_(channels)
    , samples_written_(0) {
    if (!fp_) return;
    if (sample_rate <= 0.0 || channels < 1 || channels > 32) {
        std::fclose(fp_);
        fp_ = nullptr;
        return;
    }
    write_header_placeholder();
}

FrameDumper::~FrameDumper() {
    close();
}

void FrameDumper::close() {
    if (!fp_) return;
    patch_header_finalize();
    std::fclose(fp_);
    fp_ = nullptr;
}

void FrameDumper::write_header_placeholder() {
    // We emit a WAV header with WAVE_FORMAT_IEEE_FLOAT (0x0003) and 32-bit
    // float samples. ChunkSize and Subchunk2Size are placeholders that we
    // overwrite in patch_header_finalize().
    const std::uint16_t format_code     = 0x0003;
    const std::uint16_t channels        = static_cast<std::uint16_t>(channels_);
    const std::uint32_t sr              = static_cast<std::uint32_t>(sample_rate_);
    const std::uint16_t bits_per_sample = 32;
    const std::uint16_t block_align     = static_cast<std::uint16_t>(channels_ * 4);
    const std::uint32_t byte_rate       = static_cast<std::uint32_t>(sr) * block_align;

    auto w32 = [&](std::uint32_t v) { std::fwrite(&v, 4, 1, fp_); };
    auto w16 = [&](std::uint16_t v) { std::fwrite(&v, 2, 1, fp_); };
    auto wstr = [&](const char* s) { std::fwrite(s, 1, 4, fp_); };

    wstr("RIFF");
    w32(36);                     // placeholder ChunkSize
    wstr("WAVE");
    wstr("fmt ");
    w32(16);                     // PCM/IEEE_FLOAT fmt chunk size
    w16(format_code);
    w16(channels);
    w32(sr);
    w32(byte_rate);
    w16(block_align);
    w16(bits_per_sample);
    wstr("data");
    w32(0);                      // placeholder data size
}

void FrameDumper::patch_header_finalize() {
    if (!fp_) return;
    const std::uint64_t data_bytes = samples_written_ * 4u;
    // Don't blow the 32-bit WAV size limit silently.
    const std::uint64_t total_bytes = 36 + data_bytes;
    const std::uint32_t chunk_size  =
        (total_bytes > 0xFFFFFFFEull)
            ? 0xFFFFFFFEu
            : static_cast<std::uint32_t>(total_bytes);
    const std::uint32_t data_size   =
        (data_bytes  > 0xFFFFFFFEull)
            ? 0xFFFFFFFEu
            : static_cast<std::uint32_t>(data_bytes);

    std::fflush(fp_);
    std::fseek(fp_, 4, SEEK_SET);
    std::fwrite(&chunk_size, 4, 1, fp_);
    std::fseek(fp_, 40, SEEK_SET);
    std::fwrite(&data_size, 4, 1, fp_);
}

void FrameDumper::write(const float* x, std::size_t n) {
    if (!fp_ || !x || n == 0) return;
    // Sanitize NaN/Inf to 0 so we don't poison anyone's listening tools.
    constexpr std::size_t kChunk = 1024;
    float buf[kChunk];
    while (n > 0) {
        const std::size_t k = std::min(n, kChunk);
        for (std::size_t i = 0; i < k; ++i) {
            buf[i] = std::isfinite(x[i]) ? x[i] : 0.0f;
        }
        const std::size_t wrote = std::fwrite(buf, sizeof(float), k, fp_);
        if (wrote != k) {
            // Disk full or similar — bail; subsequent writes are no-ops.
            std::fclose(fp_);
            fp_ = nullptr;
            return;
        }
        samples_written_ += wrote;
        x += k;
        n -= k;
    }
}

// ======================================================================
// AsciiMeter
// ======================================================================
AsciiMeter::AsciiMeter(std::string label, int width)
    : label_(std::move(label))
    , width_(width < 4 ? 4 : (width > 200 ? 200 : width)) {
    reset();
}

void AsciiMeter::reset() noexcept {
    peak_bits_.store(float_bits(0.0f), std::memory_order_relaxed);
    rms_bits_ .store(float_bits(0.0f), std::memory_order_relaxed);
}

void AsciiMeter::update(const float* x, std::size_t n) noexcept {
    if (!x || n == 0) return;
    float peak = 0.0f;
    double sq = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        const float v = x[i];
        if (!std::isfinite(v)) continue;
        const float a = std::fabs(v);
        if (a > peak) peak = a;
        sq += static_cast<double>(v) * static_cast<double>(v);
    }
    const float rms = static_cast<float>(std::sqrt(sq / static_cast<double>(n)));
    // Decaying peak: 90% old, 10% new for visual smoothing.
    const float old_peak = bits_float(peak_bits_.load(std::memory_order_relaxed));
    const float new_peak = std::max(0.9f * old_peak, peak);
    peak_bits_.store(float_bits(new_peak), std::memory_order_relaxed);
    rms_bits_.store(float_bits(rms),       std::memory_order_relaxed);
}

void AsciiMeter::render(std::ostream& os, bool inline_update) const {
    const float peak = bits_float(peak_bits_.load(std::memory_order_relaxed));
    const float rms  = bits_float(rms_bits_ .load(std::memory_order_relaxed));

    // Map to dBFS, clamp to [-60, 0]
    auto to_db_pos = [](float v) -> float {
        if (v <= 1e-6f) return 0.0f;
        const float db = 20.0f * std::log10(v);
        // db < 0 typically; normalize to 0..1 over -60..0 dBFS
        const float t = (db + 60.0f) / 60.0f;
        if (t < 0.0f) return 0.0f;
        if (t > 1.0f) return 1.0f;
        return t;
    };
    const float t_rms  = to_db_pos(rms);
    const float t_peak = to_db_pos(peak);
    const int n_rms  = static_cast<int>(t_rms  * static_cast<float>(width_));
    const int n_peak = static_cast<int>(t_peak * static_cast<float>(width_));

    std::string bar(static_cast<std::size_t>(width_), ' ');
    for (int i = 0; i < n_rms && i < width_; ++i) bar[static_cast<std::size_t>(i)] = '#';
    if (n_peak < width_ && n_peak >= 0)
        bar[static_cast<std::size_t>(n_peak)] = '|';

    auto db_str = [](float v) {
        char buf[16];
        if (v <= 1e-6f) std::snprintf(buf, sizeof(buf), " -inf dB");
        else            std::snprintf(buf, sizeof(buf), "%+5.1f dB",
                                      20.0 * std::log10(static_cast<double>(v)));
        return std::string(buf);
    };

    if (inline_update) os << "\r";
    if (!label_.empty()) os << "[" << label_ << "] ";
    os << "|" << bar << "| rms=" << db_str(rms) << "  peak=" << db_str(peak);
    if (!inline_update) os << "\n";
    os.flush();
}

}  // namespace aboba
