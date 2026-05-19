// SPDX-License-Identifier: GPL-3.0-or-later
//
// Debug tools for framework users.
//
// All tools here are OPT-IN. None of them are instantiated by default
// anywhere in the Aboba pipeline; you must construct and wire them in
// yourself. They are designed to answer common questions when something
// goes wrong:
//
//   "Is my audio reaching this stage?"
//       -> FrameDumper. Writes input/output to a WAV you can inspect.
//
//   "Why is processing slow?"
//       -> PerfProbe. Wrap a section, get min/max/mean/p50/p95/p99 timings.
//
//   "Are there NaNs / clipping / silence I should know about?"
//       -> SignalInspector. Counts NaN, Inf, clip, silence events.
//
//   "What's actually coming out, in real time?"
//       -> AsciiMeter. Renders a peak/RMS meter on stderr.
//
// Realtime safety:
//   * PerfProbe::record() is lock-free, allocation-free, wait-free.
//   * SignalInspector::inspect() is wait-free and allocation-free.
//   * FrameDumper writes to disk and is NOT realtime-safe — use for offline
//     processing or wrap it in your own ringbuffer + drain thread.
//   * AsciiMeter writes to stderr — generally avoid from audio thread.
//
// All tools sanitize inputs (NaN/Inf become 0) so they never propagate
// poisoned values into your data.
#pragma once

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <ostream>
#include <string>

namespace aboba {

// ======================================================================
// PerfProbe — lock-free, allocation-free timer with log-quartile histogram.
//
// Designed to be safe from inside the realtime audio callback. The
// histogram has 64 fixed buckets covering ~1 µs to ~65 ms in quarter-
// decade (2^0.25) steps. Each `record()` does one log2 + one atomic
// increment.
//
// Typical usage:
//
//   PerfProbe vocoder_probe("vocoder");
//
//   // in your audio thread:
//   {
//       PerfProbe::Scope s(vocoder_probe);
//       vocoder.process(in, out, n);
//   }   // duration recorded here
//
//   // from a control / UI thread, any time:
//   auto stats = vocoder_probe.stats();
//   std::printf("mean=%.2f µs, p99=%.2f µs\n", stats.mean_us, stats.p99_us);
// ======================================================================
class PerfProbe {
public:
    static constexpr std::size_t kBuckets = 64;

    struct Stats {
        std::uint64_t count = 0;
        double min_us  = 0.0;
        double max_us  = 0.0;
        double mean_us = 0.0;
        double p50_us  = 0.0;
        double p95_us  = 0.0;
        double p99_us  = 0.0;
        double total_us = 0.0;
    };

    explicit PerfProbe(std::string name);

    // Record a duration in seconds. Called from any thread; lock-free.
    void record(double seconds) noexcept;

    // Reset all counters. Cheap but not atomic against concurrent records.
    void reset() noexcept;

    // Snapshot counters (may be slightly inconsistent if records are in
    // flight, but always self-consistent and never UB).
    Stats stats() const noexcept;

    const std::string& name() const noexcept { return name_; }

    // Print a one-line or multi-line report.
    void print(std::ostream& os) const;

    // ------------------------------------------------------------------
    // RAII timer. Construct at the start of a section, destructor records.
    // Constructor only stores `clock::now()` — no allocations, no atomics.
    class Scope {
    public:
        explicit Scope(PerfProbe& p) noexcept
            : probe_(&p), start_(std::chrono::steady_clock::now()) {}
        Scope(const Scope&) = delete;
        Scope& operator=(const Scope&) = delete;
        ~Scope() noexcept {
            const auto end = std::chrono::steady_clock::now();
            const auto dt  = std::chrono::duration<double>(end - start_).count();
            probe_->record(dt);
        }
    private:
        PerfProbe* probe_;
        std::chrono::steady_clock::time_point start_;
    };

private:
    std::string name_;

    // Bucket layout: bucket i covers [base * 2^(i/4), base * 2^((i+1)/4))
    // with base = 1 µs. So bucket 0 is [1 µs, ~1.19 µs), bucket 4 is
    // [2 µs, ~2.38 µs), bucket 8 is [4 µs, ~4.76 µs), ..., bucket 63 is
    // [~65 ms, ~77 ms). Anything above goes into the top bucket.
    std::atomic<std::uint64_t> buckets_[kBuckets];
    std::atomic<std::uint64_t> count_;
    // Running min/max stored as bits of double via atomic uint64.
    std::atomic<std::uint64_t> total_ns_;
    std::atomic<std::uint64_t> min_ns_;
    std::atomic<std::uint64_t> max_ns_;
};


// ======================================================================
// SignalInspector — counts pathologies in a signal stream.
//
// Lock-free, allocation-free, safe from any thread including audio. Counts
// are 64-bit so they don't roll over within decades of use.
//
// Usage:
//
//   SignalInspector si;
//   // In your audio callback:
//   si.inspect(out, n);
//   // Later:
//   auto s = si.stats();
//   if (s.nan_count > 0) std::cerr << "NaN found!\n";
//
// Silence detection: a "silence block" is one where every sample of an
// inspect() call has |x| < silence_threshold. Tune via set_silence_threshold.
// ======================================================================
class SignalInspector {
public:
    struct Stats {
        std::uint64_t samples_total   = 0;
        std::uint64_t nan_count       = 0;
        std::uint64_t inf_count       = 0;
        std::uint64_t clip_count      = 0;   // |x| > 1.0
        std::uint64_t silence_blocks  = 0;
        std::uint64_t blocks_total    = 0;
        float min       = 0.0f;
        float max       = 0.0f;
        float peak_abs  = 0.0f;
        double sum_abs  = 0.0;  // -> mean_abs = sum_abs / samples_total
        double sum_sq   = 0.0;  // -> rms = sqrt(sum_sq / samples_total)
    };

    SignalInspector();

    void set_silence_threshold(float t) noexcept { silence_threshold_ = t; }
    void set_clip_threshold   (float t) noexcept { clip_threshold_    = t; }

    // Inspect a block of samples. Wait-free.
    void inspect(const float* x, std::size_t n) noexcept;

    Stats stats() const noexcept;
    void  reset() noexcept;
    void  print(std::ostream& os) const;

private:
    float silence_threshold_ = 1e-5f;
    float clip_threshold_    = 0.999f;

    std::atomic<std::uint64_t> samples_;
    std::atomic<std::uint64_t> nan_count_;
    std::atomic<std::uint64_t> inf_count_;
    std::atomic<std::uint64_t> clip_count_;
    std::atomic<std::uint64_t> silence_blocks_;
    std::atomic<std::uint64_t> blocks_total_;
    // min/max/peak/sum stored as atomic bits for lock-free updates
    std::atomic<std::uint32_t> min_bits_;
    std::atomic<std::uint32_t> max_bits_;
    std::atomic<std::uint32_t> peak_bits_;
    std::atomic<std::uint64_t> sum_abs_bits_;
    std::atomic<std::uint64_t> sum_sq_bits_;
};


// ======================================================================
// FrameDumper — write samples to a WAV file (32-bit float, mono).
//
// NOT realtime-safe. Use for offline processing or behind a drain thread.
// Constructor opens the file and writes a placeholder header; destructor
// goes back and patches in the final sample count.
//
// Usage:
//
//   FrameDumper d("/tmp/stage_3.wav", 48000.0);
//   d.write(buf, n);
//   // ... (auto-finalizes when d goes out of scope)
// ======================================================================
class FrameDumper {
public:
    FrameDumper(const std::string& path, double sample_rate, int channels = 1);
    ~FrameDumper();
    FrameDumper(const FrameDumper&) = delete;
    FrameDumper& operator=(const FrameDumper&) = delete;

    bool good() const noexcept { return fp_ != nullptr; }

    // Write n SAMPLES (not frames). For mono these are the same. For
    // multi-channel you're responsible for interleaving.
    void write(const float* x, std::size_t n);

    void close();

private:
    void write_header_placeholder();
    void patch_header_finalize();

    std::FILE* fp_;
    double     sample_rate_;
    int        channels_;
    std::uint64_t samples_written_;
};


// ======================================================================
// AsciiMeter — renders a peak/RMS bar to a stream.
//
// Useful for "is anything happening at all" sanity checks. Renders a fixed-
// width bar like:
//
//   [vocoder] |##############       . .|  rms=-12 dB  peak=-3 dB
//
// Call render() at a low rate (say 30 Hz) to avoid drowning the terminal.
// NOT realtime-safe (writes to stderr).
// ======================================================================
class AsciiMeter {
public:
    explicit AsciiMeter(std::string label = "", int width = 30);

    // Update internal envelope. Call from audio thread is OK (no I/O here).
    void update(const float* x, std::size_t n) noexcept;

    // Render one line. The carriage-return-terminated form (default) lets
    // you call this repeatedly on the same terminal line.
    void render(std::ostream& os, bool inline_update = true) const;

    void reset() noexcept;

private:
    std::string label_;
    int width_;
    // Use atomics so update/render can be called from different threads.
    std::atomic<std::uint32_t> peak_bits_;
    std::atomic<std::uint32_t> rms_bits_;
};

}  // namespace aboba
