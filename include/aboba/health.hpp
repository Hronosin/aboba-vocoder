// SPDX-License-Identifier: GPL-3.0-or-later
//
// HealthMonitor — runtime watchdog for audio processors.
//
// The threat model: an audio callback that takes longer than its budget
// causes underruns. Repeated underruns cause glitches, crackles, or in the
// worst case a full audio-stack hang (PulseAudio/PipeWire/CoreAudio start
// disconnecting clients that miss too many deadlines).
//
// HealthMonitor wraps a user processor with three layers of protection:
//
//   1. PER-CALL TIMING. Every invocation is timed. We track a moving max
//      and a count of "over-budget" calls.
//
//   2. AUTOMATIC BYPASS. If the over-budget rate exceeds a threshold within
//      a sliding window, we transparently switch the processor off — input
//      is passed straight to output. The user can re-enable manually or
//      let it re-engage automatically after a cool-down.
//
//   3. STATS. All counters are atomic and lock-free. You can read them
//      from any thread without disturbing the audio thread.
//
// HealthMonitor is itself lock-free in the audio callback path. The only
// synchronization is atomic loads/stores.
#pragma once

#include "streaming.hpp"

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>

namespace aboba {

struct HealthStats {
    std::uint64_t calls            = 0;  // total callback invocations
    std::uint64_t over_budget      = 0;  // calls exceeding budget
    std::uint64_t bypassed         = 0;  // calls served via bypass
    std::uint64_t exceptions       = 0;  // callbacks that threw
    std::uint64_t auto_bypass_trips = 0; // times we auto-tripped to bypass

    // Moving stats over the recent window (microseconds)
    std::uint32_t last_us      = 0;
    std::uint32_t max_us       = 0;  // since last reset
    std::uint32_t budget_us    = 0;  // configured budget

    bool bypass_active         = false;
};

struct HealthConfig {
    // Budget per process() call in microseconds. Typically:
    //   block_size / sample_rate * 1e6 * safety_factor
    // with safety_factor ~0.5 (use half the wall-clock budget).
    std::uint32_t budget_us = 5000;

    // Sliding window length over which we measure over-budget rate.
    std::uint32_t window_calls = 100;

    // If more than `trip_count` calls in the last `window_calls` go over
    // budget, auto-bypass kicks in.
    std::uint32_t trip_count = 10;

    // After this many consecutive on-budget calls in bypass mode, we
    // re-engage the processor. Set to UINT32_MAX to require manual recover().
    std::uint32_t recover_calls = 200;

    // Hard ceiling: if a single call exceeds this (microseconds), we trip
    // bypass immediately regardless of window stats. This catches "totally
    // stuck" pathologies that the moving-average would miss for too long.
    std::uint32_t hard_ceiling_us = 50000;  // 50 ms is already catastrophic
};

class HealthMonitor {
public:
    // The processor type matches StreamingPhaseVocoder::process — but we
    // accept any callable with that signature so users can wrap anything.
    using Processor =
        std::function<void(const float* in, float* out, std::size_t n)>;

    HealthMonitor(Processor processor, HealthConfig cfg = HealthConfig{});

    // Run one callback. Times it, applies budget logic, may bypass.
    // SAFE TO CALL FROM AUDIO THREAD. No allocations, no locks.
    void process(const float* in, float* out, std::size_t n);

    // --- Stats / control. Safe from any thread. -----------------------
    HealthStats snapshot() const;
    void reset_stats();
    void force_bypass(bool on);   // manual override
    bool is_bypassed() const;

    // Update config (e.g. when block size changes)
    void set_budget_us(std::uint32_t budget_us);

private:
    void enter_bypass(const char* reason);
    void try_recover();

    Processor processor_;

    // Config is read-mostly; we use atomics so the audio thread can pick up
    // changes without locks.
    std::atomic<std::uint32_t> budget_us_;
    std::atomic<std::uint32_t> hard_ceiling_us_;
    std::uint32_t window_calls_;
    std::uint32_t trip_count_;
    std::uint32_t recover_calls_;

    // Ring buffer of recent timings (microseconds). Size = window_calls_.
    std::unique_ptr<std::uint32_t[]> window_;
    std::size_t                      window_pos_ = 0;
    std::uint32_t                    window_over_count_ = 0;
    std::uint32_t                    consecutive_clean_ = 0;

    // Bypass state — atomic so external readers can observe.
    std::atomic<bool> bypassed_{false};
    std::atomic<bool> manual_bypass_{false};

    // Stats counters (relaxed atomics: monotonic, eventually-visible)
    std::atomic<std::uint64_t> n_calls_{0};
    std::atomic<std::uint64_t> n_over_{0};
    std::atomic<std::uint64_t> n_bypassed_{0};
    std::atomic<std::uint64_t> n_exceptions_{0};
    std::atomic<std::uint64_t> n_trips_{0};
    std::atomic<std::uint32_t> last_us_{0};
    std::atomic<std::uint32_t> max_us_{0};
};

}  // namespace aboba
