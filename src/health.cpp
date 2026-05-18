// SPDX-License-Identifier: GPL-3.0-or-later
#include "aboba/health.hpp"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <stdexcept>

namespace aboba {

namespace {

inline std::uint32_t microseconds_since(
    std::chrono::steady_clock::time_point start) {
    const auto now = std::chrono::steady_clock::now();
    const auto us = std::chrono::duration_cast<std::chrono::microseconds>(
        now - start).count();
    if (us < 0) return 0;
    if (us > static_cast<long long>(UINT32_MAX)) return UINT32_MAX;
    return static_cast<std::uint32_t>(us);
}

// Atomic max update (no compare_exchange loop primitive in std)
inline void atomic_update_max(std::atomic<std::uint32_t>& target,
                              std::uint32_t value) {
    std::uint32_t prev = target.load(std::memory_order_relaxed);
    while (value > prev &&
           !target.compare_exchange_weak(prev, value,
                                         std::memory_order_relaxed)) {
        // prev was updated by compare_exchange_weak on failure
    }
}

}  // namespace

HealthMonitor::HealthMonitor(Processor processor, HealthConfig cfg)
    : processor_(std::move(processor)),
      budget_us_(cfg.budget_us),
      hard_ceiling_us_(cfg.hard_ceiling_us),
      window_calls_(cfg.window_calls),
      trip_count_(cfg.trip_count),
      recover_calls_(cfg.recover_calls) {
    if (!processor_) {
        throw std::invalid_argument("HealthMonitor: processor is null");
    }
    if (window_calls_ == 0) window_calls_ = 1;

    // Allocate the ring buffer once. Zero-initialized.
    window_.reset(new std::uint32_t[window_calls_]());
}

void HealthMonitor::process(const float* in, float* out, std::size_t n) {
    if (n == 0) return;

    n_calls_.fetch_add(1, std::memory_order_relaxed);

    const bool manual = manual_bypass_.load(std::memory_order_acquire);
    const bool auto_bypass = bypassed_.load(std::memory_order_acquire);

    if (manual || auto_bypass) {
        // Bypass path: copy input to output as fast as possible.
        if (in != out) std::memcpy(out, in, n * sizeof(float));
        n_bypassed_.fetch_add(1, std::memory_order_relaxed);

        // While bypassed, we still time a NOP-equivalent: just the memcpy.
        // This lets us detect when the system itself is healthy again.
        // We schedule possible auto-recovery only if it wasn't a manual.
        if (!manual && auto_bypass) try_recover();
        return;
    }

    // --- Active path: time the call ---------------------------------
    const auto t0 = std::chrono::steady_clock::now();

    try {
        processor_(in, out, n);
    } catch (...) {
        // Never propagate exceptions out of an audio callback.
        // Output silence and trip bypass — something is very wrong.
        std::fill(out, out + n, 0.0f);
        n_exceptions_.fetch_add(1, std::memory_order_relaxed);
        enter_bypass("processor threw exception");
        return;
    }

    const std::uint32_t us = microseconds_since(t0);
    last_us_.store(us, std::memory_order_relaxed);
    atomic_update_max(max_us_, us);

    // --- Update sliding window ---------------------------------------
    const std::uint32_t budget   = budget_us_.load(std::memory_order_relaxed);
    const std::uint32_t hard_cap = hard_ceiling_us_.load(std::memory_order_relaxed);

    // Hard-cap: a single nuclear-bad call trips bypass immediately.
    if (us >= hard_cap) {
        n_over_.fetch_add(1, std::memory_order_relaxed);
        enter_bypass("call exceeded hard ceiling");
        return;
    }

    const bool over = (us > budget);

    // Ring buffer accounting: subtract the slot we're about to overwrite,
    // add the new sample.
    const std::uint32_t prev = window_[window_pos_];
    const bool prev_over     = (prev > budget);
    if (prev_over) {
        // window_over_count_ can't go below zero by construction
        window_over_count_ -= (window_over_count_ > 0) ? 1u : 0u;
    }
    window_[window_pos_] = us;
    if (over) {
        ++window_over_count_;
        n_over_.fetch_add(1, std::memory_order_relaxed);
    }
    window_pos_ = (window_pos_ + 1) % window_calls_;

    if (window_over_count_ >= trip_count_) {
        enter_bypass("over-budget rate exceeded threshold");
    }
}

void HealthMonitor::enter_bypass(const char* /*reason*/) {
    // We don't log from the audio callback (logging may allocate / take locks).
    // The fact of bypass is observable via stats.
    if (!bypassed_.exchange(true, std::memory_order_release)) {
        n_trips_.fetch_add(1, std::memory_order_relaxed);
        consecutive_clean_ = 0;
    }
}

void HealthMonitor::try_recover() {
    // Called from process() while in auto-bypass mode. We've already done
    // a memcpy, which is essentially free; check if the system has been
    // calm for long enough that we'd like to re-engage.
    if (recover_calls_ == UINT32_MAX) return;  // manual-only recover

    if (++consecutive_clean_ >= recover_calls_) {
        // Re-engage. Clear the window so we don't immediately re-trip on
        // stale over-budget counts from before the trip.
        std::fill(window_.get(), window_.get() + window_calls_, 0u);
        window_pos_ = 0;
        window_over_count_ = 0;
        consecutive_clean_ = 0;
        bypassed_.store(false, std::memory_order_release);
    }
}

HealthStats HealthMonitor::snapshot() const {
    HealthStats s;
    s.calls             = n_calls_.load(std::memory_order_relaxed);
    s.over_budget       = n_over_.load(std::memory_order_relaxed);
    s.bypassed          = n_bypassed_.load(std::memory_order_relaxed);
    s.exceptions        = n_exceptions_.load(std::memory_order_relaxed);
    s.auto_bypass_trips = n_trips_.load(std::memory_order_relaxed);
    s.last_us           = last_us_.load(std::memory_order_relaxed);
    s.max_us            = max_us_.load(std::memory_order_relaxed);
    s.budget_us         = budget_us_.load(std::memory_order_relaxed);
    s.bypass_active     = bypassed_.load(std::memory_order_acquire)
                       || manual_bypass_.load(std::memory_order_acquire);
    return s;
}

void HealthMonitor::reset_stats() {
    n_calls_.store(0, std::memory_order_relaxed);
    n_over_.store(0, std::memory_order_relaxed);
    n_bypassed_.store(0, std::memory_order_relaxed);
    n_exceptions_.store(0, std::memory_order_relaxed);
    n_trips_.store(0, std::memory_order_relaxed);
    last_us_.store(0, std::memory_order_relaxed);
    max_us_.store(0, std::memory_order_relaxed);
    // Window state belongs to the audio thread — don't touch it from here.
}

void HealthMonitor::force_bypass(bool on) {
    manual_bypass_.store(on, std::memory_order_release);
}

bool HealthMonitor::is_bypassed() const {
    return bypassed_.load(std::memory_order_acquire)
        || manual_bypass_.load(std::memory_order_acquire);
}

void HealthMonitor::set_budget_us(std::uint32_t budget_us) {
    budget_us_.store(budget_us, std::memory_order_relaxed);
}

}  // namespace aboba
