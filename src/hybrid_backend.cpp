// SPDX-License-Identifier: GPL-3.0-or-later
#include "aboba/hybrid_backend.hpp"

#include <algorithm>
#include <cstring>
#include <cstdio>
#include <future>
#include <limits>
#include <stdexcept>

namespace aboba {

namespace {

constexpr float kNaN = std::numeric_limits<float>::quiet_NaN();

inline bool is_nan(float x) noexcept { return x != x; }

inline std::chrono::steady_clock::time_point::rep now_rep() noexcept {
    return std::chrono::steady_clock::now().time_since_epoch().count();
}

}  // namespace

// --------------------------------------------------------------
HybridBackend::PerChild::PerChild() : healthy(true), unhealthy_since(0),
                                      calls_routed(0), calls_completed(0),
                                      calls_failed(0) {
    for (auto& v : avg_us)       v.store(kNaN, std::memory_order_relaxed);
    for (auto& v : sample_count) v.store(0, std::memory_order_relaxed);
}

std::size_t HybridBackend::size_bucket(std::size_t fft_size) noexcept {
    // 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, >=65536
    if (fft_size <  128)  return 0;
    if (fft_size <  256)  return 1;
    if (fft_size <  512)  return 2;
    if (fft_size <  1024) return 3;
    if (fft_size <  2048) return 4;
    if (fft_size <  4096) return 5;
    if (fft_size <  8192) return 6;
    if (fft_size < 16384) return 7;
    if (fft_size < 32768) return 8;
    if (fft_size < 65536) return 9;
    return 10;
}

// --------------------------------------------------------------
HybridBackend::HybridBackend(std::vector<HybridChild> children_in,
                             HybridBackendConfig cfg)
    : cfg_(cfg) {
    if (children_in.empty()) {
        throw std::invalid_argument("HybridBackend: at least one child required");
    }
    children_.reserve(children_in.size());
    for (auto& c : children_in) {
        if (!c.backend) {
            throw std::invalid_argument("HybridBackend: null child backend");
        }
        if (c.label.empty()) c.label = c.backend->name();
        auto p = std::make_unique<PerChild>();
        p->meta = std::move(c);
        children_.push_back(std::move(p));
    }
    // Sort by priority ascending — lower number = higher priority.
    std::sort(children_.begin(), children_.end(),
              [](const std::unique_ptr<PerChild>& a,
                 const std::unique_ptr<PerChild>& b) {
                  return a->meta.priority < b->meta.priority;
              });
}

void HybridBackend::refresh_health() {
    const auto now = std::chrono::steady_clock::now();
    for (auto& c : children_) {
        if (!c->healthy.load(std::memory_order_relaxed)) {
            const auto raw = c->unhealthy_since.load(std::memory_order_relaxed);
            const std::chrono::steady_clock::time_point t{
                std::chrono::steady_clock::duration{raw}};
            if (now - t >= cfg_.health_cooldown) {
                c->healthy.store(true, std::memory_order_relaxed);
                if (cfg_.log_decisions) {
                    std::lock_guard<std::mutex> lk(log_mtx_);
                    std::fprintf(stderr,
                        "[hybrid] backend '%s' returning from cooldown\n",
                        c->meta.label.c_str());
                }
            }
        }
    }
}

void HybridBackend::record_latency(std::size_t child_idx, std::size_t fft_size,
                                   float latency_us) {
    if (child_idx >= children_.size()) return;
    auto& c = *children_[child_idx];
    const std::size_t b = size_bucket(fft_size);

    // Exponential moving average. Equivalent weight to N=cost_window samples.
    const float n = static_cast<float>(std::max<std::size_t>(cfg_.cost_window, 1));
    const float alpha = 1.0f / n;

    const float prev = c.avg_us[b].load(std::memory_order_relaxed);
    const float next = is_nan(prev)
        ? latency_us
        : (prev * (1.0f - alpha) + latency_us * alpha);
    c.avg_us[b].store(next, std::memory_order_relaxed);
    c.sample_count[b].fetch_add(1, std::memory_order_relaxed);
}

void HybridBackend::record_failure(std::size_t child_idx) {
    if (child_idx >= children_.size()) return;
    auto& c = *children_[child_idx];
    c.calls_failed.fetch_add(1, std::memory_order_relaxed);
    c.healthy.store(false, std::memory_order_relaxed);
    c.unhealthy_since.store(now_rep(), std::memory_order_relaxed);
    if (cfg_.log_decisions) {
        std::lock_guard<std::mutex> lk(log_mtx_);
        std::fprintf(stderr, "[hybrid] backend '%s' marked unhealthy\n",
                     c.meta.label.c_str());
    }
}

int HybridBackend::pick_preferred(std::size_t fft_size) {
    refresh_health();

    if (cfg_.mode == HybridMode::StrictPriority) {
        // First healthy in priority order
        for (std::size_t i = 0; i < children_.size(); ++i) {
            if (children_[i]->healthy.load(std::memory_order_relaxed)) {
                return static_cast<int>(i);
            }
        }
        return -1;
    }

    // Adaptive: among ELIGIBLE & HEALTHY backends, pick lowest avg_us for
    // this size bucket. Backends without samples yet get a low score
    // (epsilon) so they're tried first to gather data.
    const std::size_t bucket = size_bucket(fft_size);

    int best_idx = -1;
    float best_score = std::numeric_limits<float>::infinity();

    for (std::size_t i = 0; i < children_.size(); ++i) {
        const auto& c = *children_[i];
        if (!c.meta.eligible_for_adaptive)         continue;
        if (!c.healthy.load(std::memory_order_relaxed)) continue;

        const float avg = c.avg_us[bucket].load(std::memory_order_relaxed);
        // Untried backend: assign 1.0us so we'll try it once
        const float score = is_nan(avg) ? 1.0f : avg;
        if (score < best_score) {
            best_score = score;
            best_idx = static_cast<int>(i);
        }
    }

    if (best_idx >= 0) return best_idx;

    // Nothing adaptive-eligible & healthy. Fall back to first healthy
    // regardless of eligibility.
    for (std::size_t i = 0; i < children_.size(); ++i) {
        if (children_[i]->healthy.load(std::memory_order_relaxed)) {
            return static_cast<int>(i);
        }
    }
    return -1;
}

void HybridBackend::build_try_order(std::size_t fft_size,
                                    std::vector<int>& out) {
    out.clear();
    const int preferred = pick_preferred(fft_size);

    if (cfg_.mode == HybridMode::AdaptiveNoFailover) {
        if (preferred >= 0) out.push_back(preferred);
        return;
    }

    // Adaptive-with-failover OR strict-priority: build list of all
    // healthy backends in priority order, with `preferred` moved to front.
    if (preferred >= 0) out.push_back(preferred);
    for (std::size_t i = 0; i < children_.size(); ++i) {
        if (static_cast<int>(i) == preferred) continue;
        if (children_[i]->healthy.load(std::memory_order_relaxed)) {
            out.push_back(static_cast<int>(i));
        }
    }
    // Last resort: include unhealthy ones at the very end (in case ALL
    // are unhealthy — better to try than to fail silently)
    if (out.empty()) {
        for (std::size_t i = 0; i < children_.size(); ++i) {
            out.push_back(static_cast<int>(i));
        }
    }
}

void HybridBackend::invoke(std::size_t child_idx, Op op,
                           const void* input, void* output,
                           std::size_t fft_size, std::size_t batch) {
    auto& c = *children_[child_idx];
    c.calls_routed.fetch_add(1, std::memory_order_relaxed);

    const auto t0 = std::chrono::steady_clock::now();
    switch (op) {
        case Op::R2C:
            c.meta.backend->fft_r2c(
                static_cast<const float*>(input),
                static_cast<std::complex<float>*>(output),
                fft_size);
            break;
        case Op::C2R:
            c.meta.backend->fft_c2r(
                static_cast<const std::complex<float>*>(input),
                static_cast<float*>(output),
                fft_size);
            break;
        case Op::R2C_BATCH:
            c.meta.backend->fft_r2c_batch(
                static_cast<const float*>(input),
                static_cast<std::complex<float>*>(output),
                fft_size, batch);
            break;
        case Op::C2R_BATCH:
            c.meta.backend->fft_c2r_batch(
                static_cast<const std::complex<float>*>(input),
                static_cast<float*>(output),
                fft_size, batch);
            break;
    }
    const auto t1 = std::chrono::steady_clock::now();
    const auto us = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
    record_latency(child_idx, fft_size, static_cast<float>(us));
    c.calls_completed.fetch_add(1, std::memory_order_relaxed);
}

void HybridBackend::dispatch_chain(Op op, const void* input, void* output,
                                   std::size_t fft_size, std::size_t batch) {
    total_calls_.fetch_add(1, std::memory_order_relaxed);

    std::vector<int> order;
    build_try_order(fft_size, order);
    if (order.empty()) {
        throw std::runtime_error(
            "HybridBackend: no backends available to dispatch call");
    }

    std::string last_err;
    for (std::size_t attempt = 0; attempt < order.size(); ++attempt) {
        const std::size_t idx = static_cast<std::size_t>(order[attempt]);
        if (attempt > 0) {
            failover_invocations_.fetch_add(1, std::memory_order_relaxed);
            if (cfg_.log_decisions) {
                std::lock_guard<std::mutex> lk(log_mtx_);
                std::fprintf(stderr, "[hybrid] failing over to '%s' (attempt %zu)\n",
                             children_[idx]->meta.label.c_str(), attempt);
            }
        }
        try {
            invoke(idx, op, input, output, fft_size, batch);
            return;  // success
        } catch (const std::exception& e) {
            last_err = e.what();
            record_failure(idx);
            // If failover is disabled, propagate immediately.
            if (cfg_.mode == HybridMode::AdaptiveNoFailover) {
                throw;
            }
            // Otherwise continue to next backend in chain
        }
    }
    // All backends failed
    throw std::runtime_error(
        "HybridBackend: all backends failed; last error: " + last_err);
}

void HybridBackend::fft_r2c(const float* input, std::complex<float>* output,
                            std::size_t fft_size) {
    dispatch_chain(Op::R2C, input, output, fft_size, 1);
}

void HybridBackend::fft_c2r(const std::complex<float>* input, float* output,
                            std::size_t fft_size) {
    dispatch_chain(Op::C2R, input, output, fft_size, 1);
}

void HybridBackend::fft_r2c_batch(const float* input, std::complex<float>* output,
                                  std::size_t fft_size, std::size_t batch) {
    // Multi-channel parallelism: split batch across eligible backends.
    if (batch < cfg_.split_batch_threshold) {
        dispatch_chain(Op::R2C_BATCH, input, output, fft_size, batch);
        return;
    }

    // Collect eligible healthy backends in priority order.
    refresh_health();
    std::vector<int> workers;
    for (std::size_t i = 0; i < children_.size(); ++i) {
        if (children_[i]->meta.eligible_for_adaptive &&
            children_[i]->healthy.load(std::memory_order_relaxed)) {
            workers.push_back(static_cast<int>(i));
        }
    }
    if (workers.empty()) {
        // No eligible healthy workers — single dispatch (and failover)
        dispatch_chain(Op::R2C_BATCH, input, output, fft_size, batch);
        return;
    }
    if (cfg_.max_parallel_workers > 0 &&
        static_cast<int>(workers.size()) > cfg_.max_parallel_workers) {
        workers.resize(static_cast<std::size_t>(cfg_.max_parallel_workers));
    }
    if (workers.size() == 1) {
        dispatch_chain(Op::R2C_BATCH, input, output, fft_size, batch);
        return;
    }

    // Split batch evenly. Each worker processes [start, start+chunk).
    const std::size_t n_workers = workers.size();
    const std::size_t base = batch / n_workers;
    const std::size_t rem  = batch % n_workers;
    const std::size_t bins = fft_size / 2 + 1;

    std::vector<std::future<std::string>> futures;
    futures.reserve(n_workers);

    std::size_t cursor = 0;
    for (std::size_t w = 0; w < n_workers; ++w) {
        const std::size_t my_batch = base + (w < rem ? 1 : 0);
        if (my_batch == 0) break;
        const std::size_t off_in  = cursor * fft_size;
        const std::size_t off_out = cursor * bins;
        const int idx = workers[w];
        const float* in_ptr  = input  + off_in;
        std::complex<float>* out_ptr = output + off_out;
        const std::size_t fs = fft_size;
        const std::size_t bs = my_batch;
        futures.push_back(std::async(std::launch::async,
            [this, idx, in_ptr, out_ptr, fs, bs]() -> std::string {
                try {
                    invoke(static_cast<std::size_t>(idx), Op::R2C_BATCH,
                           in_ptr, out_ptr, fs, bs);
                    return std::string();
                } catch (const std::exception& e) {
                    record_failure(static_cast<std::size_t>(idx));
                    // Retry sequentially on remaining healthy backends
                    try {
                        dispatch_chain(Op::R2C_BATCH, in_ptr, out_ptr, fs, bs);
                        return std::string();
                    } catch (const std::exception& e2) {
                        return std::string("split worker failed: ") + e2.what();
                    }
                }
            }));
        cursor += my_batch;
    }
    // Collect results
    std::string err;
    for (auto& f : futures) {
        std::string e = f.get();
        if (!e.empty() && err.empty()) err = std::move(e);
    }
    if (!err.empty()) {
        throw std::runtime_error("HybridBackend (batch split): " + err);
    }
}

void HybridBackend::fft_c2r_batch(const std::complex<float>* input,
                                  float* output, std::size_t fft_size,
                                  std::size_t batch) {
    if (batch < cfg_.split_batch_threshold) {
        dispatch_chain(Op::C2R_BATCH, input, output, fft_size, batch);
        return;
    }
    refresh_health();
    std::vector<int> workers;
    for (std::size_t i = 0; i < children_.size(); ++i) {
        if (children_[i]->meta.eligible_for_adaptive &&
            children_[i]->healthy.load(std::memory_order_relaxed)) {
            workers.push_back(static_cast<int>(i));
        }
    }
    if (workers.empty()) {
        dispatch_chain(Op::C2R_BATCH, input, output, fft_size, batch);
        return;
    }
    if (cfg_.max_parallel_workers > 0 &&
        static_cast<int>(workers.size()) > cfg_.max_parallel_workers) {
        workers.resize(static_cast<std::size_t>(cfg_.max_parallel_workers));
    }
    if (workers.size() == 1) {
        dispatch_chain(Op::C2R_BATCH, input, output, fft_size, batch);
        return;
    }

    const std::size_t n_workers = workers.size();
    const std::size_t base = batch / n_workers;
    const std::size_t rem  = batch % n_workers;
    const std::size_t bins = fft_size / 2 + 1;

    std::vector<std::future<std::string>> futures;
    futures.reserve(n_workers);

    std::size_t cursor = 0;
    for (std::size_t w = 0; w < n_workers; ++w) {
        const std::size_t my_batch = base + (w < rem ? 1 : 0);
        if (my_batch == 0) break;
        const std::size_t off_in  = cursor * bins;
        const std::size_t off_out = cursor * fft_size;
        const int idx = workers[w];
        const std::complex<float>* in_ptr = input + off_in;
        float* out_ptr = output + off_out;
        const std::size_t fs = fft_size;
        const std::size_t bs = my_batch;
        futures.push_back(std::async(std::launch::async,
            [this, idx, in_ptr, out_ptr, fs, bs]() -> std::string {
                try {
                    invoke(static_cast<std::size_t>(idx), Op::C2R_BATCH,
                           in_ptr, out_ptr, fs, bs);
                    return std::string();
                } catch (const std::exception& e) {
                    record_failure(static_cast<std::size_t>(idx));
                    try {
                        dispatch_chain(Op::C2R_BATCH, in_ptr, out_ptr, fs, bs);
                        return std::string();
                    } catch (const std::exception& e2) {
                        return std::string("split worker failed: ") + e2.what();
                    }
                }
            }));
        cursor += my_batch;
    }
    std::string err;
    for (auto& f : futures) {
        std::string e = f.get();
        if (!e.empty() && err.empty()) err = std::move(e);
    }
    if (!err.empty()) {
        throw std::runtime_error("HybridBackend (batch split): " + err);
    }
}

HybridStats HybridBackend::stats() const {
    HybridStats s;
    s.total_calls = total_calls_.load(std::memory_order_relaxed);
    s.failover_invocations = failover_invocations_.load(std::memory_order_relaxed);
    for (auto& c : children_) {
        s.backend_names.push_back(c->meta.label);
        s.calls_routed.push_back(c->calls_routed.load(std::memory_order_relaxed));
        s.calls_completed.push_back(c->calls_completed.load(std::memory_order_relaxed));
        s.calls_failed.push_back(c->calls_failed.load(std::memory_order_relaxed));
        s.healthy.push_back(c->healthy.load(std::memory_order_relaxed));
        // Average across populated buckets, weighted by sample count
        double weighted_sum = 0.0;
        std::uint64_t total_samples = 0;
        for (std::size_t b = 0; b < kNumSizeBuckets; ++b) {
            const float v = c->avg_us[b].load(std::memory_order_relaxed);
            const std::uint32_t cnt = c->sample_count[b].load(std::memory_order_relaxed);
            if (!is_nan(v) && cnt > 0) {
                weighted_sum += static_cast<double>(v) * cnt;
                total_samples += cnt;
            }
        }
        s.avg_latency_us.push_back(total_samples > 0
            ? static_cast<float>(weighted_sum / static_cast<double>(total_samples))
            : kNaN);
    }
    return s;
}

void HybridBackend::set_backend_healthy(std::size_t idx, bool healthy) {
    if (idx >= children_.size()) return;
    children_[idx]->healthy.store(healthy, std::memory_order_relaxed);
    if (!healthy) {
        children_[idx]->unhealthy_since.store(now_rep(), std::memory_order_relaxed);
    }
}

}  // namespace aboba
