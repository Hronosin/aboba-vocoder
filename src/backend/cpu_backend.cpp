// SPDX-License-Identifier: GPL-3.0-or-later
// CPU backend using FFTW3. Universal fallback path.
#include "aboba/backend.hpp"

#include <fftw3.h>
#include <complex>
#include <cstring>
#include <mutex>
#include <stdexcept>
#include <unordered_map>

namespace aboba {

namespace {

// FFTW plan creation is not thread-safe; execution is.
// We cache plans keyed by (size, direction, batch) to avoid re-planning.
std::mutex g_fftw_plan_mutex;

struct PlanKey {
    std::size_t size;
    std::size_t batch;
    bool        forward;

    bool operator==(const PlanKey& o) const {
        return size == o.size && batch == o.batch && forward == o.forward;
    }
};

struct PlanKeyHash {
    std::size_t operator()(const PlanKey& k) const {
        return std::hash<std::size_t>()(k.size)
             ^ (std::hash<std::size_t>()(k.batch) << 1)
             ^ (std::hash<bool>()(k.forward) << 2);
    }
};

}  // namespace

class CpuBackend : public Backend {
public:
    CpuBackend() {
        // Enable multi-threading. Pick a sane default.
        // TODO: expose thread count via config.
        fftwf_init_threads();
        fftwf_plan_with_nthreads(2);
    }

    ~CpuBackend() override {
        std::lock_guard<std::mutex> lock(g_fftw_plan_mutex);
        for (auto& kv : plan_cache_) {
            fftwf_destroy_plan(kv.second);
        }
        fftwf_cleanup_threads();
    }

    void fft_r2c(const float* input,
                 std::complex<float>* output,
                 std::size_t fft_size) override {
        fft_r2c_batch(input, output, fft_size, 1);
    }

    void fft_c2r(const std::complex<float>* input,
                 float* output,
                 std::size_t fft_size) override {
        fft_c2r_batch(input, output, fft_size, 1);
    }

    void fft_r2c_batch(const float* input,
                       std::complex<float>* output,
                       std::size_t fft_size,
                       std::size_t batch) override {
        auto plan = get_or_make_plan_r2c(fft_size, batch);
        // FFTW guru execute is thread-safe.
        fftwf_execute_dft_r2c(
            plan,
            const_cast<float*>(input),
            reinterpret_cast<fftwf_complex*>(output));
    }

    void fft_c2r_batch(const std::complex<float>* input,
                       float* output,
                       std::size_t fft_size,
                       std::size_t batch) override {
        auto plan = get_or_make_plan_c2r(fft_size, batch);
        fftwf_execute_dft_c2r(
            plan,
            reinterpret_cast<fftwf_complex*>(
                const_cast<std::complex<float>*>(input)),
            output);
    }

    BackendType type() const override { return BackendType::CPU; }
    const char* name() const override { return "CPU (FFTW3)"; }

private:
    fftwf_plan get_or_make_plan_r2c(std::size_t fft_size, std::size_t batch) {
        std::lock_guard<std::mutex> lock(g_fftw_plan_mutex);
        PlanKey key{fft_size, batch, true};
        auto it = plan_cache_.find(key);
        if (it != plan_cache_.end()) return it->second;

        // Allocate scratch buffers for planning. FFTW measures with real data.
        const std::size_t n_in  = fft_size * batch;
        const std::size_t n_out = (fft_size / 2 + 1) * batch;
        float* in  = fftwf_alloc_real(n_in);
        fftwf_complex* out = fftwf_alloc_complex(n_out);

        int n[1] = { static_cast<int>(fft_size) };
        fftwf_plan plan = fftwf_plan_many_dft_r2c(
            /*rank=*/1, n, /*howmany=*/static_cast<int>(batch),
            in,  nullptr, /*istride=*/1, /*idist=*/static_cast<int>(fft_size),
            out, nullptr, /*ostride=*/1, /*odist=*/static_cast<int>(fft_size/2 + 1),
            FFTW_ESTIMATE);  // ESTIMATE for fast cold-start; switch to MEASURE for steady-state

        fftwf_free(in);
        fftwf_free(out);

        if (!plan) throw std::runtime_error("FFTW r2c plan creation failed");
        plan_cache_[key] = plan;
        return plan;
    }

    fftwf_plan get_or_make_plan_c2r(std::size_t fft_size, std::size_t batch) {
        std::lock_guard<std::mutex> lock(g_fftw_plan_mutex);
        PlanKey key{fft_size, batch, false};
        auto it = plan_cache_.find(key);
        if (it != plan_cache_.end()) return it->second;

        const std::size_t n_in  = (fft_size / 2 + 1) * batch;
        const std::size_t n_out = fft_size * batch;
        fftwf_complex* in = fftwf_alloc_complex(n_in);
        float* out = fftwf_alloc_real(n_out);

        int n[1] = { static_cast<int>(fft_size) };
        fftwf_plan plan = fftwf_plan_many_dft_c2r(
            1, n, static_cast<int>(batch),
            in,  nullptr, 1, static_cast<int>(fft_size/2 + 1),
            out, nullptr, 1, static_cast<int>(fft_size),
            FFTW_ESTIMATE);

        fftwf_free(in);
        fftwf_free(out);

        if (!plan) throw std::runtime_error("FFTW c2r plan creation failed");
        plan_cache_[key] = plan;
        return plan;
    }

    std::unordered_map<PlanKey, fftwf_plan, PlanKeyHash> plan_cache_;
};

// Factory hook (used by factory.cpp)
std::unique_ptr<Backend> make_cpu_backend() {
    return std::make_unique<CpuBackend>();
}

}  // namespace aboba
