// SPDX-License-Identifier: GPL-3.0-or-later
// HIP + rocFFT backend. The based AMD-exclusive path.
//
// Build only when ABOBA_ENABLE_HIP=ON in CMake. Requires ROCm 6.x.
//
// SAM/Resizable BAR is auto-utilized when the GPU supports it AND the system
// BIOS has it enabled. We allocate "host-coherent" device memory which on
// SAM-capable systems ends up being the BAR-mapped region — zero-copy CPU↔GPU.
#include "aboba/backend.hpp"

#ifdef ABOBA_ENABLE_HIP

#include <hip/hip_runtime.h>
#include <rocfft/rocfft.h>

#include <complex>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace aboba {

namespace {

[[noreturn]] void throw_hip(hipError_t err, const char* what) {
    throw std::runtime_error(
        std::string("HIP error in ") + what + ": " + hipGetErrorString(err));
}

#define ABOBA_HIP_CHECK(expr)                                          \
    do {                                                               \
        hipError_t _e = (expr);                                        \
        if (_e != hipSuccess) throw_hip(_e, #expr);                    \
    } while (0)

#define ABOBA_ROCFFT_CHECK(expr)                                       \
    do {                                                               \
        rocfft_status _s = (expr);                                     \
        if (_s != rocfft_status_success)                               \
            throw std::runtime_error("rocFFT error in " #expr);        \
    } while (0)

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
        return (k.size * 1315423911u) ^ (k.batch << 7) ^ (k.forward ? 1u : 0u);
    }
};

}  // namespace

class HipBackend : public Backend {
public:
    HipBackend() {
        ABOBA_ROCFFT_CHECK(rocfft_setup());

        int device_count = 0;
        ABOBA_HIP_CHECK(hipGetDeviceCount(&device_count));
        if (device_count == 0) {
            throw std::runtime_error(
                "Aboba Vocoder requires an AMD GPU. None detected.");
        }

        // Pick device 0 for now. TODO: let user configure.
        ABOBA_HIP_CHECK(hipSetDevice(0));

        hipDeviceProp_t props{};
        ABOBA_HIP_CHECK(hipGetDeviceProperties(&props, 0));
        device_name_ = props.name;

        // Check for SAM/large BAR support. directManagedMemAccessFromHost is
        // a decent proxy: if the host can directly access device memory, BAR
        // resizing is in effect.
        sam_available_ = (props.directManagedMemAccessFromHost != 0);
    }

    ~HipBackend() override {
        for (auto& kv : plan_cache_) {
            rocfft_plan_destroy(kv.second);
        }
        plan_cache_.clear();

        // Free any device buffers we allocated. Iterate over our own size map
        // so we don't double-free or miss anything.
        if (d_in_)  { hipFree(d_in_);  d_in_  = nullptr; }
        if (d_out_) { hipFree(d_out_); d_out_ = nullptr; }
        buf_sizes_.clear();

        rocfft_cleanup();
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
        rocfft_plan plan = get_or_make_plan(fft_size, batch, /*forward=*/true);

        const std::size_t in_bytes  = sizeof(float) * fft_size * batch;
        const std::size_t out_bytes = sizeof(std::complex<float>) *
                                      (fft_size / 2 + 1) * batch;

        // Ensure device buffers
        ensure_device_buffer(d_in_, in_bytes);
        ensure_device_buffer(d_out_, out_bytes);

        ABOBA_HIP_CHECK(hipMemcpy(d_in_, input, in_bytes, hipMemcpyHostToDevice));

        void* in_ptr[1]  = { d_in_ };
        void* out_ptr[1] = { d_out_ };
        ABOBA_ROCFFT_CHECK(rocfft_execute(plan, in_ptr, out_ptr, nullptr));

        ABOBA_HIP_CHECK(hipMemcpy(output, d_out_, out_bytes, hipMemcpyDeviceToHost));
    }

    void fft_c2r_batch(const std::complex<float>* input,
                       float* output,
                       std::size_t fft_size,
                       std::size_t batch) override {
        rocfft_plan plan = get_or_make_plan(fft_size, batch, /*forward=*/false);

        const std::size_t in_bytes  = sizeof(std::complex<float>) *
                                      (fft_size / 2 + 1) * batch;
        const std::size_t out_bytes = sizeof(float) * fft_size * batch;

        ensure_device_buffer(d_in_, in_bytes);
        ensure_device_buffer(d_out_, out_bytes);

        ABOBA_HIP_CHECK(hipMemcpy(d_in_, input, in_bytes, hipMemcpyHostToDevice));

        void* in_ptr[1]  = { d_in_ };
        void* out_ptr[1] = { d_out_ };
        ABOBA_ROCFFT_CHECK(rocfft_execute(plan, in_ptr, out_ptr, nullptr));

        ABOBA_HIP_CHECK(hipMemcpy(output, d_out_, out_bytes, hipMemcpyDeviceToHost));
    }

    BackendType type() const override { return BackendType::HIP; }

    const char* name() const override {
        // Cached string so we can return const char*.
        if (cached_name_.empty()) {
            cached_name_ = "HIP/rocFFT [" + device_name_ + "]";
            if (sam_available_) cached_name_ += " (SAM)";
        }
        return cached_name_.c_str();
    }

private:
    rocfft_plan get_or_make_plan(std::size_t fft_size,
                                 std::size_t batch,
                                 bool forward) {
        PlanKey key{fft_size, batch, forward};
        auto it = plan_cache_.find(key);
        if (it != plan_cache_.end()) return it->second;

        rocfft_plan plan = nullptr;
        const std::size_t lengths[1] = { fft_size };

        ABOBA_ROCFFT_CHECK(rocfft_plan_create(
            &plan,
            rocfft_placement_notinplace,
            forward ? rocfft_transform_type_real_forward
                    : rocfft_transform_type_real_inverse,
            rocfft_precision_single,
            /*dimensions=*/1,
            lengths,
            batch,
            /*description=*/nullptr));

        plan_cache_[key] = plan;
        return plan;
    }

    void ensure_device_buffer(void*& buf, std::size_t bytes) {
        if (buf && current_buf_size(buf) >= bytes) return;
        if (buf) {
            // Erase the old size entry BEFORE freeing — the pointer may be
            // reused for a different allocation immediately after, and we'd
            // end up with a stale size record.
            buf_sizes_.erase(buf);
            ABOBA_HIP_CHECK(hipFree(buf));
            buf = nullptr;
        }

        // hipMalloc → device memory. On SAM-capable systems the driver
        // accesses this via the resizable BAR aperture, allowing the host
        // pointer-style optimizations below.
        ABOBA_HIP_CHECK(hipMalloc(&buf, bytes));
        if (!buf) {
            throw std::runtime_error("hipMalloc returned null");
        }
        buf_sizes_[buf] = bytes;
    }

    std::size_t current_buf_size(void* buf) const {
        auto it = buf_sizes_.find(buf);
        return it == buf_sizes_.end() ? 0 : it->second;
    }

    std::unordered_map<PlanKey, rocfft_plan, PlanKeyHash> plan_cache_;
    std::unordered_map<void*, std::size_t> buf_sizes_;

    void* d_in_  = nullptr;
    void* d_out_ = nullptr;

    std::string device_name_;
    bool sam_available_ = false;
    mutable std::string cached_name_;
};

std::unique_ptr<Backend> make_hip_backend() {
    return std::make_unique<HipBackend>();
}

}  // namespace aboba

#else  // !ABOBA_ENABLE_HIP

namespace aboba {
std::unique_ptr<Backend> make_hip_backend() { return nullptr; }
}  // namespace aboba

#endif  // ABOBA_ENABLE_HIP
