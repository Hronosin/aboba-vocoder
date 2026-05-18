// SPDX-License-Identifier: GPL-3.0-or-later
#include "aboba/backend.hpp"
#include "aboba/platform.hpp"

#include <cstdio>
#include <stdexcept>

namespace aboba {

// Defined in backend/cpu_backend.cpp and backend/hip_backend.cpp.
std::unique_ptr<Backend> make_cpu_backend();
std::unique_ptr<Backend> make_hip_backend();

std::unique_ptr<Backend> create_backend(BackendType type) {
    switch (type) {
        case BackendType::CPU:
            return make_cpu_backend();
        case BackendType::HIP: {
            auto b = make_hip_backend();
            if (!b) {
                throw std::runtime_error(
                    "HIP backend not available. Rebuild with -DABOBA_ENABLE_HIP=ON, "
                    "and ensure ROCm + an AMD GPU are present.");
            }
            return b;
        }
    }
    throw std::runtime_error("Unknown backend type");
}

std::unique_ptr<Backend> create_best_backend() {
    PlatformInfo info = detect_platform();

    // Try HIP first if there's an AMD GPU. We don't try HIP on NVIDIA-only
    // systems — that would just print confusing rocFFT errors. CPU it is.
    if (info.has_amd()) {
        try {
            auto hip = make_hip_backend();
            if (hip) {
                print_platform_banner(info, hip->name());
                return hip;
            }
        } catch (const std::exception& e) {
            std::fprintf(stderr,
                "[aboba] HIP available but init failed (%s), falling back to CPU\n",
                e.what());
        }
    }

    auto cpu = make_cpu_backend();
    print_platform_banner(info, cpu->name());
    return cpu;
}

}  // namespace aboba
