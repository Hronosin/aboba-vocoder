# SPDX-License-Identifier: GPL-3.0-or-later
cmake_minimum_required(VERSION 3.21)
project(aboba_vocoder
    VERSION 0.1.0
    DESCRIPTION "AMD-exclusive vocoder framework. Aboba."
    LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Default to Release if no build type set — audio code should never run debug
# by default, the math gets too slow.
if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# --- Options ---------------------------------------------------------------
option(ABOBA_ENABLE_HIP    "Build the HIP/rocFFT GPU backend (requires ROCm)" OFF)
option(ABOBA_ENABLE_REALTIME "Build the PortAudio-based realtime engine"     ON)
option(ABOBA_BUILD_EXAMPLES "Build example programs" ON)
option(ABOBA_ENABLE_ASAN   "Build with AddressSanitizer (dev only)"          OFF)
option(ABOBA_ENABLE_UBSAN  "Build with UndefinedBehaviorSanitizer (dev only)" OFF)
option(ABOBA_NATIVE_ARCH   "Use -march=native for max SIMD throughput"       OFF)

# --- Sanitizers ------------------------------------------------------------
if (ABOBA_ENABLE_ASAN)
    add_compile_options(-fsanitize=address -fno-omit-frame-pointer)
    add_link_options(-fsanitize=address)
endif()
if (ABOBA_ENABLE_UBSAN)
    add_compile_options(-fsanitize=undefined -fno-omit-frame-pointer)
    add_link_options(-fsanitize=undefined)
endif()

# --- Warnings (paranoia mode) ---------------------------------------------
if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(
        -Wall -Wextra -Wpedantic
        -Wshadow -Wconversion -Wsign-conversion
        -Wnon-virtual-dtor -Wold-style-cast
        -Wcast-align -Wunused
        -Woverloaded-virtual -Wdouble-promotion
        -Wformat=2
    )
    if (ABOBA_NATIVE_ARCH)
        add_compile_options(-march=native)
    endif()
    # Audio loops benefit from these in Release.
    add_compile_options($<$<CONFIG:Release>:-ffast-math>
                       $<$<CONFIG:Release>:-funroll-loops>)
endif()

# --- Dependencies ----------------------------------------------------------
# FFTW3 (single precision)
find_package(PkgConfig REQUIRED)
pkg_check_modules(FFTW3F REQUIRED IMPORTED_TARGET fftw3f)
# We also need the threads variant. If your distro splits it, link explicitly.
find_library(FFTW3F_THREADS_LIB fftw3f_threads)

if (ABOBA_ENABLE_REALTIME)
    pkg_check_modules(PORTAUDIO REQUIRED IMPORTED_TARGET portaudio-2.0)
endif()

# --- Library ---------------------------------------------------------------
set(ABOBA_SOURCES
    src/stft.cpp
    src/phase_vocoder.cpp
    src/streaming.cpp
    src/platform.cpp
    src/health.cpp
    src/self_check.cpp
    src/backend/cpu_backend.cpp
    src/backend/hip_backend.cpp
    src/backend/factory.cpp
)

if (ABOBA_ENABLE_REALTIME)
    list(APPEND ABOBA_SOURCES src/realtime.cpp)
endif()

add_library(aboba STATIC ${ABOBA_SOURCES})
target_include_directories(aboba PUBLIC include)
target_link_libraries(aboba
    PUBLIC PkgConfig::FFTW3F
)
if (FFTW3F_THREADS_LIB)
    target_link_libraries(aboba PUBLIC ${FFTW3F_THREADS_LIB})
endif()
if (ABOBA_ENABLE_REALTIME)
    target_link_libraries(aboba PUBLIC PkgConfig::PORTAUDIO)
    target_compile_definitions(aboba PUBLIC ABOBA_ENABLE_REALTIME=1)
endif()

# --- HIP backend (optional) ------------------------------------------------
if (ABOBA_ENABLE_HIP)
    # ROCm typically installs to /opt/rocm
    if (NOT DEFINED ROCM_PATH)
        set(ROCM_PATH "/opt/rocm" CACHE PATH "Path to ROCm install")
    endif()

    list(APPEND CMAKE_PREFIX_PATH "${ROCM_PATH}")
    find_package(hip     REQUIRED CONFIG)
    find_package(rocfft  REQUIRED CONFIG)

    target_compile_definitions(aboba PUBLIC ABOBA_ENABLE_HIP=1)
    target_link_libraries(aboba PUBLIC hip::host roc::rocfft)

    message(STATUS "Aboba: HIP backend ENABLED (ROCm at ${ROCM_PATH})")
else()
    message(STATUS "Aboba: HIP backend disabled (CPU-only build). "
                    "Pass -DABOBA_ENABLE_HIP=ON to enable.")
endif()

# --- Examples --------------------------------------------------------------
if (ABOBA_BUILD_EXAMPLES)
    add_executable(aboba_pitch_shift examples/pitch_shift.cpp)
    target_link_libraries(aboba_pitch_shift PRIVATE aboba)

    add_executable(aboba_stream_file examples/stream_file.cpp)
    target_link_libraries(aboba_stream_file PRIVATE aboba)

    add_executable(aboba_stress_test examples/stress_test.cpp)
    target_link_libraries(aboba_stress_test PRIVATE aboba)

    add_executable(aboba_dynamic_test examples/dynamic_test.cpp)
    target_link_libraries(aboba_dynamic_test PRIVATE aboba)
    # Threads for the concurrent stats-read test
    find_package(Threads REQUIRED)
    target_link_libraries(aboba_dynamic_test PRIVATE Threads::Threads)

    add_executable(aboba_info examples/info.cpp)
    target_link_libraries(aboba_info PRIVATE aboba)

    if (ABOBA_ENABLE_REALTIME)
        add_executable(aboba_realtime_voice examples/realtime_voice.cpp)
        target_link_libraries(aboba_realtime_voice PRIVATE aboba)
    endif()
endif()

# --- Install ---------------------------------------------------------------
install(TARGETS aboba
        ARCHIVE DESTINATION lib
        LIBRARY DESTINATION lib)
install(DIRECTORY include/aboba DESTINATION include)
