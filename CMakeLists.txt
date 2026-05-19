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
option(ABOBA_BUILD_PYTHON   "Build pybind11 Python bindings (requires Python + pybind11)" OFF)
option(ABOBA_ENABLE_ASAN   "Build with AddressSanitizer (dev only)"          OFF)
option(ABOBA_ENABLE_UBSAN  "Build with UndefinedBehaviorSanitizer (dev only)" OFF)
option(ABOBA_AUTO_ARCH     "Auto-detect CPU features and apply optimal flags" ON)
option(ABOBA_NATIVE_ARCH   "(legacy) Force -march=native; implied by ABOBA_AUTO_ARCH" OFF)

# --- Arch detection --------------------------------------------------------
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(AbobaArch)

if (ABOBA_AUTO_ARCH)
    aboba_detect_cpu()
    aboba_detect_amd_gpus(ABOBA_AMD_GPU_TARGETS)
    aboba_print_detection_report()
endif()

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
    if (ABOBA_AUTO_ARCH AND DEFINED ABOBA_CPU_FLAGS)
        add_compile_options(${ABOBA_CPU_FLAGS})
    elseif (ABOBA_NATIVE_ARCH)
        add_compile_options(-march=native)
    endif()
    # Release optimizations — SAFE subset only. We rely on std::isfinite()
    # everywhere, so we must NOT enable -ffast-math (which implies
    # -ffinite-math-only and lets the compiler delete those checks).
    add_compile_options(
        $<$<CONFIG:Release>:-funroll-loops>
        $<$<CONFIG:Release>:-fno-math-errno>
        $<$<CONFIG:Release>:-fno-trapping-math>
    )
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
    src/sysinfo.cpp
    src/debug.cpp
    src/yin.cpp
    src/formant_vocoder.cpp
    src/voice_character.cpp
    src/musical_scale.cpp
    src/pitch_corrector.cpp
    src/voice_config.cpp
    src/noise_reduction.cpp
    src/reverb.cpp
    src/pipeline.cpp
    src/backend/cpu_backend.cpp
    src/backend/hip_backend.cpp
    src/backend/factory.cpp
)

if (ABOBA_ENABLE_REALTIME)
    list(APPEND ABOBA_SOURCES src/realtime.cpp)
endif()

add_library(aboba STATIC ${ABOBA_SOURCES})
target_include_directories(aboba PUBLIC include)
# Always build with -fPIC so the static library can be linked into shared
# objects (Python bindings, VST3 plugins, etc).
set_target_properties(aboba PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_link_libraries(aboba
    PUBLIC PkgConfig::FFTW3F
)
target_compile_definitions(aboba PUBLIC
    ABOBA_VERSION_STRING="${PROJECT_VERSION}"
)
if (FFTW3F_THREADS_LIB)
    target_link_libraries(aboba PUBLIC ${FFTW3F_THREADS_LIB})
endif()
if (ABOBA_ENABLE_REALTIME)
    target_link_libraries(aboba PUBLIC PkgConfig::PORTAUDIO)
    target_compile_definitions(aboba PUBLIC ABOBA_ENABLE_REALTIME=1 ABOBA_HAS_REALTIME=1)
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

    target_compile_definitions(aboba PUBLIC ABOBA_ENABLE_HIP=1 ABOBA_HAS_HIP=1)
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

    # Test/example executables use lots of literals and ad-hoc numeric
    # code that triggers cosmetic warnings (-Wconversion, -Wdouble-promotion,
    # -Wsign-conversion). The library proper is strict; relax for tests so
    # we keep the test code readable. This applies ONLY to executables, NOT
    # to the library code itself.
    function(aboba_test_executable target src)
        add_executable(${target} ${src})
        target_link_libraries(${target} PRIVATE aboba)
        if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
            target_compile_options(${target} PRIVATE
                -Wno-conversion -Wno-sign-conversion
                -Wno-double-promotion -Wno-old-style-cast
                -Wno-unused-result -Wno-unused-variable)
        endif()
    endfunction()

    aboba_test_executable(aboba_stress_test  examples/stress_test.cpp)
    aboba_test_executable(aboba_dynamic_test examples/dynamic_test.cpp)
    find_package(Threads REQUIRED)
    target_link_libraries(aboba_dynamic_test PRIVATE Threads::Threads)
    aboba_test_executable(aboba_quality_test examples/quality_test.cpp)
    aboba_test_executable(aboba_debug_test   examples/debug_test.cpp)
    target_link_libraries(aboba_debug_test   PRIVATE Threads::Threads)

    aboba_test_executable(aboba_inspect examples/inspect.cpp)
    aboba_test_executable(aboba_info    examples/info.cpp)

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

# --- Python bindings -------------------------------------------------------
if (ABOBA_BUILD_PYTHON)
    find_package(Python 3.8 REQUIRED COMPONENTS Interpreter Development.Module)
    find_package(pybind11 CONFIG QUIET)
    if (NOT pybind11_FOUND)
        # Fall back to scripts shipped with the pip-installed package.
        execute_process(
            COMMAND ${Python_EXECUTABLE} -c "import pybind11; print(pybind11.get_cmake_dir())"
            OUTPUT_VARIABLE PYBIND11_CMAKE_DIR
            OUTPUT_STRIP_TRAILING_WHITESPACE)
        if (PYBIND11_CMAKE_DIR)
            set(pybind11_DIR ${PYBIND11_CMAKE_DIR})
            find_package(pybind11 CONFIG REQUIRED)
        else()
            message(FATAL_ERROR
                "ABOBA_BUILD_PYTHON=ON but pybind11 not found. "
                "Install: pip install pybind11")
        endif()
    endif()

    pybind11_add_module(aboba_py python/aboba_py.cpp)
    target_link_libraries(aboba_py PRIVATE aboba)
    set_target_properties(aboba_py PROPERTIES OUTPUT_NAME "aboba")
    # Relax warning level for binding code — pybind generates lots of
    # type promotions that aren't worth chasing.
    if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(aboba_py PRIVATE
            -Wno-conversion -Wno-sign-conversion
            -Wno-double-promotion -Wno-old-style-cast
            -Wno-unused-parameter)
    endif()
endif()
