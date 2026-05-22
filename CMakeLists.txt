# SPDX-License-Identifier: GPL-3.0-or-later
cmake_minimum_required(VERSION 3.21)
project(aboba_vocoder
    VERSION 0.1.0
    DESCRIPTION "AMD-exclusive vocoder framework. Aboba."
    LANGUAGES C CXX
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
option(ABOBA_ENABLE_VULKAN "Build the Vulkan compute backend (cross-vendor GPU)" OFF)
option(ABOBA_ENABLE_REALTIME "Build the PortAudio-based realtime engine"     ON)
option(ABOBA_BUILD_EXAMPLES "Build example programs" ON)
option(ABOBA_BUILD_PYTHON   "Build pybind11 Python bindings (requires Python + pybind11)" OFF)
option(ABOBA_BUILD_SHARED_C "Build aboba_c shared library (for game-engine / FFI integration)" ON)
option(ABOBA_BUILD_VST3    "Build VST3 plugin (requires VST3 SDK)" OFF)
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

    # ---- Compiler-level paranoia hardening -------------------------------
    # These flags exist BECAUSE bugs that slip past code review will
    # eventually happen. They are belt-and-suspenders defenses, not
    # primary safety. They cost ~1-3% performance in release builds; we
    # accept that tradeoff for security and stability.
    #
    # -fstack-protector-strong: guard pages on stack frames with arrays/
    #   variables of size > 8. Detects classic buffer overflows.
    # -D_FORTIFY_SOURCE=2: enables compile-time + runtime checks on
    #   strcpy/sprintf/memcpy family. Catches buffer overflows that miss
    #   the static analyzer.
    # -fno-strict-aliasing: a few of our SIMD/FFT routines technically
    #   alias; disabling strict aliasing prevents wrong codegen there.
    # -fno-delete-null-pointer-checks: in C++ this is mostly benign but
    #   we don't want the optimizer "proving" our null checks redundant.
    # -fwrapv: signed integer overflow wraps (defined behavior). We do
    #   NOT rely on UB for performance anywhere.
    # -fno-omit-frame-pointer in debug+release for tractable crash logs.
    add_compile_options(
        -fstack-protector-strong
        -fno-strict-aliasing
        -fno-delete-null-pointer-checks
        -fwrapv
        $<$<NOT:$<CONFIG:Debug>>:-D_FORTIFY_SOURCE=2>
        $<$<CONFIG:Release>:-fno-omit-frame-pointer>
    )
    # CFI / shadow-stack on supported toolchains
    include(CheckCXXCompilerFlag)
    check_cxx_compiler_flag(-fcf-protection=full HAVE_CF_PROTECTION)
    if (HAVE_CF_PROTECTION)
        # Control-flow integrity: protects indirect calls/jumps against
        # ROP/JOP attacks. Intel CET on supported hardware.
        add_compile_options(-fcf-protection=full)
    endif()
    check_cxx_compiler_flag(-fstack-clash-protection HAVE_STACK_CLASH)
    if (HAVE_STACK_CLASH)
        # Detects stack clash via unmapped guard pages.
        add_compile_options(-fstack-clash-protection)
    endif()
    # Linker hardening
    add_link_options(
        -Wl,-z,relro,-z,now    # full RELRO: GOT read-only after load
        -Wl,-z,noexecstack     # NX stack
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
    src/lowlatency.cpp
    src/mock_backend.cpp
    src/hybrid_backend.cpp
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
set_target_properties(aboba PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    # Hide symbols by default so the C-ABI shared library doesn't leak
    # C++ mangled names. Symbols intended for export are tagged via
    # ABOBA_API (default visibility attribute).
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON
)
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

# --- Vulkan backend --------------------------------------------------------
if (ABOBA_ENABLE_VULKAN)
    find_package(Vulkan REQUIRED)

    target_sources(aboba PRIVATE src/vulkan_backend.cpp)
    target_compile_definitions(aboba PUBLIC ABOBA_ENABLE_VULKAN=1 ABOBA_HAS_VULKAN=1)
    target_link_libraries(aboba PUBLIC Vulkan::Vulkan)
    # SPIR-V headers are in src/vulkan_shaders_gen/
    target_include_directories(aboba PRIVATE src)

    message(STATUS "Aboba: Vulkan backend ENABLED")
else()
    message(STATUS "Aboba: Vulkan backend disabled. "
                    "Pass -DABOBA_ENABLE_VULKAN=ON to enable.")
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

    aboba_test_executable(aboba_hybrid_test  examples/hybrid_test.cpp)
    target_link_libraries(aboba_hybrid_test  PRIVATE Threads::Threads)

    if (ABOBA_ENABLE_VULKAN)
        aboba_test_executable(aboba_vulkan_test examples/vulkan_test.cpp)
    endif()

    # Pure C test of the C ABI. Built as C, linked against the shared
    # libaboba_c only — verifies that the ABI surface is truly C-callable.
    if (ABOBA_BUILD_SHARED_C)
        add_executable(aboba_c_abi_test examples/c_abi_test.c)
        # IMPORTANT: this test compiles as C, links against the C ABI
        # shared library, and has NO direct dependency on the C++ static
        # library. That ensures we're testing the actual ABI boundary.
        set_target_properties(aboba_c_abi_test PROPERTIES
            C_STANDARD 99 C_STANDARD_REQUIRED ON LINKER_LANGUAGE C)
        target_link_libraries(aboba_c_abi_test PRIVATE aboba_c m)
        target_include_directories(aboba_c_abi_test PRIVATE include)
        if (CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
            target_compile_options(aboba_c_abi_test PRIVATE
                -Wall -Wextra -Wpedantic
                -Wno-unused-result)
        endif()

        # Adversarial / fuzz-style paranoia regression suite. Linked
        # against the C ABI shared library (same boundary as a real
        # caller). Compiled as C++ because it uses std::vector for
        # convenience.
        add_executable(aboba_adversarial_test examples/adversarial_test.cpp)
        target_link_libraries(aboba_adversarial_test PRIVATE aboba_c)
        target_include_directories(aboba_adversarial_test PRIVATE include)
        if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
            target_compile_options(aboba_adversarial_test PRIVATE
                -Wno-conversion -Wno-sign-conversion
                -Wno-double-promotion -Wno-old-style-cast)
        endif()
    endif()

    aboba_test_executable(aboba_inspect examples/inspect.cpp)
    aboba_test_executable(aboba_info    examples/info.cpp)

    if (ABOBA_ENABLE_REALTIME)
        add_executable(aboba_realtime_voice examples/realtime_voice.cpp)
        target_link_libraries(aboba_realtime_voice PRIVATE aboba)
    endif()
endif()


if (ABOBA_BUILD_SHARED_C)
    # The shared C-ABI library. This is what game engines and other
    # languages link against via dlopen / LoadLibrary / dotnet / etc.
    #
    # Why static linkage of the C++ library inside:
    #   We bundle libaboba.a (PIC) into this .so so the C-ABI .so is
    #   fully self-contained — engine integrators don't need to ship
    #   libaboba.a separately.
    add_library(aboba_c SHARED
        src/aboba_c.cpp
    )
    target_link_libraries(aboba_c PRIVATE aboba)
    target_include_directories(aboba_c PUBLIC include)

    # Default visibility hidden — we control what's exported via ABOBA_API.
    set_target_properties(aboba_c PROPERTIES
        CXX_VISIBILITY_PRESET hidden
        VISIBILITY_INLINES_HIDDEN ON
        OUTPUT_NAME "aboba_c")

    # Compatibility: define ABOBA_C_BUILDING when compiling this .cpp so
    # the ABOBA_API macro resolves to dllexport on Windows.
    target_compile_definitions(aboba_c PRIVATE ABOBA_C_BUILDING)

    install(TARGETS aboba_c
            ARCHIVE DESTINATION lib
            LIBRARY DESTINATION lib
            RUNTIME DESTINATION bin)
    install(FILES include/aboba_c.h DESTINATION include)
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

# --- VST3 plugin -----------------------------------------------------------
if (ABOBA_BUILD_VST3)
    add_subdirectory(vst3_wrapper)
endif()
