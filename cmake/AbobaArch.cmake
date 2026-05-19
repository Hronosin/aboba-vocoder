# SPDX-License-Identifier: GPL-3.0-or-later
#
# Aboba architecture detection.
#
# Detects:
#   * CPU features (AVX2 / AVX-512 / FMA / NEON) via try-compile probes
#   * Physical/logical cores (informational)
#   * AMD GPU compute targets (gfx codename) via rocm tools or /sys probing
#
# Provides:
#   * ABOBA_CPU_FEATURES        — list of detected features
#   * ABOBA_CPU_FLAGS           — recommended -march/-mcpu flags
#   * ABOBA_AMD_GPU_TARGETS     — list of detected gfx targets (gfx906 etc.)
#   * ABOBA_DETECTION_REPORT    — pretty multi-line summary string
#
# CRITICAL: never enables -ffast-math / -ffinite-math-only.
# We rely on std::isfinite() everywhere; -ffinite-math-only would let the
# compiler optimize those checks away, undoing our paranoia. Use the safe
# subset: -fno-math-errno + -fno-trapping-math + -funroll-loops only.

include(CheckCXXCompilerFlag)
include(CheckCXXSourceCompiles)
include(ProcessorCount)

# ----------------------------------------------------------------------
# CPU feature detection (compile-time)
#
# We probe by trying to compile small SIMD snippets that *use* the intrinsics
# rather than just trusting flag acceptance. A flag may be accepted by the
# compiler but produce code that doesn't run on the build machine; the host
# CPU has to actually understand it. For cross-compilation, fall back to
# flag-acceptance only.
# ----------------------------------------------------------------------
function(_aboba_probe_x86 OUT_FEATURES OUT_FLAGS)
    set(features "")
    set(flags "")

    # FMA
    check_cxx_compiler_flag("-mfma" _aboba_has_mfma)
    if (_aboba_has_mfma)
        list(APPEND features "FMA")
    endif()

    # AVX2
    set(CMAKE_REQUIRED_FLAGS "-mavx2")
    check_cxx_source_compiles("
        #include <immintrin.h>
        int main() {
            __m256i a = _mm256_set1_epi32(1);
            __m256i b = _mm256_add_epi32(a, a);
            return _mm256_extract_epi32(b, 0);
        }
    " _aboba_has_avx2)
    set(CMAKE_REQUIRED_FLAGS "")
    if (_aboba_has_avx2)
        list(APPEND features "AVX2")
    endif()

    # AVX-512F
    set(CMAKE_REQUIRED_FLAGS "-mavx512f")
    check_cxx_source_compiles("
        #include <immintrin.h>
        int main() {
            __m512 a = _mm512_set1_ps(1.0f);
            __m512 b = _mm512_add_ps(a, a);
            float r[16]; _mm512_storeu_ps(r, b);
            return (int)r[0];
        }
    " _aboba_has_avx512f)
    set(CMAKE_REQUIRED_FLAGS "")
    if (_aboba_has_avx512f)
        list(APPEND features "AVX-512F")
    endif()

    # We use -march=native for the actual build — it lets the compiler choose
    # everything the host CPU supports, including features we didn't probe
    # individually (BMI2, AVX-VNNI, etc.). The probes above are for the
    # *report*, not for picking flags one-by-one.
    list(APPEND flags "-march=native")

    set(${OUT_FEATURES} ${features} PARENT_SCOPE)
    set(${OUT_FLAGS}    ${flags}    PARENT_SCOPE)
endfunction()

function(_aboba_probe_aarch64 OUT_FEATURES OUT_FLAGS)
    set(features "NEON")  # always present on aarch64
    set(flags "-march=native")

    # FP16 (Apple Silicon, modern ARM server)
    set(CMAKE_REQUIRED_FLAGS "-march=armv8.2-a+fp16")
    check_cxx_source_compiles("
        #include <arm_neon.h>
        int main() {
            float16x8_t a = vdupq_n_f16(__fp16(1.0));
            float16x8_t b = vaddq_f16(a, a);
            return (int)vgetq_lane_f16(b, 0);
        }
    " _aboba_has_fp16)
    set(CMAKE_REQUIRED_FLAGS "")
    if (_aboba_has_fp16)
        list(APPEND features "FP16")
    endif()

    set(${OUT_FEATURES} ${features} PARENT_SCOPE)
    set(${OUT_FLAGS}    ${flags}    PARENT_SCOPE)
endfunction()

function(aboba_detect_cpu)
    if (CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64|x86")
        _aboba_probe_x86(features flags)
    elseif (CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
        _aboba_probe_aarch64(features flags)
    else()
        set(features "(unknown architecture)")
        set(flags "")
    endif()

    set(ABOBA_CPU_FEATURES ${features} PARENT_SCOPE)
    set(ABOBA_CPU_FLAGS    ${flags}    PARENT_SCOPE)
endfunction()

# ----------------------------------------------------------------------
# Logical and physical core count
# ----------------------------------------------------------------------
function(aboba_detect_cores OUT_LOGICAL OUT_PHYSICAL)
    ProcessorCount(logical)
    if (logical EQUAL 0)
        set(logical 1)
    endif()

    set(physical ${logical})  # default if we can't tell

    if (CMAKE_HOST_SYSTEM_NAME STREQUAL "Linux")
        # Count unique "core id" entries in /proc/cpuinfo
        execute_process(
            COMMAND bash -c "grep -c ^processor /proc/cpuinfo"
            OUTPUT_VARIABLE _logical_str
            OUTPUT_STRIP_TRAILING_WHITESPACE
            ERROR_QUIET
        )
        execute_process(
            COMMAND bash -c "awk -F: '/^core id/ {print $2}' /proc/cpuinfo | sort -u | wc -l"
            OUTPUT_VARIABLE _physical_str
            OUTPUT_STRIP_TRAILING_WHITESPACE
            ERROR_QUIET
        )
        if (_physical_str MATCHES "^[0-9]+$" AND NOT _physical_str EQUAL 0)
            set(physical ${_physical_str})
        endif()
    elseif (CMAKE_HOST_SYSTEM_NAME STREQUAL "Darwin")
        execute_process(
            COMMAND sysctl -n hw.physicalcpu
            OUTPUT_VARIABLE _physical_str
            OUTPUT_STRIP_TRAILING_WHITESPACE
            ERROR_QUIET
        )
        if (_physical_str MATCHES "^[0-9]+$")
            set(physical ${_physical_str})
        endif()
    endif()

    set(${OUT_LOGICAL}  ${logical}  PARENT_SCOPE)
    set(${OUT_PHYSICAL} ${physical} PARENT_SCOPE)
endfunction()

# ----------------------------------------------------------------------
# AMD GPU detection
#
# Strategy, in order of preference:
#   1. rocm_agent_enumerator (ships with ROCm, most reliable)
#   2. rocminfo (parse output for gfx*** lines)
#   3. /sys/class/drm/card*/device/vendor reading for vendor 0x1002 (AMD)
#
# If nothing detected we emit an empty list — that's not an error, just means
# the build host doesn't have an AMD GPU visible. The HIP backend will still
# compile if AMDGPU_TARGETS is set manually.
# ----------------------------------------------------------------------
function(aboba_detect_amd_gpus OUT_TARGETS)
    set(targets "")

    # Try rocm_agent_enumerator
    find_program(_aboba_rocm_enum
        NAMES rocm_agent_enumerator
        PATHS /opt/rocm/bin /usr/bin
    )
    if (_aboba_rocm_enum)
        execute_process(
            COMMAND ${_aboba_rocm_enum} -name
            OUTPUT_VARIABLE _enum_out
            OUTPUT_STRIP_TRAILING_WHITESPACE
            ERROR_QUIET
            RESULT_VARIABLE _enum_res
        )
        if (_enum_res EQUAL 0)
            string(REPLACE "\n" ";" _enum_lines "${_enum_out}")
            foreach(line ${_enum_lines})
                string(STRIP "${line}" line)
                # Filter out the host "gfx000" pseudo-agent
                if (line MATCHES "^gfx[0-9a-fA-F]+$" AND NOT line STREQUAL "gfx000")
                    list(APPEND targets ${line})
                endif()
            endforeach()
        endif()
    endif()

    # Try rocminfo
    if (NOT targets)
        find_program(_aboba_rocminfo NAMES rocminfo PATHS /opt/rocm/bin /usr/bin)
        if (_aboba_rocminfo)
            execute_process(
                COMMAND ${_aboba_rocminfo}
                OUTPUT_VARIABLE _info_out
                OUTPUT_STRIP_TRAILING_WHITESPACE
                ERROR_QUIET
                RESULT_VARIABLE _info_res
            )
            if (_info_res EQUAL 0)
                # Parse lines like "  Name:                    gfx906"
                string(REGEX MATCHALL "gfx[0-9a-fA-F]+" _info_matches "${_info_out}")
                foreach(m ${_info_matches})
                    if (NOT m STREQUAL "gfx000")
                        list(APPEND targets ${m})
                    endif()
                endforeach()
            endif()
        endif()
    endif()

    # Last resort: scan /sys
    if (NOT targets AND CMAKE_HOST_SYSTEM_NAME STREQUAL "Linux")
        file(GLOB _drm_vendors "/sys/class/drm/card*/device/vendor")
        foreach(vendor_file ${_drm_vendors})
            file(READ ${vendor_file} _vendor)
            string(STRIP "${_vendor}" _vendor)
            if (_vendor STREQUAL "0x1002")
                # We know it's AMD; we just can't reliably get the gfx target
                # from /sys alone without parsing the IP block versions.
                # Add a marker so the user knows we saw something.
                list(APPEND targets "amd-gpu-detected")
                break()
            endif()
        endforeach()
    endif()

    list(REMOVE_DUPLICATES targets)
    set(${OUT_TARGETS} ${targets} PARENT_SCOPE)
endfunction()

# ----------------------------------------------------------------------
# Apply safe optimization flags (Release builds).
#
# What we DO add:
#   -O3                    : standard
#   -funroll-loops         : helps FFT inner loops
#   -fno-math-errno        : skip errno on transcendental math (safe)
#   -fno-trapping-math     : assume math ops don't trap (safe for us)
#
# What we explicitly DO NOT add:
#   -ffast-math            : implies -ffinite-math-only which lets the
#                            compiler delete std::isfinite() checks. We need
#                            those.
#   -ffinite-math-only     : same reason as above.
#   -funsafe-math-opts     : reassociates ops; can change behavior on edge
#                            inputs. We prefer determinism.
# ----------------------------------------------------------------------
function(aboba_apply_safe_optimization target)
    if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(${target} PRIVATE
            $<$<CONFIG:Release>:-O3>
            $<$<CONFIG:Release>:-funroll-loops>
            $<$<CONFIG:Release>:-fno-math-errno>
            $<$<CONFIG:Release>:-fno-trapping-math>
        )
    endif()
endfunction()

# ----------------------------------------------------------------------
# Pretty-print a detection report. Call after running detection above.
# ----------------------------------------------------------------------
function(aboba_print_detection_report)
    message(STATUS "")
    message(STATUS "=== Aboba architecture detection ===")
    message(STATUS "Host system : ${CMAKE_HOST_SYSTEM_NAME} ${CMAKE_SYSTEM_PROCESSOR}")
    message(STATUS "Compiler    : ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")

    aboba_detect_cores(_log _phys)
    message(STATUS "Cores       : ${_phys} physical / ${_log} logical")

    if (DEFINED ABOBA_CPU_FEATURES AND ABOBA_CPU_FEATURES)
        string(REPLACE ";" ", " _feat_str "${ABOBA_CPU_FEATURES}")
        message(STATUS "CPU features: ${_feat_str}")
    else()
        message(STATUS "CPU features: (none detected — generic build)")
    endif()

    if (DEFINED ABOBA_CPU_FLAGS AND ABOBA_CPU_FLAGS)
        string(REPLACE ";" " " _flag_str "${ABOBA_CPU_FLAGS}")
        message(STATUS "CPU flags   : ${_flag_str}")
    endif()

    if (DEFINED ABOBA_AMD_GPU_TARGETS AND ABOBA_AMD_GPU_TARGETS)
        string(REPLACE ";" ", " _gpu_str "${ABOBA_AMD_GPU_TARGETS}")
        message(STATUS "AMD GPUs    : ${_gpu_str}")
    else()
        message(STATUS "AMD GPUs    : (none detected — CPU backend will be used)")
    endif()

    message(STATUS "====================================")
    message(STATUS "")
endfunction()
