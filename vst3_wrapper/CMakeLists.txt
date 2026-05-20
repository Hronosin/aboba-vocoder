# SPDX-License-Identifier: GPL-3.0-or-later
#
# Aboba VST3 wrapper build.
#
# This is a separate sub-project intentionally — it requires the official
# Steinberg VST3 SDK which is not bundled here. To build:
#
#   1. Download the VST3 SDK from https://www.steinberg.net/developers/
#      (or `git clone https://github.com/steinbergmedia/vst3sdk.git`).
#   2. Set the VST3_SDK_PATH environment variable or CMake cache.
#   3. Configure with:
#        cmake -S . -B build -DABOBA_BUILD_VST3=ON \
#                            -DVST3_SDK_PATH=/path/to/vst3sdk
#   4. Build:
#        cmake --build build --target Aboba
#
# The output is a platform-specific bundle:
#   * Linux:   build/Aboba.vst3/Contents/x86_64-linux/Aboba.so
#   * macOS:   build/Aboba.vst3/Contents/MacOS/Aboba
#   * Windows: build/Aboba.vst3/Contents/x86_64-win/Aboba.vst3
#
# Install paths (where DAWs look):
#   * Linux:   ~/.vst3/ or /usr/lib/vst3/
#   * macOS:   ~/Library/Audio/Plug-Ins/VST3/ or /Library/Audio/Plug-Ins/VST3/
#   * Windows: %COMMONPROGRAMFILES%\VST3\
#
# License compatibility:
#   The Steinberg VST3 SDK is dual-licensed (GPLv3 OR Steinberg's
#   proprietary). Since Aboba is GPLv3, we use the SDK under GPLv3 terms.
#   Anyone redistributing this plugin must do so under GPLv3.

cmake_minimum_required(VERSION 3.20)

# This file assumes it's being included from the top-level CMakeLists when
# ABOBA_BUILD_VST3=ON. If standalone, set up minimal project context.
if (NOT TARGET aboba)
    project(aboba_vst3 LANGUAGES CXX)
    set(CMAKE_CXX_STANDARD 17)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
    message(FATAL_ERROR
        "Aboba VST3 wrapper requires the parent project. Configure from the "
        "top-level directory with -DABOBA_BUILD_VST3=ON.")
endif()

# ---- Locate the VST3 SDK -----------------------------------------------
if (NOT DEFINED VST3_SDK_PATH)
    if (DEFINED ENV{VST3_SDK_PATH})
        set(VST3_SDK_PATH "$ENV{VST3_SDK_PATH}")
    else()
        message(FATAL_ERROR
            "Aboba VST3: VST3_SDK_PATH not set. Pass "
            "-DVST3_SDK_PATH=/path/to/vst3sdk to cmake, or set the env var.")
    endif()
endif()

if (NOT EXISTS "${VST3_SDK_PATH}/pluginterfaces/vst/ivstaudioprocessor.h")
    message(FATAL_ERROR
        "Aboba VST3: VST3_SDK_PATH='${VST3_SDK_PATH}' does not look like a "
        "VST3 SDK directory (no pluginterfaces/vst/ivstaudioprocessor.h).")
endif()

message(STATUS "Aboba VST3: using SDK at ${VST3_SDK_PATH}")

# Bring in the SDK. The official VST3 SDK ships its own CMakeLists which
# defines targets like `sdk` and `base`. We add it as a subdirectory and
# link against those targets.
set(SMTG_RUN_VST_VALIDATOR OFF CACHE BOOL "" FORCE)
set(SMTG_CREATE_PLUGIN_LINK OFF CACHE BOOL "" FORCE)
set(SMTG_ENABLE_VSTGUI_SUPPORT OFF CACHE BOOL "" FORCE)
set(SMTG_RENAME_ASSERT OFF CACHE BOOL "" FORCE)
add_subdirectory("${VST3_SDK_PATH}" "${CMAKE_BINARY_DIR}/vst3sdk" EXCLUDE_FROM_ALL)

# ---- Aboba VST3 plugin target -----------------------------------------
set(plugin_target Aboba)

set(plugin_sources
    AbobaVST3Processor.cpp
    AbobaVST3Processor.h
    AbobaVST3Controller.cpp
    AbobaVST3Controller.h
    AbobaVST3Factory.cpp
)

# `smtg_add_vst3plugin` is the SDK's helper that creates a properly-shaped
# bundle. If for some reason it's unavailable, fall back to a plain shared
# library — DAWs are typically forgiving on Linux.
if (COMMAND smtg_add_vst3plugin)
    smtg_add_vst3plugin(${plugin_target} ${plugin_sources})
    target_link_libraries(${plugin_target} PRIVATE sdk aboba)
else()
    add_library(${plugin_target} MODULE ${plugin_sources})
    target_link_libraries(${plugin_target} PRIVATE sdk aboba)
    set_target_properties(${plugin_target} PROPERTIES
        SUFFIX ".vst3"
        PREFIX "")
endif()

target_include_directories(${plugin_target} PRIVATE
    "${VST3_SDK_PATH}"
    "${CMAKE_CURRENT_SOURCE_DIR}"
    "${CMAKE_SOURCE_DIR}/include")

# Modest warning config — the SDK headers themselves are not pristine, so
# we relax aggressive checks for the wrapper translation units.
if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(${plugin_target} PRIVATE
        -Wall -Wextra
        -Wno-unused-parameter
        -Wno-multichar
        -Wno-deprecated-declarations)
endif()

# Ensure aboba is built PIC (we already set that in top-level)
set_target_properties(${plugin_target} PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON)

message(STATUS "Aboba VST3: plugin target '${plugin_target}' configured")
