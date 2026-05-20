# Aboba Vocoder — VST3 Plugin

A VST3 plugin wrapper around the Aboba voice processing framework. Use it
in any modern DAW: REAPER, Ableton Live, FL Studio, Cubase, Studio One,
Bitwig, Logic Pro, Cakewalk, etc.

## License

GPLv3. The Steinberg VST3 SDK is used under its GPLv3 dual license. Any
plugin binary built from this code is distributed under GPLv3 — that
means you can use it commercially (DAW projects, songs, livestreams) but
you cannot statically link this code into closed-source software.

## Pre-built binaries

This repository does NOT ship pre-built `.vst3` files. You must build
them yourself (see below). This is because:

1. The VST3 SDK requires a one-time download from Steinberg.
2. Cross-compiling for all three OSes from a single machine is messy.
3. GPL distribution responsibilities are clearer when each user builds
   from source.

## Build instructions

### Step 1 — Get the VST3 SDK

```bash
git clone https://github.com/steinbergmedia/vst3sdk.git
cd vst3sdk
git submodule update --init --recursive
export VST3_SDK_PATH=$PWD
cd ..
```

(The SDK is roughly 50 MB.)

### Step 2 — Build the Aboba VST3 plugin

```bash
cd aboba-vocoder
cmake -S . -B build \
    -DABOBA_BUILD_VST3=ON \
    -DVST3_SDK_PATH=$VST3_SDK_PATH \
    -DCMAKE_BUILD_TYPE=Release
cmake --build build --target Aboba -j
```

The resulting bundle is at `build/VST3/Aboba.vst3/` (or similar — check
the build output).

### Step 3 — Install

Copy or symlink the `.vst3` bundle to your DAW's plugin folder:

| OS | Default user folder |
|---|---|
| Linux | `~/.vst3/` |
| macOS | `~/Library/Audio/Plug-Ins/VST3/` |
| Windows | `%COMMONPROGRAMFILES%\VST3\` |

```bash
# Linux example:
mkdir -p ~/.vst3
cp -r build/VST3/Aboba.vst3 ~/.vst3/
```

Then rescan plugins in your DAW.

## Parameters

| Name | Range | Default | Notes |
|---|---|---|---|
| Pitch | -24 .. +24 st | 0 | Real-time pitch shift |
| Formant | -12 .. +12 st | 0 | Anti-helium formant shift |
| Character | 14 presets | Neutral | Voice presets (DeepMale, AnimeGirl, Robot, etc.) |
| Reverb | on/off | off | Algorithmic reverb |
| Reverb Wet | 0..100% | 20% | Reverb mix level |
| Autotune | on/off | off | Snap pitch to musical scale |
| Scale | 8 scales | Chromatic | Major, Minor, Pentatonic, Blues, etc. |
| Tune Strength | 0..100% | 100% | T-Pain effect at 100%, natural at lower |
| Bypass | on/off | off | Honored by the host's bypass button |

All parameters are automatable.

## Architecture

The plugin is a thin VST3 wrapper around the `aboba::VoicePipeline` C++
object:

```
DAW audio thread
    ↓
Aboba VST3 Processor (catches VST3 boilerplate, drains param changes)
    ↓
aboba::VoicePipeline (the actual DSP)
    ↓
Backend (CPU/FFTW or HIP/AMD GPU)
```

Stereo I/O is downmixed to mono before processing (Aboba is mono-only)
and the result is duplicated across L/R channels. If you need true
stereo processing, instantiate the plugin twice on separate mono buses.

## Latency

The processor reports `latency_samples()` from the pipeline to the DAW.
The host will compensate for this automatically (delay other tracks).

At the default Balanced profile and 48 kHz, this is ~2048 samples = 42 ms.

If you need lower latency, modify `AbobaVST3Processor.cpp` to construct
the pipeline with `QualityProfile::Performance` (smaller FFT size,
~21 ms) or write a low-latency variant using the `LowLatencyPipeline`
class (sub-2 ms but reduced quality — see `aboba/lowlatency.hpp`).

## Stability policy

The Aboba VST3 processor uses the same "stability > speed" policy as the
C ABI bridge. If the pipeline throws an exception during process(), the
processor falls back to passthrough for that block rather than crashing
the DAW. The exception is silently swallowed; if you want logging, hook
into `pipeline_->vocoder_stats()` from a debug build.

## Building for other platforms

The VST3 wrapper compiles identically on Linux, macOS, and Windows. The
SDK's `smtg_add_vst3plugin` helper handles platform-specific bundle
generation. Just run the same CMake commands inside a suitable build
environment:

* Linux: gcc 11+ or clang 13+
* macOS: clang 14+ (Xcode 14+)
* Windows: MSVC 2019+

For cross-compilation, see CMake's standard toolchain file documentation.
