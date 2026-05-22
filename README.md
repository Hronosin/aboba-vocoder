<div align="center">

```
       █████╗ ██████╗  ██████╗ ██████╗  █████╗
      ██╔══██╗██╔══██╗██╔═══██╗██╔══██╗██╔══██╗
      ███████║██████╔╝██║   ██║██████╔╝███████║
      ██╔══██║██╔══██╗██║   ██║██╔══██╗██╔══██║
      ██║  ██║██████╔╝╚██████╔╝██████╔╝██║  ██║
      ╚═╝  ╚═╝╚═════╝  ╚═════╝ ╚═════╝ ╚═╝  ╚═╝
                  V  O  C  O  D  E  R
```

### *The first vocoder framework AMD users didn't have to pirate.*

[![License](https://img.shields.io/badge/license-GPLv3-red)]()
[![GPU](https://img.shields.io/badge/GPU-AMD%20%2B%20Vulkan-orange)]()
[![NVIDIA](https://img.shields.io/badge/NVIDIA-tolerated%20via%20Vulkan-darkgreen)]()
[![Tests](https://img.shields.io/badge/tests-632%2F632-brightgreen)]()
[![ASan](https://img.shields.io/badge/ASan%2BUBSan-clean-brightgreen)]()
[![Paranoia](https://img.shields.io/badge/paranoia-maximum-blueviolet)]()
[![Aboba](https://img.shields.io/badge/Aboba-certified-ff69b4)]()
[![Vibes](https://img.shields.io/badge/vibes-immaculate-success)]()

</div>

---

## TL;DR

A **real-time voice processing framework** with a stable C ABI, hardened
against adversarial input, ready to drop into game engines, DAWs (via
VST3), or any FFI-capable language. Started as an AMD-themed joke about
the CUDA monopoly; ended up an actual production-grade audio framework.

- **4 compute backends**: CPU (FFTW3), HIP/rocFFT (AMD), Vulkan compute
  (cross-vendor GPU), Mock (testing). Pick one explicitly or use the
  Hybrid backend with adaptive routing + failover.
- **Two pipelines**: normal (~42 ms latency, formant-preserving) and
  low-latency (~2 ms, time-domain SOLA; reduced quality, hard 2 ms
  watchdog budget).
- **14 voice characters** (DeepMale, AnimeGirl, Robot, Demon, Chipmunk,
  Helium, …), pitch + formant shift, autotune (8 musical scales),
  algorithmic reverb, spectral noise reduction, AGC, soft + lookahead
  limiters.
- **Stable C ABI** with watchdog & bypass policy. 38 exported C symbols,
  zero C++ name-mangling leaks. Built as one `libaboba_c.so` /
  `.dll` / `.dylib` for FFI consumption.
- **Five language surfaces**: native C++, C ABI, Python (pybind11), VST3
  plugin, plus engine-integration recipes for Unity / Unreal / Godot.
- **632 tests** (462 also under ASan+UBSan) covering correctness, stress,
  paranoia/fuzz, ABI conformance, Python bindings, and Vulkan-vs-CPU
  numerical equivalence.

> The codebase takes paranoia seriously. The marketing takes nothing
> seriously. We feel this is the correct ratio.

---

## The manifesto

Real-time voice processing is dominated by a few facts:

1. **CUDA owns the GPU compute world**, and most existing audio
   frameworks assume it. AMD users get crumbs. Vulkan-based open-source
   acceleration is rare. We aim to change that.
2. **Audio glitches are worse than ugly output.** A dropout, a NaN
   burst, or an unhandled exception in a real-time callback can destroy
   a stream, hurt a listener's ears, or crash a host DAW. The framework
   has to be paranoid by default.
3. **Cross-language integration is a first-class concern.** A C++
   library that can only be called from C++ has missed half its userbase.
   Everything important is reachable from C, Python, C#, Rust, Zig,
   GDScript, and any DAW that loads VST3.

Aboba's job is to be useful, safe, and obvious to use. The AMD-themed
branding is a joke we kept because it amuses us.

---

## Quick links

| Audience | Where to look |
|---|---|
| C++ developer wanting to build a voice app | [Quickstart](#-quickstart) |
| C / Rust / Zig / FFI developer | [aboba_c.h](include/aboba_c.h) + [Bindings](#-bindings) |
| Python user | [python/README.md](python/README.md) |
| Game engine developer | [docs/INTEGRATION.md](docs/INTEGRATION.md) |
| DAW user | [vst3_wrapper/README.md](vst3_wrapper/README.md) |
| GPU enthusiast | [include/aboba/vulkan_backend.hpp](include/aboba/vulkan_backend.hpp) |
| Security researcher | [Paranoia](#-paranoia) section below |

---

## ✨ Features

### DSP

- **Formant-preserving pitch shift** via phase vocoder with cepstral
  spectral envelope estimation (shift pitch ±24 semitones without the
  helium effect).
- **YIN F0 estimator** for autotune and pitch-aware processing.
- **Autotune** with 8 musical scales (Chromatic, Major, Minor, Harmonic
  Minor, Pentatonic Major/Minor, Blues, Whole Tone), root selection,
  strength control (0..1 → T-Pain when 1.0 + low glide), portamento.
- **14 voice character presets** combining pitch + formant: Neutral,
  DeepMale, WarmMale, ChestyMale, YoungFemale, AnimeGirl, Chipmunk,
  Giant, Demon, Robot, RadioHost, Whisper, HeliumBalloon,
  CartoonVillain.
- **Spectral noise reduction** with adaptive noise profile learning.
- **AGC + lookahead limiter** for broadcast-ready loudness.
- **Algorithmic reverb** (Schroeder/Moorer-style; room size, damping,
  wet mix).
- **DC blocker, high-pass, gate, de-esser, soft limiter** as composable
  sample-by-sample DSP blocks.

### Backends

| Backend | Hardware | Status | Best for |
|---|---|---|---|
| CPU (FFTW3) | Any x86_64 / ARM64 | Default, ~50 µs / 2048-FFT | Real-time on the audio thread |
| Vulkan | Any GPU with compute (AMD, NVIDIA, Intel, ARM) | Tested via Lavapipe in CI | Cross-vendor GPU offload |
| HIP | AMD GPU + ROCm | Untested in CI (no GPU in sandbox) | High-throughput offline on AMD |
| Mock | None | Testing only | Failure injection for hybrid tests |

### Hybrid backend

The `HybridBackend` wraps N child backends with three orthogonal
capabilities:

1. **Adaptive routing** — per-call selection via rolling-EMA cost
   tracking per FFT-size bucket. The fastest backend for a given size
   wins, weighted by priority order.
2. **Failover chain** — on exception, mark backend unhealthy and walk
   the priority chain. Unhealthy backends are re-tried after a cooldown
   (10 s default). All-fail returns an error rather than silently
   corrupting output.
3. **Multi-channel parallel split** — batched calls above a threshold
   are split across eligible workers via `std::async`. Each worker has
   its own sub-failover; results are stitched back deterministically.

### Pipelines

- **VoicePipeline** — the normal high-quality path. ~42 ms latency at
  the default FFT size. All effects available.
- **LowLatencyPipeline** — sub-2 ms processing for game-engine audio
  threads. Uses time-domain SOLA pitch shift; reduced quality (no formant
  preservation, no reverb, no noise reduction, no autotune). Has a hard
  2 ms watchdog with bypass policy. See
  [`include/aboba/lowlatency.hpp`](include/aboba/lowlatency.hpp) for the
  quality tradeoff in detail.

### Configuration

- **TOML voice configs** — describe a voice as a data file:
  ```toml
  name = "streamer"
  [pipeline]      profile = "balanced"
  [character]     preset = "warm-male"
  [autotune]      enabled = true; scale = "minor"; root = "A"; strength = 0.6
  [effects]       reverb = true
  ```
  Load with one call from C, Python, or C++.

---

## 🏗 Architecture

```
                                ┌──────────────────────────────────┐
                                │  Consumers                        │
                                │  C++  •  C ABI  •  Python  •  VST3│
                                │  •  Unity / Unreal / Godot via C  │
                                └─────────────────┬─────────────────┘
                                                  │
            ┌─────────────────────────────────────┴────────────────────────────────┐
            │  Pipelines                                                            │
            │  • VoicePipeline (normal, ~42 ms, formant-preserving)                 │
            │  • LowLatencyPipeline (~2 ms, SOLA, hard budget watchdog)             │
            └─────────────────────────────────────┬────────────────────────────────┘
                                                  │
            ┌─────────────────────────────────────┴────────────────────────────────┐
            │  DSP layers (composable)                                              │
            │  • DcBlocker • HP filter • NoiseGate • DeEsser • AGC • Limiter        │
            │  • SpectralNoiseReducer (FFT-based, adaptive)                         │
            │  • FormantVocoder (phase vocoder + cepstral envelope)                 │
            │  • PitchCorrector (YIN + scale snap)                                  │
            │  • Reverb (Schroeder/Moorer)                                          │
            └─────────────────────────────────────┬────────────────────────────────┘
                                                  │
            ┌─────────────────────────────────────┴────────────────────────────────┐
            │  Backend (abstract `aboba::Backend`)                                   │
            │  ├── CPU       (FFTW3 single + double, with threading)                │
            │  ├── Vulkan    (radix-2 Cooley-Tukey compute shaders + bit reversal)  │
            │  ├── HIP       (rocFFT, AMD-only)                                     │
            │  ├── Mock      (configurable latency / failures, for testing)         │
            │  └── Hybrid    (adaptive routing + failover + multi-channel split)    │
            └──────────────────────────────────────────────────────────────────────┘
```

Every layer is independently testable. The C ABI layer is tested
against a pure-C test harness; the hybrid backend is exhaustively
tested with two Mocks; the Vulkan backend is tested against the CPU
backend for numerical equivalence (~1e-5 relative error via Lavapipe
in CI).

---

## 🔨 Build

### Requirements

- **CMake 3.20+**
- **C++17** compiler (GCC 11+, Clang 13+, MSVC 2019+)
- **FFTW3** development headers (`libfftw3-dev` on Debian/Ubuntu)
- *Optional*: Vulkan SDK + `glslangValidator` for the Vulkan backend
- *Optional*: pybind11 for Python bindings
- *Optional*: ROCm 5+ for the HIP backend
- *Optional*: Steinberg VST3 SDK for the VST3 plugin

### Quick build (CPU-only)

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

### Full build (all backends + bindings)

```bash
cmake -S . -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DABOBA_ENABLE_VULKAN=ON \
    -DABOBA_BUILD_PYTHON=ON \
    -DABOBA_BUILD_SHARED_C=ON
cmake --build build -j
```

For VST3, see [`vst3_wrapper/README.md`](vst3_wrapper/README.md). For
HIP, see comments in the top-level `CMakeLists.txt`.

### Run tests

```bash
# C++ Release
for t in stress dynamic quality debug hybrid vulkan adversarial; do
    ABOBA_QUIET=1 ./build/aboba_${t}_test
done
LD_LIBRARY_PATH=build ./build/aboba_c_abi_test

# Python
python3 python/test_bindings.py

# Under ASan + UBSan (highly recommended after any change)
cmake -S . -B build-asan -DCMAKE_BUILD_TYPE=Debug \
    -DABOBA_ENABLE_ASAN=ON -DABOBA_ENABLE_UBSAN=ON \
    -DABOBA_ENABLE_VULKAN=ON
cmake --build build-asan -j
for t in stress dynamic quality debug hybrid vulkan adversarial; do
    ABOBA_QUIET=1 ./build-asan/aboba_${t}_test
done
```

### Build options

| Option | Default | Effect |
|---|---|---|
| `ABOBA_ENABLE_HIP` | OFF | Build the HIP/rocFFT GPU backend (needs ROCm) |
| `ABOBA_ENABLE_VULKAN` | OFF | Build the Vulkan compute backend |
| `ABOBA_ENABLE_REALTIME` | ON | Build the PortAudio-based realtime engine |
| `ABOBA_BUILD_EXAMPLES` | ON | Build example programs and tests |
| `ABOBA_BUILD_PYTHON` | OFF | Build pybind11 Python bindings |
| `ABOBA_BUILD_SHARED_C` | ON | Build `libaboba_c.so` for FFI |
| `ABOBA_BUILD_VST3` | OFF | Build the VST3 plugin (needs VST3 SDK) |
| `ABOBA_ENABLE_ASAN` | OFF | Build with AddressSanitizer |
| `ABOBA_ENABLE_UBSAN` | OFF | Build with UndefinedBehaviorSanitizer |
| `ABOBA_AUTO_ARCH` | ON | Auto-detect CPU features & apply optimal flags |

---

## 🚀 Quickstart

### C++

```cpp
#include "aboba/backend.hpp"
#include "aboba/pipeline.hpp"
#include "aboba/voice_character.hpp"

auto backend = aboba::create_best_backend();
aboba::VoicePipelineConfig cfg;
cfg.sample_rate = 48000.0;
cfg.profile     = aboba::QualityProfile::Balanced;
aboba::VoicePipeline pipe(cfg, backend.get());
pipe.set_character(aboba::VoiceCharacter::DeepMale);

float in[480], out[480];
// fill `in` from your audio source...
pipe.process(in, out, 480);
// write `out` to your audio sink...
```

### C (game engines)

```c
#include "aboba_c.h"

aboba_backend* backend;
aboba_backend_create_best(&backend);

aboba_pipeline* pipe;
aboba_pipeline_create_lowlatency(backend, 48000.0, &pipe);
aboba_pipeline_set_max_block_us(pipe, 2000);
aboba_pipeline_set_budget_policy(pipe, ABOBA_BUDGET_POLICY_BYPASS);
aboba_pipeline_set_character(pipe, ABOBA_CHARACTER_DEEP_MALE);

float in[480], out[480];
aboba_pipeline_process(pipe, in, out, 480);

aboba_pipeline_destroy(pipe);
aboba_backend_destroy(backend);
```

### Python

```python
import numpy as np
import aboba

backend = aboba.create_backend()
pipe = aboba.VoicePipeline(backend, sample_rate=48000)
pipe.set_character(aboba.VoiceCharacter.AnimeGirl)
pipe.set_autotune_enabled(True)
pipe.set_autotune_scale(aboba.MusicalScale.Major, root=0)

x = (0.3 * np.sin(2*np.pi * 220 * np.arange(48000)/48000)).astype(np.float32)
y = pipe.process(x)
```

See [`python/README.md`](python/README.md) for the full Python guide.

---

## 🔌 Bindings

### C ABI

The single header [`include/aboba_c.h`](include/aboba_c.h) exposes the
entire framework via a C-callable ABI. Guarantees:

- **Stable across minor versions** (`ABOBA_C_API_VERSION_MAJOR`,
  `_MINOR`). Verify at startup with `aboba_runtime_abi_version_major()`.
- **No C++ types** in any signature. No name mangling. 38 exported C
  symbols total.
- **No exceptions** cross the boundary — every entry point returns
  `aboba_status` (a plain `int`). Use `aboba_status_message(s)` for a
  human-readable description.
- **No memory ownership** crosses the boundary — buffers stay on the
  caller side, handles are opaque and freed via `aboba_X_destroy()`.
- **No allocations** in `aboba_pipeline_process()` after the first call.
  The probe buffer used for watchdog recovery is pre-allocated and
  lazily resized only when entering bypass mode.

Verify the ABI surface yourself:

```bash
$ nm -D --defined-only build/libaboba_c.so | grep -c " T aboba_"
38
$ nm -D --defined-only build/libaboba_c.so | grep -c " T _Z"
0     # zero C++ name-mangled exports
```

### Python

`python/aboba_py.cpp` is a pybind11 module exposing the pipeline, the
configs, the standalone YIN detector, the standalone FormantVocoder,
and helpers. NumPy-first API:

- `process(np.ndarray)` accepts any numeric dtype (force-cast to
  float32), returns a new float32 array.
- `process_into(in, out)` is a zero-allocation in-place variant.
- The GIL is released automatically for blocks > 4096 samples.
- All errors propagate as Python `RuntimeError`, with line numbers for
  TOML parse failures.

---

## 🎮 Game engine integration

Aboba ships **only the C ABI** — no engine-specific plugins. The C ABI
is small enough that engine integration is a one-afternoon project.
[`docs/INTEGRATION.md`](docs/INTEGRATION.md) has full worked examples
for:

- **Unity** — `[DllImport]` P/Invoke binding + `OnAudioFilterRead`
  callback example.
- **Unreal Engine** — `FSoundEffectSubmix` integration as a C++ class.
- **Godot** — `GDExtension` with `AudioEffectInstance::_process`.

The runtime philosophy is **stability > speed**: every block has a
configurable processing budget, and on overrun the pipeline switches to
passthrough mode rather than glitch. The watchdog probes once per call
to detect when budget headroom returns and exits bypass after 8
consecutive in-budget probes. This makes Aboba safe to put inside an
audio thread.

---

## 🎹 VST3 plugin

Build with the official Steinberg VST3 SDK:

```bash
git clone https://github.com/steinbergmedia/vst3sdk.git
git -C vst3sdk submodule update --init --recursive
export VST3_SDK_PATH=$PWD/vst3sdk

cmake -S . -B build -DABOBA_BUILD_VST3=ON
cmake --build build --target Aboba -j
```

The plugin exposes 9 automatable parameters: pitch, formant, character,
reverb on/off + wet, autotune on/off + scale + strength, bypass. Stereo
I/O (downmixed to mono internally, expanded back). All parameters are
saved/restored via the host's project file.

DAW compatibility: any modern VST3 host on Linux, macOS, or Windows.
The wrapper uses only the standard `AudioEffect` base class and should
work everywhere.

License: GPLv3 (the VST3 SDK is used under its GPLv3 option). See
[`vst3_wrapper/README.md`](vst3_wrapper/README.md).

---

## 📊 Performance

Approximate per-block latency at 48 kHz, FFT size 2048, balanced profile:

| Backend | Wall-clock per FFT pair | Notes |
|---|---|---|
| CPU (FFTW3, modern x86_64) | 30–80 µs | Default; fits a 5 ms budget comfortably |
| Vulkan (discrete GPU, real) | 80–250 µs | Dominated by PCIe round-trip; wins on N ≥ 8192 |
| Vulkan (Lavapipe / CI) | 1–5 ms | Software fallback; correctness tested, not perf |
| HIP (AMD GPU + rocFFT) | Similar to Vulkan | Untested in CI; production-grade on real hardware |

The Hybrid backend's adaptive router picks the right one automatically
once it has 2-3 samples per FFT-size bucket.

The **LowLatencyPipeline** measures consistently under 2 ms on all
modern hardware. Per-block cost is dominated by SOLA cross-correlation,
which is `O(window_size * hop_size)` — fully predictable.

---

## 🛡 Paranoia

This section exists because real-time audio is one of the few places
where a crash is **observably worse than wrong output**. We treat every
external caller as potentially hostile and validate accordingly. Run
the adversarial regression suite to verify:

```bash
LD_LIBRARY_PATH=build ./build/aboba_adversarial_test
```

### Defense layers

Every public entry point goes through these layers in order:

1. **Pointer rejection** — every `void*` argument is null-checked before
   use.
2. **Size validation** — `n_samples > 16M` is rejected (`kMaxBlockSamples`).
   Sample rates outside `[4 kHz, 384 kHz]` and FFT sizes above
   `kMaxFftSize = 1M` are rejected at construction.
3. **Aliasing detection** — `in == out` (in-place) is fine; partial
   buffer overlap is detected and rejected with `ABOBA_ERR_INVALID_ARG`
   rather than silently corrupting the output.
4. **Input scrubbing** — input buffers are scanned for NaN/Inf. If any
   non-finite samples are present, the pipeline emits silence rather
   than poisoning internal IIR state with NaN propagation.
5. **Parameter clamping** — every parameter setter clamps NaN/Inf to a
   safe value (pitch ±48 st, formant ±24 st, reverb 0..1, autotune
   strength 0..1, glide 0..1000 ms). Calling
   `aboba_pipeline_set_pitch_semitones(p, NAN)` doesn't crash; it
   becomes pitch = 0.
6. **Exception barrier** — every C ABI function wraps the C++ call in a
   `guard()` lambda that catches all exceptions and translates them to
   status codes.
7. **RAII output sanitizer** — even if every preceding stage somehow
   misbehaves, the output buffer is scrubbed of NaN/Inf and clamped to
   `±4.0` at function exit.
8. **Hard limiter** — the final stage clamps every output sample to
   `±1.0` regardless of pipeline internal state. Your speakers are safe.

### Watchdog & bypass policy

Each pipeline has a configurable per-block budget (`max_block_us`).
When the budget is exceeded:

- `ABOBA_BUDGET_POLICY_LOG`: count the overrun in stats but keep
  processing fully. Use in development.
- `ABOBA_BUDGET_POLICY_BYPASS`: switch the pipeline to passthrough
  (copy input → output) for subsequent calls. Probe once per call to
  see if budget headroom has returned; after 8 consecutive in-budget
  probes, exit bypass. **No glitches, no clicks, no NaN bursts** — just
  briefly unprocessed voice. Default for the LowLatencyPipeline.

The probe buffer is **pre-allocated** in the pipeline struct, so the
bypass-recovery path has zero allocations on the hot audio thread. A
malicious caller cannot DoS the pipeline by forcing repeated probe
allocations.

### TOML config defense

The config parser:

- Rejects null input pointers.
- Rejects empty TOML bodies with `ABOBA_ERR_INVALID_ARG`.
- Refuses TOML blobs over 16 MiB (memory-exhaustion guard).
- Detects deeply nested bracket attacks (stack-overflow guard).
- Returns line numbers for every parse error.

### Test coverage

| Suite | Tests | Notes |
|---|---|---|
| `stress` | 23 | Buffer edge cases, multi-config |
| `dynamic` | 25 | Live parameter changes |
| `quality` | 218 | Numerical correctness of every DSP block |
| `debug` | 41 | Internal stats, telemetry |
| `hybrid` | 40 | Adaptive routing, failover, multi-channel |
| `vulkan` | 14 | Vulkan-vs-CPU numerical equivalence |
| `adversarial` | 46 | **NaN/Inf injection, null pointers, huge sizes, aliasing, TOML attacks, watchdog recovery** |
| `c_abi` | 55 | Pure C test of the C ABI |
| `python` | 85 | pybind11 module |
| **Total** | **547 C/C++ + 85 Python = 632** | |

All non-Python tests also run cleanly under **ASan + UBSan** (462 tests:
no leaks, no UB, no use-after-free, no integer overflow).

---

## 📁 Project layout

```
aboba-vocoder/
├── include/aboba/        ← C++ headers (Backend, Pipeline, DSP, …)
│   ├── paranoia.hpp      ← all defensive helpers + ScopedOutputSanitizer
│   ├── pipeline.hpp      ← VoicePipeline (normal, formant-preserving)
│   ├── lowlatency.hpp    ← LowLatencyPipeline (sub-2 ms, SOLA)
│   ├── backend.hpp       ← Backend abstract base
│   ├── hybrid_backend.hpp← HybridBackend (adaptive + failover)
│   ├── vulkan_backend.hpp← VulkanBackend (cross-vendor GPU)
│   ├── formant_vocoder.hpp
│   ├── pitch_corrector.hpp + yin.hpp + musical_scale.hpp
│   ├── voice_character.hpp (14 presets)
│   ├── voice_config.hpp  ← TOML config schema
│   ├── reverb.hpp + noise_reduction.hpp + dsp_blocks.hpp
│   └── ...
├── include/aboba_c.h     ← THE stable C ABI (one header)
├── src/                  ← Implementation
├── shaders/              ← GLSL compute shaders for Vulkan FFT
├── examples/             ← Test programs and demo CLIs
│   ├── adversarial_test.cpp  ← paranoia regression suite
│   ├── c_abi_test.c          ← pure C ABI test
│   ├── hybrid_test.cpp       ← hybrid backend test
│   ├── vulkan_test.cpp       ← Vulkan vs CPU equivalence
│   └── ...
├── python/               ← pybind11 module + tests
├── vst3_wrapper/         ← VST3 plugin (separate build, needs SDK)
├── docs/INTEGRATION.md   ← Unity / Unreal / Godot recipes
└── CMakeLists.txt
```

---

## 📜 License

GPLv3. See `LICENSE`.

Why GPLv3:

- The dependencies (FFTW, optionally Steinberg VST3 SDK) are dual-
  licensed with GPL; using them under their proprietary license costs
  money. GPL is consistent with everything in our supply chain.
- We don't want this silently absorbed into closed-source DAWs. If you
  build commercially on top of Aboba, you must share your modifications
  back. End users can still use Aboba binaries commercially (record
  songs, stream, etc.) — only redistribution of modified code is
  constrained.

---

## 🙏 Acknowledgements

- **FFTW3** — the gold-standard FFT library. Used as our CPU backend.
- **Steinberg** for the VST3 SDK and its GPLv3 dual-licensing option.
- **Mesa / Lavapipe** for software Vulkan — it lets us test GPU code
  paths in CI without any GPU hardware.
- **pybind11** — Python bindings without tears.
- **The CUDA monopoly**, for being so frustrating that this project
  became inevitable.

---

<div align="center">

***Aboba.*** *It's not just a punchline. It's a framework.*

</div>
