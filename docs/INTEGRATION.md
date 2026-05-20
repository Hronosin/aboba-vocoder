# Aboba Vocoder — Game Engine Integration Guide

## Overview

`libaboba_c.so` (`.dll` on Windows, `.dylib` on macOS) is a **stable C ABI** shared library. Anything that can do C FFI can integrate Aboba:

* C / C++ engines (Unreal, custom)
* C# game runtimes (Unity, MonoGame, Godot Mono) via P/Invoke
* Rust via `extern "C"` bindings or `bindgen`
* Zig, D, Swift, etc.

We **deliberately do not ship** engine-specific wrappers. The C ABI is small (~38 functions), well documented, and stable — anyone competent at integrating a sound library can wire it up in an afternoon.

## ABI guarantees

* **No breaking changes** within a major version (`ABOBA_C_API_VERSION_MAJOR`).
* **No exceptions** cross the boundary — all functions return `aboba_status` (a plain `int`).
* **No C++ types** in any signature.
* **No memory ownership** crosses the boundary — buffers stay on the caller's side, handles are opaque and freed via `aboba_X_destroy()`.
* **No allocations** from inside `aboba_pipeline_process()` after construction.
* **Stability over speed**: the pipeline has a built-in watchdog that **bypasses to passthrough** if a block exceeds the budget, rather than glitch.

## Build & link

```bash
cmake -S . -B build -DABOBA_BUILD_SHARED_C=ON
cmake --build build -j
# Resulting library: build/libaboba_c.so
# Resulting header:  include/aboba_c.h
```

Verify only C-callable symbols are exported (no C++ name mangling leaks):

```bash
nm -D --defined-only build/libaboba_c.so | grep " T " | head
# All symbols should start with `aboba_`
```

## Quickstart — C

```c
#include "aboba_c.h"
#include <stdio.h>

int main(void) {
    aboba_backend* backend = NULL;
    if (aboba_backend_create_best(&backend) != ABOBA_OK) return 1;

    aboba_pipeline* pipe = NULL;
    aboba_pipeline_create(backend, 48000.0, ABOBA_PROFILE_BALANCED, &pipe);
    aboba_pipeline_set_character(pipe, ABOBA_CHARACTER_DEEP_MALE);

    float in[480], out[480];   /* 10ms blocks */
    /* ... fill `in` from your audio source ... */
    aboba_pipeline_process(pipe, in, out, 480);
    /* ... write `out` to your audio sink ... */

    aboba_pipeline_destroy(pipe);
    aboba_backend_destroy(backend);
    return 0;
}
```

Build:
```bash
cc -std=c99 -Iinclude my_game.c -L. -laboba_c -o my_game
```

## Real-time / low-latency mode

For audio threads where 2ms is the budget (Unity audio callback, Unreal SubmixFX, etc), use `aboba_pipeline_create_lowlatency()`:

```c
aboba_pipeline_create_lowlatency(backend, 48000.0, &pipe);
aboba_pipeline_set_max_block_us(pipe, 1500);  /* tighter than default */
aboba_pipeline_set_pitch_semitones(pipe, -2.0f);
```

The low-latency pipeline:
* **Skips the formant vocoder** (uses time-domain SOLA — quality is lower; see `aboba/lowlatency.hpp` for details)
* **Skips noise reduction, autotune, reverb** (all FFT-based)
* **Has built-in 2ms watchdog** by default
* Latency is `~window_size` samples (192 by default = ~4ms @ 48kHz)

Check `aboba/lowlatency.hpp` for the full quality tradeoff explanation.

## Watchdog & stability policy

Set the budget and policy *once* at startup:

```c
aboba_pipeline_set_max_block_us(pipe, 2000);
aboba_pipeline_set_budget_policy(pipe, ABOBA_BUDGET_POLICY_BYPASS);
```

Behavior:
* **`ABOBA_BUDGET_POLICY_LOG`**: a block exceeding budget is processed normally but the `bypassed_blocks` counter increments. Use this in development.
* **`ABOBA_BUDGET_POLICY_BYPASS`**: when a block exceeds budget, the pipeline switches to **passthrough** (copies input to output) for subsequent calls. It "probes" once per call to see if it can recover budget headroom; after 8 consecutive in-budget probes it returns to full processing. **No glitches, no clicks, no NaN — just unprocessed voice briefly.**

This is the "stability > speed" policy. Use it in production.

Read counters from your engine's stats UI:

```c
aboba_pipeline_stats st;
aboba_pipeline_get_stats(pipe, &st);
printf("blocks: %llu, bypassed: %llu (%.1f%%), last: %.0f us, p99: %.0f us\n",
    (unsigned long long)st.total_blocks,
    (unsigned long long)st.bypassed_blocks,
    100.0 * (double)st.bypassed_blocks / (double)st.total_blocks,
    (double)st.last_block_us,
    (double)st.p99_block_us);
```

## Unity (C# / P/Invoke)

```csharp
using System;
using System.Runtime.InteropServices;

public static class Aboba {
    private const string DLL = "aboba_c";   // libaboba_c.so / aboba_c.dll

    public enum Profile { Quality = 0, Balanced = 1, Performance = 2 }
    public enum Character { Neutral = 0, DeepMale = 1, /* ... */ AnimeGirl = 5 }

    [DllImport(DLL)] public static extern int aboba_backend_create_best(out IntPtr b);
    [DllImport(DLL)] public static extern void aboba_backend_destroy(IntPtr b);
    [DllImport(DLL)] public static extern int aboba_pipeline_create(
        IntPtr b, double sr, Profile p, out IntPtr pipe);
    [DllImport(DLL)] public static extern int aboba_pipeline_create_lowlatency(
        IntPtr b, double sr, out IntPtr pipe);
    [DllImport(DLL)] public static extern void aboba_pipeline_destroy(IntPtr pipe);
    [DllImport(DLL)] public static extern int aboba_pipeline_set_character(
        IntPtr pipe, Character c);
    [DllImport(DLL)] public static extern int aboba_pipeline_set_pitch_semitones(
        IntPtr pipe, float st);
    [DllImport(DLL)] public static extern int aboba_pipeline_process(
        IntPtr pipe, float[] input, float[] output, UIntPtr n);
}

// In your AudioBehaviour:
private IntPtr _backend, _pipe;

void Start() {
    Aboba.aboba_backend_create_best(out _backend);
    Aboba.aboba_pipeline_create_lowlatency(_backend, 48000.0, out _pipe);
    Aboba.aboba_pipeline_set_character(_pipe, Aboba.Character.DeepMale);
}

void OnAudioFilterRead(float[] data, int channels) {
    // Unity buffers are interleaved. Downmix and process.
    int n = data.Length / channels;
    float[] mono   = new float[n];
    float[] outBuf = new float[n];
    for (int i = 0; i < n; ++i) mono[i] = data[i * channels];
    Aboba.aboba_pipeline_process(_pipe, mono, outBuf, (UIntPtr)n);
    for (int i = 0; i < n; ++i)
        for (int c = 0; c < channels; ++c)
            data[i * channels + c] = outBuf[i];
}

void OnDestroy() {
    Aboba.aboba_pipeline_destroy(_pipe);
    Aboba.aboba_backend_destroy(_backend);
}
```

* Place `libaboba_c.so` / `aboba_c.dll` in `Assets/Plugins/<platform>/`
* For Unity ≥ 2021 with `il2cpp`, you may need `[MonoPInvokeCallback]` for any callbacks (this API has none, so you're fine)
* `OnAudioFilterRead` runs on a non-Unity thread — that's fine, Aboba handles are single-threaded but each `_pipe` is used only from this one thread

## Unreal Engine (C++)

In your `Build.cs`:

```csharp
PublicIncludePaths.Add(Path.Combine(ModuleDirectory, "ThirdParty/Aboba/include"));
PublicAdditionalLibraries.Add(Path.Combine(ModuleDirectory,
    "ThirdParty/Aboba/lib", "libaboba_c.so"));
```

In your `SubmixEffect`:

```cpp
extern "C" {
    #include "aboba_c.h"
}

class FAbobaSubmixEffect : public FSoundEffectSubmix {
    void OnInitData(const FSoundEffectSubmixInitData& InitData) override {
        aboba_backend_create_best(&Backend);
        aboba_pipeline_create_lowlatency(Backend, InitData.SampleRate, &Pipeline);
        aboba_pipeline_set_character(Pipeline, ABOBA_CHARACTER_DEEP_MALE);
    }

    void OnProcessAudio(const FSoundEffectSubmixInputData& In,
                       FSoundEffectSubmixOutputData& Out) override {
        const int32 NumFrames = In.NumFrames;
        aboba_pipeline_process(Pipeline, In.AudioBuffer->GetData(),
                                         Out.AudioBuffer->GetData(),
                                         (size_t)NumFrames);
    }

    void OnPresetChanged() override { /* re-apply settings */ }

    aboba_backend* Backend = nullptr;
    aboba_pipeline* Pipeline = nullptr;

    virtual ~FAbobaSubmixEffect() override {
        aboba_pipeline_destroy(Pipeline);
        aboba_backend_destroy(Backend);
    }
};
```

## Godot (GDExtension)

In your `*.cpp`:

```cpp
extern "C" { #include "aboba_c.h" }
#include <godot_cpp/classes/audio_effect_instance.hpp>

class AbobaEffectInstance : public godot::AudioEffectInstance {
    GDCLASS(AbobaEffectInstance, AudioEffectInstance);
    aboba_backend* backend_ = nullptr;
    aboba_pipeline* pipe_ = nullptr;
public:
    AbobaEffectInstance() {
        aboba_backend_create_best(&backend_);
        aboba_pipeline_create_lowlatency(backend_, 48000.0, &pipe_);
    }
    ~AbobaEffectInstance() {
        aboba_pipeline_destroy(pipe_);
        aboba_backend_destroy(backend_);
    }
    void _process(const AudioFrame* in, AudioFrame* out, int32_t n) override {
        // Downmix in[] (stereo) -> mono, process, expand back to stereo
        std::vector<float> mono(n), processed(n);
        for (int32_t i = 0; i < n; ++i)
            mono[i] = 0.5f * (in[i].left + in[i].right);
        aboba_pipeline_process(pipe_, mono.data(), processed.data(), (size_t)n);
        for (int32_t i = 0; i < n; ++i) {
            out[i].left = out[i].right = processed[i];
        }
    }
};
```

## Threading rules

Each `aboba_pipeline*` handle is **single-threaded** — use it only from one thread at a time. The handle does internal locking only for stats reads, not the process path (which would defeat real-time guarantees).

You can have multiple independent pipelines on multiple threads:

```c
// Worker thread A
aboba_pipeline_process(pipe_a, ...);

// Worker thread B (simultaneously, with a DIFFERENT pipe handle)
aboba_pipeline_process(pipe_b, ...);
```

Querying stats (`aboba_pipeline_get_stats`, `aboba_pipeline_get_character`, etc) IS thread-safe and can be done from your engine's monitoring/UI thread.

## ABI version check

At startup, verify ABI compatibility:

```c
if (aboba_runtime_abi_version_major() != ABOBA_C_API_VERSION_MAJOR) {
    fprintf(stderr, "Aboba ABI mismatch: header=%d, runtime=%d\n",
        ABOBA_C_API_VERSION_MAJOR, aboba_runtime_abi_version_major());
    abort();
}
```

## Status code handling

Every function returns `aboba_status`. Use `aboba_status_message(s)` for a human-readable description:

```c
aboba_status s = aboba_pipeline_create(backend, 48000.0,
                                        ABOBA_PROFILE_BALANCED, &pipe);
if (s != ABOBA_OK) {
    fprintf(stderr, "create failed: %s\n", aboba_status_message(s));
    return 1;
}
```

## Loading configs from disk

TOML configs let you ship voice presets as data files instead of recompiling:

```c
aboba_config* cfg = NULL;
if (aboba_config_load_file("voices/streamer.toml", &cfg) != ABOBA_OK) {
    fprintf(stderr, "Config error at line %d: %s\n",
        aboba_config_last_error_line(),
        aboba_config_last_error());
}

aboba_pipeline* pipe = NULL;
aboba_pipeline_create_from_config(backend, cfg, &pipe);
aboba_config_destroy(cfg);  // pipe keeps its own copy
```

See `python/README.md` for the TOML schema.
