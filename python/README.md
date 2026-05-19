# Aboba Vocoder — Python bindings

Pythonic interface to the Aboba real-time voice framework. NumPy-first.

## Installation

```bash
# 1. Install pybind11
pip install pybind11

# 2. Build with Python support enabled
cd /path/to/aboba-vocoder
cmake -S . -B build -DABOBA_BUILD_PYTHON=ON
cmake --build build -j

# 3. The compiled module is at build/aboba.cpython-*.so
#    Add to your PYTHONPATH:
export PYTHONPATH=$PWD/build:$PYTHONPATH
```

## Quickstart

```python
import numpy as np
import aboba

# 1) Backend (auto-picks HIP/AMD if available, else CPU)
backend = aboba.create_backend()
print("Backend:", backend.name())

# 2) Build a pipeline
pipe = aboba.VoicePipeline(
    backend,
    sample_rate=48000,
    profile=aboba.QualityProfile.Balanced
)

# 3) Pick a character or set pitch/formant manually
pipe.set_character(aboba.VoiceCharacter.DeepMale)
# OR:
# pipe.set_pitch_semitones(-3.0)
# pipe.set_formant_semitones(-2.0)

# 4) Process some audio (1D float32 numpy array)
sr = 48000
t = np.arange(sr) / sr
x = (0.3 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)

y = pipe.process(x)
print("Output RMS:", np.sqrt(np.mean(y**2)))
```

## Loading a TOML config

```python
cfg = aboba.load_voice_config("voice.toml")
pipe = aboba.pipeline_from_config(cfg, backend)

# Or build a config in code:
cfg = aboba.VoiceConfig()
cfg.name = "my-voice"
cfg.has_character = True
cfg.character = aboba.VoiceCharacter.AnimeGirl
cfg.autotune_enabled = True
cfg.autotune_scale = aboba.MusicalScale.Major
cfg.autotune_root = 0  # C major
toml = cfg.serialize()
print(toml)
```

## Autotune (lightweight)

```python
pipe.set_autotune_enabled(True)
pipe.set_autotune_scale(aboba.MusicalScale.Minor, root=9)  # A minor
pipe.set_autotune_strength(0.7)   # 0..1
pipe.set_autotune_glide_ms(30.0)  # T-Pain effect at 0, natural at 30+

# Combine with manual pitch shift; they ADD:
pipe.set_pitch_semitones(2.0)
```

## Standalone YIN F0 detection (research / offline)

```python
y = aboba.YinDetector(sample_rate=48000, f0_min_hz=60.0, f0_max_hz=1500.0)
r = y.detect(audio_samples)
print(f"F0 = {r.f0_hz:.1f} Hz, voiced = {r.is_voiced}")
```

## Standalone FormantVocoder (no pipeline)

```python
voc = aboba.FormantVocoder(backend, fft_size=2048, hop_size=512,
                            profile=aboba.QualityProfile.Quality)
voc.set_pitch_semitones(7.0)        # +7 semitones
voc.set_formant_semitones(0.0)      # preserve formants (anti-helium)
y = voc.process(x)
```

## Reverb

```python
pipe.set_reverb_enabled(True)
pipe.set_reverb_room_size(0.6)   # 0..1
pipe.set_reverb_damping(0.4)     # 0..1, higher = darker tail
pipe.set_reverb_wet(0.25)        # 0..1, mix level
```

## In-place processing

```python
out_buf = np.zeros_like(x)
pipe.process_into(x, out_buf)   # no allocation
```

For long blocks (>4096 samples) the GIL is released automatically.

## Diagnostics

```python
stats = pipe.vocoder_stats()
print("Voiced/unvoiced frames:", stats.frames_voiced, stats.frames_unvoiced)
print("Last F0:", stats.last_f0_hz)

at = pipe.autotune_stats()
print("Last input/target:", at.last_input_f0_hz, at.last_target_f0_hz)
```

## Running tests

```bash
PYTHONPATH=build python3 python/test_bindings.py
```
