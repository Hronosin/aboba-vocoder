#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""
Python binding tests for Aboba Vocoder.

Run with:
    PYTHONPATH=build python3 python/test_bindings.py

Exits 0 on all pass, 1 if any fail.
"""

import os
import sys
import math
import textwrap
import tempfile
import numpy as np

# Locate the .so. Try common spots:
_here = os.path.dirname(os.path.abspath(__file__))
_root = os.path.dirname(_here)
for cand in (
    os.path.join(_root, "build"),
    os.path.join(_root, "build-asan"),
    _here,
):
    if any(f.startswith("aboba.cpython") for f in os.listdir(cand) if os.path.isfile(os.path.join(cand, f))):
        sys.path.insert(0, cand)
        break

import aboba

failures = 0
total = 0

def check(cond, label):
    global failures, total
    total += 1
    if cond:
        print(f"  PASS  {label}")
    else:
        print(f"  FAIL  {label}")
        failures += 1


def sine(hz, sr=48000, dur=0.5, amp=0.3):
    n = int(sr * dur)
    t = np.arange(n) / sr
    return (amp * np.sin(2 * math.pi * hz * t)).astype(np.float32)


def voice_like(f0, sr=48000, dur=1.0, amp=0.2):
    """Sum of harmonics, decaying amplitude."""
    n = int(sr * dur)
    t = np.arange(n) / sr
    out = np.zeros(n, dtype=np.float32)
    harm = np.array([1.0, 0.6, 0.4, 0.25, 0.15, 0.1, 0.07, 0.05], dtype=np.float32)
    for h, h_amp in enumerate(harm):
        hz = f0 * (h + 1)
        if hz > sr * 0.45:
            break
        out += (amp * h_amp * np.sin(2 * math.pi * hz * t)).astype(np.float32)
    return out


# ====================================================================
print("\n=== Module surface ===")
check(aboba.__version__ == "0.1.0", "version is 0.1.0")
check(len(list(aboba.QualityProfile.__members__)) == 3, "3 quality profiles")
check(len(list(aboba.VoiceCharacter.__members__)) == 14, "14 voice characters")
check(len(list(aboba.MusicalScale.__members__)) == 9, "9 musical scales")

# ====================================================================
print("\n=== Backend ===")
backend = aboba.create_backend()
check(backend is not None, "create_backend returns object")
check("CPU" in backend.name() or "HIP" in backend.name(),
      f"backend name reasonable: {backend.name()}")

# ====================================================================
print("\n=== Hz/MIDI helpers ===")
check(abs(aboba.hz_to_midi(440.0) - 69.0) < 0.001, "hz_to_midi(440)=69")
check(abs(aboba.midi_to_hz(69.0) - 440.0) < 0.1, "midi_to_hz(69)=440")
# A4 -> C4 = 9 semitones down
check(abs(aboba.midi_to_hz(60.0) - 261.626) < 1.0, "midi_to_hz(60)≈261.6 (C4)")

# ====================================================================
print("\n=== character_id / character_from_id ===")
for ch in aboba.VoiceCharacter.__members__.values():
    cid = aboba.character_id(ch)
    back = aboba.character_from_id(cid)
    check(back == ch, f"  roundtrip '{cid}'")
check(aboba.character_from_id("DEEP-MALE") == aboba.VoiceCharacter.DeepMale,
      "case-insensitive lookup")

# ====================================================================
print("\n=== snap_to_scale ===")
# C major: C D E F G A B
snapped = aboba.snap_to_scale(61.0, aboba.MusicalScale.Major, root_semitones=0)
check(abs(snapped - 60.0) < 1.5,
      f"C# (61) in C-major -> nearest scale tone (got {snapped})")

# ====================================================================
print("\n=== YinDetector ===")
yin = aboba.YinDetector(sample_rate=48000)
for target in [110.0, 220.0, 440.0]:
    s = sine(target, dur=0.5)
    r = yin.detect(s)
    err_cents = 1200.0 * math.log2(r.f0_hz / target) if r.f0_hz > 0 else 9999.0
    check(abs(err_cents) < 10.0,
          f"YIN detects {target} Hz (got {r.f0_hz:.1f}, err {err_cents:+.1f} cents)")
    check(r.is_voiced, f"  marked as voiced")

# ====================================================================
print("\n=== FormantVocoder ===")
voc = aboba.FormantVocoder(backend, fft_size=2048, hop_size=512,
                            sample_rate=48000, profile=aboba.QualityProfile.Balanced)
voc.set_pitch_semitones(5.0)
voc.set_formant_semitones(0.0)

x = voice_like(180.0, dur=1.5)
y = voc.process(x)
check(isinstance(y, np.ndarray), "process returns ndarray")
check(y.dtype == np.float32, "output dtype is float32")
check(y.shape == x.shape, "output shape matches input")
check(np.all(np.isfinite(y)), "output is finite")
check(np.max(np.abs(y)) < 5.0, "output amplitude reasonable")

stats = voc.stats()
check(stats.frames_total > 0, "stats.frames_total > 0")

# ====================================================================
print("\n=== PitchCorrector ===")
corr = aboba.PitchCorrector(sample_rate=48000,
                             scale=aboba.MusicalScale.Major,
                             root=0, strength=1.0, glide_ms=0.0)
# C#4 (277 Hz) in C major should snap ~1 semitone
out = corr.analyze(sine(277.183, dur=0.5))
check(abs(abs(out) - 1.0) < 0.5,
      f"PitchCorrector: C#4 in C major -> ~±1 st correction (got {out:+.2f})")

corr.set_strength(0.0)
out2 = corr.analyze(sine(277.183, dur=0.5))
check(abs(out2) < 0.1, "strength=0 -> ~0 correction")

# ====================================================================
print("\n=== VoicePipeline basic ===")
pipe = aboba.VoicePipeline(backend, sample_rate=48000,
                            profile=aboba.QualityProfile.Balanced)
pipe.set_pitch_semitones(3.0)

x = voice_like(150.0, dur=1.5)
y = pipe.process(x)
check(y.shape == x.shape, "pipeline output shape matches")
check(np.all(np.isfinite(y)), "pipeline output finite")
check(np.max(np.abs(y)) < 1.01, "pipeline output bounded by limiter")

# ====================================================================
print("\n=== VoicePipeline characters ===")
pipe.set_character(aboba.VoiceCharacter.AnimeGirl)
check(pipe.current_character == aboba.VoiceCharacter.AnimeGirl,
      "current_character reflects set")
y = pipe.process(voice_like(150.0, dur=0.5))
check(np.all(np.isfinite(y)), "anime preset output finite")

# Try every character
for name, ch in aboba.VoiceCharacter.__members__.items():
    pipe.reset()
    pipe.set_character(ch)
    y = pipe.process(voice_like(180.0, dur=0.3))
    ok = np.all(np.isfinite(y)) and np.max(np.abs(y)) < 1.01
    check(ok, f"  character '{name}' produces clean output")

# ====================================================================
print("\n=== VoicePipeline autotune ===")
pipe.reset()
pipe.set_autotune_enabled(True)
pipe.set_autotune_scale(aboba.MusicalScale.Major, root=0)
pipe.set_autotune_strength(1.0)
pipe.set_autotune_glide_ms(20.0)

# Slightly off-key tone
x = sine(277.0, dur=2.0, amp=0.15)
y = pipe.process(x)
check(np.all(np.isfinite(y)), "autotune output finite")
check(pipe.autotune_enabled, "autotune flag readable")

st = pipe.autotune_stats()
check(st.analyses_total > 0, "autotune ran at least once")

# ====================================================================
print("\n=== VoicePipeline reverb ===")
pipe.reset()
pipe.set_reverb_enabled(True)
pipe.set_reverb_room_size(0.7)
pipe.set_reverb_damping(0.3)
pipe.set_reverb_wet(0.25)
y = pipe.process(voice_like(150.0, dur=1.0))
check(np.all(np.isfinite(y)), "reverb output finite")

# ====================================================================
print("\n=== VoicePipeline process_into (in-place) ===")
x = voice_like(200.0, dur=0.5)
y = np.zeros_like(x)
pipe.reset()
pipe.process_into(x, y)
check(np.any(y != 0), "process_into wrote into output buffer")

# Wrong-size buffer should raise
try:
    pipe.process_into(x, np.zeros(100, dtype=np.float32))
    check(False, "wrong-size process_into raises")
except (ValueError, RuntimeError):
    check(True, "wrong-size process_into raises")

# ====================================================================
print("\n=== Input validation ===")
# Non-1D should raise
try:
    pipe.process(np.zeros((4, 4), dtype=np.float32))
    check(False, "2D input raises")
except (ValueError, RuntimeError):
    check(True, "2D input raises")

# float64 should be force-cast to float32 silently (we used forcecast)
y = pipe.process(np.zeros(1024, dtype=np.float64))
check(y.dtype == np.float32, "float64 input is converted to float32 output")
check(np.all(np.isfinite(y)), "  converted output is finite")

# ====================================================================
print("\n=== VoiceConfig (TOML) ===")
toml_text = textwrap.dedent('''
name = "py-test"
description = "From Python"

[pipeline]
profile = "balanced"
sample_rate = 48000

[character]
preset = "warm-male"

[autotune]
enabled = true
scale = "minor"
root = "A"
strength = 0.6
glide_ms = 40.0

[effects]
reverb = true

[reverb]
room_size = 0.5
damping = 0.4
wet = 0.2
''').strip()

cfg = aboba.parse_voice_config(toml_text)
check(cfg.name == "py-test", "config.name parsed")
check(cfg.has_character, "config.has_character True")
check(cfg.character == aboba.VoiceCharacter.WarmMale, "config.character")
check(cfg.autotune_enabled, "config.autotune_enabled")
check(cfg.autotune_scale == aboba.MusicalScale.Minor, "config.autotune_scale")
check(cfg.autotune_root == 9, "config.autotune_root = A (9)")
check(abs(cfg.autotune_strength - 0.6) < 0.01, "config.autotune_strength")
check(cfg.reverb, "config.reverb")

# Build pipeline from config
pipe2 = aboba.pipeline_from_config(cfg, backend)
check(pipe2.current_character == aboba.VoiceCharacter.WarmMale,
      "pipeline_from_config applied character")
check(pipe2.autotune_enabled, "pipeline_from_config applied autotune")
check(pipe2.reverb_enabled, "pipeline_from_config applied reverb")

# Process audio through it
y = pipe2.process(voice_like(150.0, dur=1.0))
check(np.all(np.isfinite(y)), "configured pipeline produces clean audio")

# Round-trip via serialize
text2 = cfg.serialize()
cfg2 = aboba.parse_voice_config(text2)
check(cfg2.name == cfg.name, "serialize round-trip preserves name")
check(cfg2.character == cfg.character, "serialize round-trip preserves character")
check(cfg2.autotune_scale == cfg.autotune_scale, "serialize round-trip preserves scale")

# Load from file
with tempfile.NamedTemporaryFile(mode='w', suffix='.toml', delete=False) as f:
    f.write(toml_text)
    tmp_path = f.name
try:
    cfg3 = aboba.load_voice_config(tmp_path)
    check(cfg3.name == "py-test", "load_voice_config from file works")
finally:
    os.unlink(tmp_path)

# Error path
try:
    aboba.parse_voice_config("name = wrong\n[autotune]\nenabled = yes\n")
    check(False, "invalid TOML raises")
except RuntimeError as e:
    check("line" in str(e).lower(), f"error message mentions line ({e})")

# ====================================================================
print("\n=== GIL release on long blocks (smoke test) ===")
# Just verify a large processing call doesn't crash. Real GIL-release
# verification needs threading. This is a sanity check.
big = np.zeros(48000 * 3, dtype=np.float32)  # 3 seconds
big[::4] = 0.1  # impulses
y = pipe.process(big)
check(y.shape == big.shape, "3-second buffer processes")

# ====================================================================
print("\n========================================")
if failures == 0:
    print(f"Total: {total}/{total} passed \u2713")
    sys.exit(0)
else:
    print(f"Total: {total - failures}/{total} passed - {failures} FAILED \u2717")
    sys.exit(1)
