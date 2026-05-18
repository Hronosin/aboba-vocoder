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
[![GPU](https://img.shields.io/badge/GPU-AMD%20only-red)]()
[![NVIDIA](https://img.shields.io/badge/NVIDIA-not%20today-darkgreen)]()
[![SAM](https://img.shields.io/badge/SAM-supported-blueviolet)]()
[![Aboba](https://img.shields.io/badge/Aboba-certified-ff69b4)]()
[![Vibes](https://img.shields.io/badge/vibes-immaculate-success)]()

</div>

---

## The Manifesto

For too long, AMD users have been treated as second-class citizens in the audio world. Every tutorial, every TTS repo, every voice converter, every cool research paper — they all end with the same line:

> *"Just buy an NVIDIA card lol"*

**No.**

We have 24 GB of VRAM on a 7900 XTX. We have Smart Access Memory. We have ROCm. We have hands. We will write our own software, thank you very much, and we will release it under GPLv3 so nobody can EULA us into a corner ever again.

**Aboba Vocoder** is a C++ vocoder framework that runs natively on AMD GPUs through HIP + rocFFT, exploits Resizable BAR / SAM for low-latency CPU↔GPU transfers, and falls back to a brutally optimized FFTW3 CPU path when no AMD GPU is present.

**It does not, and will not, support NVIDIA. This is not an oversight. It is a feature.**

---

## What It Does (already, today, right now, on your machine)

🎤 **Real-time voice changing** — mic → pitch shift → virtual cable → OBS / Discord / your therapist's Zoom call

🎼 **Offline pitch & time manipulation** — pitch-shift, time-stretch, the works. For TTS post-processing or just for the bit.

⚡ **Streaming STFT/iSTFT engine** — block sizes from 64 to 4096 samples. You pick the latency/quality tradeoff. We don't tell you what's "good enough".

🔬 **Phase vocoder accuracy: ±5 cents across ±12 semitones.** That's musically tighter than your average drummer.

🏎️ **30× realtime on a single CPU core.** You could run thirty simultaneous voice changers and your laptop wouldn't even spin up its fan.

🔴 **AMD GPU acceleration** through HIP + rocFFT, with automatic SAM detection. Or CPU fallback. Never CUDA. Especially not CUDA.

---

## Performance

Benchmark of the streaming pitch shifter, single thread, x86_64, fft=2048, hop=512:

| Block size | CPU per block | Budget @ 48 kHz | Headroom |
|-----------:|--------------:|----------------:|---------:|
|         64 |      ~0.05 ms |         1.33 ms |  **~26×** |
|        256 |      ~0.18 ms |         5.33 ms |  **~30×** |
|       1024 |      ~0.65 ms |        21.33 ms |  **~33×** |

Translation for non-DSP people: this software uses **3% of one CPU core** to do what every YouTube tutorial says you need a $1600 graphics card for.

---

## Quick Start (the impatient version)

```bash
# Linux with apt. For other OSes see "Slow Start" below.
sudo apt install build-essential cmake pkg-config libfftw3-dev portaudio19-dev

git clone https://github.com/YOUR_USERNAME_HERE/aboba-vocoder
cd aboba-vocoder
cmake -S . -B build && cmake --build build -j

# List your audio devices
./build/aboba_realtime_voice --list

# Become a 4-semitone-higher version of yourself
./build/aboba_realtime_voice --in 0 --out 0 --semitones 4
```

---

## Slow Start (the "I want to understand what I'm doing" version)

### 1. Build options

```bash
cmake -S . -B build \
    -DABOBA_ENABLE_HIP=ON \         # Build the AMD GPU backend (needs ROCm 6.x)
    -DABOBA_ENABLE_REALTIME=ON \    # Build the PortAudio realtime engine
    -DABOBA_BUILD_EXAMPLES=ON       # Build the demo programs
```

The CPU build is universal. The HIP build requires ROCm at `/opt/rocm` (or pass `-DROCM_PATH=/path/to/rocm`).

### 2. Routing audio to OBS / Discord

Aboba doesn't care where its output goes. You need a **virtual audio device** that OBS/Discord can read from:

| OS       | Recommended virtual device                                |
|----------|-----------------------------------------------------------|
| Linux    | PipeWire / PulseAudio null sink (`pactl load-module module-null-sink`) |
| Windows  | [VB-CABLE](https://vb-audio.com/Cable/) (free)            |
| macOS    | [BlackHole](https://existential.audio/blackhole/) (free)  |

Then point Aboba's output to that device, and point OBS/Discord's input to that device. Done.

### 3. Example: real-time voice shift

```bash
./build/aboba_realtime_voice \
    --in 1 \              # your microphone
    --out 5 \             # your virtual cable
    --sr 48000 \          # sample rate
    --block 256 \         # 256 samples = ~5 ms callback budget
    --semitones 7         # perfect fifth up
```

### 4. Example: offline TTS post-processing

```cpp
#include "aboba/backend.hpp"
#include "aboba/phase_vocoder.hpp"

auto backend = aboba::create_best_backend();
aboba::PhaseVocoder pv(2048, 512, backend.get());

std::vector<float> output;
pv.pitch_shift(input.data(), input.size(), /*semitones=*/-3.0f, output);
```

---

## Architecture

```
        ┌─────────────────────────────────────────┐
        │         your application                │
        │  (TTS pipeline / voice changer / DAW)   │
        └────────────────┬────────────────────────┘
                         │
        ┌────────────────▼────────────────────────┐
        │       Aboba public API (C++17)          │
        │   PhaseVocoder    StreamingPhaseVocoder │
        │   STFT            RealtimeEngine        │
        └────────────────┬────────────────────────┘
                         │
        ┌────────────────▼────────────────────────┐
        │            Backend interface            │
        │     fft_r2c / fft_c2r / batched ver.    │
        └─────┬────────────────────────┬──────────┘
              │                        │
   ┌──────────▼──────────┐  ┌──────────▼──────────┐
   │    CPU backend      │  │    HIP backend      │
   │  FFTW3, AVX, MT     │  │  rocFFT on AMD GPU  │
   │  (works anywhere)   │  │  + SAM autodetect   │
   └─────────────────────┘  └─────────────────────┘
```

The whole thing is structured so that adding a new backend is one file. Want a Vulkan compute backend? Write one. Want to port to a TPU? Be our guest. **Want to add CUDA?** Read the manifesto again. *(See "Contributing" for the polite rejection process.)*

---

## Aboba vs. The Competition

| | Aboba Vocoder | WORLD | NVIDIA CUDA Stack |
|---|---|---|---|
| **Price of entry** | Your existing AMD GPU | Free | $1599 + soul |
| **License** | GPLv3 | Modified BSD | EULA (which you'll definitely read in full) |
| **Realtime support** | ✅ Built-in | ⚠️ Possible with effort | ✅ but only if you read 2847 pages of cuDNN release notes |
| **GPU acceleration** | ✅ AMD only by design | ❌ CPU only | ✅ Green tax included |
| **Documentation** | This README | Excellent | "See the example, then cry" |
| **Sound quality** | ±5 cents @ ±12 st | Reference quality | Depends on your model and three PhD students |
| **NVIDIA support** | Never | N/A | Yes (it's their entire business) |
| **Has been featured in a Two Minute Papers video** | Not yet | No | Yes (and we're salty about it) |

---

## FAQ

**Q: What does "Aboba" mean?**
A: Aboba.

**Q: No really, what does it mean?**
A: It's a sound. A vibe. A philosophical position. An entire worldview. Mostly it just sounds funny.

**Q: Will you ever add NVIDIA support?**
A: No.

**Q: But—**
A: No.

**Q: I have an RTX 5090 and—**
A: This software respectfully asks you to leave.

**Q: Is this a serious project?**
A: The code is serious. The pitch accuracy is ±5 cents. The README is not. The license is GPLv3 which is **very** serious.

**Q: Can I use this commercially?**
A: Yes. You must publish your modifications under GPLv3. If that's a problem, this project is not for you, and frankly neither is the open-source ecosystem.

**Q: Will you accept corporate sponsorship from AMD?**
A: We're hoping. Lisa, if you're reading this, our DMs are open.

**Q: I want professional support / SLA / enterprise tier.**
A: Aboba.

**Q: How does it compare to WORLD?**
A: WORLD is genuinely excellent academic software and we respect Morise-sensei deeply. We are faster on CPU and we work on GPU. That's it. That's the difference. There is no shade here.

**Q: How does it compare to neural vocoders like HiFi-GAN, Vocos, BigVGAN?**
A: Different tool. We're a *phase* vocoder — DSP, not deep learning. For mel→waveform synthesis with neural vocoders, just export them to ONNX and run with `onnxruntime-rocm`. We may add a built-in neural backend later. We may not.

**Q: Why don't you just contribute to an existing project?**
A: We tried. They wanted CUDA-first. We left.

**Q: I want to contribute but I have an NVIDIA card. What do I do?**
A: You're welcome here. Write CPU code. Test CPU code. Optimize CPU code. We will gently close any PR that adds CUDA dependencies. The first time politely. The second time with feeling.

**Q: Why GPLv3 specifically?**
A: Because if a corporation wants to embed this in a closed product, they should have to write their own. We've all had enough of "free as in beer, then suddenly it's a $50/month SaaS subscription" software. Not this time.

**Q: Will there be a fork with a normal README and proper docs?**
A: We hope so. That's the entire plan.

**Q: I'm offended by the tone of this README.**
A: That's not a question. But yes, this is by design.

**Q: Is the author okay?**
A: The author is rofling.

---

## Roadmap

### Shipped 🚀

- [x] Backend abstraction
- [x] STFT / iSTFT engine
- [x] Offline phase vocoder (time-stretch + pitch-shift)
- [x] Streaming phase vocoder (Bernsee bin-shift)
- [x] PortAudio real-time engine
- [x] FFTW3 CPU backend
- [x] HIP/rocFFT GPU backend with SAM detection
- [x] This unhinged README

### Soon™

- [ ] Mel-spectrogram → audio (true WORLD replacement for TTS)
- [ ] Python bindings via pybind11
- [ ] Polyphase sinc resampler (replace the placeholder linear one)
- [ ] Formant-preserving pitch shift
- [ ] VST3 plugin wrapper
- [ ] Lock-free ring buffers for sub-millisecond jitter

### Eventually

- [ ] Neural vocoder backend (ONNX Runtime on ROCm)
- [ ] Real-time formant shifting independent of pitch
- [ ] JACK / ASIO direct backends
- [ ] Documentation that someone other than the author can read

### Will Never Happen

- [ ] CUDA support
- [ ] Closed-source commercial fork licensing
- [ ] A "Pro" tier
- [ ] Re-licensing to MIT so Big Tech can ingest us into their training data
- [ ] An NPM package called `aboba-react-vocoder-hooks`

---

## Contributing

We accept PRs from anyone. Including, yes, NVIDIA users. The rules:

1. **No CUDA dependencies.** Not even optional ones. Not even with `#ifdef`. We mean it.
2. **No closed-source dependencies.** If it's not redistributable under GPLv3, it doesn't go in.
3. **CPU code should work everywhere.** x86, ARM, RISC-V — if it's not vendor-specific GPU code, it should be portable.
4. **GPU code is HIP-only.** AMD has Vulkan Compute too if HIP is unsuitable. Not CUDA.
5. **Code style:** clang-format with the included config. Don't argue with the robot.
6. **Have fun.** This is a community project. If it stops being fun, we've failed.

PRs that violate rule 1 will be closed with a polite reference to this section. PRs that violate rule 1 a second time will be closed with a less polite reference.

---

## Acknowledgments

Real, sincere acknowledgments because credit matters:

- **The FFTW team** — for 30 years of being the best damn FFT library on earth.
- **PortAudio contributors** — for making cross-platform audio I/O suck slightly less.
- **AMD's ROCm team** — for actually building an open GPU compute stack.
- **Stephan Bernsee** — for the phase vocoder pitch-shift algorithm that this project's streaming engine is based on.
- **Masanori Morise** — for WORLD, a genuinely beautiful piece of academic software.
- **The Free Software Foundation** — for making GPL exist.
- **Every AMD user who got told to "just buy NVIDIA"** — this one's for you.

---

## License

```
Aboba Vocoder
Copyright (C) 2026 Aboba Vocoder contributors

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

[...standard GPL preamble continues...]
```

See [LICENSE](LICENSE) for the full text.

---

<div align="center">

### *"Aboba."*
— Aboba Vocoder, probably

🔴 **AMD inside.** 🔴

</div>
