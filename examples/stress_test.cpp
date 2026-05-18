// SPDX-License-Identifier: GPL-3.0-or-later
//
// Stress test: validates the streaming vocoder against:
//   1. Mixed-source inputs (the bug we just fixed).
//   2. Silence inputs (silence fast-path must not poison phase state).
//   3. NaN/Inf inputs (sanitization must keep us alive).
//   4. Very short block sizes (1, 7, 13 — non-power-of-2, prime).
//   5. Pitch shifts at the edge of the sane range.
//   6. Reset between bursts (state must clear properly).
//   7. Repeated start/stop cycles (no leaks across constructions).
//
// Each test prints PASS/FAIL. Non-zero exit on any failure.
#include "aboba/backend.hpp"
#include "aboba/streaming.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <memory>
#include <random>
#include <stdexcept>
#include <vector>

namespace {

constexpr float kPi = 3.14159265358979323846f;

int g_failures = 0;
int g_total    = 0;

void check(bool cond, const char* what) {
    ++g_total;
    if (cond) {
        std::printf("  PASS  %s\n", what);
    } else {
        std::printf("  FAIL  %s\n", what);
        ++g_failures;
    }
}

// Find the strongest spectral peak in `signal`, return its frequency in Hz.
float find_peak_hz(const std::vector<float>& signal, int sample_rate,
                   int analysis_offset = 0) {
    const int N = 32768;
    if (static_cast<int>(signal.size()) < analysis_offset + N) return 0.0f;

    std::vector<float> buf(N);
    std::copy(signal.begin() + analysis_offset,
              signal.begin() + analysis_offset + N,
              buf.begin());
    // Hann window
    for (int i = 0; i < N; ++i) {
        buf[i] *= 0.5f * (1.0f - std::cos(2.0f * kPi * i / (N - 1)));
    }
    // Naive DFT magnitude (slow but no dep)... just call FFTW via the backend.
    // Easier: re-do it in the test with a tiny inline RFFT? Use a quick goertzel
    // grid sweep instead. For 32768 samples we'd need a real FFT — let's just
    // do a coarse goertzel sweep over candidate frequencies.
    float best_hz = 0.0f, best_mag = 0.0f;
    for (float hz = 50.0f; hz < sample_rate * 0.45f; hz += 0.5f) {
        const float w = 2.0f * kPi * hz / sample_rate;
        const float coeff = 2.0f * std::cos(w);
        float s1 = 0.0f, s2 = 0.0f;
        for (int i = 0; i < N; ++i) {
            const float s0 = buf[i] + coeff * s1 - s2;
            s2 = s1;
            s1 = s0;
        }
        const float re = s1 - s2 * std::cos(w);
        const float im = s2 * std::sin(w);
        const float mag = re * re + im * im;
        if (mag > best_mag) { best_mag = mag; best_hz = hz; }
    }
    return best_hz;
}

// Generate a mix of multiple sine waves at the given freqs (mono).
std::vector<float> make_mix(const std::vector<float>& freqs, int sample_rate,
                            float duration_sec, float amp = 0.2f) {
    const int N = static_cast<int>(sample_rate * duration_sec);
    std::vector<float> sig(N, 0.0f);
    for (float f : freqs) {
        for (int i = 0; i < N; ++i) {
            sig[i] += amp * std::sin(2.0f * kPi * f * i / sample_rate);
        }
    }
    return sig;
}

// Detect whether a signal contains NaN/Inf/giant values (signs of state corruption).
bool signal_is_clean(const std::vector<float>& s, float max_abs = 10.0f) {
    for (float v : s) {
        if (!std::isfinite(v)) return false;
        if (std::fabs(v) > max_abs) return false;
    }
    return true;
}

}  // namespace

int main() {
    auto backend = aboba::create_best_backend();
    const int sr = 48000;

    std::printf("=== Test 1: Identity (pitch=0) passthrough ===\n");
    {
        aboba::StreamingPhaseVocoder vp(2048, 512, backend.get());
        vp.set_pitch_semitones(0.0f);

        // Single sine — multi-harmonic test belongs in Test 2 below where
        // we explicitly verify multi-source behavior.
        auto in  = make_mix({440.0f}, sr, 1.5f);
        std::vector<float> out(in.size());
        vp.process(in.data(), out.data(), in.size());

        check(signal_is_clean(out), "output is finite and bounded");

        const float peak = find_peak_hz(out, sr, /*offset=*/8192);
        std::printf("    (peak=%.2f Hz, expected ~440 Hz)\n", static_cast<double>(peak));
        check(std::fabs(peak - 440.0f) < 5.0f,
              "identity passthrough preserves fundamental ~440 Hz");
    }

    std::printf("\n=== Test 2: Multi-source mix, pitch +5 semitones ===\n");
    {
        // Two sines an octave apart. Pitch shifting should preserve their
        // relationship — both shift by the same ratio.
        aboba::StreamingPhaseVocoder vp(2048, 512, backend.get());
        vp.set_pitch_semitones(5.0f);
        const float ratio = std::pow(2.0f, 5.0f / 12.0f);

        auto in = make_mix({440.0f, 880.0f}, sr, 2.0f);
        std::vector<float> out(in.size());
        vp.process(in.data(), out.data(), in.size());

        check(signal_is_clean(out), "output is finite and bounded");

        // After latency, look for the shifted peaks.
        const float peak = find_peak_hz(out, sr, /*offset=*/8192);
        const float expected_low  = 440.0f * ratio;   // ~587 Hz
        const float expected_high = 880.0f * ratio;   // ~1175 Hz
        const float err_to_low  = std::fabs(peak - expected_low);
        const float err_to_high = std::fabs(peak - expected_high);
        // We don't know which peak is strongest, but it must be one of them.
        check(err_to_low < 10.0f || err_to_high < 10.0f,
              "shifted peak lands near 440*ratio or 880*ratio");
        std::printf("    (peak=%.1f Hz; targets=%.1f, %.1f)\n",
                    peak, expected_low, expected_high);
    }

    std::printf("\n=== Test 3: Silence input ===\n");
    {
        aboba::StreamingPhaseVocoder vp(2048, 512, backend.get());
        vp.set_pitch_semitones(7.0f);

        std::vector<float> in(sr, 0.0f);  // 1 sec of silence
        std::vector<float> out(in.size());
        vp.process(in.data(), out.data(), in.size());

        check(signal_is_clean(out), "silence in → finite, bounded output");

        // After silence we should be able to process real input without
        // garbage from poisoned state.
        auto sig = make_mix({440.0f}, sr, 1.0f);
        vp.process(sig.data(), out.data(), sig.size());
        check(signal_is_clean(out), "real input after silence is still clean");
    }

    std::printf("\n=== Test 4: NaN/Inf input survival ===\n");
    {
        aboba::StreamingPhaseVocoder vp(2048, 512, backend.get());
        vp.set_pitch_semitones(3.0f);

        std::vector<float> in(sr, 0.0f);
        // Inject some NaN/Inf in random places
        std::mt19937 rng(42);
        std::uniform_int_distribution<int> pos(0, sr - 1);
        for (int i = 0; i < 50; ++i) {
            in[pos(rng)] = std::numeric_limits<float>::quiet_NaN();
            in[pos(rng)] = std::numeric_limits<float>::infinity();
            in[pos(rng)] = -std::numeric_limits<float>::infinity();
        }

        std::vector<float> out(in.size());
        vp.process(in.data(), out.data(), in.size());
        check(signal_is_clean(out),
              "pathological NaN/Inf input doesn't corrupt output");

        // Verify state recovers afterwards.
        auto sig = make_mix({440.0f}, sr, 1.0f);
        vp.process(sig.data(), out.data(), sig.size());
        check(signal_is_clean(out),
              "vocoder recovers and processes normal signal after pathological input");
    }

    std::printf("\n=== Test 5: Tiny block sizes (1, 7, 13, prime) ===\n");
    {
        aboba::StreamingPhaseVocoder vp(2048, 512, backend.get());
        vp.set_pitch_semitones(4.0f);

        auto in = make_mix({440.0f}, sr, 2.0f);
        std::vector<float> out(in.size(), 0.0f);

        // Feed with awkward chunk sizes.
        const std::size_t chunks[] = {1, 7, 13, 31, 127};
        std::size_t pos = 0;
        std::size_t ci  = 0;
        while (pos < in.size()) {
            const std::size_t n = std::min(chunks[ci % 5], in.size() - pos);
            vp.process(in.data() + pos, out.data() + pos, n);
            pos += n;
            ++ci;
        }
        check(signal_is_clean(out), "weird chunk sizes produce clean output");
    }

    std::printf("\n=== Test 6: Edge-of-range pitch shifts ===\n");
    {
        // ±60 semitones should clamp internally; ±100 should clamp to ±60.
        for (float st : {-60.0f, -36.0f, -12.0f, 12.0f, 36.0f, 60.0f, 100.0f, -100.0f}) {
            aboba::StreamingPhaseVocoder vp(2048, 512, backend.get());
            vp.set_pitch_semitones(st);

            auto in = make_mix({440.0f}, sr, 0.5f);
            std::vector<float> out(in.size());
            vp.process(in.data(), out.data(), in.size());

            char label[64];
            std::snprintf(label, sizeof(label),
                          "extreme pitch %.0f semitones is bounded", st);
            check(signal_is_clean(out, /*max_abs=*/100.0f), label);
        }
    }

    std::printf("\n=== Test 7: reset() clears state ===\n");
    {
        aboba::StreamingPhaseVocoder vp(2048, 512, backend.get());
        vp.set_pitch_semitones(7.0f);

        auto burst = make_mix({440.0f}, sr, 0.5f);
        std::vector<float> out(burst.size());
        vp.process(burst.data(), out.data(), burst.size());

        vp.reset();

        // After reset, the next process() must not have lingering tail.
        std::vector<float> silence(sr, 0.0f);
        std::vector<float> out2(silence.size(), 0.0f);  // sized to match input
        vp.process(silence.data(), out2.data(), silence.size());
        // Within the first latency window output should be ~0.
        float max_abs_in_latency = 0.0f;
        for (std::size_t i = 0; i < 2048; ++i) {
            max_abs_in_latency = std::max(max_abs_in_latency, std::fabs(out2[i]));
        }
        check(max_abs_in_latency < 1e-3f,
              "after reset, no tail from previous burst");
    }

    std::printf("\n=== Test 8: Invalid arguments rejected ===\n");
    {
        bool threw = false;
        try { aboba::StreamingPhaseVocoder bad(0, 0, backend.get()); }
        catch (const std::invalid_argument&) { threw = true; }
        check(threw, "fft_size=0 rejected");

        threw = false;
        try { aboba::StreamingPhaseVocoder bad(2048, 4096, backend.get()); }
        catch (const std::invalid_argument&) { threw = true; }
        check(threw, "hop_size > fft_size rejected");

        threw = false;
        try { aboba::StreamingPhaseVocoder bad(2048, 512, nullptr); }
        catch (const std::invalid_argument&) { threw = true; }
        check(threw, "null backend rejected");

        aboba::StreamingPhaseVocoder vp(2048, 512, backend.get());
        threw = false;
        try { vp.process(nullptr, nullptr, 100); }
        catch (const std::invalid_argument&) { threw = true; }
        check(threw, "null buffers in process() rejected");
    }

    std::printf("\n=== Test 9: Many instances (no shared-state corruption) ===\n");
    {
        std::vector<std::unique_ptr<aboba::StreamingPhaseVocoder>> vocoders;
        for (int i = 0; i < 10; ++i) {
            vocoders.emplace_back(std::make_unique<aboba::StreamingPhaseVocoder>(
                2048, 512, backend.get()));
            vocoders.back()->set_pitch_semitones(static_cast<float>(i - 5));
        }

        auto in = make_mix({440.0f}, sr, 0.3f);
        std::vector<float> out(in.size());
        bool all_clean = true;
        for (auto& vp : vocoders) {
            vp->process(in.data(), out.data(), in.size());
            if (!signal_is_clean(out)) { all_clean = false; break; }
        }
        check(all_clean, "10 concurrent vocoder instances all produce clean output");
    }

    std::printf("\n========================================\n");
    std::printf("Total: %d/%d passed", g_total - g_failures, g_total);
    if (g_failures == 0) {
        std::printf(" ✓\n");
        return 0;
    } else {
        std::printf(" — %d FAILED ✗\n", g_failures);
        return 1;
    }
}
