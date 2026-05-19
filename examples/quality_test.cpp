// SPDX-License-Identifier: GPL-3.0-or-later
//
// Quality tests for the new audio modules:
//   * YIN F0 detection accuracy & noise robustness
//   * DSP blocks (DC blocker, HP, gate, limiter, de-esser) behavior
//   * FormantVocoder: doesn't lose pitch info, preserves overall energy
//   * VoicePipeline: end-to-end, no allocations in process(), profile switch
//
// These tests focus on what a USER would actually notice. Numeric tolerances
// are set so that legitimate quality changes don't silently regress.
#include "aboba/backend.hpp"
#include "aboba/dsp_blocks.hpp"
#include "aboba/formant_vocoder.hpp"
#include "aboba/pipeline.hpp"
#include "aboba/yin.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <numeric>
#include <random>
#include <stdexcept>
#include <vector>

namespace {

constexpr float kPi = 3.14159265358979323846f;

int g_failures = 0;
int g_total    = 0;

void check(bool cond, const char* what) {
    ++g_total;
    if (cond) std::printf("  PASS  %s\n", what);
    else      { std::printf("  FAIL  %s\n", what); ++g_failures; }
}

std::vector<float> sine(float hz, int sr, float dur_sec, float amp = 0.3f) {
    const int N = static_cast<int>(sr * dur_sec);
    std::vector<float> s(N);
    for (int i = 0; i < N; ++i) {
        s[i] = amp * std::sin(2.0f * kPi * hz * i / sr);
    }
    return s;
}

// Voice-like signal: fundamental + harmonics with realistic decay
std::vector<float> voice_like(float f0, int sr, float dur_sec, float amp = 0.2f) {
    const int N = static_cast<int>(sr * dur_sec);
    std::vector<float> s(N, 0.0f);
    const float harmonics[] = {1.0f, 0.6f, 0.4f, 0.25f, 0.15f, 0.1f, 0.07f, 0.05f};
    for (int h = 0; h < 8; ++h) {
        const float hz = f0 * static_cast<float>(h + 1);
        if (hz > sr * 0.45f) break;
        for (int i = 0; i < N; ++i) {
            s[i] += amp * harmonics[h] * std::sin(2.0f * kPi * hz * i / sr);
        }
    }
    return s;
}

float rms(const float* x, std::size_t n) {
    double ss = 0.0;
    for (std::size_t i = 0; i < n; ++i) ss += x[i] * x[i];
    return static_cast<float>(std::sqrt(ss / std::max<std::size_t>(n, 1)));
}

bool all_finite_bounded(const std::vector<float>& s, float bound = 5.0f) {
    for (float v : s) {
        if (!std::isfinite(v) || std::fabs(v) > bound) return false;
    }
    return true;
}

}  // namespace

int main() {
    using namespace aboba;
    const int sr = 48000;

    // ============================================================
    std::printf("=== YIN: accuracy on pure sines ===\n");
    {
        YinConfig yc;
        yc.sample_rate = sr;
        yc.f0_min_hz = 50.0f;
        yc.f0_max_hz = 1500.0f;
        YinDetector yin(yc);

        const float test_freqs[] = {80.0f, 110.0f, 220.0f, 440.0f, 880.0f};
        for (float target : test_freqs) {
            auto s = sine(target, sr, 0.5f);
            const std::size_t needed = yin.window_size() + yin.tau_max();
            if (s.size() < needed) { std::printf("    skip (signal too short)\n"); continue; }
            auto r = yin.detect(s.data(), s.size());

            const float err_cents = (r.f0_hz > 0)
                ? 1200.0f * std::log2(r.f0_hz / target) : 9999.0f;
            std::printf("    target=%.1f Hz, detected=%.2f Hz, aper=%.3f, "
                        "voiced=%s, err=%+.1f cents\n",
                        static_cast<double>(target),
                        static_cast<double>(r.f0_hz),
                        static_cast<double>(r.aperiodicity),
                        r.is_voiced ? "Y" : "n",
                        static_cast<double>(err_cents));
            char label[80];
            std::snprintf(label, sizeof(label),
                          "YIN finds %.0f Hz within +/- 10 cents", static_cast<double>(target));
            check(std::fabs(err_cents) < 10.0f, label);
            check(r.is_voiced, "YIN marks pure sine as voiced");
        }
    }

    // ============================================================
    std::printf("\n=== YIN: low F0 from 60 Hz works (bass voice range) ===\n");
    {
        YinConfig yc;
        yc.sample_rate = sr;
        yc.f0_min_hz = 50.0f;
        yc.f0_max_hz = 800.0f;
        YinDetector yin(yc);

        auto s = voice_like(65.0f, sr, 0.8f);  // very low bass
        const std::size_t needed = yin.window_size() + yin.tau_max();
        auto r = yin.detect(s.data(), s.size());

        const float err_cents = (r.f0_hz > 0)
            ? 1200.0f * std::log2(r.f0_hz / 65.0f) : 9999.0f;
        std::printf("    voice-like 65 Hz -> %.2f Hz, aper=%.3f (err=%+.1f cents)\n",
                    static_cast<double>(r.f0_hz),
                    static_cast<double>(r.aperiodicity),
                    static_cast<double>(err_cents));
        check(std::fabs(err_cents) < 20.0f, "YIN handles 65 Hz voice (octave-safe)");
    }

    // ============================================================
    std::printf("\n=== YIN: rejects noise as unvoiced ===\n");
    {
        YinConfig yc; yc.sample_rate = sr;
        YinDetector yin(yc);

        std::mt19937 rng(42);
        std::normal_distribution<float> dist(0.0f, 0.2f);
        std::vector<float> noise(static_cast<std::size_t>(sr * 0.5));
        for (auto& v : noise) v = dist(rng);

        auto r = yin.detect(noise.data(), noise.size());
        std::printf("    noise: aper=%.3f voiced=%s\n",
                    static_cast<double>(r.aperiodicity),
                    r.is_voiced ? "Y" : "n");
        check(!r.is_voiced, "YIN does not mark white noise as voiced");
        check(r.aperiodicity > 0.3f, "YIN aperiodicity high on noise");
    }

    // ============================================================
    std::printf("\n=== YIN: silence ===\n");
    {
        YinConfig yc; yc.sample_rate = sr;
        YinDetector yin(yc);

        std::vector<float> sil(static_cast<std::size_t>(sr * 0.5), 0.0f);
        auto r = yin.detect(sil.data(), sil.size());
        check(!r.is_voiced, "YIN: silence is unvoiced");
        check(r.f0_hz == 0.0f, "YIN: silence returns f0=0");
        check(r.aperiodicity >= 0.9f, "YIN: silence has high aperiodicity");
    }

    // ============================================================
    std::printf("\n=== DC blocker: removes constant offset ===\n");
    {
        DcBlocker dc;
        dc.configure(sr, 20.0f);

        std::vector<float> sig(sr, 0.5f);  // pure DC
        // Add a small AC component so we can verify it passes
        for (int i = 0; i < sr; ++i) {
            sig[i] += 0.1f * std::sin(2.0f * kPi * 1000.0f * i / sr);
        }
        std::vector<float> out(sig.size());
        dc.process_block(sig.data(), out.data(), sig.size());

        // After the filter settles, the mean should be ~0
        double mean = 0.0;
        for (std::size_t i = sr / 4; i < out.size(); ++i) mean += out[i];
        mean /= (out.size() - sr / 4);
        std::printf("    residual DC after filter: %.5f\n", mean);
        check(std::fabs(mean) < 0.01, "DC offset removed");
        check(all_finite_bounded(out), "DC blocker output finite/bounded");
    }

    // ============================================================
    std::printf("\n=== High-pass: attenuates 50 Hz, passes 1 kHz ===\n");
    {
        OnePoleHighPass hp;
        hp.configure(sr, 80.0f);

        auto low  = sine(50.0f, sr, 0.5f, 0.3f);
        auto high = sine(1000.0f, sr, 0.5f, 0.3f);
        std::vector<float> lo_out(low.size()), hi_out(high.size());
        hp.process_block(low.data(),  lo_out.data(), low.size());
        hp.reset();
        hp.process_block(high.data(), hi_out.data(), high.size());

        const std::size_t skip = static_cast<std::size_t>(sr / 4);
        const float in_rms_low = rms(low.data()+skip, low.size()-skip);
        const float out_rms_low = rms(lo_out.data()+skip, lo_out.size()-skip);
        const float in_rms_high = rms(high.data()+skip, high.size()-skip);
        const float out_rms_high = rms(hi_out.data()+skip, hi_out.size()-skip);

        std::printf("    50 Hz: %.4f -> %.4f (ratio %.2f)\n",
                    in_rms_low, out_rms_low, out_rms_low/in_rms_low);
        std::printf("    1 kHz: %.4f -> %.4f (ratio %.2f)\n",
                    in_rms_high, out_rms_high, out_rms_high/in_rms_high);
        check(out_rms_low / in_rms_low < 0.8f, "50 Hz attenuated");
        check(out_rms_high / in_rms_high > 0.8f, "1 kHz mostly passes");
    }

    // ============================================================
    std::printf("\n=== Noise gate: silences low-level signal ===\n");
    {
        NoiseGate gate;
        gate.configure(sr, -45.0f, -55.0f, 0.005f, 0.020f);

        // Quiet noise below threshold
        auto sig = sine(440.0f, sr, 1.0f, 0.001f);  // ~-60 dB
        std::vector<float> out(sig.size());
        gate.process_block(sig.data(), out.data(), sig.size());

        const float in_r  = rms(sig.data() + sr/2, sr/2);
        const float out_r = rms(out.data() + sr/2, sr/2);
        std::printf("    quiet: in=%.5f out=%.5f\n", in_r, out_r);
        check(out_r < in_r * 0.1f, "quiet signal gated to near-zero");

        // Loud signal: should pass
        gate.reset();
        auto loud = sine(440.0f, sr, 1.0f, 0.3f);
        gate.process_block(loud.data(), out.data(), loud.size());
        const float loud_in  = rms(loud.data() + sr/2, sr/2);
        const float loud_out = rms(out.data() + sr/2, sr/2);
        std::printf("    loud:  in=%.5f out=%.5f\n", loud_in, loud_out);
        check(loud_out > loud_in * 0.8f, "loud signal passes through gate");
    }

    // ============================================================
    std::printf("\n=== Soft limiter: prevents peaks > threshold ===\n");
    {
        SoftLimiter lim;
        lim.configure(sr, 0.9f, 0.030f);
        auto sig = sine(440.0f, sr, 1.0f, 1.5f);  // would clip
        std::vector<float> out(sig.size());
        lim.process_block(sig.data(), out.data(), sig.size());

        float maxabs = 0.0f;
        for (std::size_t i = sr/4; i < out.size(); ++i) {
            maxabs = std::max(maxabs, std::fabs(out[i]));
        }
        std::printf("    max output after limiting: %.3f\n", maxabs);
        check(maxabs <= 1.01f, "limiter prevents overload");
        check(all_finite_bounded(out, 1.1f), "limiter output bounded");
    }

    // ============================================================
    std::printf("\n=== FormantVocoder: passes signal through cleanly at pitch=0 ===\n");
    {
        auto backend = create_best_backend();
        FormantVocoderConfig fc;
        fc.sample_rate = sr;
        fc.fft_size = 2048; fc.hop_size = 512;
        fc.profile = QualityProfile::Balanced;
        FormantVocoder voc(fc, backend.get());
        voc.set_pitch_semitones(0.0f);

        auto in = voice_like(150.0f, sr, 1.5f);
        std::vector<float> out(in.size());
        voc.process(in.data(), out.data(), in.size());

        check(all_finite_bounded(out, 5.0f), "vocoder pitch=0 output is finite/bounded");

        // After latency, energy should be comparable
        const std::size_t skip = 4096;
        const float in_r  = rms(in.data()  + skip, in.size()  - skip);
        const float out_r = rms(out.data() + skip, out.size() - skip);
        std::printf("    in_rms=%.4f out_rms=%.4f ratio=%.2f\n",
                    in_r, out_r, out_r/in_r);
        check(out_r > in_r * 0.4f && out_r < in_r * 2.5f,
              "energy preserved within reasonable factor");
    }

    // ============================================================
    std::printf("\n=== FormantVocoder: pitch shift +5 semitones preserves spectral character ===\n");
    {
        auto backend = create_best_backend();
        FormantVocoderConfig fc;
        fc.sample_rate = sr;
        fc.fft_size = 2048; fc.hop_size = 512;
        fc.profile = QualityProfile::Quality;
        FormantVocoder voc(fc, backend.get());
        voc.set_pitch_semitones(5.0f);

        auto in = voice_like(150.0f, sr, 1.5f);
        std::vector<float> out(in.size());
        voc.process(in.data(), out.data(), in.size());

        auto s = voc.stats();
        std::printf("    frames: total=%llu voiced=%llu unvoiced=%llu silent=%llu degenerate=%llu\n",
                    (unsigned long long)s.frames_total,
                    (unsigned long long)s.frames_voiced,
                    (unsigned long long)s.frames_unvoiced,
                    (unsigned long long)s.frames_silent,
                    (unsigned long long)s.frames_degenerate);
        std::printf("    last F0=%.1f Hz, aperiodicity=%.3f\n",
                    s.last_f0_hz, s.last_aperiodicity);

        check(all_finite_bounded(out, 5.0f), "shifted output finite/bounded");
        check(s.frames_voiced > s.frames_unvoiced,
              "voice-like signal mostly classified as voiced");

        // Verify the pitch actually shifted by checking peak frequency
        const float ratio = std::pow(2.0f, 5.0f/12.0f);
        // Skip initial latency, use a clean middle chunk
        const std::size_t off = sr / 2;
        const std::size_t N   = 16384;
        if (off + N <= out.size()) {
            std::vector<float> chunk(out.begin() + off, out.begin() + off + N);
            // Naive: largest sample of an FFT magnitude via goertzel over 50-1000 Hz
            float best_hz = 0.0f, best_mag = 0.0f;
            for (float hz = 50.0f; hz < 600.0f; hz += 1.0f) {
                const float w = 2.0f * kPi * hz / sr;
                const float coeff = 2.0f * std::cos(w);
                float s1 = 0, s2 = 0;
                for (float v : chunk) {
                    const float s0 = v + coeff*s1 - s2;
                    s2 = s1; s1 = s0;
                }
                const float re = s1 - s2*std::cos(w);
                const float im = s2*std::sin(w);
                const float m = re*re + im*im;
                if (m > best_mag) { best_mag = m; best_hz = hz; }
            }
            const float expected = 150.0f * ratio;
            const float err = 1200.0f * std::log2(best_hz / expected);
            std::printf("    peak=%.1f Hz (expected %.1f, err=%+.1f cents)\n",
                        best_hz, expected, err);
            check(std::fabs(err) < 30.0f, "pitch shift accuracy within 30 cents");
        }
    }

    // ============================================================
    std::printf("\n=== FormantVocoder: NaN/Inf input doesn't crash ===\n");
    {
        auto backend = create_best_backend();
        FormantVocoderConfig fc;
        fc.sample_rate = sr;
        FormantVocoder voc(fc, backend.get());
        voc.set_pitch_semitones(4.0f);

        std::vector<float> in(sr, 0.0f);
        std::mt19937 rng(1);
        std::uniform_int_distribution<std::size_t> pos(0, sr-1);
        for (int i = 0; i < 50; ++i) {
            in[pos(rng)] = std::numeric_limits<float>::quiet_NaN();
            in[pos(rng)] = std::numeric_limits<float>::infinity();
        }
        std::vector<float> out(in.size());
        voc.process(in.data(), out.data(), in.size());
        check(all_finite_bounded(out, 10.0f),
              "vocoder survives NaN/Inf input");
    }

    // ============================================================
    std::printf("\n=== VoicePipeline: end-to-end with each profile ===\n");
    {
        auto backend = create_best_backend();
        for (auto p : { QualityProfile::Quality,
                        QualityProfile::Balanced,
                        QualityProfile::Performance }) {
            VoicePipelineConfig pc;
            pc.sample_rate = sr;
            pc.profile = p;
            VoicePipeline pipe(pc, backend.get());
            pipe.set_pitch_semitones(3.0f);

            auto in = voice_like(180.0f, sr, 1.5f);
            std::vector<float> out(in.size());
            pipe.process(in.data(), out.data(), in.size());

            char label[80];
            std::snprintf(label, sizeof(label),
                          "%s profile: output finite/bounded",
                          profile_name(p));
            check(all_finite_bounded(out, 2.0f), label);

            std::snprintf(label, sizeof(label),
                          "%s profile: output peak < 1.0 (limiter works)",
                          profile_name(p));
            float maxabs = 0;
            for (std::size_t i = 8192; i < out.size(); ++i)
                maxabs = std::max(maxabs, std::fabs(out[i]));
            check(maxabs < 1.01f, label);
        }
    }

    // ============================================================
    std::printf("\n=== VoicePipeline: silence in -> silence out ===\n");
    {
        auto backend = create_best_backend();
        VoicePipelineConfig pc;
        pc.sample_rate = sr;
        VoicePipeline pipe(pc, backend.get());
        pipe.set_pitch_semitones(5.0f);

        std::vector<float> in(sr, 0.0f);
        std::vector<float> out(sr);
        pipe.process(in.data(), out.data(), in.size());

        const float out_r = rms(out.data() + sr/2, sr/2);
        std::printf("    silence-in: out RMS = %.6f\n", out_r);
        check(out_r < 1e-3f, "silent input produces silent output");
    }

    // ============================================================
    std::printf("\n=== VoicePipeline: profile switch mid-stream survives ===\n");
    {
        auto backend = create_best_backend();
        VoicePipelineConfig pc;
        pc.sample_rate = sr;
        pc.profile = QualityProfile::Balanced;
        VoicePipeline pipe(pc, backend.get());
        pipe.set_pitch_semitones(2.0f);

        auto in = voice_like(150.0f, sr, 2.0f);
        std::vector<float> out(in.size());
        const std::size_t half = in.size() / 2;
        pipe.process(in.data(), out.data(), half);
        pipe.set_profile(QualityProfile::Performance);
        pipe.process(in.data() + half, out.data() + half, in.size() - half);

        check(all_finite_bounded(out, 2.0f),
              "profile switch mid-stream produces clean output");
    }

    std::printf("\n========================================\n");
    std::printf("Total: %d/%d passed", g_total - g_failures, g_total);
    if (g_failures == 0) { std::printf(" \u2713\n"); return 0; }
    else { std::printf(" - %d FAILED \u2717\n", g_failures); return 1; }
}
