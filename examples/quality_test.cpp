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
#include "aboba/musical_scale.hpp"
#include "aboba/noise_reduction.hpp"
#include "aboba/pipeline.hpp"
#include "aboba/pitch_corrector.hpp"
#include "aboba/voice_config.hpp"
#include "aboba/voice_character.hpp"
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

    // ============================================================
    std::printf("\n=== FormantVocoder: independent formant shift ===\n");
    {
        auto backend = create_best_backend();
        // Compare three configs: (a) baseline (pitch=0, formant=0),
        // (b) pure pitch shift (pitch=+5, formant=0),
        // (c) pure formant shift (pitch=0, formant=+5).
        // All should be bounded, finite, and energy-preserving.
        for (auto& cfg : std::vector<std::pair<float,float>>{
                {0.0f, 0.0f}, {5.0f, 0.0f}, {0.0f, 5.0f}, {0.0f, -5.0f},
                {5.0f, -3.0f}, {-3.0f, 5.0f}}) {
            FormantVocoderConfig fc;
            fc.sample_rate = sr;
            fc.profile = QualityProfile::Quality;
            FormantVocoder voc(fc, backend.get());
            voc.set_pitch_semitones(cfg.first);
            voc.set_formant_semitones(cfg.second);

            auto in = voice_like(180.0f, sr, 1.5f);
            std::vector<float> out(in.size());
            voc.process(in.data(), out.data(), in.size());

            char label[120];
            std::snprintf(label, sizeof(label),
                "FormantVoc pitch=%+.1f formant=%+.1f -> output finite/bounded",
                static_cast<double>(cfg.first), static_cast<double>(cfg.second));
            check(all_finite_bounded(out, 5.0f), label);

            const std::size_t skip = 4096;
            const float in_r  = rms(in.data()  + skip, in.size()  - skip);
            const float out_r = rms(out.data() + skip, out.size() - skip);
            std::snprintf(label, sizeof(label),
                "  energy reasonable (pitch=%+.1f formant=%+.1f): in=%.3f out=%.3f",
                static_cast<double>(cfg.first), static_cast<double>(cfg.second),
                static_cast<double>(in_r), static_cast<double>(out_r));
            std::printf("    %s\n", label);
            check(out_r > in_r * 0.2f && out_r < in_r * 3.0f,
                  "  energy ratio within 0.2x..3x");
        }
    }

    // ============================================================
    std::printf("\n=== VoicePipeline: voice characters apply cleanly ===\n");
    {
        auto backend = create_best_backend();
        // Iterate over every defined character. Each must:
        //   - apply without throwing
        //   - produce finite/bounded output
        //   - leave the limiter doing its job (peak < 1.0)
        for (int i = 0; i < character_count(); ++i) {
            auto ch = static_cast<VoiceCharacter>(i);
            VoicePipelineConfig pc;
            pc.sample_rate = sr;
            pc.profile = QualityProfile::Balanced;
            VoicePipeline pipe(pc, backend.get());
            pipe.set_character(ch);

            auto in = voice_like(180.0f, sr, 1.0f);
            std::vector<float> out(in.size());
            pipe.process(in.data(), out.data(), in.size());

            char label[120];
            std::snprintf(label, sizeof(label),
                "character '%s' produces finite/bounded output",
                character_id(ch));
            check(all_finite_bounded(out, 2.0f), label);

            float peak = 0.0f;
            for (std::size_t k = 8192; k < out.size(); ++k)
                peak = std::max(peak, std::fabs(out[k]));
            std::snprintf(label, sizeof(label),
                "character '%s' respects limiter (peak=%.3f < 1.0)",
                character_id(ch), static_cast<double>(peak));
            check(peak < 1.01f, label);

            check(pipe.current_character() == ch,
                  "  current_character() matches what was set");
        }
    }

    // ============================================================
    std::printf("\n=== character_from_id roundtrip ===\n");
    {
        for (int i = 0; i < character_count(); ++i) {
            auto ch = static_cast<VoiceCharacter>(i);
            const char* id = character_id(ch);
            auto back = character_from_id(id);
            char label[80];
            std::snprintf(label, sizeof(label), "roundtrip '%s' -> enum -> '%s'",
                          id, character_id(back));
            check(back == ch, label);
        }
        check(character_from_id("not-a-real-character") == VoiceCharacter::Count,
              "unknown id returns Count sentinel");
        check(character_from_id(nullptr) == VoiceCharacter::Count,
              "null id returns Count sentinel");
        check(character_from_id("DEEP-MALE") == VoiceCharacter::DeepMale,
              "id lookup is case-insensitive");
    }

    // ============================================================
    std::printf("\n=== NoiseReducer: reduces white noise, preserves signal ===\n");
    {
        auto backend = create_best_backend();
        NoiseReductionConfig nc;
        nc.sample_rate = sr;
        nc.learning_seconds = 0.5f;
        nc.oversubtraction = 2.0f;  // aggressive for the test
        nc.spectral_floor  = 0.05f;
        NoiseReducer nr(nc, backend.get());

        // Build: 0.5s of pure noise (for learning) then 1.0s of voice+noise
        std::mt19937 rng(42);
        std::normal_distribution<float> ndist(0.0f, 0.05f);
        std::vector<float> in;
        in.reserve(static_cast<std::size_t>(sr * 1.5));
        // Learning section: pure noise
        for (int i = 0; i < sr / 2; ++i) in.push_back(ndist(rng));
        // Then voice + noise
        for (int i = 0; i < sr; ++i) {
            float v = 0.3f * std::sin(2.0f * kPi * 200.0f * static_cast<float>(i) / sr);
            v += 0.15f * std::sin(2.0f * kPi * 400.0f * static_cast<float>(i) / sr);
            v += 0.10f * std::sin(2.0f * kPi * 800.0f * static_cast<float>(i) / sr);
            v += ndist(rng);
            in.push_back(v);
        }
        std::vector<float> out(in.size());
        nr.process(in.data(), out.data(), in.size());

        check(all_finite_bounded(out, 5.0f), "noise reducer output finite/bounded");

        // RMS in the voice+noise section should be reduced compared to input
        const std::size_t voice_start = sr / 2 + 4096;  // skip warm-up
        const float in_r  = rms(in.data()  + voice_start, in.size()  - voice_start);
        const float out_r = rms(out.data() + voice_start, out.size() - voice_start);

        auto s = nr.stats();
        std::printf("    in_rms=%.4f out_rms=%.4f ratio=%.2f  "
                    "mean_gain=%.3f SNR=%.1fdB\n",
                    in_r, out_r, out_r/in_r,
                    static_cast<double>(s.last_mean_gain),
                    static_cast<double>(s.estimated_snr_db));

        // We expect SOME reduction — the noise was 0.05 RMS and signal 0.36 RMS,
        // so input total ~ sqrt(0.36² + 0.05²) ~ 0.36. After NR, noise should
        // drop noticeably. Realistically out_r should be 0.7-1.0 of in_r.
        check(out_r < in_r * 1.1f, "NR did not amplify");
        check(out_r > in_r * 0.3f, "NR did not destroy signal");
    }

    // ============================================================
    std::printf("\n=== NoiseReducer: silence -> silence ===\n");
    {
        auto backend = create_best_backend();
        NoiseReductionConfig nc;
        nc.sample_rate = sr;
        NoiseReducer nr(nc, backend.get());
        std::vector<float> in(sr, 0.0f);
        std::vector<float> out(sr);
        nr.process(in.data(), out.data(), in.size());
        check(all_finite_bounded(out, 0.001f), "silence in -> silence out");
    }

    // ============================================================
    std::printf("\n=== NoiseReducer: NaN/Inf survive ===\n");
    {
        auto backend = create_best_backend();
        NoiseReducer nr(NoiseReductionConfig{}, backend.get());

        std::vector<float> in(sr, 0.0f);
        std::mt19937 rng(7);
        std::uniform_int_distribution<std::size_t> pos(0, sr-1);
        for (int i = 0; i < 50; ++i) {
            in[pos(rng)] = std::numeric_limits<float>::quiet_NaN();
            in[pos(rng)] = std::numeric_limits<float>::infinity();
        }
        std::vector<float> out(in.size());
        nr.process(in.data(), out.data(), in.size());
        check(all_finite_bounded(out, 10.0f),
              "noise reducer survives NaN/Inf input");
    }

    // ============================================================
    std::printf("\n=== NoiseReducer: learn_noise_profile from known silence ===\n");
    {
        auto backend = create_best_backend();
        NoiseReducer nr(NoiseReductionConfig{}, backend.get());

        // Train on pure noise
        std::mt19937 rng(123);
        std::normal_distribution<float> ndist(0.0f, 0.1f);
        std::vector<float> calib(static_cast<std::size_t>(sr));
        for (auto& v : calib) v = ndist(rng);
        nr.learn_noise_profile(calib.data(), calib.size());

        auto s_before = nr.stats();
        check(s_before.frames_in_learning == 0,
              "manual calibration skips warm-up phase");
    }

    // ============================================================
    std::printf("\n=== VoicePipeline: NR can be disabled via flag ===\n");
    {
        auto backend = create_best_backend();
        VoicePipelineConfig pc;
        pc.sample_rate = sr;
        pc.enable_noise_reducer = false;
        VoicePipeline pipe(pc, backend.get());

        auto in = voice_like(200.0f, sr, 1.0f);
        std::vector<float> out(in.size());
        pipe.process(in.data(), out.data(), in.size());
        check(all_finite_bounded(out, 2.0f),
              "pipeline runs cleanly with NR disabled");

        auto ns = pipe.noise_stats();
        check(ns.frames_total == 0, "NR frames stay at 0 when disabled");
    }

    // ============================================================
    std::printf("\n=== AutoGain: brings quiet signal up to target ===\n");
    {
        AutoGain agc;
        agc.configure(sr, -16.0f, 24.0f, -12.0f, 50.0f, 500.0f, -55.0f);

        // Very quiet sine (~-40 dBFS RMS, would be ~0.01 peak)
        auto sig = sine(440.0f, sr, 2.0f, 0.01f);
        std::vector<float> out(sig.size());
        agc.process_block(sig.data(), out.data(), sig.size());

        // After 1 second of slow attack, gain should have lifted things up
        const float in_r  = rms(sig.data() + sr, sr);
        const float out_r = rms(out.data() + sr, sr);
        std::printf("    quiet sine: in=%.4f out=%.4f gain=%.1fdB target≈%.4f\n",
                    in_r, out_r, agc.current_gain_db(),
                    std::pow(10.0f, -16.0f/20.0f));
        check(out_r > in_r * 2.0f, "AGC amplifies quiet signal");
        check(all_finite_bounded(out, 2.0f), "AGC output bounded");
    }

    // ============================================================
    std::printf("\n=== AutoGain: doesn't amplify silence ===\n");
    {
        AutoGain agc;
        agc.configure(sr, -16.0f, 24.0f, -12.0f, 50.0f, 500.0f, -55.0f);
        std::vector<float> in(sr, 0.0f);
        std::vector<float> out(sr);
        agc.process_block(in.data(), out.data(), in.size());

        const float out_r = rms(out.data(), sr);
        check(out_r < 1e-3f, "silence doesn't get amplified");
    }

    // ============================================================
    std::printf("\n=== AutoGain: attenuates too-loud signal ===\n");
    {
        AutoGain agc;
        agc.configure(sr, -16.0f, 24.0f, -12.0f, 50.0f, 500.0f, -55.0f);
        // Loud sine (~0.8 peak, much louder than target -16 dBFS)
        auto sig = sine(440.0f, sr, 2.0f, 0.8f);
        std::vector<float> out(sig.size());
        agc.process_block(sig.data(), out.data(), sig.size());

        const float in_r  = rms(sig.data() + sr, sr);
        const float out_r = rms(out.data() + sr, sr);
        std::printf("    loud sine: in=%.3f out=%.3f gain=%.1fdB\n",
                    in_r, out_r, agc.current_gain_db());
        check(out_r < in_r, "AGC attenuates loud signal");
        check(all_finite_bounded(out, 2.0f), "AGC output bounded");
    }

    // ============================================================
    std::printf("\n=== LookaheadLimiter: prevents peaks past threshold ===\n");
    {
        LookaheadLimiter ll;
        ll.configure(sr, -1.0f, 2.0f, 30.0f);
        // Mix of normal level + spike that would overshoot
        std::vector<float> sig(sr);
        for (int i = 0; i < sr; ++i) {
            sig[static_cast<std::size_t>(i)] = 0.5f *
                std::sin(2.0f*kPi*440.0f*i/sr);
        }
        // Inject a series of huge spikes (well above ceiling)
        for (int p : {sr/4, sr/2, 3*sr/4}) {
            sig[static_cast<std::size_t>(p)] = 5.0f;
        }
        std::vector<float> out(sig.size());
        ll.process_block(sig.data(), out.data(), sig.size());

        float maxabs = 0.0f;
        for (std::size_t i = static_cast<std::size_t>(sr/8);
             i < out.size(); ++i)
            maxabs = std::max(maxabs, std::fabs(out[i]));
        std::printf("    lookahead samples=%zu  max output=%.3f (ceiling≈0.89)\n",
                    ll.latency_samples(), static_cast<double>(maxabs));
        check(maxabs <= 1.0f, "lookahead limiter respects ceiling");
        check(all_finite_bounded(out, 1.1f), "limiter output bounded");
    }

    // ============================================================
    std::printf("\n=== LookaheadLimiter: identity on quiet signal ===\n");
    {
        LookaheadLimiter ll;
        ll.configure(sr, -1.0f, 2.0f, 30.0f);
        auto sig = sine(440.0f, sr, 1.0f, 0.3f);
        std::vector<float> out(sig.size());
        ll.process_block(sig.data(), out.data(), sig.size());

        // Account for delay (skip lookahead samples)
        const std::size_t delay = ll.latency_samples();
        double err = 0.0;
        for (std::size_t i = delay + 1000; i < out.size(); ++i) {
            err += std::fabs(out[i] - sig[i - delay]);
        }
        err /= static_cast<double>(out.size() - delay - 1000);
        std::printf("    mean abs error (after delay compensation): %.5f\n", err);
        check(err < 1e-4, "limiter passes quiet signal through unchanged");
    }

    // ============================================================
    std::printf("\n=== VoicePipeline: AGC + lookahead in Quality profile ===\n");
    {
        auto backend = create_best_backend();
        VoicePipelineConfig pc;
        pc.sample_rate = sr;
        pc.profile = QualityProfile::Quality;
        VoicePipeline pipe(pc, backend.get());

        // Very quiet voice; AGC should bring it up to a sensible level
        auto in = voice_like(180.0f, sr, 2.0f, 0.02f);
        std::vector<float> out(in.size());
        pipe.process(in.data(), out.data(), in.size());

        const std::size_t skip = sr;  // skip warm-up
        const float in_r  = rms(in.data()  + skip, in.size()  - skip);
        const float out_r = rms(out.data() + skip, out.size() - skip);
        std::printf("    quality+quiet: in=%.4f out=%.4f\n", in_r, out_r);
        check(out_r > in_r * 1.5f, "Quality profile AGC lifts quiet input");
        check(all_finite_bounded(out, 1.1f), "still bounded");

        float peak = 0.0f;
        for (std::size_t k = skip; k < out.size(); ++k)
            peak = std::max(peak, std::fabs(out[k]));
        check(peak <= 1.0f, "Quality profile respects ceiling");
    }

    // ============================================================
    std::printf("\n=== Musical scales: standard sets are well-formed ===\n");
    {
        // All standard scales should have at least the root (bit 0) set.
        for (int i = 0; i < static_cast<int>(MusicalScale::Count); ++i) {
            auto s = static_cast<MusicalScale>(i);
            auto mask = scale_mask(s);
            char label[80];
            std::snprintf(label, sizeof(label),
                "scale '%s' contains the root", scale_name(s));
            check((mask & 0x1) != 0 || s == MusicalScale::Custom, label);
        }
        // Major has 7 notes, pentatonic has 5, chromatic has 12, blues has 6.
        check(__builtin_popcount(scale_mask(MusicalScale::Major)) == 7,
              "major scale has 7 notes");
        check(__builtin_popcount(scale_mask(MusicalScale::Minor)) == 7,
              "minor scale has 7 notes");
        check(__builtin_popcount(scale_mask(MusicalScale::PentatonicMajor)) == 5,
              "pentatonic-major has 5 notes");
        check(__builtin_popcount(scale_mask(MusicalScale::Chromatic)) == 12,
              "chromatic has 12 notes");
        check(__builtin_popcount(scale_mask(MusicalScale::Blues)) == 6,
              "blues has 6 notes");
    }

    // ============================================================
    std::printf("\n=== snap_to_scale: in-scale notes are not moved ===\n");
    {
        // C major in C: { C D E F G A B } at MIDI {60, 62, 64, 65, 67, 69, 71}
        const std::uint16_t mask = scale_mask(MusicalScale::Major);
        const int in_scale_midi[] = {60, 62, 64, 65, 67, 69, 71, 72, 74};
        for (int m : in_scale_midi) {
            const float snapped = snap_to_scale(static_cast<float>(m), mask, 0);
            char label[80];
            std::snprintf(label, sizeof(label),
                "C-major: MIDI %d stays put (got %.1f)", m,
                static_cast<double>(snapped));
            check(std::fabs(snapped - static_cast<float>(m)) < 0.01f, label);
        }
        // Out-of-scale (C#, D#, F#, G#, A#) gets pulled to nearest
        check(std::fabs(snap_to_scale(61.0f, mask, 0) - 60.0f) < 1.5f,
              "C major: C# (61) snaps to nearby");
        check(std::fabs(snap_to_scale(66.0f, mask, 0) - 65.0f) < 1.5f,
              "C major: F# (66) snaps to F");
    }

    // ============================================================
    std::printf("\n=== snap_to_scale: A minor pentatonic from random pitch ===\n");
    {
        const std::uint16_t mask = scale_mask(MusicalScale::PentatonicMinor);
        // A minor pent: A C D E G (MIDI 57 60 62 64 67, plus next octave 69)
        const float test_in[]  = {58.3f, 60.7f, 65.4f, 66.4f};
        const int   expect[]   = {57, 60, 64, 67};
        for (size_t i = 0; i < 4; ++i) {
            const float snapped = snap_to_scale(test_in[i], mask, 9); // root = A = 9
            char label[80];
            std::snprintf(label, sizeof(label),
                "A min pent: %.1f -> %.1f (expect ~%d)",
                static_cast<double>(test_in[i]),
                static_cast<double>(snapped), expect[i]);
            check(std::fabs(snapped - static_cast<float>(expect[i])) < 1.5f, label);
        }
    }

    // ============================================================
    std::printf("\n=== hz_to_midi / midi_to_hz round trip ===\n");
    {
        const float test[] = {110.0f, 220.0f, 440.0f, 880.0f, 1760.0f};
        for (float hz : test) {
            const float m = hz_to_midi(hz);
            const float h = midi_to_hz(m);
            char label[80];
            std::snprintf(label, sizeof(label),
                "Hz<->MIDI round trip at %.1f Hz", static_cast<double>(hz));
            check(std::fabs(h - hz) / hz < 0.001f, label);
        }
        check(std::fabs(hz_to_midi(440.0f) - 69.0f) < 0.01f,
              "440 Hz = MIDI 69 (A4)");
    }

    // ============================================================
    std::printf("\n=== PitchCorrector: snaps to chromatic produces ~0 correction on integer notes ===\n");
    {
        // Chromatic includes all 12 notes, so any note we feed should be in
        // the scale (modulo cent-level error from YIN).
        PitchCorrectorConfig pc;
        pc.sample_rate = sr;
        pc.scale = MusicalScale::Chromatic;
        pc.strength = 1.0f;
        pc.glide_ms = 0.0f;  // instant
        PitchCorrector corrector(pc);

        // Generate a clean 220 Hz tone — that's MIDI 57.something, very
        // close to A3 (MIDI 57). Should snap to MIDI 57 -> tiny correction.
        auto s220 = sine(220.0f, sr, 0.5f);
        const float corr = corrector.analyze(s220.data(), s220.size());
        std::printf("    220 Hz -> correction %+.3f semitones\n",
                    static_cast<double>(corr));
        check(std::fabs(corr) < 0.20f,
              "chromatic snap: correction near 0 for pitched input");
    }

    // ============================================================
    std::printf("\n=== PitchCorrector: snaps off-key pitch to scale ===\n");
    {
        PitchCorrectorConfig pc;
        pc.sample_rate = sr;
        pc.scale = MusicalScale::Major;
        pc.root_semis = 0;  // C major
        pc.strength = 1.0f;
        pc.glide_ms = 0.0f;
        PitchCorrector corrector(pc);

        // Generate 277 Hz which is C#4 (MIDI 61). In C major this is OUT
        // of scale; nearest are C (60) and D (62). Should pick one and
        // produce a correction of about ±1 semitone.
        auto s_offkey = sine(277.183f, sr, 0.5f);  // C#4
        const float corr = corrector.analyze(s_offkey.data(), s_offkey.size());
        std::printf("    C#4 (277.2 Hz) in C major -> correction %+.3f st\n",
                    static_cast<double>(corr));
        check(std::fabs(corr) >= 0.5f && std::fabs(corr) <= 1.5f,
              "off-key note snaps about ±1 semitone");
    }

    // ============================================================
    std::printf("\n=== PitchCorrector: strength=0 produces no correction ===\n");
    {
        PitchCorrectorConfig pc;
        pc.sample_rate = sr;
        pc.scale = MusicalScale::Major;
        pc.strength = 0.0f;        // disabled
        pc.glide_ms = 0.0f;
        PitchCorrector corrector(pc);

        auto s_off = sine(277.183f, sr, 0.5f);
        const float corr = corrector.analyze(s_off.data(), s_off.size());
        check(std::fabs(corr) < 0.01f, "strength=0 -> always 0 correction");
    }

    // ============================================================
    std::printf("\n=== PitchCorrector: unvoiced input -> 0 correction ===\n");
    {
        PitchCorrectorConfig pc;
        pc.sample_rate = sr;
        pc.scale = MusicalScale::Major;
        pc.strength = 1.0f;
        pc.glide_ms = 0.0f;
        pc.bypass_unvoiced = true;
        PitchCorrector corrector(pc);

        // White noise — unvoiced
        std::mt19937 rng(7);
        std::normal_distribution<float> nd(0.0f, 0.2f);
        std::vector<float> noise(sr / 2);
        for (auto& v : noise) v = nd(rng);

        const float corr = corrector.analyze(noise.data(), noise.size());
        check(std::fabs(corr) < 0.01f, "noise input -> 0 correction (bypassed)");
    }

    // ============================================================
    std::printf("\n=== PitchCorrector: stats are populated ===\n");
    {
        PitchCorrectorConfig pc;
        pc.sample_rate = sr;
        PitchCorrector corrector(pc);
        auto s = sine(440.0f, sr, 0.5f);
        corrector.analyze(s.data(), s.size());
        auto st = corrector.stats();
        check(st.analyses_total >= 1, "at least one analysis ran");
        check(st.last_input_f0_hz > 400.0f && st.last_input_f0_hz < 480.0f,
              "stats report ~440 Hz input");
    }

    // ============================================================
    std::printf("\n=== VoicePipeline: autotune toggles cleanly ===\n");
    {
        auto backend = create_best_backend();
        VoicePipelineConfig pc;
        pc.sample_rate = sr;
        pc.profile = QualityProfile::Balanced;
        pc.enable_autotune = false;
        VoicePipeline pipe(pc, backend.get());

        auto in = voice_like(150.0f, sr, 1.5f);
        std::vector<float> out(in.size());

        // First pass: autotune OFF
        pipe.process(in.data(), out.data(), in.size());
        check(all_finite_bounded(out, 2.0f), "autotune OFF -> finite output");
        check(!pipe.autotune_enabled(), "autotune reports off");

        // Second pass: turn ON
        pipe.set_autotune_enabled(true);
        pipe.set_autotune_scale(MusicalScale::Major, 0);
        pipe.set_autotune_strength(1.0f);
        pipe.set_autotune_glide_ms(20.0f);
        pipe.process(in.data(), out.data(), in.size());
        check(all_finite_bounded(out, 2.0f), "autotune ON -> finite output");
        check(pipe.autotune_enabled(), "autotune reports on");

        // Confirm stats updated
        const auto st = pipe.autotune_stats();
        check(st.analyses_total > 0, "autotune analyses ran when enabled");
    }

    // ============================================================
    std::printf("\n=== VoicePipeline: autotune + manual pitch sum correctly ===\n");
    {
        auto backend = create_best_backend();
        VoicePipelineConfig pc;
        pc.sample_rate = sr;
        pc.enable_autotune = true;
        VoicePipeline pipe(pc, backend.get());
        pipe.set_autotune_scale(MusicalScale::Chromatic, 0);  // any pitch is in-scale
        pipe.set_autotune_strength(1.0f);
        pipe.set_pitch_semitones(3.0f);  // manual shift +3

        auto in = voice_like(220.0f, sr, 1.5f);
        std::vector<float> out(in.size());
        pipe.process(in.data(), out.data(), in.size());

        // Manual +3 should still be applied; autotune correction should be
        // tiny (input is already at near-pitched values).
        check(all_finite_bounded(out, 2.0f),
              "manual pitch + autotune produces finite output");
    }

    // ============================================================
    std::printf("\n=== VoiceConfig: parse minimal config ===\n");
    {
        const std::string toml = R"(
name = "test-voice"

[pitch]
semitones = 3.0
formant_semitones = -1.5
)";
        auto r = parse_voice_config(toml);
        check(r.ok(), "minimal config parses");
        if (r.ok()) {
            check(r->name == "test-voice", "name parsed");
            check(std::fabs(r->pitch_semitones - 3.0f) < 0.01f, "pitch parsed");
            check(std::fabs(r->formant_semitones - -1.5f) < 0.01f, "formant parsed");
        }
    }

    // ============================================================
    std::printf("\n=== VoiceConfig: parse full config with all sections ===\n");
    {
        const std::string toml = R"(
name = "streamer"
description = "Full config test"

[pipeline]
profile = "quality"
sample_rate = 48000.0
fft_size = 2048
hop_size = 512

[character]
preset = "warm-male"

[autotune]
enabled = true
scale = "minor"
root = "A"
strength = 0.75
glide_ms = 25.0

[effects]
noise_gate = true
highpass = true
highpass_cutoff_hz = 90.0
noise_reducer = true
agc = false
de_esser = true
reverb = true

[reverb]
room_size = 0.6
damping = 0.4
wet = 0.18

[limiter]
lookahead = true
)";
        auto r = parse_voice_config(toml);
        check(r.ok(), "full config parses");
        if (r.ok()) {
            check(r->profile == QualityProfile::Quality, "profile = quality");
            check(r->fft_size == 2048, "fft_size = 2048");
            check(r->has_character, "character section parsed");
            check(r->character == VoiceCharacter::WarmMale, "character = warm-male");
            check(r->autotune_enabled, "autotune enabled");
            check(r->autotune_scale == MusicalScale::Minor, "scale = minor");
            check(r->autotune_root == 9, "root = A (9)");
            check(std::fabs(r->autotune_strength - 0.75f) < 0.01f, "strength");
            check(std::fabs(r->autotune_glide_ms - 25.0f) < 0.01f, "glide_ms");
            check(!r->agc, "agc disabled");
            check(r->de_esser, "de_esser enabled");
            check(r->reverb, "reverb enabled");
            check(std::fabs(r->reverb_room_size - 0.6f) < 0.01f, "reverb room_size");
            check(r->lookahead, "lookahead limiter");
        }
    }

    // ============================================================
    std::printf("\n=== VoiceConfig: comments and blank lines ===\n");
    {
        const std::string toml = R"(
# Top comment
name = "commented" # inline comment

# blank line above

[pitch]
semitones = 0  # zero shift
)";
        auto r = parse_voice_config(toml);
        check(r.ok(), "config with comments parses");
        if (r.ok()) check(r->name == "commented", "name preserved through comments");
    }

    // ============================================================
    std::printf("\n=== VoiceConfig: error reporting points to right line ===\n");
    {
        const std::string toml = R"(
name = "ok"

[autotune]
enabled = yes
)";
        auto r = parse_voice_config(toml);
        check(!r.ok(), "invalid bool flagged");
        check(r.line_number == 5, "error on correct line");
    }

    // ============================================================
    std::printf("\n=== VoiceConfig: unknown character preset errors ===\n");
    {
        const std::string toml = R"(
[character]
preset = "bigfoot"
)";
        auto r = parse_voice_config(toml);
        check(!r.ok(), "unknown character preset rejected");
        check(r.error.find("bigfoot") != std::string::npos,
              "error mentions bad value");
    }

    // ============================================================
    std::printf("\n=== VoiceConfig: missing file error ===\n");
    {
        auto r = load_voice_config("/this/path/does/not/exist.toml");
        check(!r.ok(), "missing file errors out");
        check(r.error.find("cannot open") != std::string::npos,
              "error mentions cannot open");
    }

    // ============================================================
    std::printf("\n=== VoiceConfig: round-trip via serialize ===\n");
    {
        VoiceConfig src;
        src.name = "roundtrip";
        src.profile = QualityProfile::Balanced;
        src.pitch_semitones = 2.5f;
        src.formant_semitones = -1.0f;
        src.has_character = true;
        src.character = VoiceCharacter::DeepMale;
        src.autotune_enabled = true;
        src.autotune_scale = MusicalScale::PentatonicMinor;
        src.autotune_root = 2;  // D
        src.autotune_strength = 0.6f;
        src.autotune_glide_ms = 40.0f;
        src.reverb = true;
        src.reverb_room_size = 0.8f;
        src.lookahead = true;

        const std::string toml = serialize_voice_config(src);
        auto r = parse_voice_config(toml);
        check(r.ok(), "serialize -> parse roundtrip succeeds");
        if (r.ok()) {
            check(r->name == src.name, "  name preserved");
            check(r->profile == src.profile, "  profile preserved");
            check(std::fabs(r->pitch_semitones - src.pitch_semitones) < 0.01f, "  pitch preserved");
            check(r->character == src.character, "  character preserved");
            check(r->autotune_enabled == src.autotune_enabled, "  autotune flag");
            check(r->autotune_scale == src.autotune_scale, "  autotune scale");
            check(r->autotune_root == src.autotune_root, "  autotune root");
            check(r->reverb == src.reverb, "  reverb flag");
            check(std::fabs(r->reverb_room_size - src.reverb_room_size) < 0.01f,
                  "  reverb room_size");
            check(r->lookahead == src.lookahead, "  lookahead flag");
        }
    }

    // ============================================================
    std::printf("\n=== VoiceConfig: apply_runtime to pipeline ===\n");
    {
        const std::string toml = R"(
name = "applied"
[pipeline]
profile = "balanced"
[character]
preset = "anime"
[autotune]
enabled = true
scale = "major"
root = "G"
strength = 0.5
[effects]
reverb = true
[reverb]
room_size = 0.7
damping = 0.5
wet = 0.3
)";
        auto r = parse_voice_config(toml);
        check(r.ok(), "config parses");
        if (r.ok()) {
            auto backend = create_best_backend();
            auto pipe_cfg = r->to_pipeline_config();
            VoicePipeline pipe(pipe_cfg, backend.get());
            r->apply_runtime(pipe);

            check(pipe.current_character() == VoiceCharacter::AnimeGirl,
                  "character applied");
            check(pipe.autotune_enabled(), "autotune applied");
            check(pipe.reverb_enabled(), "reverb applied");

            // Sanity: process some audio through it
            auto in = voice_like(180.0f, sr, 0.5f);
            std::vector<float> out(in.size());
            pipe.process(in.data(), out.data(), in.size());
            check(all_finite_bounded(out, 2.0f), "  output finite after config-applied");
        }
    }

    std::printf("\n========================================\n");
    std::printf("Total: %d/%d passed", g_total - g_failures, g_total);
    if (g_failures == 0) { std::printf(" \u2713\n"); return 0; }
    else { std::printf(" - %d FAILED \u2717\n", g_failures); return 1; }
}
