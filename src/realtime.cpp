// SPDX-License-Identifier: GPL-3.0-or-later
//
// PortAudio real-time engine — paranoid edition.
//
// Improvements over v0.2:
//   * Zero allocations in the audio callback. All scratch buffers are sized
//     and allocated in start(), then reused for the lifetime of the stream.
//   * Multi-channel input handling: if the device has >1 input channel, we
//     down-mix to mono before calling the user processor. Output is fanned
//     out to all output channels. Prevents silent failures with stereo mics.
//   * Hardened start()/stop() with a mutex to prevent races between threads.
//   * Clearer error messages with device names.
//   * Defensive: callback never throws, never allocates, never logs.
#include "aboba/realtime.hpp"

#include <portaudio.h>

#include <algorithm>
#include <atomic>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace aboba {

namespace {

[[noreturn]] void throw_pa(PaError err, const char* what) {
    throw std::runtime_error(
        std::string("PortAudio error in ") + what + ": " + Pa_GetErrorText(err));
}

class PaLifetime {
public:
    PaLifetime() {
        const PaError err = Pa_Initialize();
        if (err != paNoError) throw_pa(err, "Pa_Initialize");
    }
    ~PaLifetime() { Pa_Terminate(); }
    PaLifetime(const PaLifetime&) = delete;
    PaLifetime& operator=(const PaLifetime&) = delete;
};

// Single shared PortAudio init. Multiple RealtimeEngine instances are fine —
// PortAudio's static is initialized exactly once at first call and torn down
// at program exit. Thread-safe per the C++11 static-init rules.
struct PaGlobal {
    static PaLifetime& get() {
        static PaLifetime instance;
        return instance;
    }
};

}  // namespace

struct RealtimeEngine::Impl {
    PaStream*        stream    = nullptr;
    AudioProcessor   processor;
    std::atomic<bool> running{false};
    std::mutex       lifecycle_mutex;   // protects start/stop transitions

    // Channel layout (set in start(), constant during stream)
    int  in_channels  = 1;
    int  out_channels = 1;

    // Pre-allocated scratch buffers used by callback. NEVER resized after
    // start() returns. Sized for the worst-case PortAudio block.
    std::vector<float> mono_in;
    std::vector<float> mono_out;
    std::vector<float> silence;  // for output-only streams

    static int callback(const void* in,
                        void* out,
                        unsigned long frame_count,
                        const PaStreamCallbackTimeInfo*,
                        PaStreamCallbackFlags,
                        void* user_data) {
        auto* self = static_cast<Impl*>(user_data);

        // Safety net: if PortAudio asks for more frames than we sized for,
        // produce silence rather than buffer-overflow. Should never happen
        // because we pass frames_per_buffer at OpenStream time.
        if (frame_count > self->mono_in.size()) {
            std::fill(static_cast<float*>(out),
                      static_cast<float*>(out)
                          + frame_count * self->out_channels,
                      0.0f);
            return paContinue;
        }

        const float* in_interleaved = static_cast<const float*>(in);
        float*       out_interleaved = static_cast<float*>(out);

        // --- Down-mix input to mono -----------------------------------
        float* mono_in = self->mono_in.data();
        if (in_interleaved) {
            const int ch = self->in_channels;
            if (ch == 1) {
                std::copy(in_interleaved, in_interleaved + frame_count, mono_in);
            } else {
                const float inv_ch = 1.0f / static_cast<float>(ch);
                for (unsigned long i = 0; i < frame_count; ++i) {
                    float sum = 0.0f;
                    const float* row = in_interleaved + i * ch;
                    for (int c = 0; c < ch; ++c) sum += row[c];
                    mono_in[i] = sum * inv_ch;
                }
            }
        } else {
            // No input device — feed silence (pre-allocated).
            std::copy(self->silence.data(),
                      self->silence.data() + frame_count,
                      mono_in);
        }

        // --- Run user processor (mono in → mono out) ------------------
        float* mono_out = self->mono_out.data();
        if (self->processor) {
            try {
                self->processor(mono_in, mono_out, frame_count);
            } catch (...) {
                // Callbacks must never throw. On exception, output silence
                // and keep the stream alive — better than crashing the
                // whole audio graph.
                std::fill(mono_out, mono_out + frame_count, 0.0f);
            }
        } else {
            std::copy(mono_in, mono_in + frame_count, mono_out);
        }

        // --- Fan-out to interleaved output -----------------------------
        const int oc = self->out_channels;
        if (oc == 1) {
            std::copy(mono_out, mono_out + frame_count, out_interleaved);
        } else {
            for (unsigned long i = 0; i < frame_count; ++i) {
                const float s = mono_out[i];
                float* row = out_interleaved + i * oc;
                for (int c = 0; c < oc; ++c) row[c] = s;
            }
        }
        return paContinue;
    }
};

RealtimeEngine::RealtimeEngine() : impl_(std::make_unique<Impl>()) {
    PaGlobal::get();
}

RealtimeEngine::~RealtimeEngine() {
    try { stop(); } catch (...) {}
}

std::vector<DeviceInfo> RealtimeEngine::list_input_devices() {
    PaGlobal::get();
    std::vector<DeviceInfo> out;
    const int count = Pa_GetDeviceCount();
    if (count < 0) throw_pa(count, "Pa_GetDeviceCount");
    for (int i = 0; i < count; ++i) {
        const PaDeviceInfo* info = Pa_GetDeviceInfo(i);
        if (!info || info->maxInputChannels <= 0) continue;
        out.push_back({i, info->name ? info->name : "?",
                       info->maxInputChannels, info->maxOutputChannels,
                       info->defaultSampleRate});
    }
    return out;
}

std::vector<DeviceInfo> RealtimeEngine::list_output_devices() {
    PaGlobal::get();
    std::vector<DeviceInfo> out;
    const int count = Pa_GetDeviceCount();
    if (count < 0) throw_pa(count, "Pa_GetDeviceCount");
    for (int i = 0; i < count; ++i) {
        const PaDeviceInfo* info = Pa_GetDeviceInfo(i);
        if (!info || info->maxOutputChannels <= 0) continue;
        out.push_back({i, info->name ? info->name : "?",
                       info->maxInputChannels, info->maxOutputChannels,
                       info->defaultSampleRate});
    }
    return out;
}

void RealtimeEngine::start(int input_device,
                           int output_device,
                           double sample_rate,
                           std::size_t frames_per_buffer,
                           AudioProcessor processor) {
    std::lock_guard<std::mutex> lock(impl_->lifecycle_mutex);

    if (impl_->running.load()) {
        throw std::runtime_error("RealtimeEngine: already running");
    }
    if (sample_rate <= 0.0 || sample_rate > 384000.0) {
        throw std::invalid_argument("RealtimeEngine: invalid sample rate");
    }
    if (frames_per_buffer == 0 || frames_per_buffer > 65536) {
        throw std::invalid_argument("RealtimeEngine: invalid frames_per_buffer");
    }

    // Resolve devices
    PaStreamParameters in_params{};
    in_params.device = (input_device < 0)
                           ? Pa_GetDefaultInputDevice() : input_device;
    if (in_params.device == paNoDevice) {
        throw std::runtime_error("No input device available");
    }
    const PaDeviceInfo* in_info = Pa_GetDeviceInfo(in_params.device);
    if (!in_info || in_info->maxInputChannels <= 0) {
        throw std::runtime_error("Selected input device has no input channels");
    }

    PaStreamParameters out_params{};
    out_params.device = (output_device < 0)
                            ? Pa_GetDefaultOutputDevice() : output_device;
    if (out_params.device == paNoDevice) {
        throw std::runtime_error("No output device available");
    }
    const PaDeviceInfo* out_info = Pa_GetDeviceInfo(out_params.device);
    if (!out_info || out_info->maxOutputChannels <= 0) {
        throw std::runtime_error("Selected output device has no output channels");
    }

    // Choose channel counts. Prefer mono if the device supports it; otherwise
    // open with the device's actual channel count and down/up-mix in callback.
    // Some virtual cables (VB-Cable, BlackHole) only expose 2 channels.
    impl_->in_channels  = std::min(in_info->maxInputChannels,  2);
    impl_->out_channels = std::min(out_info->maxOutputChannels, 2);

    in_params.channelCount       = impl_->in_channels;
    in_params.sampleFormat       = paFloat32;
    in_params.suggestedLatency   = in_info->defaultLowInputLatency;
    in_params.hostApiSpecificStreamInfo = nullptr;

    out_params.channelCount      = impl_->out_channels;
    out_params.sampleFormat      = paFloat32;
    out_params.suggestedLatency  = out_info->defaultLowOutputLatency;
    out_params.hostApiSpecificStreamInfo = nullptr;

    // Verify the requested format is actually supported BEFORE opening.
    // This gives clearer errors than a generic OpenStream failure.
    const PaError supported =
        Pa_IsFormatSupported(&in_params, &out_params, sample_rate);
    if (supported != paFormatIsSupported) {
        throw std::runtime_error(
            std::string("Format not supported by devices: ")
            + Pa_GetErrorText(supported)
            + " (input=" + in_info->name
            + ", output=" + out_info->name + ")");
    }

    // Pre-allocate scratch buffers (no allocation in callback after this).
    impl_->mono_in.assign(frames_per_buffer, 0.0f);
    impl_->mono_out.assign(frames_per_buffer, 0.0f);
    impl_->silence.assign(frames_per_buffer, 0.0f);
    impl_->processor = std::move(processor);

    PaError err = Pa_OpenStream(
        &impl_->stream,
        &in_params,
        &out_params,
        sample_rate,
        static_cast<unsigned long>(frames_per_buffer),
        paClipOff,
        &Impl::callback,
        impl_.get());
    if (err != paNoError) {
        impl_->processor = nullptr;
        throw_pa(err, "Pa_OpenStream");
    }

    err = Pa_StartStream(impl_->stream);
    if (err != paNoError) {
        Pa_CloseStream(impl_->stream);
        impl_->stream = nullptr;
        impl_->processor = nullptr;
        throw_pa(err, "Pa_StartStream");
    }
    impl_->running = true;
}

void RealtimeEngine::stop() {
    std::lock_guard<std::mutex> lock(impl_->lifecycle_mutex);
    if (!impl_->running.load()) return;

    if (impl_->stream) {
        // Pa_StopStream blocks until the callback finishes. After it returns
        // we know the callback won't fire again on this stream, so it's safe
        // to mutate Impl state.
        Pa_StopStream(impl_->stream);
        Pa_CloseStream(impl_->stream);
        impl_->stream = nullptr;
    }
    impl_->processor = nullptr;  // drop the closure
    impl_->running = false;
}

bool RealtimeEngine::is_running() const {
    return impl_->running.load();
}

}  // namespace aboba
