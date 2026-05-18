// SPDX-License-Identifier: GPL-3.0-or-later
//
// PortAudio-based real-time audio I/O.
//
// Pipes microphone -> processor callback -> output device (which can be a
// virtual audio device for OBS / Discord). On Linux pick a PipeWire/PulseAudio
// virtual sink; on Windows use VB-Cable; on macOS use BlackHole.
#pragma once

#include <cstddef>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace aboba {

struct DeviceInfo {
    int         index;
    std::string name;
    int         max_input_channels;
    int         max_output_channels;
    double      default_sample_rate;
};

// User-supplied audio processor.
// `input` and `output` are mono float32 buffers of `n_samples` each.
using AudioProcessor =
    std::function<void(const float* input, float* output, std::size_t n_samples)>;

class RealtimeEngine {
public:
    RealtimeEngine();
    ~RealtimeEngine();

    RealtimeEngine(const RealtimeEngine&)            = delete;
    RealtimeEngine& operator=(const RealtimeEngine&) = delete;

    static std::vector<DeviceInfo> list_input_devices();
    static std::vector<DeviceInfo> list_output_devices();

    // Start the audio stream. -1 = default device.
    // Throws std::runtime_error on failure.
    void start(int input_device,
               int output_device,
               double sample_rate,
               std::size_t frames_per_buffer,
               AudioProcessor processor);

    void stop();
    bool is_running() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace aboba
