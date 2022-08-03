#include <array>
#include <string>
#include <thread>
#include <chrono>
#include <alsa/asoundlib.h>
#include <fmt/core.h>
#include <fmt/ranges.h>

#include "cereal/messaging/messaging.h"
#include "config.h"

const char *service_name = "micData";


int main(int argc, char *argv[])
{
    PubMaster pm { {service_name} };

    snd_pcm_t *pcm_handle {};          
    snd_pcm_stream_t stream_capture { SND_PCM_STREAM_CAPTURE };
    snd_pcm_hw_params_t *hwparams {};
    uint32_t sample_rate { AUDIO_SAMPLE_RATE };

    int32_t frame_size { snd_pcm_format_width(AUDIO_PCM_FORMAT)/8 * AUDIO_CHANNELS };
    snd_pcm_uframes_t frames { 500 };

    // Allocate the hw_params struct
    snd_pcm_hw_params_alloca(&hwparams);

    for(int retry = 0;; ++retry) {
        int ret = snd_pcm_open(&pcm_handle, AUDIO_DEVICE_NAME, stream_capture, 0);
        if (ret == -EBUSY) {
            if (retry < 10) {
                fmt::print(stderr, "Got EBUSY opening PCM device {}, retrying in 1 sec...\n", AUDIO_DEVICE_NAME);
                std::this_thread::sleep_for(std::chrono::seconds(1));
            }
            else {
                fmt::print(stderr, "Unable to open PCM device {}\n", AUDIO_DEVICE_NAME);
                return EXIT_FAILURE;
            }
        }
        else if (ret < 0) {
            fmt::print(stderr, "Error opening PCM device {} - {}\n", AUDIO_DEVICE_NAME, ret);
            return EXIT_FAILURE;
        }
        else {
            fmt::print(stderr, "Opened PCM device {}\n", AUDIO_DEVICE_NAME);
            break;
        }
    }

    // Init hwparams with full configuration space 
    if (snd_pcm_hw_params_any(pcm_handle, hwparams) < 0) {
        fmt::print(stderr, "Can not configure this PCM device.\n");
        return EXIT_FAILURE;
    }

    // We can do interleaved access, in case we ever go to stereo, but for now it doesn't matter
    if (snd_pcm_hw_params_set_access(pcm_handle, hwparams, SND_PCM_ACCESS_RW_INTERLEAVED) < 0) {
        fmt::print(stderr, "Error setting access.\n");
        return EXIT_FAILURE;
    }

    // Set sample format
    if (snd_pcm_hw_params_set_format(pcm_handle, hwparams, AUDIO_PCM_FORMAT) < 0) {
        fmt::print(stderr, "Error setting format.\n");
        return EXIT_FAILURE;
    }

    // Set and confirm sample rate
    if (snd_pcm_hw_params_set_rate_near(pcm_handle, hwparams, &sample_rate, 0) < 0) {
        fmt::print(stderr, "Error setting rate.\n");
        return EXIT_FAILURE;
    }

    if (sample_rate != AUDIO_SAMPLE_RATE) {
        fmt::print(stderr, "The rate {} Hz is not supported by your hardware.\n  ==> Could use {} Hz instead.\n", AUDIO_SAMPLE_RATE, sample_rate);
        return EXIT_FAILURE;
    }

    // Set number of channels
    static_assert(AUDIO_CHANNELS == 1, "This code only supports mono audio for now. You'd need to adjust it for stereo.");

    if (snd_pcm_hw_params_set_channels(pcm_handle, hwparams, AUDIO_CHANNELS) < 0) {
        fmt::print(stderr, "Error setting channels.\n");
        return EXIT_FAILURE;
    }

    if (snd_pcm_hw_params_set_period_size_near(pcm_handle, hwparams, &frames, 0) < 0) {
        fmt::print(stderr, "Error setting period size ({}).\n", frames);
        return EXIT_FAILURE; 
    }
  
    // Actually set the HW parameters
    if (snd_pcm_hw_params(pcm_handle, hwparams) < 0) {
        fmt::print(stderr, "Error setting HW params.\n");
        return EXIT_FAILURE;
    }

    fmt::print("Audio device {} successfully opened, with {} frames of size {}\n", AUDIO_DEVICE_NAME, frames, frame_size);

    int pcmreturn;
    auto buf = kj::heapArray<int32_t>(frames * AUDIO_CHANNELS);
    auto fbuf = kj::heapArray<float>(frames * AUDIO_CHANNELS);

    for (;;) {
        pcmreturn = snd_pcm_readi(pcm_handle, buf.begin(), frames);
        if (pcmreturn == -EPIPE) {
            fmt::print(stderr, "overrun occurred\n");
            snd_pcm_prepare(pcm_handle);
            return EXIT_FAILURE;
        } else if (pcmreturn < 0) {
            fmt::print(stderr, "error from read: {}\n", snd_strerror(pcmreturn));    
            return EXIT_FAILURE;
        }

        fmt::print("read {} frames\n", pcmreturn);
        fmt::print("{:032b}\n", buf[0]);

        // Convert to float
        for (size_t i { 0 }; i < buf.size(); i++) {
            fbuf[i] = buf[i] / static_cast<float>(std::numeric_limits<int32_t>::max());
        }

        MessageBuilder msg;
        auto event = msg.initEvent(true);
        auto mdat = event.initMicData();
        mdat.setMic(cereal::AudioData::MicrophonePlacement::MAIN_BODY);
        mdat.setChannel(0);
        mdat.setData(kj::ArrayPtr<float>(fbuf.begin(), fbuf.end()));
        
        auto words = capnp::messageToFlatArray(msg);
        auto bytes = words.asBytes();
        pm.send(service_name, bytes.begin(), bytes.size());
    }

    snd_pcm_drain(pcm_handle);
    snd_pcm_close(pcm_handle);
    return EXIT_SUCCESS;
}