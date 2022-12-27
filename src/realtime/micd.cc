#include <array>
#include <string>
#include <thread>
#include <algorithm>
#include <chrono>
#include <alsa/asoundlib.h>
#include <fmt/core.h>
#include <fmt/ranges.h>

#include "cereal/messaging/messaging.h"

#include "util.h"
#include "config.h"

const char *service_name = "micData";
ExitHandler do_exit;

int main(int argc, char *argv[])
{
    PubMaster pm { {service_name} };

    snd_pcm_t *pcm_handle {};          
    snd_pcm_stream_t stream_capture { SND_PCM_STREAM_CAPTURE };
    snd_pcm_hw_params_t *hwparams {};
    uint32_t sample_rate { AUDIO_SAMPLE_RATE };
    snd_pcm_format_t pcm_format {snd_pcm_format_value(AUDIO_PCM_FORMAT)};

    int32_t frame_size { snd_pcm_format_width(pcm_format)/8 * AUDIO_CHANNELS };
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

    // We can do interleaved access, which is the more common way to use ALSA
    if (snd_pcm_hw_params_set_access(pcm_handle, hwparams, SND_PCM_ACCESS_RW_INTERLEAVED) < 0) {
        fmt::print(stderr, "Error setting access.\n");
        return EXIT_FAILURE;
    }

    // Verify and set sample format
    if (pcm_format == SND_PCM_FORMAT_UNKNOWN) {
        fmt::print(stderr, "Error setting PCM format {} not recognized.\n", AUDIO_PCM_FORMAT);
        return EXIT_FAILURE;
    }

    if (snd_pcm_hw_params_set_format(pcm_handle, hwparams, pcm_format) < 0) {
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
    bool firstcheck = true;
    auto buf = kj::heapArray<int32_t>(frames * AUDIO_CHANNELS);
    auto ch0 = kj::heapArray<float>(frames);

    std::fill(ch0.begin(), ch0.end(), 0.0f);

    while (!do_exit) {
        pcmreturn = snd_pcm_readi(pcm_handle, buf.begin(), frames);
        if (pcmreturn == -EPIPE) {
            fmt::print(stderr, "overrun occurred\n");
            snd_pcm_prepare(pcm_handle);
            return EXIT_FAILURE;
        } else if (pcmreturn < 0) {
            fmt::print(stderr, "error from read: {}\n", snd_strerror(pcmreturn));    
            return EXIT_FAILURE;
        }
        else if (pcmreturn != frames) {
            // For now, we only support dealing with full buffers
            fmt::print(stderr, "read incomplete buffer: size {} / {}\n", pcmreturn, frames);    

            if (do_exit){
                break;
            }
            else {
                return EXIT_FAILURE;
            }
        }

        // Convert only channel 0 to float
        for (size_t frame { 0 }; frame < pcmreturn; ++frame) {
            ch0[frame] = buf[frame * 2] / static_cast<float>(std::numeric_limits<int32_t>::max());
        }

        // Assert that the first frame has correct looking data
        if (firstcheck) {
            firstcheck = false;

            auto count = std::count_if(ch0.begin(), ch0.end(), [](float s) { return s != 0.0f; });
            fmt::print("Non zero count {}\n", count);

            float min = *std::min_element(ch0.begin(), ch0.end());
            float max = *std::max_element(ch0.begin(), ch0.end());
            
            if (count < frames * 0.90) {
                fmt::print(stderr, "microphone does not appear to be connected\n");    
                return EXIT_FAILURE; 
            }

            if (min > -0.1f || max < 0.1f) {
                fmt::print(stderr, "microphone samples appear very low valued, so it may not be plugged in\naborting early so you can check\n");    
                return EXIT_FAILURE; 
            }
        }

        //fmt::print("{} = {:032b} = {}\n", buf[0], buf[0], ch0[0]);

        MessageBuilder msg;
        auto event = msg.initEvent(true);
        auto mdat = event.initMicData();
        mdat.setMic(cereal::AudioData::MicrophonePlacement::MAIN_BODY);
        mdat.setChannel(0);
        mdat.setData(kj::ArrayPtr<float>(ch0.begin(), ch0.end()));
        
        auto words = capnp::messageToFlatArray(msg);
        auto bytes = words.asBytes();
        pm.send(service_name, bytes.begin(), bytes.size());
    }

    fmt::print("Min sample: {}  Max sample: {}\n", minSample, maxSample);

    snd_pcm_drain(pcm_handle);
    snd_pcm_close(pcm_handle);
    return EXIT_SUCCESS;
}