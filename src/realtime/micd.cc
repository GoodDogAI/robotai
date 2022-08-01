#include <cstdlib>
#include <array>
#include <string>
#include <alsa/asoundlib.h>
#include <fmt/core.h>
#include <fmt/ranges.h>

#include "config.h"

int main(int argc, char *argv[])
{
    snd_pcm_t *pcm_handle;          
    snd_pcm_stream_t stream_capture = SND_PCM_STREAM_CAPTURE;
    snd_pcm_hw_params_t *hwparams;
    uint32_t sample_rate = AUDIO_SAMPLE_RATE;

    int32_t frame_size = snd_pcm_format_width(AUDIO_PCM_FORMAT)/8 * AUDIO_CHANNELS;
    snd_pcm_uframes_t frames = 500;

    // Allocate the hw_params struct
    snd_pcm_hw_params_alloca(&hwparams);

    if (snd_pcm_open(&pcm_handle, AUDIO_DEVICE_NAME, stream_capture, 0) < 0) {
        fmt::print(stderr, "Error opening PCM device {} - {}\n", AUDIO_DEVICE_NAME, errno);
        return EXIT_FAILURE;
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
    auto buf = std::make_unique<int32_t[]>(frames * AUDIO_CHANNELS);
  
    for (;;) {
        pcmreturn = snd_pcm_readi(pcm_handle, buf.get(), frames);
        if (pcmreturn == -EPIPE) {
            fmt::print(stderr, "overrun occurred\n");
            snd_pcm_prepare(pcm_handle);
            return EXIT_FAILURE;
        } else if (pcmreturn < 0) {
            fmt::print(stderr, "error from read: {}\n", snd_strerror(pcmreturn));    
            return EXIT_FAILURE;
        }

        fmt::print("read {} frames\n", pcmreturn);
        fmt::print("{} {} {} {} \n", buf[0], buf[1], buf[2], buf[3]);
    }

    snd_pcm_drain(pcm_handle);
    snd_pcm_close(pcm_handle);
    return EXIT_SUCCESS;
}