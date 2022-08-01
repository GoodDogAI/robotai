#include <cstdlib>
#include <string>
#include <alsa/asoundlib.h>
#include <fmt/core.h>

#include "config.h"

int main(int argc, char *argv[])
{
    snd_pcm_t *pcm_handle;          
    snd_pcm_stream_t stream_capture = SND_PCM_STREAM_CAPTURE;
    snd_pcm_hw_params_t *hwparams;
    int sample_rate = AUDIO_SAMPLE_RATE;

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

    if (rate != AUDIO_SAMPLE_RATE) {
        fmt::print(stderr, "The rate {} Hz is not supported by your hardware.\n 
                            ==> Could use {} Hz instead.\n", rate, exact_rate);
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}