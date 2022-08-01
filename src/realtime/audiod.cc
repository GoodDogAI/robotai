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

    constexpr int32_t periods = 1;
    constexpr snd_pcm_uframes_t periodsize = 8192; 

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
    if (snd_pcm_hw_params_set_channels(pcm_handle, hwparams, 1) < 0) {
        fmt::print(stderr, "Error setting channels.\n");
        return EXIT_FAILURE;
    }

    // Set number of periods. Periods used to be called fragments. 
    if (snd_pcm_hw_params_set_periods(pcm_handle, hwparams, periods, 0) < 0) {
        fmt::print(stderr, "Error setting periods.\n");
        return EXIT_FAILURE;
    }

    if (snd_pcm_hw_params_set_buffer_size(pcm_handle, hwparams, (periodsize * periods)>>2) < 0) {
        fmt::print(stderr, "Error setting buffersize.\n");
        return EXIT_FAILURE;
    }
  
  
    // Actually set the HW parameters
    if (snd_pcm_hw_params(pcm_handle, hwparams) < 0) {
        fmt::print(stderr, "Error setting HW params.\n");
        return EXIT_FAILURE;
    }

    fmt::print("Audio device {} successfully opened\n", AUDIO_DEVICE_NAME);

    int pcmreturn;
    std::array<uint8_t, periodsize> buf;

    while ((pcmreturn = snd_pcm_readi(pcm_handle, static_cast<void*>(&buf), periodsize>>2)) < 0) {
        snd_pcm_prepare(pcm_handle);
        fmt::print("{}", buf);
    }

    return EXIT_SUCCESS;
}