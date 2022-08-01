#pragma once

#define CAMERA_WIDTH 1280
#define CAMERA_HEIGHT 720
#define CAMERA_FPS 15

#define LOG_DURATION_SECONDS 60
#define LOG_PATH "/media/card"
#define LOG_SERVICE "http://jake-training-box.jakepoz.gmail.com.beta.tailscale.net:8000"

#define ENCODER_BITRATE 8'000'000
#define ENCODER_DEV "/dev/nvhost-msenc"
#define ENCODER_COMP_NAME "NVENC"

#define AUDIO_DEVICE_NAME "hw:tegrasndt19xmob,0"
#define AUDIO_PCM_FORMAT SND_PCM_FORMAT_S32_LE
#define AUDIO_SAMPLE_RATE 48'000
#define AUDIO_CHANNELS 1
