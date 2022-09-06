import os
from src.config.dotdict import dotdict

DEVICE_CONFIG = dotdict({
    "CAMERA_WIDTH": 1280,
    "CAMERA_HEIGHT": 720,
    "CAMERA_FPS": 15,

    "LOG_DURATION_SECONDS": 60,
    "LOG_PATH": "/media/card",
    "LOG_SERVICE": "http://jake-training-box.jakepoz.gmail.com.beta.tailscale.net:8000",

    "MODEL_STORAGE_PATH": "/home/robot/models",
    "MODEL_SERVICE": "http://jake-training-box.jakepoz.gmail.com.beta.tailscale.net:8000",

    "ENCODER_BITRATE": 8_000_000,
    "ENCODER_DEV": "/dev/nvhost-msenc",
    "ENCODER_COMP_NAME": "NVENC",

    "AUDIO_DEVICE_NAME": "hw:APE,0",
    "AUDIO_PCM_FORMAT": "S32_LE",
    "AUDIO_SAMPLE_RATE": 48_000,
    "AUDIO_CHANNELS": 2,

    "ODRIVE_SERIAL_PORT": "/dev/ttyACM0",
    "ODRIVE_BAUD_RATE": 115_200,
    "ODRIVE_MAX_SPEED": 2.0,
 
    "SIMPLEBGC_SERIAL_PORT": "/dev/ttyTHS0",
    "SIMPLEBGC_BAUD_RATE": 115_200,
})


HOST_CONFIG = dotdict({
    "RECORD_DIR": os.environ.get("ROBOTAI_RECORD_DIR", "/media/storage/robotairecords"),
    "CACHE_DIR": os.environ.get("ROBOTAI_CACHE_DIR", "/media/storage/robotaicache"),

    "DEFAULT_BRAIN_CONFIG": "orange-hippo-1",

    "DEFAULT_DECODE_GPU_ID": 0,
})

BRAIN_CONFIGS = dotdict({
    "orange-hippo-1": {
        "models": {
            #"brain_model": "orange-hippo-1",
            "vision_model": "yolov7-tiny-s53",
        },

        "audio_resampler": "test1",
    }
})

# Configures how models turn from pytorch checkpoints into runnable objects on the host and device
MODEL_CONFIGS = dotdict({
    "yolov7-tiny-s53": {
        "type": "vision",
        "load_fn": "src.train.yolov7.load.load_yolov7",
        "checkpoint": "/home/jake/robotai/_checkpoints/yolov7-tiny.pt",

        # Input dimensions must be divisible by the stride
        # In current situations, the image will be cropped to the nearest multiple of the stride
        "dimension_stride": 32,

        "intermediate_layer": "input.236",
        "intermediate_slice": 53,
    }
})


WEB_CONFIG = dotdict({
   # Unknown
})

