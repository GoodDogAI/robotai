import os

DEVICE_CONFIG = {
    "CAMERA_WIDTH": 1280,
    "CAMERA_HEIGHT": 720,
    "CAMERA_FPS": 15,

    "LOG_DURATION_SECONDS": 60,
    "LOG_PATH": "/media/card",
    "LOG_SERVICE": "http://jake-training-box.jakepoz.gmail.com.beta.tailscale.net:8000",

    "ENCODER_BITRATE": 8_000_000,
    "ENCODER_DEV": "/dev/nvhost-msenc",
    "ENCODER_COMP_NAME": "NVENC",

    "AUDIO_DEVICE_NAME": "hw:APE,0",
    "AUDIO_PCM_FORMAT": "SND_PCM_FORMAT_S32_LE",
    "AUDIO_SAMPLE_RATE": 48_000,
    "AUDIO_CHANNELS": 2,

    "ODRIVE_SERIAL_PORT": "/dev/ttyACM0",
    "ODRIVE_BAUD_RATE": 115_200,
    "ODRIVE_MAX_SPEED": 2.0,
 
    "SIMPLEBGC_SERIAL_PORT": "/dev/ttyTHS0",
    "SIMPLEBGC_BAUD_RATE": 115_200,
}


HOST_CONFIG = {
    "RECORD_DIR": os.environ.get("ROBOTAI_RECORD_DIR", "/media/storage/robotairecords"),
    "CACHE_DIR": os.environ.get("ROBOTAI_CACHE_DIR", "/media/storage/robotaicache"),

    "DEFAULT_VISION_CONFIG": "yolov7-tiny-s53"
}

# Vision subsystem configuration
VISION_CONFIGS = {
    "yolov7-tiny-s53": {
        "load_fn": "src.train.yolov7.load_yolov7",
        "checkpoint": "/home/jake/robotai/_checkpoints/yolov7-tiny.pt",

        # Input dimensions must be divisible by the stride
        # In current situations, the image will be cropped to the nearest multiple of the stride
        "dimension_stride": 32,

        "intermediate_layer": "input.236",
        "intermediate_slice": 53,
    }
}


WEB_CONFIG = {
   "WEB_VIDEO_DECODE_GPU_ID": 0,
}
