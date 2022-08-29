import os
import torch
from pathlib import Path

from src.train.yolov7.load import load_yolov7


RECORD_DIR = os.environ.get("ROBOTAI_RECORD_DIR", "/media/storage/robotairecords/converted")
CACHE_DIR = os.environ.get("ROBOTAI_CACHE_DIR", "/media/storage/robotaicache")

# Make sure that the required directories exist when this config file gets loaded
Path(RECORD_DIR).mkdir(parents=True, exist_ok=True)
Path(CACHE_DIR).mkdir(parents=True, exist_ok=True)


# Vision subsystem configuration
VISION_INTERMEDIATE_CONFIG = "yolov7-tiny-s53"

VISION_CONFIGS = {
    "yolov7-tiny-s53": {
        "load_fn": load_yolov7,
        "checkpoint": "/home/jake/robotai/_checkpoints/yolov7-tiny.pt",

        # Input dimensions must be divisible by the stride
        # In current situations, the image will be cropped to the nearest multiple of the stride
        "dimension_stride": 32,

        "intermediate_layer": "input.236",
        "intermediate_slice": 53,
    }
}
