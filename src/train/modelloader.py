import torch
import onnx
import os

import polygraphy
import polygraphy.backend.trt

from src.train.config_train import VISION_CONFIGS, CACHE_DIR
from src.include.config import load_realtime_config

CONFIG = load_realtime_config()
DECODE_WIDTH = int(CONFIG["CAMERA_WIDTH"])
DECODE_HEIGHT = int(CONFIG["CAMERA_HEIGHT"])


# Loads a preconfigured model from a pytorch checkpoint,
# if needed, it builds a new tensorRT engine, and verifies that the model results are identical
def load_vision_model(config: str) -> polygraphy.backend.trt.TrtRunner:
    config = VISION_CONFIGS[config]
    assert config is not None, "Unable to find config"

    batch_size = 1
    img_size = (DECODE_HEIGHT, DECODE_WIDTH)
    device = "cuda:0"

    # Load the original pytorch model
    model = config["load_fn"](config["checkpoint"])

    img = torch.zeros(batch_size, 3, *img_size).to(device)  # image size(1,3,320,192) iDetection

    y = model(img)  # dry run

    # Convert that to ONNX
    onnx_path = os.path.join(CACHE_DIR, "onnx", os.path.basename(config["checkpoint"]).replace(".pt", ".onnx"))
    print('\nStarting ONNX export with onnx %s...' % onnx.__version__)


