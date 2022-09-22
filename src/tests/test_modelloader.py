

import torch
import tensorrt as trt
import unittest

from src.config import HOST_CONFIG, BRAIN_CONFIGS
from src.train.modelloader import create_and_validate_onnx, create_and_validate_trt


class TestModelLoaderTRT(unittest.TestCase):
    def setUp(self) -> None:
        self.sampleVisionConfig = {
            "type": "vision",
            "load_fn": "src.train.yolov7.load.load_yolov7",
            "input_format": "rgb",
            "checkpoint": "/home/jake/robotai/_checkpoints/yolov7-tiny.pt",

            # Input dimensions must be divisible by the stride
            # In current situations, the image will be cropped to the nearest multiple of the stride
            "dimension_stride": 32,

            "intermediate_layer": "input.219", # Another option to try could be onnx::Conv_254
            "intermediate_slice": 53,
        }

        self.sampleRewardConfig = {
            "type": "reward",
            "load_fn": "src.train.yolov7.load.load_yolov7",
            "input_format": "rgb",
            "checkpoint": "/home/jake/robotai/_checkpoints/yolov7-tiny.pt",

            "class_weights": {
                "person": 3,
                "spoon": 10,
            },
            "global_reward_scale": 0.10,

            "detection_layer": "488",
            "detection_threshold": 0.50,
            "iou_threshold": 0.50,
        }

    def test_onnx_vision(self):
        create_and_validate_onnx(self.sampleVisionConfig, skip_cache=True)

    def test_trt_vision(self):
        onnx_path = create_and_validate_onnx(self.sampleVisionConfig)
        create_and_validate_trt(onnx_path, skip_cache=True)

    def test_onnx_reward(self):
        create_and_validate_onnx(self.sampleRewardConfig, skip_cache=True)