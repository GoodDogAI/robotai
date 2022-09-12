

import torch
import tensorrt as trt
import unittest

from src.config import HOST_CONFIG, BRAIN_CONFIGS
from src.train.modelloader import create_and_validate_onnx, create_and_validate_trt


class TestModelLoaderTRT(unittest.TestCase):
    def test_onnx(self):
        create_and_validate_onnx(BRAIN_CONFIGS[HOST_CONFIG.DEFAULT_BRAIN_CONFIG]["models"]["vision_model"], skip_cache=True)

    def test_trt(self):
        create_and_validate_trt(BRAIN_CONFIGS[HOST_CONFIG.DEFAULT_BRAIN_CONFIG]["models"]["vision_model"], skip_cache=False)
