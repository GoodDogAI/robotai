

import torch
import tensorrt as trt
import unittest

from src.config import HOST_CONFIG, BRAIN_CONFIGS
from src.train.modelloader import load_vision_model


class TestModelLoaderTRT(unittest.TestCase):
    def test_basic(self):
        load_vision_model(BRAIN_CONFIGS[HOST_CONFIG.DEFAULT_BRAIN_CONFIG]["vision_model"])