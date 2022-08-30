

import torch
import tensorrt as trt
import unittest

from src.config import HOST_CONFIG
from src.train.modelloader import load_vision_model


class TestModelLoaderTRT(unittest.TestCase):
    def test_basic(self):
        load_vision_model(HOST_CONFIG.DEFAULT_VISION_CONFIG)