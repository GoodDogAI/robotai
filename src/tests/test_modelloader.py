

import torch
import tensorrt as trt
import unittest

from src.train.config_train import VISION_INTERMEDIATE_CONFIG
from src.train.modelloader import load_vision_model


class TestModelLoaderTRT(unittest.TestCase):
    def test_basic(self):
        load_vision_model(VISION_INTERMEDIATE_CONFIG)