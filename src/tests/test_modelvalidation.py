

import os
import tensorrt as trt
import unittest

from src.config import HOST_CONFIG, BRAIN_CONFIGS
from src.train.modelloader import create_and_validate_onnx, create_and_validate_trt
from src.logutil import validate_log

class TestModelValidation(unittest.TestCase):
    def test_vision_intermediate_video(self):
        test_path = os.path.join(HOST_CONFIG.RECORD_DIR, "unittest", "alphalog-4f642094-2022-9-14-16_24.log")
        
        with open(test_path, "rb") as f:
            validate_log(f)
