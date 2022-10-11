import os
import unittest

from src.train.modelloader import model_fullname
from src.train.arrowcache import ArrowModelCache
from src.config import HOST_CONFIG, MODEL_CONFIGS

class TestArrowModelCache(unittest.TestCase):
    def test_basic(self):
        cache = ArrowModelCache(os.path.join(HOST_CONFIG.RECORD_DIR, "unittest"), model_fullname(MODEL_CONFIGS["yolov7-tiny-s53"]))
        cache.build_cache()