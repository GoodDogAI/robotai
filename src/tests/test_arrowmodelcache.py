import os
import unittest

from src.train.modelloader import model_fullname
from src.train.arrowcache import ArrowModelCache, ArrowRLCache
from src.config import HOST_CONFIG, MODEL_CONFIGS

class TestArrowModelCache(unittest.TestCase):
    def test_basic_vision(self):
        cache = ArrowModelCache(os.path.join(HOST_CONFIG.RECORD_DIR, "unittest"), MODEL_CONFIGS["yolov7-tiny-s53"])
        cache.build_cache(force_rebuild=True)

    def test_basic_reward(self):
        cache = ArrowModelCache(os.path.join(HOST_CONFIG.RECORD_DIR, "unittest"), MODEL_CONFIGS["yolov7-tiny-prioritize_centered_nms"])
        cache.build_cache(force_rebuild=True)

class TestArrowRLCache(unittest.TestCase):
    def test_basic(self):
        cache = ArrowRLCache(os.path.join(HOST_CONFIG.RECORD_DIR, "unittest"), MODEL_CONFIGS["basic-brain-test1"])
        cache.build_cache(force_rebuild=True)