import os
import unittest

from src.train.modelloader import model_fullname
from src.train.arrowcache import ArrowModelCache, ArrowRLDataset
from src.config import HOST_CONFIG, MODEL_CONFIGS

class TestArrowModelCache(unittest.TestCase):
    def test_basic_vision(self):
        cache = ArrowModelCache(os.path.join(HOST_CONFIG.RECORD_DIR, "unittest"), MODEL_CONFIGS["yolov7-tiny-s53"], force_rebuild=True)

    def test_basic_reward(self):
        cache = ArrowModelCache(os.path.join(HOST_CONFIG.RECORD_DIR, "unittest"), MODEL_CONFIGS["yolov7-tiny-prioritize_centered_nms"], force_rebuild=True)

class TestArrowRLCache(unittest.TestCase):
    def test_basic(self):
        cache = ArrowRLDataset(os.path.join(HOST_CONFIG.RECORD_DIR, "unittest"), MODEL_CONFIGS["basic-brain-test1"])
        
        for entry in cache.generate_samples():
            print(entry)

    # Test that you wait for the obs vectors to populate, though this is somewhat helped by waiting for the first inference
    # TODO test the case of done flags being set on last entry, plus following the "on_reward_override" mode
    # TODO test reward overrides
    # TODO test next_obs mode flags