import os
import numpy as np
import unittest
import itertools

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

        last_override = False
        for entry in itertools.islice(cache.generate_samples(), 10000):
            if entry["reward_override"]:
                print(entry["reward_override"], entry["reward"], entry["done"])

            if not entry["reward_override"] and last_override:
                print("------------------")
                
            last_override = entry["reward_override"]

        samples = list(itertools.islice(cache.generate_samples(), 100))

        for index, sample in enumerate(samples[:-1]):
            np.testing.assert_array_almost_equal(sample["next_obs"], samples[index + 1]["obs"])

    # Test that you wait for the obs vectors to populate, though this is somewhat helped by waiting for the first inference
    # TODO test the case of done flags being set on last entry, plus following the "on_reward_override" mode
    # TODO test reward overrides