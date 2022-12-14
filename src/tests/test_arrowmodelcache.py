import os
import time
import tempfile
import numpy as np
import unittest
import itertools

from unittest.mock import MagicMock

from src.messaging import new_message
from src.train.modelloader import model_fullname
from src.train.arrowcache import ArrowModelCache
from src.train.rldataset import MsgVecDataset
from src.train.reward_modifiers import reward_modifier_penalize_move_backwards
from src.config import HOST_CONFIG, MODEL_CONFIGS

class TestArrowModelCache(unittest.TestCase):
    def test_basic_vision(self):
        cache = ArrowModelCache(os.path.join(HOST_CONFIG.RECORD_DIR, "unittest"), MODEL_CONFIGS["yolov7-tiny-s53"], force_rebuild=True)

    def test_basic_reward(self):
        cache = ArrowModelCache(os.path.join(HOST_CONFIG.RECORD_DIR, "unittest"), MODEL_CONFIGS["yolov7-tiny-prioritize_centered_nms"], force_rebuild=True)

