import unittest
import os

from src.web.video import load_image
from src.web.config_web import RECORD_DIR


class VPFTest(unittest.TestCase):
    def test_load(self):
        test_path = os.path.join(RECORD_DIR, "unittest", "alphalog-2022-7-28-16_54.log")
        
        img = load_image(test_path, 0)
        self.assertEqual(img.shape, (720, 1280 * 3))

        img = load_image(test_path, 20)
        self.assertEqual(img.shape, (720, 1280 * 3))

