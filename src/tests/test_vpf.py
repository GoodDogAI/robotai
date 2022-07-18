import unittest
import os

from src.web.video import load_image
from src.web.config import RECORD_DIR


class VPFTest(unittest.TestCase):
    def test_load(self):
        test_path = os.path.join(RECORD_DIR, "alphalog-2022-5-29-22_29.log")
        img = load_image(test_path, 0)

