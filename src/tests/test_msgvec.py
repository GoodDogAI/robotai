import unittest
import json

from src.msgvec.pymsgvec import PyMsgVec

class TestMsgVec(unittest.TestCase):
    def test_init(self):
        config = {"obs": [], "act": []}
        PyMsgVec(json.dumps(config).encode("utf-8"))