import unittest
import json
import os
import time
from cereal import log
from src.msgvec.pymsgvec import PyMsgVec
from src.config import BRAIN_CONFIGS, HOST_CONFIG

class TestMsgVec(unittest.TestCase):
    def test_init(self):
        config = {"obs": [], "act": []}
        PyMsgVec(json.dumps(config).encode("utf-8"))
    
    def test_failed_init(self):
        with self.assertRaises(Exception):
            PyMsgVec(b"invalid json")

    def test_feed_real_data(self):
        log_path = os.path.join(HOST_CONFIG.RECORD_DIR, "unittest", "alphalog-41a516ae-2022-9-19-2_20.log")
        default_cfg = BRAIN_CONFIGS[HOST_CONFIG.DEFAULT_BRAIN_CONFIG]
        msgvec = PyMsgVec(json.dumps(default_cfg["msgvec"]).encode("utf-8"))

        start = time.perf_counter()
        count = 0

        with open(log_path, "rb") as f:
            events = log.Event.read_multiple(f)
            for evt in events:
                msgvec.input(evt.as_builder().to_bytes())
                count += 1

        print(f"Processed {count} events in {time.perf_counter() - start} seconds")


    def test_obs_size(self):
        config = {"obs": [
            {
                "type": "msg",
                "path": "voltage.volts",
                "index": -1,
                "timeout": 0.01,
                "filter": {
                    "field": "voltage.type",
                    "op": "eq",
                    "value": "mainBattery",
                },
                "transform": {
                    "type": "rescale",
                    "msg_range": [0, 13.5],
                    "vec_range": [-1, 1],
                }
            },
        ], "act": []}
        msgvec = PyMsgVec(json.dumps(config).encode("utf-8"))
        self.assertEqual(msgvec.obs_size(), 1)

    def test_input(self):
        config = {"obs": [
            {
                "type": "msg",
                "path": "voltage.volts",
                "index": -1,
                "timeout": 0.01,
                "filter": {
                    "field": "voltage.type",
                    "op": "eq",
                    "value": "mainBattery",
                },
                "transform": {
                    "type": "rescale",
                    "msg_range": [0, 13.5],
                    "vec_range": [-1, 1],
                }
            },
        ], "act": []}
        msgvec = PyMsgVec(json.dumps(config).encode("utf-8"))

        self.assertEqual(msgvec.obs_size(), 1)
        self.assertEqual(msgvec.get_obs_vector(), [0.0])

        def _sendAndAssert(voltage, vector):
            event = log.Event.new_message()
            event.init("voltage")
            event.voltage.volts = voltage
            event.voltage.type = "mainBattery"

            self.assertTrue(msgvec.input(event.to_bytes()))

            self.assertAlmostEqual(msgvec.get_obs_vector()[0], vector, places=3)

        _sendAndAssert(0.0, -1.0)
        _sendAndAssert(13.5, 1.0)
        _sendAndAssert(6.75, 0.0)

        _sendAndAssert(-1000.0, -1.0)
        _sendAndAssert(1000.0, 1.0)