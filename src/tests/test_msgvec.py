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

    def test_init_index(self):
        config = {"obs": [
            {
                "type": "msg",
                "path": "voltage.volts",
                "index": 0,
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

        with self.assertRaises(Exception):
            PyMsgVec(json.dumps(config).encode("utf-8"))

    def test_init_multiindex(self):
        config = {"obs": [
            {
                "type": "msg",
                "path": "voltage.volts",
                "index": [-1, -2, 0],
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

        with self.assertRaises(Exception):
            PyMsgVec(json.dumps(config).encode("utf-8"))

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

        config = {"obs": [
            {
                "type": "msg",
                "path": "voltage.volts",
                "index": -4,
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


        config = {"obs": [
            {
                "type": "msg",
                "path": "voltage.volts",
                "index": [-1, -5],
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
        self.assertEqual(msgvec.obs_size(), 2)

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

    def test_two_inputs_same_message(self):
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
                    "vec_range": [-2, 2],
                }
            },
        ], "act": []}
        msgvec = PyMsgVec(json.dumps(config).encode("utf-8"))

        self.assertEqual(msgvec.obs_size(), 2)
        self.assertEqual(msgvec.get_obs_vector(), [0.0, 0.0])

        event = log.Event.new_message()
        event.init("voltage")
        event.voltage.volts = 13.5
        event.voltage.type = "mainBattery"

        self.assertTrue(msgvec.input(event.to_bytes()))
        self.assertEqual(msgvec.get_obs_vector(), [1.0, 2.0])

    def test_single_larger_index(self):
        config = {"obs": [
            {
                "type": "msg",
                "path": "voltage.volts",
                "index": -5,
                "timeout": 0.01,
                "filter": {
                    "field": "voltage.type",
                    "op": "eq",
                    "value": "mainBattery",
                },
                "transform": {
                    "type": "rescale",
                    "msg_range": [0, 100],
                    "vec_range": [0, 100],
                }
            },

        ], "act": []}
        msgvec = PyMsgVec(json.dumps(config).encode("utf-8"))

        self.assertEqual(msgvec.obs_size(), 1)
        self.assertEqual(msgvec.get_obs_vector(), [0.0])

        feeds = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        expected = [[0],
                    [0],
                    [0],
                    [0],
                    [1],
                    [2],
                    [3],
                    [4],
                    [5]]

        for feed, expect in zip(feeds, expected):
            event = log.Event.new_message()
            event.init("voltage")
            event.voltage.volts = feed
            event.voltage.type = "mainBattery"

            self.assertTrue(msgvec.input(event.to_bytes()))
            self.assertEqual(msgvec.get_obs_vector(), expect)

    def test_multi_index(self):
        config = {"obs": [
            {
                "type": "msg",
                "path": "voltage.volts",
                "index": [-1, -2, -5],
                "timeout": 0.01,
                "filter": {
                    "field": "voltage.type",
                    "op": "eq",
                    "value": "mainBattery",
                },
                "transform": {
                    "type": "rescale",
                    "msg_range": [0, 100],
                    "vec_range": [0, 100],
                }
            },

        ], "act": []}
        msgvec = PyMsgVec(json.dumps(config).encode("utf-8"))

        self.assertEqual(msgvec.obs_size(), 3)
        self.assertEqual(msgvec.get_obs_vector(), [0.0, 0.0, 0.0])

        feeds = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        expected = [[1, 0, 0],
                    [2, 1, 0],
                    [3, 2, 0],
                    [4, 3, 0],
                    [5, 4, 1],
                    [6, 5, 2],
                    [7, 6, 3],
                    [8, 7, 4],
                    [9, 8, 5]]

        for feed, expect in zip(feeds, expected):
            event = log.Event.new_message()
            event.init("voltage")
            event.voltage.volts = feed
            event.voltage.type = "mainBattery"

            self.assertTrue(msgvec.input(event.to_bytes()))
            self.assertEqual(msgvec.get_obs_vector(), expect)

    def test_act_basic(self):
        config = {"obs": [], "act": [
                {
                    "type": "msg",
                    "path": "voltage.volts",
                    "timeout": 0.01,
                    "transform": {
                        "type": "identity",
                    },
                },


            ]}
        msgvec = PyMsgVec(json.dumps(config).encode("utf-8"))

        self.assertEqual(msgvec.act_size(), 1)

        for i in range(1000):
            result = msgvec.get_action_command([i])

            # Make sure the saved result is still valid, even if you do some other stuff in between (memory testing)
            msgvec.get_action_command([0.0])
            msgvec.get_action_command([-1.0])
                
            self.assertAlmostEqual(result[0].voltage.volts, i, places=3)

    def test_duplicates(self):
        config = {"obs": [], "act": []}
        for i in range(1000):
            config["act"].append( {
                    "type": "msg",
                    "path": "voltage.volts",
                    "timeout": 0.01,
                    "transform": {
                        "type": "identity",
                    },
                });
    
        with self.assertRaises(Exception):
            msgvec = PyMsgVec(json.dumps(config).encode("utf-8"))
        


    