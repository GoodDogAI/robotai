import unittest
import json
import os
import math
import time
import random

import numpy as np

from cereal import log
from src.messaging import new_message
from src.msgvec.pymsgvec import PyMsgVec, PyTimeoutResult, PyMessageTimingMode
from src.config import HOST_CONFIG, MODEL_CONFIGS

class MsgVecBaseTest(unittest.TestCase):
    def assertMsgProcessed(self, input_result):
        self.assertTrue(input_result["msg_processed"])


class TestMsgVec(MsgVecBaseTest):
    def test_init(self):
        config = {"obs": [], "act": []}
        PyMsgVec(config, PyMessageTimingMode.REPLAY)
    
    def test_failed_init(self):
        with self.assertRaises(Exception):
            PyMsgVec(b"invalid json")

    def test_feed_real_data(self):
        log_path = os.path.join(HOST_CONFIG.RECORD_DIR, "unittest", "alphalog-41a516ae-2022-9-19-2_20.log")
        default_cfg = MODEL_CONFIGS[HOST_CONFIG.DEFAULT_BRAIN_CONFIG]
        msgvec = PyMsgVec(default_cfg["msgvec"], PyMessageTimingMode.REPLAY)

        start = time.perf_counter()
        count = 0

        with open(log_path, "rb") as f:
            events = log.Event.read_multiple(f)
            for evt in events:
                msgvec.input(evt)
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
            PyMsgVec(config, PyMessageTimingMode.REPLAY)

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
            PyMsgVec(config, PyMessageTimingMode.REPLAY)

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
        msgvec = PyMsgVec(config, PyMessageTimingMode.REPLAY)
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
        msgvec = PyMsgVec(config, PyMessageTimingMode.REPLAY)
        self.assertEqual(msgvec.obs_size(), 4)


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
        msgvec = PyMsgVec(config, PyMessageTimingMode.REPLAY)
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
        msgvec = PyMsgVec(config, PyMessageTimingMode.REPLAY)

        self.assertEqual(msgvec.obs_size(), 1)
        self.assertEqual(msgvec.get_obs_vector_raw(), [0.0])

        def _sendAndAssert(voltage, vector):
            event = log.Event.new_message()
            event.init("voltage")
            event.voltage.volts = voltage
            event.voltage.type = "mainBattery"

            self.assertMsgProcessed(msgvec.input(event))

            self.assertAlmostEqual(msgvec.get_obs_vector_raw()[0], vector, places=3)

        _sendAndAssert(0.0, -1.0)
        _sendAndAssert(13.5, 1.0)
        _sendAndAssert(6.75, 0.0)

        _sendAndAssert(-1000.0, -1.0)
        _sendAndAssert(1000.0, 1.0)

    def test_input_filter_wrong_type(self):
        config = {"obs": [
            { 
                "type": "msg",
                "path": "gyroscope.gyro.v.1",
                "index": -1,
                "timeout": 0.01,
                "filter": {
                    "field": "gyroscope.sensor",
                    "op": "eq",
                    "value": 1, 
                },
                "transform": {
                    "type": "rescale",
                    "msg_range": [-2, 2],
                    "vec_range": [-1, 1],
                },
            },
        ], "act": []}
        msgvec = PyMsgVec(config, PyMessageTimingMode.REPLAY)

        self.assertEqual(msgvec.obs_size(), 1)
        msg = new_message("gyroscope")
        msg.gyroscope.init("gyro")
        msg.gyroscope.sensor = 1
        msg.gyroscope.gyro.init("v", 3)
        msg.gyroscope.gyro.v = [1, 2, 3]
        result = msgvec.input(msg)
        self.assertTrue(result["msg_processed"])

        print(msgvec.get_obs_vector_raw())

    def test_input_reader(self):
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
        msgvec = PyMsgVec(config, PyMessageTimingMode.REPLAY)

        event = log.Event.new_message()
        event.init("voltage")
        event.voltage.volts = 13.5
        event.voltage.type = "mainBattery"

        res = msgvec.input(event.as_reader())
        print(res)



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
        msgvec = PyMsgVec(config, PyMessageTimingMode.REPLAY)

        self.assertEqual(msgvec.obs_size(), 2)
        self.assertEqual(msgvec.get_obs_vector_raw().tolist(), [0.0, 0.0])

        event = log.Event.new_message()
        event.init("voltage")
        event.voltage.volts = 13.5
        event.voltage.type = "mainBattery"

        self.assertTrue(msgvec.input(event))
        self.assertEqual(msgvec.get_obs_vector_raw().tolist(), [1.0, 2.0])

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
        msgvec = PyMsgVec(config, PyMessageTimingMode.REPLAY)

        self.assertEqual(msgvec.obs_size(), 5)
        self.assertEqual(msgvec.get_obs_vector_raw().tolist(), [0.0] * 5)

        feeds = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        expected = [[1, 0, 0, 0, 0],
                    [2, 1, 0, 0, 0],
                    [3, 2, 1, 0, 0],
                    [4, 3, 2, 1, 0],
                    [5, 4, 3, 2, 1],
                    [6, 5, 4, 3, 2],
                    [7, 6, 5, 4, 3],
                    [8, 7, 6, 5, 4],
                    [9, 8, 7, 6, 5]]

        for feed, expect in zip(feeds, expected):
            event = log.Event.new_message()
            event.init("voltage")
            event.voltage.volts = feed
            event.voltage.type = "mainBattery"

            self.assertMsgProcessed(msgvec.input(event))
            self.assertEqual(msgvec.get_obs_vector_raw().tolist(), expect)

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

            {
                "type": "msg",
                "path": "headFeedback.yawAngle",
                "index": -5,
                "timeout": 0.01,
                "transform": {
                    "type": "rescale",
                    "msg_range": [0, 100],
                    "vec_range": [0, 1000],
                }
            },

            {
                "type": "msg",
                "path": "headFeedback.pitchAngle",
                "index": [-1, -2, -5],
                "timeout": 0.01,
                "transform": {
                    "type": "rescale",
                    "msg_range": [0, 100],
                    "vec_range": [0, 1000],
                }
            },

        ], "act": []}
        msgvec = PyMsgVec(config, PyMessageTimingMode.REPLAY)

        self.assertEqual(msgvec.obs_size(), 11)
        self.assertEqual(msgvec.get_obs_vector_raw().tolist(), [0.0] * 11)

        feeds = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        expected = [[1, 0, 0, 10, 00, 00, 00, 00, 10, 0, 0],
                    [2, 1, 0, 20, 10, 00, 00, 00, 20, 10, 0],
                    [3, 2, 0, 30, 20, 10, 00, 00, 30, 20, 0],
                    [4, 3, 0, 40, 30, 20, 10, 00, 40, 30, 0],
                    [5, 4, 1, 50, 40, 30, 20, 10, 50, 40, 10],
                    [6, 5, 2, 60, 50, 40, 30, 20, 60, 50, 20],
                    [7, 6, 3, 70, 60, 50, 40, 30, 70, 60, 30],
                    [8, 7, 4, 80, 70, 60, 50, 40, 80, 70, 40],
                    [9, 8, 5, 90, 80, 70, 60, 50, 90, 80, 50]]

        for feed, expect in zip(feeds, expected):
            event = log.Event.new_message()
            event.init("voltage")
            event.voltage.volts = feed
            event.voltage.type = "mainBattery"
            self.assertMsgProcessed(msgvec.input(event))

            event = log.Event.new_message()
            event.init("headFeedback")
            event.headFeedback.pitchAngle = feed
            event.headFeedback.yawAngle = feed 
            self.assertMsgProcessed(msgvec.input(event))

            self.assertEqual(msgvec.get_obs_vector_raw().tolist(), expect)

    def test_obs_arrays(self):
        config = {"obs": [
                { 
                    "type": "msg",
                    "path": "gyroscope.gyro.v.0",
                    "index": -1,
                    "timeout": 0.125,
                    "transform": {
                        "type": "identity",
                    },
                },

            ], "act": []}

        msgvec = PyMsgVec(config, PyMessageTimingMode.REPLAY)
        self.assertEqual(msgvec.obs_size(), 1)

        msg = new_message("gyroscope")
        msg.gyroscope.init("gyro")
        msg.gyroscope.gyro.init("v", 3)
        msg.gyroscope.gyro.v = [1, 2, 3]
        self.assertMsgProcessed(msgvec.input(msg))

        self.assertEqual(msgvec.get_obs_vector_raw().tolist(), [1])

    def test_obs_array_oob(self):
        config = {"obs": [
                { 
                    "type": "msg",
                    "path": "gyroscope.gyro.v.10",
                    "index": -1,
                    "timeout": 0.125,
                    "transform": {
                        "type": "identity",
                    },
                },

            ], "act": []}

        msgvec = PyMsgVec(config, PyMessageTimingMode.REPLAY)
        self.assertEqual(msgvec.obs_size(), 1)

        msg = new_message("gyroscope")
        msg.gyroscope.init("gyro")
        msg.gyroscope.gyro.init("v", 3)
        msg.gyroscope.gyro.v = [1, 2, 3]

        with self.assertRaises(RuntimeError):
            self.assertMsgProcessed(msgvec.input(msg))

    def test_vision_vectors1(self):
        config = {"obs": [
                { 
                    "type": "msg",
                    "path": "odriveFeedback.leftMotor.vel",
                    "index": -1,
                    "timeout": 0.125,
                    "transform": {
                        "type": "identity",
                    },
                },

                {
                    "type": "msg",
                    "path": "voltage.volts",
                    "index": -1,
                    "timeout": 0.125,
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
                    "path": "headFeedback.pitchAngle",
                    "index": -1,
                    "timeout": 0.125,
                    "transform": {
                        "type": "rescale",
                        "msg_range": [-45.0, 45.0],
                        "vec_range": [-1, 1],
                    },
                },

                {
                    "type": "vision",
                    "size": 10,
                    "timeout": .1,
                    "index": -1,
                }
            ], "act": []}

        msgvec = PyMsgVec(config, PyMessageTimingMode.REPLAY)

        self.assertEqual(msgvec.get_obs_vector_raw().tolist(), [0.0] * 13)

        msgvec.input_vision(np.arange(10, 20, dtype=np.float32), 1)
        self.assertEqual(msgvec.get_obs_vector_raw().tolist(), [0.0, 0.0, 0.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0])

        msgvec.input_vision(np.arange(40, 50, dtype=np.float32), 1)
        self.assertEqual(msgvec.get_obs_vector_raw().tolist(), [0.0, 0.0, 0.0, 40.0, 41.0, 42.0, 43.0, 44.0, 45.0, 46.0, 47.0, 48.0, 49.0])

        event = log.Event.new_message()
        event.init("voltage")
        event.voltage.volts = 15
        event.voltage.type = "mainBattery"
        self.assertMsgProcessed(msgvec.input(event))

        self.assertEqual(msgvec.get_obs_vector_raw().tolist(), [0.0, 1.0, 0.0, 40.0, 41.0, 42.0, 43.0, 44.0, 45.0, 46.0, 47.0, 48.0, 49.0])
 
    def test_vision_vectors2(self):
        config = {"obs": [
                { 
                    "type": "msg",
                    "path": "odriveFeedback.leftMotor.vel",
                    "index": -1,
                    "timeout": 0.125,
                    "transform": {
                        "type": "identity",
                    },
                },

                {
                    "type": "msg",
                    "path": "voltage.volts",
                    "index": -1,
                    "timeout": 0.125,
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
                    "path": "headFeedback.pitchAngle",
                    "index": -1,
                    "timeout": 0.125,
                    "transform": {
                        "type": "rescale",
                        "msg_range": [-45.0, 45.0],
                        "vec_range": [-1, 1],
                    },
                },

                {
                    "type": "vision",
                    "timeout": 0.10,
                    "size": 10,
                    "index": -3, # Takes the last 3 vision vectors
                }
            ], "act": []}

        msgvec = PyMsgVec(config, PyMessageTimingMode.REPLAY)

        self.assertEqual(msgvec.get_obs_vector_raw().tolist(), [0.0] * 3 + [0.0] * 10 * 3)

        msgvec.input_vision(np.arange(10, 20, dtype=np.float32), 1)
        self.assertEqual(msgvec.get_obs_vector_raw().tolist(), [0.0] * 3 + list(range(10, 20)) + [0.0] * 10 * 2)

        msgvec.input_vision(np.arange(40, 50, dtype=np.float32), 1)
        self.assertEqual(msgvec.get_obs_vector_raw().tolist(), [0.0] * 3 + list(range(40,50)) + list(range(10, 20)) + [0.0] * 10)

        event = log.Event.new_message()
        event.init("voltage")
        event.voltage.volts = 15
        event.voltage.type = "mainBattery"
        self.assertMsgProcessed(msgvec.input(event))

        self.assertEqual(msgvec.get_obs_vector_raw().tolist(), [0.0, 1.0, 0.0] + list(range(40,50)) + list(range(10, 20)) + [0.0] * 10)

        msgvec.input_vision(np.arange(60, 70, dtype=np.float32), 1)
        self.assertEqual(msgvec.get_obs_vector_raw().tolist(), [0.0, 1.0, 0.0] + list(range(60, 70)) + list(range(40,50)) + list(range(10, 20)))

    def test_vision_vectors3(self):
        config = {"obs": [
                { 
                    "type": "msg",
                    "path": "odriveFeedback.leftMotor.vel",
                    "index": -1,
                    "timeout": 0.125,
                    "transform": {
                        "type": "identity",
                    },
                },

                {
                    "type": "msg",
                    "path": "voltage.volts",
                    "index": -1,
                    "timeout": 0.125,
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
                    "path": "headFeedback.pitchAngle",
                    "index": -1,
                    "timeout": 0.125,
                    "transform": {
                        "type": "rescale",
                        "msg_range": [-45.0, 45.0],
                        "vec_range": [-1, 1],
                    },
                },

                {
                    "type": "vision",
                    "size": 10,
                    "timeout": 0.10,
                    "index": [-3], # Takes the vision vector from 3 steps ago
                }
            ], "act": []}

        msgvec = PyMsgVec(config, PyMessageTimingMode.REPLAY)

        self.assertEqual(msgvec.get_obs_vector_raw().tolist(), [0.0] * 13)

        msgvec.input_vision(np.arange(10, 20, dtype=np.float32), 1)
        self.assertEqual(msgvec.get_obs_vector_raw().tolist(), [0.0] * 13)

        msgvec.input_vision(np.arange(40, 50, dtype=np.float32), 1)
        self.assertEqual(msgvec.get_obs_vector_raw().tolist(), [0.0] * 13)

        event = log.Event.new_message()
        event.init("voltage")
        event.voltage.volts = 15
        event.voltage.type = "mainBattery"
        self.assertMsgProcessed(msgvec.input(event))

        self.assertEqual(msgvec.get_obs_vector_raw().tolist(), [0.0, 1.0, 0.0] + [0.0] * 10)

        msgvec.input_vision(np.arange(60, 70, dtype=np.float32), 1)
        self.assertEqual(msgvec.get_obs_vector_raw().tolist(), [0.0, 1.0, 0.0] + list(range(10, 20)))

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
        msgvec = PyMsgVec(config, PyMessageTimingMode.REALTIME)

        self.assertEqual(msgvec.act_size(), 1)

        # Small test case, we don't support querying a zero size obs vector
        self.assertEqual(msgvec.obs_size(), 0)
        with self.assertRaises(AssertionError):
            msgvec.get_obs_vector()

        for i in range(1000):
            result = msgvec.get_action_command(np.array([i], dtype=np.float32))

            # Make sure the saved result is still valid, even if you do some other stuff in between (memory testing)
            msgvec.get_action_command(np.array([0.0], dtype=np.float32))
            msgvec.get_action_command(np.array([-1.0], dtype=np.float32))
                
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
            msgvec = PyMsgVec(config, PyMessageTimingMode.REALTIME)

    def test_populate_message(self):
        config = {"obs": [], "act": [
                {
                    "type": "msg",
                    "path": "headCommand.pitchAngle",
                    "timeout": 0.01,
                    "transform": {
                        "type": "identity",
                    },
                },

                {
                    "type": "msg",
                    "path": "headCommand.yawAngle",
                    "timeout": 0.01,
                    "transform": {
                        "type": "rescale",
                        "vec_range": [-1, 1],
                        "msg_range": [-45.0, 45.0],
                    },
                },

                {
                    "type": "msg",
                    "path": "odriveCommand.desiredVelocityLeft",
                    "timeout": 0.01,
                    "transform": {
                        "type": "identity",
                    },
                },

                {
                    "type": "msg",
                    "path": "odriveCommand.desiredVelocityRight",
                    "timeout": 0.01,
                    "transform": {
                        "type": "identity",
                    },
                },
            ]}
        msgvec = PyMsgVec(config, PyMessageTimingMode.REALTIME)

        self.assertEqual(msgvec.act_size(), 4)

        result = msgvec.get_action_command(np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32))
        print(result)

        self.assertEqual(len(result), 2)
        self.assertEqual(result[0].which(), "headCommand")
        self.assertEqual(result[1].which(), "odriveCommand")

        self.assertEqual(result[0].headCommand.pitchAngle, 1.0)
        self.assertEqual(result[0].headCommand.yawAngle, 45.0)
        self.assertEqual(result[1].odriveCommand.desiredCurrentLeft, 0.0)
        self.assertEqual(result[1].odriveCommand.desiredCurrentRight, 0.0)
        self.assertEqual(result[1].odriveCommand.desiredVelocityLeft, 3.0)
        self.assertEqual(result[1].odriveCommand.desiredVelocityRight, 4.0)
        self.assertEqual(result[1].odriveCommand.controlMode, "velocity")

    def test_act_transforms(self):
        config = {"obs": [], "act": [
            {
                "type": "msg",
                "path": "headCommand.yawAngle",
                "timeout": 0.01,
                "transform": {
                    "type": "rescale",
                    "vec_range": [-1, 1],
                    "msg_range": [-45.0, 45.0],
                },
            },

        ]}
        msgvec = PyMsgVec(config, PyMessageTimingMode.REALTIME)

        result = msgvec.get_action_command(np.array([2.0], dtype=np.float32))
        self.assertEqual(result[0].headCommand.yawAngle, 45.0)

        result = msgvec.get_action_command(np.array([-2.0], dtype=np.float32))
        self.assertEqual(result[0].headCommand.yawAngle, -45.0)

        result = msgvec.get_action_command(np.array([0.0], dtype=np.float32))
        self.assertEqual(result[0].headCommand.yawAngle, 0.0)

        result = msgvec.get_action_command(np.array([0.5], dtype=np.float32))
        self.assertEqual(result[0].headCommand.yawAngle, 45.0 / 2)

    def test_feed_acts(self):
        config = {"obs": [], "act": [
            {
                "type": "msg",
                "path": "headCommand.yawAngle",
                "timeout": 0.01,
                "transform": {
                    "type": "rescale",
                    "vec_range": [-1, 1],
                    "msg_range": [-45.0, 45.0],
                },
            },
        ]}
        msgvec = PyMsgVec(config, PyMessageTimingMode.REPLAY)

        test_params = [
            (45, 1),
            (-45, -1),
            (46, 1),
            (-46, -1),
            (0, 0),
            (45.0 / 2, 0.5)]

        for msg, vec in test_params:
            event = log.Event.new_message()
            event.init("headCommand")
            event.headCommand.yawAngle = msg
            self.assertMsgProcessed(msgvec.input(event))
            self.assertEqual(msgvec.get_act_vector().dtype, np.float32)
            self.assertAlmostEqual(msgvec.get_act_vector()[0], vec)

        # Feed in a message that doesn't exist in the config
        event = log.Event.new_message()
        event.init("voltage")
        event.voltage.volts = 13.5
        event.voltage.type = "mainBattery"

        self.assertFalse(msgvec.input(event)["msg_processed"])

    def test_action_inverse(self):
        config = {"obs": [], "act": [
            {
                "type": "msg",
                "path": "headCommand.yawAngle",
                "timeout": 0.01,
                "transform": {
                    "type": "rescale",
                    "vec_range": [-1, 1],
                    "msg_range": [-45.0, 45.0],
                },
            },
        ]}
        msgvec_replay = PyMsgVec(config, PyMessageTimingMode.REPLAY)
        msgvec_realtime = PyMsgVec(config, PyMessageTimingMode.REALTIME)


        for i in range(1000):
            f = random.uniform(-2, 2)
            messages = msgvec_realtime.get_action_command(np.array([f], dtype=np.float32))

            self.assertEqual(len(messages), 1)
            self.assertMsgProcessed(msgvec_replay.input(messages[0]))
            self.assertMsgProcessed(msgvec_realtime.input(messages[0]))

            self.assertAlmostEqual(msgvec_replay.get_act_vector()[0], min(max(f, -1), 1) , places=3)

    def test_transform_ranges_minmax(self):
        with self.assertRaises(Exception):
            config = {"obs": [], "act": [
                {
                    "type": "msg",
                    "path": "headCommand.yawAngle",
                    "timeout": 0.01,
                    "transform": {
                        "type": "rescale",
                        "vec_range": [1, -1],
                        "msg_range": [-45.0, 45.0],
                    },
                },
            ]}
            msgvec = PyMsgVec(config, PyMessageTimingMode.REPLAY)

        with self.assertRaises(Exception):
            config = {"obs": [], "act": [
                {
                    "type": "msg",
                    "path": "headCommand.yawAngle",
                    "timeout": 0.01,
                    "transform": {
                        "type": "rescale",
                        "vec_range": [-1, -1],
                        "msg_range": [-45.0, 45.0],
                    },
                },
            ]}
            msgvec = PyMsgVec(config, PyMessageTimingMode.REPLAY)

        with self.assertRaises(Exception):
            config = {"obs": [], "act": [
                {
                    "type": "msg",
                    "path": "headCommand.yawAngle",
                    "timeout": 0.01,
                    "transform": {
                        "type": "rescale",
                        "vec_range": [-1, 10],
                        "msg_range": [45.0, 40.0],
                    },
                },
            ]}
            msgvec = PyMsgVec(config, PyMessageTimingMode.REPLAY)

        with self.assertRaises(Exception):
            config = {"obs": [], "act": [
                {
                    "type": "msg",
                    "path": "headCommand.yawAngle",
                    "timeout": 0.01,
                    "transform": {
                        "type": "rescale",
                        "vec_range": [-1, 10, 3],
                        "msg_range": [32.0, 40.0],
                    },
                },
            ]}
            msgvec = PyMsgVec(config, PyMessageTimingMode.REPLAY)

    def test_obs_timeouts(self):
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
                    "msg_range": [0, 100],
                    "vec_range": [0, 100],
                }
            },

        ], "act": []}
        msgvec = PyMsgVec(config, PyMessageTimingMode.REALTIME)

        timeout, _ = msgvec.get_obs_vector()
        self.assertEqual(timeout, PyTimeoutResult.MESSAGES_NOT_READY)

        msg = new_message("voltage")
        msg.voltage.volts = 13.5
        msg.voltage.type = "mainBattery"
        msgvec.input(msg)

        timeout, _ = msgvec.get_obs_vector()
        self.assertEqual(timeout, PyTimeoutResult.MESSAGES_ALL_READY)

        time.sleep(0.02)

        timeout, _ = msgvec.get_obs_vector()
        self.assertEqual(timeout, PyTimeoutResult.MESSAGES_NOT_READY)

    def test_obs_timeouts_multiindex(self):
        config = {"obs": [
            {
                "type": "msg",
                "path": "voltage.volts",
                "index": [-1, -5, -10],
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
        msgvec = PyMsgVec(config, PyMessageTimingMode.REALTIME)

        timeout, _ = msgvec.get_obs_vector()
        self.assertEqual(timeout, PyTimeoutResult.MESSAGES_NOT_READY)

        msg = new_message("voltage")
        msg.voltage.volts = 13.5
        msg.voltage.type = "mainBattery"
        msgvec.input(msg)

        timeout, _ = msgvec.get_obs_vector()
        self.assertEqual(timeout, PyTimeoutResult.MESSAGES_PARTIALLY_READY)

        time.sleep(0.02)

        timeout, _ = msgvec.get_obs_vector()
        self.assertEqual(timeout, PyTimeoutResult.MESSAGES_NOT_READY)

        msg = new_message("voltage")
        msg.voltage.volts = 13.5
        msg.voltage.type = "mainBattery"
        msgvec.input(msg)

        timeout, _ = msgvec.get_obs_vector()
        self.assertEqual(timeout, PyTimeoutResult.MESSAGES_PARTIALLY_READY)

        for i in range(9):
            msg = new_message("voltage")
            msg.voltage.volts = 13.5
            msg.voltage.type = "mainBattery"
            msgvec.input(msg)

        timeout, vec = msgvec.get_obs_vector()
        self.assertEqual(timeout, PyTimeoutResult.MESSAGES_ALL_READY)

        time.sleep(0.02)

        timeout, _ = msgvec.get_obs_vector()
        self.assertEqual(timeout, PyTimeoutResult.MESSAGES_NOT_READY)

    def test_timeouts_multiple_msg_types(self):
        config = {"obs": [
            { 
                "type": "msg",
                "path": "odriveFeedback.leftMotor.vel",
                "index": -1,
                "timeout": 0.01,
                "transform": {
                    "type": "identity",
                },
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
                    "vec_range": [-1, 1],
                }
            },
        ], "act": []}
        msgvec = PyMsgVec(config, PyMessageTimingMode.REALTIME)

        timeout, _ = msgvec.get_obs_vector()
        self.assertEqual(timeout, PyTimeoutResult.MESSAGES_NOT_READY)

        msg = new_message("voltage")
        msg.voltage.volts = 13.5
        msg.voltage.type = "mainBattery"
        msgvec.input(msg)

        timeout, _ = msgvec.get_obs_vector()
        self.assertEqual(timeout, PyTimeoutResult.MESSAGES_NOT_READY)

        msg = new_message("odriveFeedback")
        msg.odriveFeedback.leftMotor.vel = 1.0
        msgvec.input(msg)

        timeout, _ = msgvec.get_obs_vector()
        self.assertEqual(timeout, PyTimeoutResult.MESSAGES_ALL_READY)

        time.sleep(0.02)

        timeout, _ = msgvec.get_obs_vector()
        self.assertEqual(timeout, PyTimeoutResult.MESSAGES_NOT_READY)

    def test_obs_vision_timeouts(self):
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
                    "msg_range": [0, 100],
                    "vec_range": [0, 100],
                }
            },
            {
                "type": "vision",
                "size": 1000,
                "timeout": 0.100,
                "index": [-1, -2],
            }
        ], "act": []}
        msgvec = PyMsgVec(config, PyMessageTimingMode.REALTIME)

        timeout, _ = msgvec.get_obs_vector()
        self.assertEqual(timeout, PyTimeoutResult.MESSAGES_NOT_READY)

        msg = new_message("voltage")
        msg.voltage.volts = 13.5
        msg.voltage.type = "mainBattery"
        msgvec.input(msg)

        timeout, _ = msgvec.get_obs_vector()
        self.assertEqual(timeout, PyTimeoutResult.MESSAGES_NOT_READY)

        msgvec.input_vision(np.arange(1000, dtype=np.float32), 1)

        timeout, _ = msgvec.get_obs_vector()
        self.assertEqual(timeout, PyTimeoutResult.MESSAGES_PARTIALLY_READY)

        msgvec.input_vision(np.arange(1000, dtype=np.float32), 1)

        timeout, _ = msgvec.get_obs_vector()
        self.assertEqual(timeout, PyTimeoutResult.MESSAGES_ALL_READY)

        
    def test_appcontrol_basic(self):
        config = {"obs": [], "act": [],
            "appcontrol": {
                "mode": "steering_override_v1",
                "timeout": 0.125,
            },
         }
        msgvec = PyMsgVec(config, PyMessageTimingMode.REALTIME)

        result = msgvec.get_action_command(np.array([], dtype=np.float32))
        self.assertEqual(result, [])

        msg = new_message("appControl")
        msg.appControl.connectionState = "connected"
        msgvec.input(msg)

        result = msgvec.get_action_command(np.array([], dtype=np.float32))
        self.assertEqual(result, [])

        msg = new_message("appControl")
        msg.appControl.connectionState = "connected"
        msg.appControl.motionState = "manualControl"
        msg.appControl.linearXOverride = 1.0
        msgvec.input(msg)

        result = msgvec.get_action_command(np.array([], dtype=np.float32))
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0].which(), "odriveCommand")
        self.assertEqual(result[1].which(), "headCommand")
        self.assertEqual(result[0].odriveCommand.desiredVelocityLeft, -1.0)
        self.assertEqual(result[0].odriveCommand.desiredVelocityRight, 1.0)
        self.assertEqual(result[1].headCommand.pitchAngle, 5.0)
        self.assertEqual(result[1].headCommand.yawAngle, 0.0)

        msg = new_message("appControl")
        msg.appControl.connectionState = "connected"
        msg.appControl.motionState = "manualControl"
        msg.appControl.linearXOverride = 1.0
        msg.appControl.angularZOverride = 1.0
        msgvec.input(msg)

        result = msgvec.get_action_command(np.array([], dtype=np.float32))
   
        self.assertAlmostEqual(result[0].odriveCommand.desiredVelocityLeft, 0.0, places=3)
        self.assertAlmostEqual(result[0].odriveCommand.desiredVelocityRight, 2.0, places=3)
        self.assertAlmostEqual(result[1].headCommand.pitchAngle, 5.0)
        self.assertAlmostEqual(result[1].headCommand.yawAngle, -30.0)

        time.sleep(0.15)

        result = msgvec.get_action_command(np.array([], dtype=np.float32))
        self.assertEqual(result, [])

        msg = new_message("appControl")
        msg.appControl.connectionState = "connected"
        msg.appControl.motionState = "stopAllOutputs"
        msg.appControl.linearXOverride = 1.0
        msg.appControl.angularZOverride = 1.0
        msgvec.input(msg)

        result = msgvec.get_action_command(np.array([], dtype=np.float32))
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0].which(), "odriveCommand")
        self.assertEqual(result[1].which(), "headCommand")
        self.assertEqual(result[0].odriveCommand.desiredVelocityLeft, 0.0)
        self.assertEqual(result[0].odriveCommand.desiredVelocityRight, 0.0)
        self.assertEqual(result[1].headCommand.pitchAngle, 0.0)
        self.assertEqual(result[1].headCommand.yawAngle, 0.0)

    def test_appcontrol_override(self):
        config = {"obs": [], "act": [
            {
                "type": "msg",
                "path": "odriveCommand.desiredVelocityLeft",
                "timeout": 0.01,
                "transform": {
                    "type": "rescale",
                    "vec_range": [-1, 1],
                    "msg_range": [-3, 3],
                },
            },
            {
                "type": "msg",
                "path": "odriveCommand.desiredVelocityRight",
                "timeout": 0.01,
                "transform": {
                    "type": "rescale",
                    "vec_range": [-1, 1],
                    "msg_range": [-3, 3],
                },
            },
        ],  
           "appcontrol": {
                "mode": "steering_override_v1",
                "timeout": 0.125,
            },}
        msgvec = PyMsgVec(config, PyMessageTimingMode.REALTIME)

        
        result = msgvec.get_action_command(np.array([-1.0, 1.0], dtype=np.float32))
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].which(), "odriveCommand")
        self.assertEqual(result[0].odriveCommand.desiredVelocityLeft, -3.0)
        self.assertEqual(result[0].odriveCommand.desiredVelocityRight, 3.0)

        msg = new_message("appControl")
        msg.appControl.connectionState = "connected"
        msg.appControl.motionState = "stopAllOutputs"
        msgvec.input(msg)

        result = msgvec.get_action_command(np.array([-1.0, 1.0], dtype=np.float32))
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0].which(), "odriveCommand")
        self.assertEqual(result[0].odriveCommand.desiredVelocityLeft, 0.0)
        self.assertEqual(result[0].odriveCommand.desiredVelocityRight, 0.0)

        # Not connected means it will read the motion state
        msg = new_message("appControl")
        msg.appControl.connectionState = "notConnected"
        msg.appControl.motionState = "stopAllOutputs"
        msgvec.input(msg)

        result = msgvec.get_action_command(np.array([-1.0, 1.0], dtype=np.float32))
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0].which(), "odriveCommand")
        self.assertEqual(result[0].odriveCommand.desiredVelocityLeft, 0.0)
        self.assertEqual(result[0].odriveCommand.desiredVelocityRight, 0.0)
        self.assertEqual(result[1].which(), "headCommand")
        self.assertEqual(result[1].headCommand.pitchAngle, 0.0)
        self.assertEqual(result[1].headCommand.yawAngle, 0.0)

        # Not connected means it will read the motion state
        msg = new_message("appControl")
        msg.appControl.connectionState = "notConnected"
        msg.appControl.motionState = "suspendMajorMotion"
        msgvec.input(msg)

        result = msgvec.get_action_command(np.array([-1.0, 1.0], dtype=np.float32))
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0].which(), "odriveCommand")
        self.assertEqual(result[0].odriveCommand.desiredVelocityLeft, 0.0)
        self.assertEqual(result[0].odriveCommand.desiredVelocityRight, 0.0)
        self.assertEqual(result[1].which(), "headCommand")
        baseYaw = result[1].headCommand.yawAngle
        basePitch = result[1].headCommand.pitchAngle

        # No delay should mean the same results
        msg = new_message("appControl")
        msg.appControl.connectionState = "notConnected"
        msg.appControl.motionState = "suspendMajorMotion"
        msgvec.input(msg)

        result = msgvec.get_action_command(np.array([-1.0, 1.0], dtype=np.float32))
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0].which(), "odriveCommand")
        self.assertEqual(result[0].odriveCommand.desiredVelocityLeft, 0.0)
        self.assertEqual(result[0].odriveCommand.desiredVelocityRight, 0.0)
        self.assertEqual(result[1].which(), "headCommand")
        self.assertAlmostEqual(result[1].headCommand.yawAngle, baseYaw)
        self.assertAlmostEqual(result[1].headCommand.pitchAngle, basePitch)

        # Delay should mean the head moves
        time.sleep(2.50)
        msg = new_message("appControl")
        msg.appControl.connectionState = "notConnected"
        msg.appControl.motionState = "suspendMajorMotion"
        msgvec.input(msg)

        result = msgvec.get_action_command(np.array([-1.0, 1.0], dtype=np.float32))
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0].which(), "odriveCommand")
        self.assertEqual(result[0].odriveCommand.desiredVelocityLeft, 0.0)
        self.assertEqual(result[0].odriveCommand.desiredVelocityRight, 0.0)
        self.assertEqual(result[1].which(), "headCommand")
        self.assertNotAlmostEqual(result[1].headCommand.yawAngle, baseYaw)
        self.assertNotAlmostEqual(result[1].headCommand.pitchAngle, basePitch)


    def test_override_reward_realtime(self):
        config = {"obs": [], "act": [
            {
                "type": "msg",
                "path": "odriveCommand.currentLeft",
                "timeout": 0.01,
                "transform": {
                    "type": "rescale",
                    "vec_range": [-1, 1],
                    "msg_range": [-3, 3],
                },
            },
            {
                "type": "msg",
                "path": "odriveCommand.currentRight",
                "timeout": 0.01,
                "transform": {
                    "type": "rescale",
                    "vec_range": [-1, 1],
                    "msg_range": [-3, 3],
                },
            },
        ],  
        "appcontrol": {
            "mode": "steering_override_v1",
            "timeout": 0.125,
        },
        "rew": {
            "override": {
                "positive_reward": 3.0,
                "positive_reward_timeout": 0.50,

                "negative_reward": -12.0,
                "negative_reward_timeout": 0.50,
            }
        },}
        msgvec = PyMsgVec(config, PyMessageTimingMode.REALTIME)

        # No override, no reward
        valid, rew = msgvec.get_reward()
        self.assertFalse(valid)

        # Override, positive reward
        msg = new_message("appControl")
        msg.appControl.connectionState = "connected"
        msg.appControl.rewardState = "overridePositive"
        msgvec.input(msg)

        valid, rew = msgvec.get_reward()
        self.assertTrue(valid)
        self.assertEqual(rew, 3.0)

        # Override, negative reward
        msg = new_message("appControl")
        msg.appControl.connectionState = "connected"
        msg.appControl.rewardState = "overrideNegative"
        msgvec.input(msg)

        valid, rew = msgvec.get_reward()
        self.assertTrue(valid)
        self.assertEqual(rew, -12.0)

        # Override, not connected
        msg = new_message("appControl")
        msg.appControl.connectionState = "notConnected"
        msg.appControl.rewardState = "overrideNegative"
        msgvec.input(msg)

        valid, rew = msgvec.get_reward()
        self.assertFalse(valid)
        self.assertTrue(math.isnan(rew))
        
        # Override, timeout
        msg = new_message("appControl")
        msg.appControl.connectionState = "connected"
        msg.appControl.rewardState = "overridePositive"
        msgvec.input(msg)

        valid, rew = msgvec.get_reward()
        self.assertTrue(valid)
        time.sleep(0.25)    
        valid, rew = msgvec.get_reward()
        self.assertTrue(valid)
        time.sleep(0.35)
        valid, rew = msgvec.get_reward()
        self.assertFalse(valid)

    def test_override_reward_replay(self):
        config = {"obs": [], "act": [
            {
                "type": "msg",
                "path": "odriveCommand.desiredVelocityLeft",
                "timeout": 0.01,
                "transform": {
                    "type": "rescale",
                    "vec_range": [-1, 1],
                    "msg_range": [-3, 3],
                },
            },
            {
                "type": "msg",
                "path": "odriveCommand.desiredVelocityRight",
                "timeout": 0.01,
                "transform": {
                    "type": "rescale",
                    "vec_range": [-1, 1],
                    "msg_range": [-3, 3],
                },
            },
        ],  
        "appcontrol": {
            "mode": "steering_override_v1",
            "timeout": 0.125,
        },
        "rew": {
            "override": {
                "positive_reward": 3.0,
                "positive_reward_timeout": 0.50,

                "negative_reward": -12.0,
                "negative_reward_timeout": 0.50,
            }
        },}
        msgvec = PyMsgVec(config, PyMessageTimingMode.REPLAY)

        # No override, no reward
        valid, rew = msgvec.get_reward()
        self.assertFalse(valid)

        # Override, positive reward
        msg = new_message("appControl")
        msg.appControl.connectionState = "connected"
        msg.appControl.rewardState = "overridePositive"
        msgvec.input(msg)

        valid, rew = msgvec.get_reward()
        self.assertTrue(valid)
        self.assertEqual(rew, 3.0)

        # Override, negative reward
        msg = new_message("appControl")
        msg.appControl.connectionState = "connected"
        msg.appControl.rewardState = "overrideNegative"
        msgvec.input(msg)

        valid, rew = msgvec.get_reward()
        self.assertTrue(valid)
        self.assertEqual(rew, -12.0)

        # Override, not connected
        msg = new_message("appControl")
        msg.appControl.connectionState = "notConnected"
        msg.appControl.rewardState = "overrideNegative"
        msgvec.input(msg)

        valid, rew = msgvec.get_reward()
        self.assertFalse(valid)
        self.assertTrue(math.isnan(rew))
        
        # Override, timeout
        msg = new_message("appControl")
        msg.appControl.connectionState = "connected"
        msg.appControl.rewardState = "overridePositive"
        msgvec.input(msg)

        valid, rew = msgvec.get_reward()
        self.assertTrue(valid)

        stale_msg = new_message("voltage")
        stale_msg.logMonoTime = msg.logMonoTime + 0.25 * 1e9
        msgvec.input(stale_msg)
         
        valid, rew = msgvec.get_reward()
        self.assertTrue(valid)

        stale_msg = new_message("voltage")
        stale_msg.logMonoTime = msg.logMonoTime + 0.55 * 1e9
        msgvec.input(stale_msg)
     
        valid, rew = msgvec.get_reward()
        self.assertFalse(valid)

    def test_appcontrol_override_reward_multimessages(self):
        # This test covers the case when you are live-controlling the robot, and so appcontrol messages
        # might be coming in every 100ms. But, occasionally, you may want to override the reward, and that
        # override may be for 2000ms. We used to have a bug where the override would be applied for only
        # the last message.
        config = {"obs": [], "act": [
            {
                "type": "msg",
                "path": "odriveCommand.desiredVelocityLeft",
                "timeout": 0.01,
                "transform": {
                    "type": "rescale",
                    "vec_range": [-1, 1],
                    "msg_range": [-3, 3],
                },
            },
            {
                "type": "msg",
                "path": "odriveCommand.desiredVelocityRight",
                "timeout": 0.01,
                "transform": {
                    "type": "rescale",
                    "vec_range": [-1, 1],
                    "msg_range": [-3, 3],
                },
            },
        ],  
        "appcontrol": {
            "mode": "steering_override_v1",
            "timeout": 0.125,
        },
        "rew": {
            "override": {
                "positive_reward": 3.0,
                "positive_reward_timeout": 2.00,

                "negative_reward": -12.0,
                "negative_reward_timeout": 3.00,
            }
        },}
        msgvec = PyMsgVec(config, PyMessageTimingMode.REPLAY)

        # No override, no reward
        msg = new_message("appControl")
        msg.appControl.connectionState = "connected"
        msg.appControl.rewardState = "noOverride"
        msg.appControl.motionState = "noOverride"
        msgvec.input(msg)
        valid, rew = msgvec.get_reward()
        self.assertFalse(valid)

        msg = new_message("appControl")
        msg.appControl.connectionState = "connected"
        msg.appControl.rewardState = "noOverride"
        msg.appControl.motionState = "manualControl"
        msgvec.input(msg)
        valid, rew = msgvec.get_reward()
        self.assertFalse(valid)

        posmsg = new_message("appControl")
        posmsg.appControl.connectionState = "connected"
        posmsg.appControl.rewardState = "overridePositive"
        posmsg.appControl.motionState = "manualControl"
        msgvec.input(posmsg)
        valid, rew = msgvec.get_reward()
        self.assertTrue(valid)

        msg = new_message("appControl")
        msg.appControl.connectionState = "connected"
        msg.appControl.rewardState = "noOverride"
        msg.appControl.motionState = "manualControl"
        msg.logMonoTime = posmsg.logMonoTime + 0.5 * 1e9
        msgvec.input(msg)
        valid, rew = msgvec.get_reward()
        self.assertTrue(valid)

        msg = new_message("appControl")
        msg.appControl.connectionState = "connected"
        msg.appControl.rewardState = "noOverride"
        msg.appControl.motionState = "manualControl"
        msg.logMonoTime = posmsg.logMonoTime + 1.9 * 1e9
        msgvec.input(msg)
        valid, rew = msgvec.get_reward()
        self.assertTrue(valid)

        msg = new_message("appControl")
        msg.appControl.connectionState = "connected"
        msg.appControl.rewardState = "noOverride"
        msg.appControl.motionState = "manualControl"
        msg.logMonoTime = posmsg.logMonoTime + 2.01 * 1e9
        msgvec.input(msg)
        valid, rew = msgvec.get_reward()
        self.assertFalse(valid)

        negmsg = new_message("appControl")
        negmsg.appControl.connectionState = "connected"
        negmsg.appControl.rewardState = "overrideNegative"
        negmsg.appControl.motionState = "manualControl"
        msgvec.input(negmsg)
        valid, rew = msgvec.get_reward()
        self.assertTrue(valid)

        msg = new_message("appControl")
        msg.appControl.connectionState = "connected"
        msg.appControl.rewardState = "noOverride"
        msg.appControl.motionState = "manualControl"
        msg.logMonoTime = negmsg.logMonoTime + 0.5 * 1e9
        msgvec.input(msg)
        valid, rew = msgvec.get_reward()
        self.assertTrue(valid)

        msg = new_message("appControl")
        msg.appControl.connectionState = "connected"
        msg.appControl.rewardState = "noOverride"
        msg.appControl.motionState = "manualControl"
        msg.logMonoTime = negmsg.logMonoTime + 1.9 * 1e9
        msgvec.input(msg)
        valid, rew = msgvec.get_reward()
        self.assertTrue(valid)

        msg = new_message("appControl")
        msg.appControl.connectionState = "connected"
        msg.appControl.rewardState = "noOverride"
        msg.appControl.motionState = "manualControl"
        msg.logMonoTime = negmsg.logMonoTime + 2.01 * 1e9
        msgvec.input(msg)
        valid, rew = msgvec.get_reward()
        self.assertTrue(valid)

        msg = new_message("appControl")
        msg.appControl.connectionState = "connected"
        msg.appControl.rewardState = "noOverride"
        msg.appControl.motionState = "manualControl"
        msg.logMonoTime = negmsg.logMonoTime + 3.01 * 1e9
        msgvec.input(msg)
        valid, rew = msgvec.get_reward()
        self.assertFalse(valid)


    def test_appcontrol_disconnected(self):
        config = {"obs": [], "act": [
            {
                "type": "msg",
                "path": "odriveCommand.desiredCurrentLeft",
                "timeout": 0.01,
                "transform": {
                    "type": "rescale",
                    "vec_range": [-1, 1],
                    "msg_range": [-3, 3],
                },
            },
            {
                "type": "msg",
                "path": "odriveCommand.desiredCurrentRight",
                "timeout": 0.01,
                "transform": {
                    "type": "rescale",
                    "vec_range": [-1, 1],
                    "msg_range": [-3, 3],
                },
            },
        ],  
        "appcontrol": {
            "mode": "steering_override_v1",
            "timeout": 0.125,
        },
        "rew": {
            "override": {
                "positive_reward": 3.0,
                "positive_reward_timeout": 0.50,

                "negative_reward": -12.0,
                "negative_reward_timeout": 0.50,
            }
        },}
        msgvec = PyMsgVec(config, PyMessageTimingMode.REALTIME)

        result = msgvec.get_action_command(np.array([-1.0, 1.0], dtype=np.float32))
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].which(), "odriveCommand")
        self.assertEqual(result[0].odriveCommand.desiredCurrentLeft, -3.0)
        self.assertEqual(result[0].odriveCommand.desiredCurrentRight, 3.0)

        msg = new_message("appControl")
        msg.appControl.connectionState = "notConnected"
        msg.appControl.motionState = "noOverride"
        msgvec.input(msg)

        result = msgvec.get_action_command(np.array([-1.0, 1.0], dtype=np.float32))
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].which(), "odriveCommand")
        self.assertEqual(result[0].odriveCommand.desiredCurrentLeft, -3.0)
        self.assertEqual(result[0].odriveCommand.desiredCurrentRight, 3.0)

        msg = new_message("appControl")
        msg.appControl.connectionState = "notConnected"
        msg.appControl.motionState = "suspendMajorMotion"
        msgvec.input(msg)

        result = msgvec.get_action_command(np.array([-1.0, 1.0], dtype=np.float32))
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0].which(), "odriveCommand")
        self.assertEqual(result[0].odriveCommand.desiredCurrentLeft, 0.0)
        self.assertEqual(result[0].odriveCommand.desiredCurrentRight, 0.0)

        time.sleep(0.25)

        # If the last app control messages has timed out, then you just resume operating normally
        result = msgvec.get_action_command(np.array([-1.0, 1.0], dtype=np.float32))
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].which(), "odriveCommand")
        self.assertEqual(result[0].odriveCommand.desiredCurrentLeft, -3.0)
        self.assertEqual(result[0].odriveCommand.desiredCurrentRight, 3.0)


    def test_input_ready_status(self):
        config = {"obs":  [
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
                    "msg_range": [0, 100],
                    "vec_range": [0, 100],
                }
            },
        ], "act": [
                {
                    "type": "msg",
                    "path": "headCommand.pitchAngle",
                    "timeout": 0.01,
                    "transform": {
                        "type": "identity",
                    },
                },
            ]}
        msgvec = PyMsgVec(config, PyMessageTimingMode.REPLAY)

        msg = new_message("appControl")
        msg.appControl.connectionState = "notConnected"
        msg.appControl.motionState = "stopAllOutputs"
        self.assertEqual(msgvec.input(msg), {"msg_processed": True, "act_ready": False})

        msg = new_message("headFeedback")
        self.assertEqual(msgvec.input(msg), {"msg_processed": False, "act_ready": False})

        msg = new_message("headCommand")
        self.assertEqual(msgvec.input(msg), {"msg_processed": True, "act_ready": True})

        msg = new_message("headCommand")
        self.assertEqual(msgvec.input(msg), {"msg_processed": True, "act_ready": False})

        msg = new_message("voltage")
        self.assertEqual(msgvec.input(msg), {"msg_processed": True, "act_ready": False})

        self.assertEqual(msgvec.get_obs_vector_raw(), [0.0])

        msg = new_message("headCommand")
        self.assertEqual(msgvec.input(msg), {"msg_processed": True, "act_ready": True})

    def test_input_ready_status_multimsg(self):
        config = {"obs":  [
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
                    "msg_range": [0, 100],
                    "vec_range": [0, 100],
                }
            },
        ], "act": [
                    {
                    "type": "msg",
                    "path": "odriveCommand.desiredCurrentLeft",
                    "timeout": 0.125,
                    "transform": {
                        "type": "rescale",
                        "msg_range": [-2, 2],
                        "vec_range": [-1, 1],
                    },
                },

                {
                    "type": "msg",
                    "path": "odriveCommand.desiredCurrentRight",
                    "timeout": 0.125,
                    "transform": {
                        "type": "rescale",
                        "msg_range": [-2, 2],
                        "vec_range": [-1, 1],
                    },
                },

                { 
                    "type": "msg",
                    "path": "headCommand.pitchAngle",
                    "index": -1,
                    "timeout": 0.125,
                    "transform": {
                        "type": "rescale",
                        "vec_range": [-1, 1],
                        "msg_range": [-45.0, 45.0],
                    },
                },

                { 
                    "type": "msg",
                    "path": "headCommand.yawAngle",
                    "index": -1,
                    "timeout": 0.125,
                    "transform": {
                        "type": "rescale",
                        "vec_range": [-1, 1],
                        "msg_range": [-45.0, 45.0],
                    },
                },
            ]}
        msgvec = PyMsgVec(config, PyMessageTimingMode.REPLAY)

        msg = new_message("appControl")
        msg.appControl.connectionState = "notConnected"
        msg.appControl.motionState = "stopAllOutputs"
        self.assertEqual(msgvec.input(msg), {"msg_processed": True, "act_ready": False})

        msg = new_message("headFeedback")
        self.assertEqual(msgvec.input(msg), {"msg_processed": False, "act_ready": False})

        msg = new_message("headCommand")
        self.assertEqual(msgvec.input(msg), {"msg_processed": True, "act_ready": False})

        msg = new_message("odriveCommand")
        self.assertEqual(msgvec.input(msg), {"msg_processed": True, "act_ready": True})
    
    def test_act_values_bounded(self):
        config = {"obs":  [], "act": [
                    {
                    "type": "msg",
                    "path": "odriveCommand.desiredCurrentLeft",
                    "timeout": 0.125,
                    "transform": {
                        "type": "rescale",
                        "msg_range": [-2, 2],
                        "vec_range": [-1, 1],
                    },
                },  
            ]}
        msgvec = PyMsgVec(config, PyMessageTimingMode.REPLAY)

        with self.assertRaises(RuntimeError):
            msgvec.get_action_command(np.array([math.inf], dtype=np.float32))

        with self.assertRaises(RuntimeError):
            msgvec.get_action_command(np.array([-math.inf], dtype=np.float32))

        with self.assertRaises(RuntimeError):
            msgvec.get_action_command(np.array([math.nan], dtype=np.float32))

    def test_timing_index_basic(self):
        config = {"obs":  [
            {
                "type": "msg",
                "path": "voltage.volts",
                "index": -1,
                "timing_index": -1,
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
        msgvec = PyMsgVec(config, PyMessageTimingMode.REPLAY)
        self.assertEqual(msgvec.obs_size(), 2)

        msg = new_message("voltage")
        msg.voltage.type = "mainBattery"
        msg.voltage.volts = 12.0
        msgvec.input(msg)

        # You query the obs vector instantly after getting the message
        self.assertEqual(msgvec.get_obs_vector_raw().tolist(), [12, 0.0])

        msg = new_message("voltage")
        msg.voltage.type = "mainBattery"
        msg.voltage.volts = 12.0
        msgvec.input(msg)

        # Replay mode, so timing is always relative to the last message received
        self.assertEqual(msgvec.get_obs_vector_raw().tolist(), [12, 0.0])

        omsg = new_message("odriveFeedback")
        omsg.logMonoTime = msg.logMonoTime + 1e9 * 0.005
        msgvec.input(omsg)

        # If you have delayed half of the timeout, on average should be 0.5 timing value
        self.assertEqual(msgvec.get_obs_vector_raw().tolist(), [12, 0.5])

        omsg = new_message("odriveFeedback")
        omsg.logMonoTime = msg.logMonoTime + 1e9 * 0.01
        msgvec.input(omsg)

        # If you have delayed the timeout, on average should be 1.0 timing value
        self.assertEqual(msgvec.get_obs_vector_raw().tolist(), [12, 1.0])

        # If things are really delayed, you should get a clamped 2.0 timing value
        omsg = new_message("odriveFeedback")
        omsg.logMonoTime = msg.logMonoTime + 1e9 * 1.01
        msgvec.input(omsg)

        self.assertEqual(msgvec.get_obs_vector_raw().tolist(), [12, 2.0])

    def test_timing_index_second_msg(self):
        config = {"obs":  [
            {
                "type": "msg",
                "path": "voltage.volts",
                "index": -1,
                "timing_index": -1,
                "timeout": 0.01,
                "filter": {
                    "field": "voltage.type",
                    "op": "eq",
                    "value": "mainBattery",
                },
                "transform": {
                    "type": "identity"
                }

            },
            {
                "type": "msg",
                "path": "headFeedback.pitchAngle",
                "index": -1,
                "timeout": 0.01,
                "transform": {
                    "type": "identity"
                }
            },
        ], "act": []}
        msgvec = PyMsgVec(config, PyMessageTimingMode.REPLAY)
        self.assertEqual(msgvec.obs_size(), 3)

        msg = new_message("headFeedback")
        msg.headFeedback.pitchAngle = 10.0
        msgvec.input(msg)

        msg = new_message("voltage")
        msg.voltage.type = "mainBattery"
        msg.voltage.volts = 12.0
        msgvec.input(msg)

        # You query the obs vector instantly after getting the message
        # so, you expect 0 for the timing index
        self.assertEqual(msgvec.get_obs_vector_raw().tolist(), [12, 0.0, 10])

    def test_timing_index_configs(self):
        config = {"obs":  [
            {
                "type": "msg",
                "path": "voltage.volts",
                "index": -1,
                "timing_index": -2,
                "timeout": 0.01,
                "filter": {
                    "field": "voltage.type",
                    "op": "eq",
                    "value": "mainBattery",
                },
                "transform": {
                    "type": "identity"
                }
            }
        ], "act": []}

        with self.assertRaises(Exception):
            msgvec = PyMsgVec(config, PyMessageTimingMode.REPLAY)

    def test_timing_index_multi1(self):
        config = {"obs":  [
            {
                "type": "msg",
                "path": "voltage.volts",
                "index": -3,
                "timing_index": -1,
                "timeout": 0.01,
                "filter": {
                    "field": "voltage.type",
                    "op": "eq",
                    "value": "mainBattery",
                },
                "transform": {
                    "type": "identity"
                }

            },
            {
                "type": "msg",
                "path": "headFeedback.pitchAngle",
                "index": -1,
                "timeout": 0.01,
                "transform": {
                    "type": "identity"
                }
            },
        ], "act": []}
        msgvec = PyMsgVec(config, PyMessageTimingMode.REPLAY)
        self.assertEqual(msgvec.obs_size(), 5)

        msg = new_message("headFeedback")
        msg.headFeedback.pitchAngle = 10.0
        msgvec.input(msg)

        msg = new_message("voltage")
        msg.voltage.type = "mainBattery"
        msg.voltage.volts = 12.0
        msgvec.input(msg)

        # You query the obs vector instantly after getting the message
        # so, you expect 0 for the timing index
        self.assertEqual(msgvec.get_obs_vector_raw().tolist(), [12.0, 0.0, 0.0, 0.0, 10.0])  

        msg = new_message("voltage")
        msg.voltage.type = "mainBattery"
        msg.voltage.volts = 11.0
        msgvec.input(msg)

        msg = new_message("voltage")
        msg.voltage.type = "mainBattery"
        msg.voltage.volts = 10.0
        msgvec.input(msg)

        prevTime = msg.logMonoTime
        msg = new_message("headFeedback")
        msg.headFeedback.pitchAngle = 9.0
        msg.logMonoTime = prevTime + 1e9 * 0.005
        msgvec.input(msg)
        self.assertEqual(msgvec.get_obs_vector_raw().tolist(), [10.0, 11.0, 12.0, 0.5, 9.0])  

    def test_timing_index_realtime(self):
        config = {"obs":  [
            {
                "type": "msg",
                "path": "voltage.volts",
                "index": -1,
                "timing_index": -1,
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
        msgvec = PyMsgVec(config, PyMessageTimingMode.REALTIME)
        self.assertEqual(msgvec.obs_size(), 2)

        msg = new_message("voltage")
        msg.voltage.type = "mainBattery"
        msg.voltage.volts = 12.0
        msgvec.input(msg)

        # You query the obs vector instantly after getting the message
        self.assertAlmostEqual(msgvec.get_obs_vector_raw()[1], 0.0, places=1)

        time.sleep(0.005)
        omsg = new_message("odriveFeedback")
        msgvec.input(omsg)

        # If you have delayed half of the timeout, on average should be 0.5 timing value
        self.assertAlmostEqual(msgvec.get_obs_vector_raw()[1], 0.5, places=1)

        time.sleep(0.01)
        self.assertAlmostEqual(msgvec.get_obs_vector_raw()[1], 1.5, places=1)

        time.sleep(0.01)
        self.assertAlmostEqual(msgvec.get_obs_vector_raw()[1], 2.0, places=1)

    def test_obs_and_action(self):
        config = {"obs":  [
            {
                "type": "msg",
                "path": "odriveFeedback.leftMotor.vel",
                "index": -1,
                "timeout": 0.01,
                "transform": {
                    "type": "rescale",
                    "msg_range": [0, 100],
                    "vec_range": [0, 100],
                }
            },
        ], "act": [{
                "type": "msg",
                "path": "odriveCommand.desiredVelocityLeft",
                "index": -1,
                "timeout": 0.01,
                "transform": {
                    "type": "rescale",
                    "msg_range": [0, 100],
                    "vec_range": [0, 100],
                }
            },]}
        msgvec = PyMsgVec(config, PyMessageTimingMode.REALTIME) 

        msg = new_message("odriveFeedback")
        msg.odriveFeedback.leftMotor.vel = 12.0
        print(msgvec.input(msg))

        actions = msgvec.get_action_command(np.array([0.0], dtype=np.float32))
        print(msgvec.input(actions[0]))

        self.assertEqual(msgvec.get_obs_vector_raw().tolist(), [12.0])

    def test_obs_and_action_replay(self):
        config = {"obs":  [
            {
                "type": "msg",
                "path": "odriveFeedback.leftMotor.vel",
                "index": -1,
                "timeout": 0.01,
                "transform": {
                    "type": "rescale",
                    "msg_range": [0, 100],
                    "vec_range": [0, 100],
                }
            },
        ], "act": [{
                "type": "msg",
                "path": "odriveCommand.desiredVelocityLeft",
                "index": -1,
                "timeout": 0.01,
                "transform": {
                    "type": "rescale",
                    "msg_range": [0, 100],
                    "vec_range": [0, 100],
                }
            },]}
        msgvec = PyMsgVec(config, PyMessageTimingMode.REPLAY) 

        msg = new_message("odriveFeedback")
        msg.odriveFeedback.leftMotor.vel = 1.0
        self.assertEqual(msgvec.input(msg), {'msg_processed': True, 'act_ready': False})

        msg = new_message("odriveCommand")
        msg.odriveCommand.desiredVelocityLeft = 2.0
        self.assertEqual(msgvec.input(msg), {'msg_processed': True, 'act_ready': True})

        self.assertEqual(msgvec.get_obs_vector_raw().tolist(), [1.0])
        self.assertEqual(msgvec.get_act_vector().tolist(), [2.0])

        msg = new_message("odriveFeedback")
        msg.odriveFeedback.leftMotor.vel = 3.0
        self.assertEqual(msgvec.input(msg), {'msg_processed': True, 'act_ready': False})

        msg = new_message("odriveCommand")
        msg.odriveCommand.desiredVelocityLeft = 4.0
        self.assertEqual(msgvec.input(msg), {'msg_processed': True, 'act_ready': True})

        self.assertEqual(msgvec.get_obs_vector_raw().tolist(), [3.0])
        self.assertEqual(msgvec.get_act_vector().tolist(), [4.0])

    def test_real_data_to_vector(self):
        config = {"obs": [
                { 
                    "type": "msg",
                    "path": "odriveFeedback.leftMotor.vel",
                    "index": -1,
                    "timeout": 0.125,
                    "transform": {
                        "type": "identity",
                    },
                },

                {
                    "type": "msg",
                    "path": "voltage.volts",
                    "index": -1,
                    "timeout": 0.125,
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
                    "path": "headFeedback.pitchAngle",
                    "index": -1,
                    "timeout": 0.125,
                    "transform": {
                        "type": "rescale",
                        "msg_range": [-45.0, 45.0],
                        "vec_range": [-1, 1],
                    },
                },

                {
                    "type": "vision",
                    "size": 17003,
                    "timeout": 0.100,
                    "index": -1,
                }
            ],

            "act": [
                {
                    "type": "msg",
                    "path": "odriveCommand.desiredCurrentLeft",
                    "timeout": 0.125,
                    "transform": {
                        "type": "rescale",
                        "msg_range": [-2, 2],
                        "vec_range": [-1, 1],
                    },
                },

                {
                    "type": "msg",
                    "path": "odriveCommand.desiredCurrentRight",
                    "timeout": 0.125,
                    "transform": {
                        "type": "rescale",
                        "msg_range": [-2, 2],
                        "vec_range": [-1, 1],
                    },
                },

                { 
                    "type": "msg",
                    "path": "headCommand.pitchAngle",
                    "index": -1,
                    "timeout": 0.125,
                    "transform": {
                        "type": "rescale",
                        "vec_range": [-1, 1],
                        "msg_range": [-45.0, 45.0],
                    },
                },

                { 
                    "type": "msg",
                    "path": "headCommand.yawAngle",
                    "index": -1,
                    "timeout": 0.125,
                    "transform": {
                        "type": "rescale",
                        "vec_range": [-1, 1],
                        "msg_range": [-45.0, 45.0],
                    },
                },
            ],

            "rew": {
                "override": {
                    "positive_reward": 1.0,
                    "positive_reward_timeout": 0.50,

                    "negative_reward": -1.0,
                    "negative_reward_timeout": 0.50,
                }
            },

            "appcontrol": {
                "mode": "steering_override_v1",
                "timeout": 0.125,
            },

            "done": {
                "mode": "on_reward_override",
            }
        }
        msgvec = PyMsgVec(config, PyMessageTimingMode.REPLAY)

        messages_since_last_inference = []
        seen_act_ready = False

        with open(os.path.join(HOST_CONFIG.RECORD_DIR, "unittest", "alphalog-2fd2bf40-2022-10-10-23_14.log"), "rb") as f:
            events = log.Event.read_multiple(f)
            for evt in events:
                result = msgvec.input(evt)
                messages_since_last_inference.append(evt.which())
               
                print(f"Result: {result} - {evt.which()}")

                if evt.which() == "modelInference":
                    messages_since_last_inference = []
                    seen_act_ready = False
                    timeout, obs = msgvec.get_obs_vector()

                if "headCommand" in messages_since_last_inference and "odriveCommand" in messages_since_last_inference and not seen_act_ready:
                    self.assertTrue(result["act_ready"])
                    seen_act_ready = True
                else:
                    self.assertFalse(result["act_ready"])

                if evt.which() == "appControl":
                    self.assertEqual(result["msg_processed"], True)
                    self.assertEqual(evt.appControl.rewardState, "noOverride")

                if result["act_ready"]:
                    act = msgvec.get_act_vector()
                    rew_valid, rew_value = msgvec.get_reward()
                    self.assertEqual(act.tolist(), [0.0, 0.0, 0.0, 0.0])
                    self.assertFalse(rew_valid)
                    self.assertTrue(math.isnan(rew_value))
          


class TestMsgVecRelative(MsgVecBaseTest):
    def test_relative_basic(self):
        config = {"obs": [], "act": [
            {
                "type": "relative_msg",
                "path": "headCommand.yawAngle",
                "initial": 0.0,
                "range": [-45.0, 45.0],
                "timeout": 0.01,
                "transform": {
                    "type": "rescale",
                    "vec_range": [-1, 1],
                    "msg_range": [-5.0, 5.0],
                },
            },

        ]}
        msgvec = PyMsgVec(config, PyMessageTimingMode.REALTIME)

        result = msgvec.get_action_command(np.array([0.0], dtype=np.float32))
        self.assertEqual(result[0].headCommand.yawAngle, 0.0)

        result = msgvec.get_action_command(np.array([1.0], dtype=np.float32))
        self.assertEqual(result[0].headCommand.yawAngle, 5.0)

        result = msgvec.get_action_command(np.array([-1.0], dtype=np.float32))
        self.assertEqual(result[0].headCommand.yawAngle, 0.0)

        result = msgvec.get_action_command(np.array([-1.0], dtype=np.float32))
        self.assertEqual(result[0].headCommand.yawAngle, -5.0)

        result = msgvec.get_action_command(np.array([-1.0], dtype=np.float32))
        self.assertEqual(result[0].headCommand.yawAngle, -10.0)

        result = msgvec.get_action_command(np.array([0.5], dtype=np.float32))
        self.assertEqual(result[0].headCommand.yawAngle, -7.5)

    def test_relative_all_same_type(self):
        config = {"obs": [], "act": [
            {
                "type": "relative_msg",
                "path": "headCommand.yawAngle",
                "initial": 0.0,
                "range": [-45.0, 45.0],
                "timeout": 0.01,
                "transform": {
                    "type": "rescale",
                    "vec_range": [-1, 1],
                    "msg_range": [-5.0, 5.0],
                },
            },

            {
                "type": "msg",
                "path": "headCommand.pitchAngle",
                "initial": 0.0,
                "range": [-45.0, 45.0],
                "timeout": 0.01,
                "transform": {
                    "type": "rescale",
                    "vec_range": [-1, 1],
                    "msg_range": [-5.0, 5.0],
                },
            },

        ]}

        with self.assertRaises(RuntimeError):
            msgvec = PyMsgVec(config, PyMessageTimingMode.REALTIME)

    def test_relative_initial(self):
        config = {"obs": [], "act": [
            {
                "type": "relative_msg",
                "path": "headCommand.yawAngle",
                "initial": 10.0,
                "range": [-45.0, 45.0],
                "timeout": 0.01,
                "transform": {
                    "type": "rescale",
                    "vec_range": [-1, 1],
                    "msg_range": [-5.0, 5.0],
                },
            },

        ]}
        msgvec = PyMsgVec(config, PyMessageTimingMode.REALTIME)

        result = msgvec.get_action_command(np.array([0.0], dtype=np.float32))
        self.assertEqual(result[0].headCommand.yawAngle, 10.0)

        result = msgvec.get_action_command(np.array([1.0], dtype=np.float32))
        self.assertEqual(result[0].headCommand.yawAngle, 15.0)

        result = msgvec.get_action_command(np.array([-1.0], dtype=np.float32))
        self.assertEqual(result[0].headCommand.yawAngle, 10.0)

        result = msgvec.get_action_command(np.array([-1.0], dtype=np.float32))
        self.assertEqual(result[0].headCommand.yawAngle, 5.0)

        result = msgvec.get_action_command(np.array([-1.0], dtype=np.float32))
        self.assertEqual(result[0].headCommand.yawAngle, 0.0)

        result = msgvec.get_action_command(np.array([0.5], dtype=np.float32))
        self.assertEqual(result[0].headCommand.yawAngle, 2.5)

    def test_relative_min_max(self):
        config = {"obs": [], "act": [
            {
                "type": "relative_msg",
                "path": "headCommand.yawAngle",
                "initial": 0.0,
                "range": [-45.0, 45.0],
                "timeout": 0.01,
                "transform": {
                    "type": "rescale",
                    "vec_range": [-1, 1],
                    "msg_range": [-100.0, 100.0],
                },
            },

        ]}
        msgvec = PyMsgVec(config, PyMessageTimingMode.REALTIME)

        result = msgvec.get_action_command(np.array([0.0], dtype=np.float32))
        self.assertEqual(result[0].headCommand.yawAngle, 0.0)

        result = msgvec.get_action_command(np.array([1.0], dtype=np.float32))
        self.assertEqual(result[0].headCommand.yawAngle, 45.0)

        result = msgvec.get_action_command(np.array([-1.0], dtype=np.float32))
        self.assertEqual(result[0].headCommand.yawAngle, -45.0)

        result = msgvec.get_action_command(np.array([-1.0], dtype=np.float32))
        self.assertEqual(result[0].headCommand.yawAngle, -45.0)

    def test_relative_no_transform(self):
        config = {"obs": [], "act": [
            {
                "type": "relative_msg",
                "path": "headCommand.yawAngle",
                "initial": 2.0,
                "range": [-45.0, 45.0],
                "timeout": 0.01,
                "transform": {
                    "type": "identity",
                }
            }
        ]}
        msgvec = PyMsgVec(config, PyMessageTimingMode.REALTIME)

        result = msgvec.get_action_command(np.array([0.0], dtype=np.float32))
        self.assertEqual(result[0].headCommand.yawAngle, 2.0)

        result = msgvec.get_action_command(np.array([1.0], dtype=np.float32))
        self.assertEqual(result[0].headCommand.yawAngle, 3.0)

        result = msgvec.get_action_command(np.array([-10.0], dtype=np.float32))
        self.assertEqual(result[0].headCommand.yawAngle, -7.0)

        result = msgvec.get_action_command(np.array([-100.0], dtype=np.float32))
        self.assertEqual(result[0].headCommand.yawAngle, -45.0)

    def test_relative_replay(self):
        config = {"obs": [], "act": [
            {
                "type": "relative_msg",
                "path": "headCommand.yawAngle",
                "initial": 0.0,
                "range": [-45.0, 45.0],
                "timeout": 0.01,
                "transform": {
                    "type": "rescale",
                    "vec_range": [-1, 1],
                    "msg_range": [-45.0, 45.0],
                },
            },
        ]}
        msgvec = PyMsgVec(config, PyMessageTimingMode.REPLAY)

        test_params = [
            (45, 1),
            (-45, -1),
            (46, 1),
            (-46, -1),
            (0, 1),
            (45 / 2, 0.5),
            (45 / 2, 0.0)]

        for msg, vec in test_params:
            print(f"test: {msg} -> {vec}")
            event = log.Event.new_message()
            event.init("headCommand")
            event.headCommand.yawAngle = msg
            self.assertMsgProcessed(msgvec.input(event))
            self.assertEqual(msgvec.get_act_vector().dtype, np.float32)
            self.assertAlmostEqual(msgvec.get_act_vector()[0], vec)

        # Feed in a message that doesn't exist in the config
        event = log.Event.new_message()
        event.init("voltage")
        event.voltage.volts = 13.5
        event.voltage.type = "mainBattery"

        self.assertFalse(msgvec.input(event)["msg_processed"])

    def test_relative_replay_smaller_range(self):
        config = {"obs": [], "act": [
            {
                "type": "relative_msg",
                "path": "headCommand.yawAngle",
                "initial": 0.0,
                "range": [-45.0, 45.0],
                "timeout": 0.01,
                "transform": {
                    "type": "rescale",
                    "vec_range": [-1, 1],
                    "msg_range": [-5.0, 5.0],
                },
            },
        ]}
        msgvec = PyMsgVec(config, PyMessageTimingMode.REPLAY)

        test_params = [
            (0, 0),
            (5, 1),
            (10, 1),
            (12.5, 0.5),
            (12.5, 0.0),
            (-5, -1),
            (-10, -1),
            (-12.5, -0.5),
            (-12.5, 0.0),
            (0, 1)
            ]

        for msg, vec in test_params:
            print(f"test: {msg} -> {vec}")
            event = log.Event.new_message()
            event.init("headCommand")
            event.headCommand.yawAngle = msg
            self.assertMsgProcessed(msgvec.input(event))
            self.assertEqual(msgvec.get_act_vector().dtype, np.float32)
            self.assertAlmostEqual(msgvec.get_act_vector()[0], vec)

        # Feed in a message that doesn't exist in the config
        event = log.Event.new_message()
        event.init("voltage")
        event.voltage.volts = 13.5
        event.voltage.type = "mainBattery"

        self.assertFalse(msgvec.input(event)["msg_processed"])

    def test_relative_replay_initial(self):
        config = {"obs": [], "act": [
            {
                "type": "relative_msg",
                "path": "headCommand.yawAngle",
                "initial": 20.0,
                "range": [-45.0, 45.0],
                "timeout": 0.01,
                "transform": {
                    "type": "rescale",
                    "vec_range": [-1, 1],
                    "msg_range": [-5.0, 5.0],
                },
            },
        ]}
        msgvec = PyMsgVec(config, PyMessageTimingMode.REPLAY)

        test_params = [
            (20, 0),
            (25, 1),
            (25, 0),
            (22.5, -0.5),
            ]

        for msg, vec in test_params:
            print(f"test: {msg} -> {vec}")
            event = log.Event.new_message()
            event.init("headCommand")
            event.headCommand.yawAngle = msg
            self.assertMsgProcessed(msgvec.input(event))
            self.assertEqual(msgvec.get_act_vector().dtype, np.float32)
            self.assertAlmostEqual(msgvec.get_act_vector()[0], vec)

        # Feed in a message that doesn't exist in the config
        event = log.Event.new_message()
        event.init("voltage")
        event.voltage.volts = 13.5
        event.voltage.type = "mainBattery"

        self.assertFalse(msgvec.input(event)["msg_processed"])

    def test_relative_invalid_config1(self):
        config = {"obs": [], "act": [
            {
                "type": "relative_msg",
                "path": "headCommand.yawAngle",
                "initial": 200.0,
                "range": [-45.0, 45.0],
                "timeout": 0.01,
                "transform": {
                    "type": "rescale",
                    "vec_range": [-1, 1],
                    "msg_range": [-5.0, 5.0],
                },
            },
        ]}

        with self.assertRaises(RuntimeError):
            msgvec = PyMsgVec(config, PyMessageTimingMode.REPLAY)

    def test_relative_invalid_config2(self):
        config = {"obs": [], "act": [
            {
                "type": "relative_msg",
                "path": "headCommand.yawAngle",
                "initial": 0.0,
                "range": [-1.0, -3.0],
                "timeout": 0.01,
                "transform": {
                    "type": "rescale",
                    "vec_range": [-1, 1],
                    "msg_range": [-5.0, 5.0],
                },
            },
        ]}

        with self.assertRaises(RuntimeError):
            msgvec = PyMsgVec(config, PyMessageTimingMode.REPLAY)
      
    def test_relative_manual_override(self):
        config = {"obs": [], "act": [
            {
                "type": "relative_msg",
                "path": "odriveCommand.desiredVelocityLeft",
                "initial": 0.0,
                "range": [-1.0, 1.0],
                "timeout": 0.125,
                "transform": {
                    "type": "rescale",
                    "vec_range": [-1, 1],
                    "msg_range": [-0.15, 0.15],
                },
            },
        ],
           "appcontrol": {
                "mode": "steering_override_v1",
                "timeout": 0.125,
            },}

        realtime_msgvec = PyMsgVec(config, PyMessageTimingMode.REALTIME)
        replay_msgvec = PyMsgVec(config, PyMessageTimingMode.REPLAY)

        msg = new_message("appControl")
        msg.appControl.connectionState = "connected"
        msg.appControl.motionState = "manualControl"
        msg.appControl.linearXOverride = 1.0
        realtime_msgvec.input(msg)

        result = realtime_msgvec.get_action_command(np.array([0.0], dtype=np.float32))
        for msg in result:
            replay_msgvec.input(msg)

        print(replay_msgvec.get_act_vector())
        
        msg = new_message("appControl")
        msg.appControl.connectionState = "connected"
        msg.appControl.motionState = "manualControl"
        msg.appControl.linearXOverride = 1.0
        realtime_msgvec.input(msg)

        result = realtime_msgvec.get_action_command(np.array([0.0], dtype=np.float32))
        for msg in result:
            replay_msgvec.input(msg)

        print(replay_msgvec.get_act_vector())

        # The problem in this test is the mismatch of ranges in the training data
        # For example, in typical control, outputing a maximum value of 1.0 (and 1.0 is the max because we have a tanh activation function) 
        # that will increase your velocity by 0.15 units in the message itself.
        # However, you may do something like issue a manual override, which immediately sets the velocity from 0.0 to 1.0
        # What happens is that there is no way to represent this overflow in the network, because it saturates at 1
        # With a tanh activation, the gradient shrinks to 0 out past 1.0, so it wouldn't do any good
        
        # Option 1: Change the ranges to allow for 1.0 or -1.0 to basically do a full range transform
        #  You could then penalize outputting anything too extreme
        # Option 2: Maybe keep some internal saturation counter, and output 1.0's for a few frames afterwards after a saturation
        
class TestMsgVecDiscrete(MsgVecBaseTest):
    def test_discrete_basic(self):
        config = {"obs": [], "act": [
            {
                "type": "discrete_msg",
                "path": "headCommand.yawAngle",
                "initial": 0.0,
                "range": [-45.0, 45.0],
                # It's a good idea to have zero be the first element, so an all-zero action vector is a no-op
                "choices": [0.0, -10.0, -5.0, -1.0, 1.0, 5.0, 10.0],
                "timeout": 0.01,
                "transform": {
                    "type": "identity",
                },
            },

        ]}
        msgvec = PyMsgVec(config, PyMessageTimingMode.REALTIME)

        self.assertEqual(msgvec.act_size(), 7)

        result = msgvec.get_action_command(np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32))
        self.assertEqual(result[0].headCommand.yawAngle, 0.0)
        print(result)

        result = msgvec.get_action_command(np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32))
        self.assertEqual(result[0].headCommand.yawAngle, -10.0)
        print(result)

        result = msgvec.get_action_command(np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32))
        self.assertEqual(result[0].headCommand.yawAngle, -15.0)
        print(result)

        result = msgvec.get_action_command(np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32))
        self.assertEqual(result[0].headCommand.yawAngle, -25.0)
        print(result)

        result = msgvec.get_action_command(np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32))
        self.assertEqual(result[0].headCommand.yawAngle, -35.0)
        print(result)

        result = msgvec.get_action_command(np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32))
        self.assertEqual(result[0].headCommand.yawAngle, -45.0)
        print(result)

        result = msgvec.get_action_command(np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32))
        self.assertEqual(result[0].headCommand.yawAngle, -45.0)
        print(result)

    def test_discrete_transform(self):
        config = {"obs": [], "act": [
            {
                "type": "discrete_msg",
                "path": "headCommand.yawAngle",
                "initial": 0.0,
                "range": [-45.0, 45.0],
                # It's a good idea to have zero be the first element, so an all-zero action vector is a no-op
                "choices": [0.0, -10.0, -5.0, -1.0, 1.0, 5.0, 10.0],
                "timeout": 0.01,
                "transform": {
                    "type": "rescale",
                    "vec_range": [-1, 1],
                    "msg_range": [-0.15, 0.15],
                },
            },

        ]}

        # There is no sense to transform anything in a discrete msg, because the vector values are just used to pick a discrete choice from a list
        # and not to manipulate a continuous value
        with self.assertRaises(RuntimeError):
            msgvec = PyMsgVec(config, PyMessageTimingMode.REALTIME)

    def test_discrete_multiple_actions(self):
        config = {"obs": [], "act": [
            {
                "type": "discrete_msg",
                "path": "headCommand.pitchAngle",
                "initial": 5.0,
                "range": [-45.0, 45.0],
                # It's a good idea to have zero be the first element, so an all-zero action vector is a no-op
                "choices": [0, -1, 1],
                "timeout": 0.01,
                "transform": {
                    "type": "identity",
                },
            },
            {
                "type": "discrete_msg",
                "path": "odriveCommand.desiredVelocityLeft",
                "initial": 0.0,
                "range": [-45.0, 45.0],
                # It's a good idea to have zero be the first element, so an all-zero action vector is a no-op
                "choices": [0, -2, -1, 1, 2],
                "timeout": 0.01,
                "transform": {
                    "type": "identity",
                },
            },

        ]}
        msgvec = PyMsgVec(config, PyMessageTimingMode.REALTIME)
        self.assertEqual(msgvec.act_size(), 8)

        result = msgvec.get_action_command(np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32))
        self.assertEqual(result[0].headCommand.pitchAngle, 5.0)
        self.assertEqual(result[1].odriveCommand.desiredVelocityLeft, 0.0)
        
        result = msgvec.get_action_command(np.array([0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float32))
        self.assertEqual(result[0].headCommand.pitchAngle, 4.0)
        self.assertEqual(result[1].odriveCommand.desiredVelocityLeft, 0.0)

        result = msgvec.get_action_command(np.array([0.0, 1.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0], dtype=np.float32))
        self.assertEqual(result[0].headCommand.pitchAngle, 4.0)
        self.assertEqual(result[1].odriveCommand.desiredVelocityLeft, -2.0)

