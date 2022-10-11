import unittest
import json
import os
import math
import time
import random
from cereal import log
from src.messaging import new_message
from src.msgvec.pymsgvec import PyMsgVec, PyTimeoutResult
from src.config import HOST_CONFIG, MODEL_CONFIGS



class TestMsgVec(unittest.TestCase):
    def assertMsgProcessed(self, input_result):
        self.assertTrue(input_result["msg_processed"])
    
    def test_init(self):
        config = {"obs": [], "act": []}
        PyMsgVec(json.dumps(config).encode("utf-8"))
    
    def test_failed_init(self):
        with self.assertRaises(Exception):
            PyMsgVec(b"invalid json")

    def test_feed_real_data(self):
        log_path = os.path.join(HOST_CONFIG.RECORD_DIR, "unittest", "alphalog-41a516ae-2022-9-19-2_20.log")
        default_cfg = MODEL_CONFIGS[HOST_CONFIG.DEFAULT_BRAIN_CONFIG]
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
        self.assertEqual(msgvec.get_obs_vector_raw(), [0.0])

        def _sendAndAssert(voltage, vector):
            event = log.Event.new_message()
            event.init("voltage")
            event.voltage.volts = voltage
            event.voltage.type = "mainBattery"

            self.assertMsgProcessed(msgvec.input(event.to_bytes()))

            self.assertAlmostEqual(msgvec.get_obs_vector_raw()[0], vector, places=3)

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
        self.assertEqual(msgvec.get_obs_vector_raw(), [0.0, 0.0])

        event = log.Event.new_message()
        event.init("voltage")
        event.voltage.volts = 13.5
        event.voltage.type = "mainBattery"

        self.assertTrue(msgvec.input(event.to_bytes()))
        self.assertEqual(msgvec.get_obs_vector_raw(), [1.0, 2.0])

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
        self.assertEqual(msgvec.get_obs_vector_raw(), [0.0])

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

            self.assertMsgProcessed(msgvec.input(event.to_bytes()))
            self.assertEqual(msgvec.get_obs_vector_raw(), expect)

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
        self.assertEqual(msgvec.get_obs_vector_raw(), [0.0, 0.0, 0.0])

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

            self.assertMsgProcessed(msgvec.input(event.to_bytes()))
            self.assertEqual(msgvec.get_obs_vector_raw(), expect)

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
                    "path": "odriveCommand.currentLeft",
                    "timeout": 0.01,
                    "transform": {
                        "type": "identity",
                    },
                },

                {
                    "type": "msg",
                    "path": "odriveCommand.currentRight",
                    "timeout": 0.01,
                    "transform": {
                        "type": "identity",
                    },
                },
            ]}
        msgvec = PyMsgVec(json.dumps(config).encode("utf-8"))

        self.assertEqual(msgvec.act_size(), 4)

        result = msgvec.get_action_command([1.0, 2.0, 3.0, 4.0])
        print(result)

        self.assertEqual(len(result), 2)
        self.assertEqual(result[0].which(), "headCommand")
        self.assertEqual(result[1].which(), "odriveCommand")

        self.assertEqual(result[0].headCommand.pitchAngle, 1.0)
        self.assertEqual(result[0].headCommand.yawAngle, 45.0)
        self.assertEqual(result[1].odriveCommand.currentLeft, 3.0)
        self.assertEqual(result[1].odriveCommand.currentRight, 4.0)

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
        msgvec = PyMsgVec(json.dumps(config).encode("utf-8"))

        result = msgvec.get_action_command([2.0])
        self.assertEqual(result[0].headCommand.yawAngle, 45.0)

        result = msgvec.get_action_command([-2.0])
        self.assertEqual(result[0].headCommand.yawAngle, -45.0)

        result = msgvec.get_action_command([0.0])
        self.assertEqual(result[0].headCommand.yawAngle, 0.0)

        result = msgvec.get_action_command([0.5])
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
        msgvec = PyMsgVec(json.dumps(config).encode("utf-8"))

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
            self.assertMsgProcessed(msgvec.input(event.to_bytes()))
            self.assertAlmostEqual(msgvec.get_act_vector()[0], vec)

        # Feed in a message that doesn't exist in the config
        event = log.Event.new_message()
        event.init("voltage")
        event.voltage.volts = 13.5
        event.voltage.type = "mainBattery"

        self.assertFalse(msgvec.input(event.to_bytes())["msg_processed"])

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
        msgvec = PyMsgVec(json.dumps(config).encode("utf-8"))


        for i in range(1000):
            f = random.uniform(-2, 2)
            messages = msgvec.get_action_command([f])

            self.assertEqual(len(messages), 1)
            self.assertMsgProcessed(msgvec.input(messages[0].as_builder().to_bytes()))

            self.assertAlmostEqual(msgvec.get_act_vector()[0], min(max(f, -1), 1) , places=3)

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
            msgvec = PyMsgVec(json.dumps(config).encode("utf-8"))

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
            msgvec = PyMsgVec(json.dumps(config).encode("utf-8"))

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
            msgvec = PyMsgVec(json.dumps(config).encode("utf-8"))

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
            msgvec = PyMsgVec(json.dumps(config).encode("utf-8"))

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
        msgvec = PyMsgVec(json.dumps(config).encode("utf-8"))

        timeout, _ = msgvec.get_obs_vector()
        self.assertEqual(timeout, PyTimeoutResult.MESSAGES_NOT_READY)

        msg = new_message("voltage")
        msg.voltage.volts = 13.5
        msg.voltage.type = "mainBattery"
        msgvec.input(msg.to_bytes())

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
        msgvec = PyMsgVec(json.dumps(config).encode("utf-8"))

        timeout, _ = msgvec.get_obs_vector()
        self.assertEqual(timeout, PyTimeoutResult.MESSAGES_NOT_READY)

        msg = new_message("voltage")
        msg.voltage.volts = 13.5
        msg.voltage.type = "mainBattery"
        msgvec.input(msg.to_bytes())

        timeout, _ = msgvec.get_obs_vector()
        self.assertEqual(timeout, PyTimeoutResult.MESSAGES_PARTIALLY_READY)

        time.sleep(0.02)

        timeout, _ = msgvec.get_obs_vector()
        self.assertEqual(timeout, PyTimeoutResult.MESSAGES_NOT_READY)

        msg = new_message("voltage")
        msg.voltage.volts = 13.5
        msg.voltage.type = "mainBattery"
        msgvec.input(msg.to_bytes())

        timeout, _ = msgvec.get_obs_vector()
        self.assertEqual(timeout, PyTimeoutResult.MESSAGES_PARTIALLY_READY)

        for i in range(9):
            msg = new_message("voltage")
            msg.voltage.volts = 13.5
            msg.voltage.type = "mainBattery"
            msgvec.input(msg.to_bytes())

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
        msgvec = PyMsgVec(json.dumps(config).encode("utf-8"))

        timeout, _ = msgvec.get_obs_vector()
        self.assertEqual(timeout, PyTimeoutResult.MESSAGES_NOT_READY)

        msg = new_message("voltage")
        msg.voltage.volts = 13.5
        msg.voltage.type = "mainBattery"
        msgvec.input(msg.to_bytes())

        timeout, _ = msgvec.get_obs_vector()
        self.assertEqual(timeout, PyTimeoutResult.MESSAGES_NOT_READY)

        msg = new_message("odriveFeedback")
        msg.odriveFeedback.leftMotor.vel = 1.0
        msgvec.input(msg.to_bytes())

        timeout, _ = msgvec.get_obs_vector()
        self.assertEqual(timeout, PyTimeoutResult.MESSAGES_ALL_READY)

        time.sleep(0.02)

        timeout, _ = msgvec.get_obs_vector()
        self.assertEqual(timeout, PyTimeoutResult.MESSAGES_NOT_READY)

    def test_appcontrol_basic(self):
        config = {"obs": [], "act": [],
            "appcontrol": {
                "mode": "steering_override_v1",
                "timeout": 0.125,
            },
         }
        msgvec = PyMsgVec(json.dumps(config).encode("utf-8"))

        result = msgvec.get_action_command([])
        self.assertEqual(result, [])

        msg = new_message("appControl")
        msg.appControl.connectionState = "connected"
        msgvec.input(msg.to_bytes())

        result = msgvec.get_action_command([])
        self.assertEqual(result, [])

        msg = new_message("appControl")
        msg.appControl.connectionState = "connected"
        msg.appControl.motionState = "manualControl"
        msg.appControl.linearXOverride = 1.0
        msgvec.input(msg.to_bytes())

        result = msgvec.get_action_command([])
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0].which(), "odriveCommand")
        self.assertEqual(result[1].which(), "headCommand")
        self.assertEqual(result[0].odriveCommand.currentLeft, -1.0)
        self.assertEqual(result[0].odriveCommand.currentRight, 1.0)
        self.assertEqual(result[1].headCommand.pitchAngle, 0.0)
        self.assertEqual(result[1].headCommand.yawAngle, 0.0)

        msg = new_message("appControl")
        msg.appControl.connectionState = "connected"
        msg.appControl.motionState = "manualControl"
        msg.appControl.linearXOverride = 1.0
        msg.appControl.angularZOverride = 1.0
        msgvec.input(msg.to_bytes())

        result = msgvec.get_action_command([])
   
        self.assertAlmostEqual(result[0].odriveCommand.currentLeft, 0.0, places=3)
        self.assertAlmostEqual(result[0].odriveCommand.currentRight, 2.0, places=3)
        self.assertAlmostEqual(result[1].headCommand.pitchAngle, 0.0)
        self.assertAlmostEqual(result[1].headCommand.yawAngle, -30.0)

        time.sleep(0.15)

        result = msgvec.get_action_command([])
        self.assertEqual(result, [])

        msg = new_message("appControl")
        msg.appControl.connectionState = "connected"
        msg.appControl.motionState = "stopAllOutputs"
        msg.appControl.linearXOverride = 1.0
        msg.appControl.angularZOverride = 1.0
        msgvec.input(msg.to_bytes())

        result = msgvec.get_action_command([])
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0].which(), "odriveCommand")
        self.assertEqual(result[1].which(), "headCommand")
        self.assertEqual(result[0].odriveCommand.currentLeft, 0.0)
        self.assertEqual(result[0].odriveCommand.currentRight, 0.0)
        self.assertEqual(result[1].headCommand.pitchAngle, 0.0)
        self.assertEqual(result[1].headCommand.yawAngle, 0.0)

    def test_appcontrol_override(self):
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
            },}
        msgvec = PyMsgVec(json.dumps(config).encode("utf-8"))

        
        result = msgvec.get_action_command([-1.0, 1.0])
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].which(), "odriveCommand")
        self.assertEqual(result[0].odriveCommand.currentLeft, -3.0)
        self.assertEqual(result[0].odriveCommand.currentRight, 3.0)

        msg = new_message("appControl")
        msg.appControl.connectionState = "connected"
        msg.appControl.motionState = "stopAllOutputs"
        msgvec.input(msg.to_bytes())

        result = msgvec.get_action_command([-1.0, 1.0])
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0].which(), "odriveCommand")
        self.assertEqual(result[0].odriveCommand.currentLeft, 0.0)
        self.assertEqual(result[0].odriveCommand.currentRight, 0.0)

        # Not connected means no effect on action command
        msg = new_message("appControl")
        msg.appControl.connectionState = "notConnected"
        msg.appControl.motionState = "stopAllOutputs"
        msgvec.input(msg.to_bytes())

        result = msgvec.get_action_command([-1.0, 1.0])
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].which(), "odriveCommand")
        self.assertEqual(result[0].odriveCommand.currentLeft, -3.0)
        self.assertEqual(result[0].odriveCommand.currentRight, 3.0)

    def test_override_reward(self):
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
        msgvec = PyMsgVec(json.dumps(config).encode("utf-8"))

        # No override, no reward
        valid, rew = msgvec.get_reward()
        self.assertFalse(valid)

        # Override, positive reward
        msg = new_message("appControl")
        msg.appControl.connectionState = "connected"
        msg.appControl.rewardState = "overridePositive"
        msgvec.input(msg.to_bytes())

        valid, rew = msgvec.get_reward()
        self.assertTrue(valid)
        self.assertEqual(rew, 3.0)

        # Override, negative reward
        msg = new_message("appControl")
        msg.appControl.connectionState = "connected"
        msg.appControl.rewardState = "overrideNegative"
        msgvec.input(msg.to_bytes())

        valid, rew = msgvec.get_reward()
        self.assertTrue(valid)
        self.assertEqual(rew, -12.0)

        # Override, not connected
        msg = new_message("appControl")
        msg.appControl.connectionState = "notConnected"
        msg.appControl.rewardState = "overrideNegative"
        msgvec.input(msg.to_bytes())

        valid, rew = msgvec.get_reward()
        self.assertFalse(valid)
        self.assertTrue(math.isnan(rew))
        
        # Override, timeout
        msg = new_message("appControl")
        msg.appControl.connectionState = "connected"
        msg.appControl.rewardState = "overridePositive"
        msgvec.input(msg.to_bytes())

        valid, rew = msgvec.get_reward()
        self.assertTrue(valid)
        time.sleep(0.25)    
        valid, rew = msgvec.get_reward()
        self.assertTrue(valid)
        time.sleep(0.35)
        valid, rew = msgvec.get_reward()
        self.assertFalse(valid)

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
        msgvec = PyMsgVec(json.dumps(config).encode("utf-8"))

        msg = new_message("appControl")
        msg.appControl.connectionState = "notConnected"
        msg.appControl.motionState = "stopAllOutputs"
        self.assertEqual(msgvec.input(msg.to_bytes()), {"msg_processed": True, "act_ready": False})

        msg = new_message("headFeedback")
        self.assertEqual(msgvec.input(msg.to_bytes()), {"msg_processed": False, "act_ready": False})

        msg = new_message("headCommand")
        self.assertEqual(msgvec.input(msg.to_bytes()), {"msg_processed": True, "act_ready": True})

        msg = new_message("headCommand")
        self.assertEqual(msgvec.input(msg.to_bytes()), {"msg_processed": True, "act_ready": False})

        msg = new_message("voltage")
        self.assertEqual(msgvec.input(msg.to_bytes()), {"msg_processed": True, "act_ready": False})

        self.assertEqual(msgvec.get_obs_vector_raw(), [0.0])

        msg = new_message("headCommand")
        self.assertEqual(msgvec.input(msg.to_bytes()), {"msg_processed": True, "act_ready": True})

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
                    "path": "odriveCommand.currentLeft",
                    "timeout": 0.125,
                    "transform": {
                        "type": "rescale",
                        "msg_range": [-2, 2],
                        "vec_range": [-1, 1],
                    },
                },

                {
                    "type": "msg",
                    "path": "odriveCommand.currentRight",
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
        msgvec = PyMsgVec(json.dumps(config).encode("utf-8"))

        msg = new_message("appControl")
        msg.appControl.connectionState = "notConnected"
        msg.appControl.motionState = "stopAllOutputs"
        self.assertEqual(msgvec.input(msg.to_bytes()), {"msg_processed": True, "act_ready": False})

        msg = new_message("headFeedback")
        self.assertEqual(msgvec.input(msg.to_bytes()), {"msg_processed": False, "act_ready": False})

        msg = new_message("headCommand")
        self.assertEqual(msgvec.input(msg.to_bytes()), {"msg_processed": True, "act_ready": False})

        msg = new_message("odriveCommand")
        self.assertEqual(msgvec.input(msg.to_bytes()), {"msg_processed": True, "act_ready": True})
    
    def test_act_values_bounded(self):
        config = {"obs":  [], "act": [
                    {
                    "type": "msg",
                    "path": "odriveCommand.currentLeft",
                    "timeout": 0.125,
                    "transform": {
                        "type": "rescale",
                        "msg_range": [-2, 2],
                        "vec_range": [-1, 1],
                    },
                },  
            ]}
        msgvec = PyMsgVec(json.dumps(config).encode("utf-8"))

        with self.assertRaises(RuntimeError):
            msgvec.get_action_command([math.inf])

        with self.assertRaises(RuntimeError):
            msgvec.get_action_command([-math.inf])

        with self.assertRaises(RuntimeError):
            msgvec.get_action_command([math.nan])

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
                    "index": -1,
                }
            ],

            "act": [
                {
                    "type": "msg",
                    "path": "odriveCommand.currentLeft",
                    "timeout": 0.125,
                    "transform": {
                        "type": "rescale",
                        "msg_range": [-2, 2],
                        "vec_range": [-1, 1],
                    },
                },

                {
                    "type": "msg",
                    "path": "odriveCommand.currentRight",
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
        msgvec = PyMsgVec(json.dumps(config).encode("utf-8"))

        messages_since_last_inference = []
        seen_act_ready = False

        with open(os.path.join(HOST_CONFIG.RECORD_DIR, "unittest", "alphalog-2fd2bf40-2022-10-10-23_14.log"), "rb") as f:
            events = log.Event.read_multiple(f)
            for evt in events:
                result = msgvec.input(evt.as_builder().to_bytes())
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
                    self.assertEqual(act, [0.0, 0.0, 0.0, 0.0])
                    self.assertFalse(rew_valid)
                    self.assertTrue(math.isnan(rew_value))
          


