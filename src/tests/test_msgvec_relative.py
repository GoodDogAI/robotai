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
            act = msgvec.get_act_vector()
            self.assertEqual(act.dtype, np.float32)
            self.assertAlmostEqual(act[0], vec)

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
            act = msgvec.get_act_vector()
            self.assertEqual(act.dtype, np.float32)
            self.assertAlmostEqual(act[0], vec)

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
            act = msgvec.get_act_vector()
            self.assertEqual(act.dtype, np.float32)
            self.assertAlmostEqual(act[0], vec)

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
        