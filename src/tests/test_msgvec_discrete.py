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


class TestMsgVecDiscrete(MsgVecBaseTest):
    def test_discrete_basic(self):
        config = {"obs": [], "act": [
            {
                "type": "discrete_msg",
                "path": "headCommand.yawAngle",
                "initial": 0.0,
                "range": [-45.0, 45.0],
                # It's a good idea to have zero be the first element, so an all-zero action vector is a no-op
                "choices": [-10.0, -5.0, -1.0, 1.0, 5.0, 10.0],
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

    def test_replay_basic(self):
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
        msgvec = PyMsgVec(config, PyMessageTimingMode.REPLAY)
        self.assertEqual(msgvec.act_size(), 7)

        msg = new_message("headCommand")
        msg.headCommand.yawAngle = 0.0
        print(msgvec.input(msg))

        self.assertEqual(msgvec.get_act_vector().tolist(), [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        msg = new_message("headCommand")
        msg.headCommand.yawAngle = 10.0
        print(msgvec.input(msg))

        self.assertEqual(msgvec.get_act_vector().tolist(), [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])

        msg = new_message("headCommand")
        msg.headCommand.yawAngle = 10.0
        print(msgvec.input(msg))

        self.assertEqual(msgvec.get_act_vector().tolist(), [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    def test_replay_midway(self):
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
        msgvec = PyMsgVec(config, PyMessageTimingMode.REPLAY)
        self.assertEqual(msgvec.act_size(), 7)

        msg = new_message("headCommand")
        msg.headCommand.yawAngle = 0.5
        msgvec.input(msg)

        self.assertEqual(msgvec.get_act_vector().tolist(), [0.5, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0])

        msg = new_message("headCommand")
        msg.headCommand.yawAngle = 0.5
        msgvec.input(msg)

        # Zero change from previous
        self.assertEqual(msgvec.get_act_vector().tolist(), [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        msg = new_message("headCommand")
        msg.headCommand.yawAngle = 0.0
        msgvec.input(msg)

        self.assertEqual(msgvec.get_act_vector().tolist(), [0.5, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0])
    
        msg = new_message("headCommand")
        msg.headCommand.yawAngle = 0.1
        msgvec.input(msg)

        np.testing.assert_almost_equal(msgvec.get_act_vector().tolist(), [0.9, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0])

    def test_replay_minmax(self):
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
        msgvec = PyMsgVec(config, PyMessageTimingMode.REPLAY)

        msg = new_message("headCommand")
        msg.headCommand.yawAngle = 45.0
        msgvec.input(msg)

        self.assertEqual(msgvec.get_act_vector().tolist(), [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])

        msg = new_message("headCommand")
        msg.headCommand.yawAngle = -45.0
        msgvec.input(msg)

        self.assertEqual(msgvec.get_act_vector().tolist(), [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    def test_discrete_multiple_replay(self):
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
        msgvec = PyMsgVec(config, PyMessageTimingMode.REPLAY)
        self.assertEqual(msgvec.act_size(), 8)

        msg = new_message("headCommand")
        msg.headCommand.pitchAngle = -2.0
        msgvec.input(msg)

        msg = new_message("odriveCommand")
        msg.odriveCommand.desiredVelocityLeft = -2.0
        msgvec.input(msg)

        self.assertEqual(msgvec.get_act_vector().tolist(), [0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
