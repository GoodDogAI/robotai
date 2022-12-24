import os
import time
import tempfile
import numpy as np
import unittest
import itertools

from unittest.mock import MagicMock

from src.msgvec.pymsgvec import PyMsgVec, PyTimeoutResult, PyMessageTimingMode
from src.messaging import new_message
from src.train.modelloader import model_fullname
from src.train.arrowcache import ArrowModelCache
from src.train.rldataset import MsgVecDataset
from src.train.reward_modifiers import reward_modifier_penalize_move_backwards
from src.config import HOST_CONFIG, MODEL_CONFIGS



class ManualTestMsgVecDataset(unittest.TestCase):
    def setUp(self) -> None:
        self.brain_config = {
            "type": "brain",

            "checkpoint": "/home/jake/robotai/_checkpoints/basic-brain-test1-sb3-0.zip",
            "load_fn": "src.models.stable_baselines3.load.load_stable_baselines3_actor",

            "models": {
                "vision": "yolov7-tiny-s53",
                "reward": "yolov7-tiny-prioritize_centered_nms",
            },

            "msgvec": {
                "obs": [
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
                        "path": "headFeedback.yawAngle",
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
                        "timeout": 0.100,
                        "index": -1,
                    }
                ],

                "act": [
                    {
                        "type": "discrete_msg",
                        "path": "odriveCommand.desiredVelocityLeft",
                        "initial": 0.0,
                        "range": [-0.5, 0.5],
                        "choices": [-0.1, 0.1],
                        "timeout": 0.125,
                        "transform": {
                            "type": "identity",
                        },
                    },
                    {
                        "type": "discrete_msg",
                        "path": "odriveCommand.desiredVelocityRight",
                        "initial": 0.0,
                        "range": [-0.5, 0.5],
                        "choices": [-0.1, 0.1],
                        "timeout": 0.125,
                        "transform": {
                            "type": "identity",
                        },
                    },
                ],

                "rew": {
                    "base": "reward",

                    "override": {
                        "positive_reward": 10.0,
                        "positive_reward_timeout": 2.0,

                        "negative_reward": -15.0,
                        "negative_reward_timeout": 2.0,
                    }
                },

                "appcontrol": {
                    "mode": "steering_override_v1",
                    "timeout": 0.300,
                },

                "done": {
                    "mode": "on_reward_override",
                }
            }
        }


    def test_discrete_messages_replay(self):
        with tempfile.TemporaryDirectory() as td, open(os.path.join(td, "test.log"), "w") as f:
            cache = MsgVecDataset(td, self.brain_config)
            cache.vision_cache = MagicMock()
            cache.vision_cache.get.return_value = np.zeros((10), dtype=np.float32)
            cache.reward_cache = MagicMock()
            cache.reward_cache.get.return_value = 0.0

            # Messages get ready
            msg = new_message("odriveFeedback")
            msg.write(f)
            
            msg = new_message("headFeedback")
            msg.write(f)
            
            # Inference occurs
            msg = new_message("modelInference")
            msg.write(f)

            f.flush()
            self.assertEqual(list(cache.generate_dataset()), [])

            # Action vector gets written out
            msg = new_message("odriveCommand")
            msg.write(f)

            msg = new_message("modelInference")
            msg.write(f)
            msg = new_message("odriveCommand")
            msg.odriveCommand.desiredVelocityLeft = 0.1
            msg.write(f)

            msg = new_message("modelInference")
            msg.write(f)
            msg = new_message("odriveCommand")
            msg.odriveCommand.desiredVelocityLeft = 0.0
            msg.odriveCommand.desiredVelocityRight = 0.1
            msg.write(f)
            

            # The last output is always discarded, because you need to "next_obs" in most RL algorithms
            msg = new_message("modelInference")
            msg.write(f)
            msg = new_message("odriveCommand")
            msg.write(f)


            samples = list(cache.generate_dataset(shuffle_within_group=False))
            self.assertEqual(len(samples), 3)

            np.testing.assert_almost_equal(samples[0]["act"].tolist(), [1, 0, 0, 0, 0])
            np.testing.assert_almost_equal(samples[1]["act"].tolist(), [0, 0, 1, 0, 0])
            np.testing.assert_almost_equal(samples[2]["act"].tolist(), [0, 0.5, 0, 0, 0.5])

    def test_discrete_messages_backforth(self):
        msgvec = PyMsgVec(self.brain_config["msgvec"], PyMessageTimingMode.REALTIME)

        inputs = [
            "odriveFeedback",
            "headFeedback",
            "vision",

            "odriveFeedback",
            "headFeedback",
            "vision",
        ]

        # Tests cycling between msgvec and the logs

        with tempfile.TemporaryDirectory() as td, open(os.path.join(td, "test.log"), "w") as f:
            cache = MsgVecDataset(td, self.brain_config)
            cache.vision_cache = MagicMock()
            cache.vision_cache.get.return_value = np.zeros((10), dtype=np.float32)
            cache.reward_cache = MagicMock()
            cache.reward_cache.get.return_value = 0.0

            for inp in inputs:
                if inp == "vision":
                    msgvec.input_vision(np.arange(0, 10, dtype=np.float32), 1)
                    rdy, _ = msgvec.get_obs_vector()
                    self.assertEqual(rdy, PyTimeoutResult.MESSAGES_ALL_READY)

                    msg = new_message("modelInference")
                    msg.write(f)

                    for act in msgvec.get_action_command(np.array([0.0] * msgvec.act_size(), dtype=np.float32)):
                        act.as_builder().write(f)
                        print(msgvec.input(act))
                else:
                    msg = new_message(inp)
                    msg.write(f)
                    print(msgvec.input(msg))

            f.flush()
            samples = list(cache.generate_dataset(shuffle_within_group=False))
            self.assertEqual(len(samples), 1)

            np.testing.assert_almost_equal(samples[0]["act"].tolist(), [1, 0, 0, 0, 0])

    def test_discrete_messages_override_controls(self):
        msgvec = PyMsgVec(self.brain_config["msgvec"], PyMessageTimingMode.REALTIME)

        msg = new_message("appControl")
        msg.appControl.connectionState = "connected"
        msg.appControl.motionState = "manualControl"
        msg.appControl.linearXOverride = 1.0
        msgvec.input(msg)

        print(msgvec.get_action_command(np.array([0.0] * msgvec.act_size(), dtype=np.float32)))

        raise NotImplementedError("Still need to finish this")


            