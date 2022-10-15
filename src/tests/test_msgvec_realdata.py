import unittest
import json
import os
import math
import time
import random
from cereal import log
from src.messaging import new_message
from src.msgvec.pymsgvec import PyMsgVec, PyTimeoutResult, PyMessageTimingMode
from src.config import HOST_CONFIG, MODEL_CONFIGS



class TestMsgVecRealData(unittest.TestCase):
    def assertMsgProcessed(self, input_result):
        self.assertTrue(input_result["msg_processed"])
    
    def test_real_data_rewards(self):
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
        msgvec = PyMsgVec(json.dumps(config).encode("utf-8"), PyMessageTimingMode.REPLAY)

        messages_since_last_inference = []
        expect_next_reward_positive = False
        seen_act_ready = False

        with open(os.path.join(HOST_CONFIG.RECORD_DIR, "unittest", "alphalog-7d8256e7-2022-10-15-20_12.log"), "rb") as f:
            events = log.Event.read_multiple(f)
            for evt in events:
                result = msgvec.input(evt.as_builder().to_bytes())
                messages_since_last_inference.append(evt.which())
               
                print(f"Result: {result} - {evt.which()}")

                if evt.which() == "headCameraState":
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
                    if evt.appControl.rewardState == "overridePositive":
                        expect_next_reward_positive = True
                        print("Expecting positive reward")

                if result["act_ready"]:
                    act = msgvec.get_act_vector()
                    rew_valid, rew_value = msgvec.get_reward()
                    print(f"Act: {act} - Reward: {rew_valid} {rew_value}")

                    # if expect_next_reward_positive:
                    #     self.assertTrue(rew_valid)
                    #     self.assertGreater(rew_value, 0)
                        
                    
          


