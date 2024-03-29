import os
import time
import tempfile
import numpy as np
import unittest
import itertools

from unittest.mock import MagicMock

from src.messaging import new_message
from src.train.modelloader import model_fullname
from src.train.arrowcache import ArrowModelCache
from src.train.rldataset import MsgVecDataset
from src.train.reward_modifiers import reward_modifier_penalize_move_backwards
from src.config import HOST_CONFIG, MODEL_CONFIGS

class TestMsgVecDataset(unittest.TestCase):
    def test_current_config(self):
        cache = MsgVecDataset(os.path.join(HOST_CONFIG.RECORD_DIR, "unittest"), MODEL_CONFIGS["basic-brain-test1"])

        last_override = False
        last_done = False
        # We set shuffle_within_group to False so that we can see each log's samples linearly, and make sure that various flags and overrides are set correctly
        for entry in itertools.islice(cache.generate_dataset(shuffle_within_group=False), 10000):
            if not entry["reward_override"] and last_override:
                self.assertTrue(last_done)
                
            last_override = entry["reward_override"]
            last_done = entry["done"]

        samples = list(itertools.islice(cache.generate_dataset(shuffle_within_group=False), 100))

        for index, sample in enumerate(samples[:-1]):
            np.testing.assert_array_almost_equal(sample["next_obs"], samples[index + 1]["obs"])

    def test_current_config_timing(self):
        cache = MsgVecDataset(os.path.join(HOST_CONFIG.RECORD_DIR, "unittest", "oct22testgroup"), MODEL_CONFIGS["basic-brain-test1"])

        # cache.vision_cache = MagicMock()
        # cache.vision_cache.get.return_value = np.zeros((17003), dtype=np.float32)
        # cache.reward_cache = MagicMock()
        # cache.reward_cache.get.return_value = np.zeros((1), dtype=np.float32)
        counter = 0
        for entry in cache.generate_dataset():
            counter += 1

        # Read it twice to make sure the cache is warmed up
        # counter = 0
        # for entry in cache.generate_samples():
        #     counter += 1


        start = time.perf_counter()
        counter = 0
        for entry in cache.generate_dataset():
            counter += 1
        end = time.perf_counter()

        print(f"Time to generate {counter} samples: {end - start}")
        print(f"Samples per second: {counter / (end - start)}")

    def test_missing_frames(self):
        cache = MsgVecDataset(os.path.join(HOST_CONFIG.RECORD_DIR, "unittest", "oct22testgroup"), MODEL_CONFIGS["basic-brain-test1"])

        # Check that each frame key is incrementing by 1
        last_frame = None
        for entry in cache.generate_dataset(shuffle_within_group=False):
            frame = int(entry["key"].split('-')[-1])
            print(entry["key"], frame)
            if last_frame is not None:
                self.assertEqual(frame, last_frame + 1)
            last_frame = frame
            

    def test_on_reward_override_done(self):
        brain_config = {
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
                    "index": -3,
                    "timeout": 0.125,
                    "transform": {
                        "type": "identity",
                    },
                }, 

                { 
                    "type": "msg",
                    "path": "headFeedback.yawAngle",
                    "index": -3,
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
                    "path": "odriveCommand.desiredVelocityLeft",
                    "timeout": 0.125,
                    "transform": {
                        "type": "rescale",
                        "msg_range": [-0.5, 0.5],
                        "vec_range": [-1, 1],
                    },
                },

                {
                    "type": "msg",
                    "path": "odriveCommand.desiredVelocityRight",
                    "timeout": 0.125,
                    "transform": {
                        "type": "rescale",
                        "msg_range": [-0.5, 0.5],
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
    
        cache = MsgVecDataset(os.path.join(HOST_CONFIG.RECORD_DIR, "unittest"), brain_config)


        last_override = False
        last_done = None
        for entry in cache.generate_dataset(shuffle_within_group=False):
            if entry["reward_override"]:
                print(entry["reward_override"], entry["reward"], entry["done"])

            if not entry["reward_override"] and last_override:
                print("-----------------")
                self.assertTrue(last_done)
                
            # Make sure the datatypes line up
            self.assertTrue(isinstance(entry["done"], bool))
            self.assertTrue(isinstance(entry["reward_override"], bool))
            self.assertTrue(isinstance(entry["reward"], float))

            last_override = entry["reward_override"]
            last_done = entry["done"]
            