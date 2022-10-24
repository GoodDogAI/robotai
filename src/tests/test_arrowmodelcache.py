import os
import time
import tempfile
import numpy as np
import unittest
import itertools

from unittest.mock import MagicMock

from src.messaging import new_message
from src.train.modelloader import model_fullname
from src.train.arrowcache import ArrowModelCache, ArrowRLDataset
from src.config import HOST_CONFIG, MODEL_CONFIGS

class TestArrowModelCache(unittest.TestCase):
    def test_basic_vision(self):
        cache = ArrowModelCache(os.path.join(HOST_CONFIG.RECORD_DIR, "unittest"), MODEL_CONFIGS["yolov7-tiny-s53"], force_rebuild=True)

    def test_basic_reward(self):
        cache = ArrowModelCache(os.path.join(HOST_CONFIG.RECORD_DIR, "unittest"), MODEL_CONFIGS["yolov7-tiny-prioritize_centered_nms"], force_rebuild=True)

class TestArrowRLCache(unittest.TestCase):
    def test_current_config(self):
        cache = ArrowRLDataset(os.path.join(HOST_CONFIG.RECORD_DIR, "unittest"), MODEL_CONFIGS["basic-brain-test1"])

        last_override = False
        # We set shuffle_within_group to False so that we can see each log's samples linearly, and make sure that various flags and overrides are set correctly
        for entry in itertools.islice(cache.generate_samples(shuffle_within_group=False), 10000):
            if entry["reward_override"]:
                print(entry["reward_override"], entry["reward"], entry["done"])

            if not entry["reward_override"] and last_override:
                self.assertTrue
                
            last_override = entry["reward_override"]

        samples = list(itertools.islice(cache.generate_samples(shuffle_within_group=False), 100))

        for index, sample in enumerate(samples[:-1]):
            np.testing.assert_array_almost_equal(sample["next_obs"], samples[index + 1]["obs"])

    def test_current_config_timing(self):
        cache = ArrowRLDataset(os.path.join(HOST_CONFIG.RECORD_DIR, "unittest"), MODEL_CONFIGS["basic-brain-test1"])

        # cache.vision_cache = MagicMock()
        # cache.vision_cache.get.return_value = np.zeros((17003), dtype=np.float32)
        # cache.reward_cache = MagicMock()
        # cache.reward_cache.get.return_value = np.zeros((1), dtype=np.float32)
        counter = 0
        for entry in cache.generate_samples():
            counter += 1

        # Read it twice to make sure the cache is warmed up
        # counter = 0
        # for entry in cache.generate_samples():
        #     counter += 1


        start = time.perf_counter()
        counter = 0
        for entry in cache.generate_samples():
            counter += 1
        end = time.perf_counter()

        print(f"Time to generate {counter} samples: {end - start}")
        print(f"Samples per second: {counter / (end - start)}")

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
    
        cache = ArrowRLDataset(os.path.join(HOST_CONFIG.RECORD_DIR, "unittest"), brain_config)


        last_override = False
        last_done = None
        for entry in cache.generate_samples(shuffle_within_group=False):
            if entry["reward_override"]:
                print(entry["reward_override"], entry["reward"], entry["done"])

            if not entry["reward_override"] and last_override:
                print("-----------------")
                self.assertTrue(last_done)
                
            last_override = entry["reward_override"]
            last_done = entry["done"]

class ManualTestArrowRLCache(unittest.TestCase):
    def setUp(self) -> None:
        self. brain_config = {
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

    def test_single_observation(self):
        with tempfile.TemporaryDirectory() as td, open(os.path.join(td, "test.log"), "w") as f:
            cache = ArrowRLDataset(td, self.brain_config)
            cache.vision_cache = MagicMock()
            cache.vision_cache.get.return_value = np.zeros((17003), dtype=np.float32)
            cache.reward_cache = MagicMock()
            cache.reward_cache.get.return_value = np.zeros((1), dtype=np.float32)

            # Messages get ready
            msg = new_message("odriveFeedback")
            msg.write(f)
            
            msg = new_message("headFeedback")
            msg.write(f)
            
            # Inference occurs
            msg = new_message("modelInference")
            msg.write(f)

            f.flush()
            self.assertEqual(list(cache.generate_samples()), [])

            # Action vector gets written out
            msg = new_message("odriveCommand")
            msg.write(f)
            
            # Still blank because you need two datapoints to make a valid log
            f.flush()
            self.assertEqual(list(cache.generate_samples()), [])

            msg = new_message("modelInference")
            msg.write(f)
            msg = new_message("odriveCommand")
            msg.write(f)

            samples = list(cache.generate_samples())
            self.assertEqual(len(samples), 1)

            self.assertTrue(samples[0]["done"])
            

    # Test that you wait for the obs vectors to populate, though this is somewhat helped by waiting for the first inference
    def test_wait_for_obs(self):
        with tempfile.TemporaryDirectory() as td, open(os.path.join(td, "test.log"), "w") as f:
            cache = ArrowRLDataset(td, self.brain_config)
            cache.vision_cache = MagicMock()
            cache.vision_cache.get.return_value = np.zeros((17003), dtype=np.float32)
            cache.reward_cache = MagicMock()
            cache.reward_cache.get.return_value = np.zeros((1), dtype=np.float32)

            # Messages get ready
            msg = new_message("odriveFeedback")
            msg.write(f)

            # Not sending any headback, so no samples should be generated            
            # msg = new_message("headFeedback")
            # msg.write(f)
            
            # Inference occurs
            msg = new_message("modelInference")
            msg.write(f)

            f.flush()
            self.assertEqual(list(cache.generate_samples()), [])

            # Action vector gets written out
            msg = new_message("odriveCommand")
            msg.write(f)
            
            # Still blank because you need two datapoints to make a valid log
            f.flush()
            self.assertEqual(list(cache.generate_samples()), [])

            msg = new_message("modelInference")
            msg.write(f)
            msg = new_message("odriveCommand")
            msg.write(f)

            samples = list(cache.generate_samples())
            self.assertEqual(len(samples), 0)
            
    