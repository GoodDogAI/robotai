import tempfile
import os
import time
import copy
import tensorrt as trt
import unittest

from src.config import YOLOV7_CLASS_NAMES
from src.train.modelloader import create_and_validate_onnx, create_and_validate_trt, model_fullname, get_vision_model_size


class TestModelLoaderTRT(unittest.TestCase):
    def setUp(self) -> None:
        self.sampleVisionConfig = {
            "type": "vision",
            "load_fn": "src.models.yolov7.load.load_yolov7",
            "input_format": "rgb",
            "checkpoint": "/home/jake/robotai/_checkpoints/yolov7-tiny.pt",

            # Input dimensions must be divisible by the stride
            # In current situations, the image will be cropped to the nearest multiple of the stride
            "dimension_stride": 32,

            "intermediate_layer": "input.219", # Another option to try could be onnx::Conv_254
            "intermediate_slice": 53,
        }

        self.sampleVisionConfigMultiLayers = {
            "type": "vision",
            "load_fn": "src.models.yolov7.load.load_yolov7",
            "input_format": "rgb",
            "checkpoint": "/home/jake/robotai/_checkpoints/yolov7-tiny.pt",

            # Input dimensions must be divisible by the stride
            # In current situations, the image will be cropped to the nearest multiple of the stride
            "dimension_stride": 32,

            "intermediate_layer": ["onnx::Conv_351", "onnx::Conv_379", "onnx::Conv_365"], 
            "intermediate_slice": 53,
        }

        self.sampleRewardConfig =  {
            "type": "reward",
            "load_fn": "src.models.yolov7.load.load_yolov7",
            "input_format": "rgb",
            "checkpoint": "/home/jake/robotai/_checkpoints/yolov7.pt",
            "class_names": YOLOV7_CLASS_NAMES,

            # Input dimensions must be divisible by the stride
            # In current situations, the image will be cropped to the nearest multiple of the stride
            "dimension_stride": 32,

            "max_detections": 100,
            "iou_threshold": 0.45,

            "reward_module": "src.train.reward.SumCenteredObjectsPresentReward",

            "reward_kwargs": {
                "class_weights": {
                    "person": 3,
                    "spoon": 10,
                },
                "reward_scale": 0.10,
                "center_epsilon": 0.1,  
            }
        }

        self.sampleBrainConfig = {
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
                            "msg_range": [0, 15],
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
                        "path": "odriveCommand.desiredVelocityLeft",
                        "timeout": 0.125,
                        "transform": {
                            "type": "rescale",
                            "msg_range": [-2, 2],
                            "vec_range": [-1, 1],
                        },
                    },

                    {
                        "type": "msg",
                        "path": "odriveCommand.desiredVelocityRight",
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
                    "base": "reward",

                    "override": {
                        "positive_reward": 1.0,
                        "positive_reward_timeout": 0.0667,

                        "negative_reward": -1.0,
                        "negative_reward_timeout": 0.0667,
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

    def test_onnx_vision(self):
        create_and_validate_onnx(self.sampleVisionConfig, skip_cache=True)

    def test_get_vision_model_size(self):
        fullname = model_fullname(self.sampleVisionConfig)
        self.assertEqual(get_vision_model_size(fullname), 17003)

    def test_trt_vision(self):
        onnx_path = create_and_validate_onnx(self.sampleVisionConfig)
        create_and_validate_trt(onnx_path, skip_cache=True)

    def test_onnx_reward(self):
        create_and_validate_onnx(self.sampleRewardConfig, skip_cache=True)

    def test_trt_reward(self):
        onnx_path = create_and_validate_onnx(self.sampleRewardConfig)
        create_and_validate_trt(onnx_path, skip_cache=True)

    def test_multilayer_intermediate_concat(self):
        onnx_path = create_and_validate_onnx(self.sampleVisionConfigMultiLayers, skip_cache=True)
        create_and_validate_trt(onnx_path, skip_cache=True)

    def test_stable_baselines_actor(self):
        onnx_path = create_and_validate_onnx(self.sampleBrainConfig, skip_cache=True)
        create_and_validate_trt(onnx_path, skip_cache=True)

    def test_stable_baselines_normalized_actor(self):
        config = copy.deepcopy(self.sampleBrainConfig)
        config["checkpoint"] = "/home/jake/robotai/_checkpoints/basic-brain-test1-sb3-run55.zip"
        onnx_path = create_and_validate_onnx(config, skip_cache=True)
        create_and_validate_trt(onnx_path, skip_cache=True)

    def test_model_fullname(self):
        with tempfile.TemporaryDirectory() as td:
            with open(os.path.join(td, "test.pt"), "wb") as f:
                f.write(b"test")

            config = {
                "type": "vision",
                "load_fn": "src.models.yolov7.load.load_yolov7",
                "input_format": "rgb",
                "checkpoint": os.path.join(td, "test.pt"),

                # Input dimensions must be divisible by the stride
                # In current situations, the image will be cropped to the nearest multiple of the stride
                "dimension_stride": 32,

                "intermediate_layer": "input.219", # Another option to try could be onnx::Conv_254
                "intermediate_slice": 53,
            }

            f1 = model_fullname(config)
            f2 = model_fullname(config)
            self.assertEqual(f1, f2)

            new_config = copy.deepcopy(config)
            f3 = model_fullname(new_config)
            self.assertEqual(f1, f3)

            new_config["intermediate_slice"] = 54
            f4 = model_fullname(new_config)
            self.assertNotEqual(f1, f4)

            # Need to sleep because file mtime is used to determine if the model needs to be rehashed
            time.sleep(0.01)
            with open(os.path.join(td, "test.pt"), "wb") as f:
                f.write(b"omgwerwer")

            f5 = model_fullname(config)
            self.assertNotEqual(f1, f5)

            # Need to sleep because file mtime is used to determine if the model needs to be rehashed
            time.sleep(0.01)
            with open(os.path.join(td, "test.pt"), "wb") as f:
                f.write(b"test")

            f6 = model_fullname(config)
            self.assertEqual(f1, f6)
    