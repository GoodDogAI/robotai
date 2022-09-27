import unittest
import time
import torch
import random
import numpy as np
import pyarrow as pa
import pandas as pd
from torch.utils.data import DataLoader

from polygraphy.cuda import DeviceView

from src.config import YOLOV7_CLASS_NAMES, HOST_CONFIG
from src.train.modelloader import load_vision_model, model_fullname
from src.train.videoloader import build_datapipe
from src.train.videodataset import VideoFrameDataset

class VideoLoaderTest(unittest.TestCase):
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

    def test_pyarrow(self):
        with load_vision_model(model_fullname(self.sampleVisionConfig)) as intermediate_engine, \
             load_vision_model(model_fullname(self.sampleRewardConfig)) as reward_engine:

            ds = VideoFrameDataset(base_path=HOST_CONFIG.RECORD_DIR)
            ds.download_and_prepare()

            def mapfn(example):
                feed = {
                    "y": np.expand_dims(example["y"], 0),
                    "uv": np.expand_dims(example["uv"], 0),
                }

                intermediates = intermediate_engine.infer(feed, copy_outputs_to_host=True)
                rewards = reward_engine.infer(feed, copy_outputs_to_host=True)
                
                return {
                    "intermediate": intermediates["intermediate"],
                    "reward": rewards["reward"],
                }
        

            ds = ds.as_dataset()
            print(ds)

            ds = ds.with_format("numpy").map(mapfn, writer_batch_size=32, remove_columns=["y", "uv"], cache_file_names={"train": "train-mapped.arrow", "validation": "validation-mapped.arrow"})
            print(ds)
        



if __name__ == '__main__':
    unittest.main()
