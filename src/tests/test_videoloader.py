import unittest
import time
import torch
import numpy as np
from torch.utils.data import DataLoader

from polygraphy.cuda import DeviceView

from src.config import YOLOV7_CLASS_NAMES
from src.train.modelloader import load_vision_model, model_fullname
from src.train.videoloader import build_datapipe

class VideoLoaderTest(unittest.TestCase):
    max_elements = 1000

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

    def test_basic(self):
        start = time.perf_counter()
        count = 0

        for x in build_datapipe().header(self.max_elements):
            count += 1

        print(f"Took {time.perf_counter() - start:0.2f} seconds")
        print(f"Loaded {count} frames")
        print(f"{count / (time.perf_counter() - start):0.1f} fps")
  
    def test_train(self):
        datapipe = build_datapipe().header(self.max_elements)
        dl = DataLoader(dataset=datapipe, batch_size=1)

        with load_vision_model(model_fullname(self.sampleVisionConfig)) as intermediate_engine, \
             load_vision_model(model_fullname(self.sampleRewardConfig)) as reward_engine:

            start = time.perf_counter()
            count = 0

            for y, uv in dl:
                # Use DeviceView to keep tensors on the GPU
                intermediates = intermediate_engine.infer({"y": DeviceView(y.data_ptr(), y.shape, np.float32),
                                                           "uv": DeviceView(uv.data_ptr(), uv.shape, np.float32)}, copy_outputs_to_host=False)

                rewards = reward_engine.infer({"y": DeviceView(y.data_ptr(), y.shape, np.float32),
                                               "uv": DeviceView(uv.data_ptr(), uv.shape, np.float32)}, copy_outputs_to_host=False)

                count += 1

        print(f"Took {time.perf_counter() - start:0.2f} seconds")
        print(f"Loaded {count} frames")
        print(f"{count / (time.perf_counter() - start):0.1f} fps")







if __name__ == '__main__':
    unittest.main()
