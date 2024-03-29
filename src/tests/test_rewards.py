import os
import torch
import tensorrt as trt
import unittest
import importlib
import numpy as np

from PIL import Image


from src.config import HOST_CONFIG, YOLOV7_CLASS_NAMES
from src.train.modelloader import model_fullname, load_vision_model, create_pt_model
from src.train.onnx_yuv import png_to_nv12m
from src.train.reward import SumCenteredObjectsPresentReward
from src.utils.draw_bboxes import draw_bboxes_pil


class TestRewards(unittest.TestCase):
    def setUp(self) -> None:
        self.sampleRewardConfig = {
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

        # If the model is not cached, create and validate it
        self.model_fullname = model_fullname(self.sampleRewardConfig)

    def test_reward_inference(self):
        png_path = os.path.join(HOST_CONFIG.RECORD_DIR, "unittest", "horses.png")
        with open(png_path, "rb") as f1:
            y, uv = png_to_nv12m(f1)

        with load_vision_model(self.model_fullname) as engine:
             trt_outputs = engine.infer({"y": y, "uv": uv})

        img = Image.open(png_path)
        img = draw_bboxes_pil(img, trt_outputs["bboxes"], self.sampleRewardConfig)
        
        # SAVE PIL Image to file
        img.save(os.path.join(HOST_CONFIG.RECORD_DIR, "unittest", "horses_with_bboxes.png"))

    def test_reward_pt_to_trt_real_image(self):
        # The modelloader tests things with random fake images, so it's reward values are always around 0, and the reward is not fully tested
        # So, you always want to have a separate test that uses a real image and checks that the reward values match up
        png_path = os.path.join(HOST_CONFIG.RECORD_DIR, "unittest", "horses.png")
        with open(png_path, "rb") as f1:
            y, uv = png_to_nv12m(f1)

        pt_model = create_pt_model(self.sampleRewardConfig)
        device = "cuda"

        y_pt = torch.from_numpy(y).to(device)
        uv_pt = torch.from_numpy(uv).to(device)

        reward, bboxes, raw_detections = pt_model(y=y_pt, uv=uv_pt)
        reward = reward.cpu().detach().numpy()

        # Check this against the TRT
        with load_vision_model(self.model_fullname) as engine:
             trt_outputs = engine.infer({"y": y, "uv": uv})

             self.assertAlmostEqual(reward, trt_outputs["reward"], places=3)
        
    def test_sum_centered_objects(self):
        center_epsilon = 0.1

        fn = SumCenteredObjectsPresentReward(width=100, height=100, reward_scale=1.0, center_epsilon=center_epsilon)

        # Perfectly centered object should have reward of 1 / center_epsilon (the epsilon is to prevent division by zero)
        self.assertEqual(fn(torch.tensor([50, 50, 100, 100, 1.0, 1.0])), 10.0)

        # Twice as many objects should be twice as much reward
        self.assertEqual(fn(torch.tensor([ [50, 50, 100, 100, 1.0, 1.0],
                                           [50, 50, 100, 100, 1.0, 1.0]])), 20.0)


        # Distance on either axis should be the same reward
        self.assertAlmostEqual(fn(torch.tensor([0, 50, 100, 100, 1.0, 1.0])).item(), 1.667, places=2)
        self.assertAlmostEqual(fn(torch.tensor([50, 0, 100, 100, 1.0, 1.0])).item(), 1.667, places=2)

        # Width and height should be the same reward
        self.assertAlmostEqual(fn(torch.tensor([0, 50, 50, 100, 1.0, 1.0])).item(), 1.667, places=2)
        self.assertAlmostEqual(fn(torch.tensor([0, 50, 100, 50, 1.0, 1.0])).item(), 1.667, places=2)

        # Getting farther from the center
        self.assertAlmostEqual(fn(torch.tensor([0, 0, 100, 100, 1.0, 1.0])).item(), 1.239, places=2)

        # Two random objects, close one to center should have higher reward
        for i in range(100):
            t1 = torch.rand(6)
            t2 = torch.rand(6)
            t1[2:] = torch.tensor([100, 100, 1.0, 1.0])
            t2[2:] = torch.tensor([100, 100, 1.0, 1.0])

            d1 = torch.sqrt((t1[0] - 50) ** 2 + (t1[1] - 50) ** 2)
            d2 = torch.sqrt((t2[0] - 50) ** 2 + (t2[1] - 50) ** 2)

            if d1 < d2:
                self.assertGreater(fn(t1), fn(t2))
            else:
                self.assertLess(fn(t1), fn(t2))
         
        # Test that rewards are multiplied by the class probability
        self.assertEqual(fn(torch.tensor([50, 50, 100, 100, 1.0, 0.5])), 5.0)

        # Test that rewards are multiplied by the detection probability
        self.assertEqual(fn(torch.tensor([50, 50, 100, 100, 0.5, 0.5])), 2.5)

        # Test the global reward scale
        fn_scaled = SumCenteredObjectsPresentReward(width=100, height=100, reward_scale=10.0, center_epsilon=center_epsilon)
        self.assertEqual(fn_scaled(torch.tensor([50, 50, 100, 100, 1.0, 1.0])), 100.0)

    def test_sum_centered_class_weights(self):
        center_epsilon = 0.1

        fn = SumCenteredObjectsPresentReward(width=100, height=100, reward_scale=1.0, class_names=["apples", "oranges", "pears"], class_weights={"oranges": 5}, center_epsilon=center_epsilon)

        self.assertEqual(fn(torch.tensor([50, 50, 100, 100, 1.0, 1.0, 0.0, 0.0])), 10.0)
        self.assertEqual(fn(torch.tensor([50, 50, 100, 100, 1.0, 0.0, 1.0, 0.0])), 50.0)
        self.assertEqual(fn(torch.tensor([50, 50, 100, 100, 1.0, 0.0, 0.9, 1.0])), 10.0)

        self.assertEqual(fn(torch.tensor([50, 50, 100, 100, 0.1, 1.0, 0.0, 0.0])), 1.0)
        self.assertEqual(fn(torch.tensor([50, 50, 100, 100, 0.1, 0.0, 1.0, 0.0])), 5.0)
        self.assertEqual(fn(torch.tensor([50, 50, 100, 100, 0.1, 0.0, 0.9, 1.0])), 1.0)

        # Test with multiple classes
        self.assertEqual(fn(torch.tensor([[50, 50, 100, 100, 1.0, 1.0, 0.0, 0.0],
                                          [50, 50, 100, 100, 1.0, 1.0, 0.0, 0.0]])), 20.0)

        self.assertEqual(fn(torch.tensor([[50, 50, 100, 100, 1.0, 1.0, 0.0, 0.0],
                                          [50, 50, 100, 100, 1.0, 0.0, 1.0, 0.0],
                                          [50, 50, 100, 100, 1.0, 1.0, 0.0, 0.0]])), 70.0)

