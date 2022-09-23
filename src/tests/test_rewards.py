import os
import torch
import tensorrt as trt
import unittest
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from src.config import HOST_CONFIG, YOLOV7_CLASS_NAMES
from src.train.modelloader import model_fullname, load_vision_model
from src.train.onnx_yuv import png_to_nv12m
from src.train.reward import SumCenteredObjectsPresentReward


def draw_bboxes_pil(png_path, bboxes, class_names) -> Image:
    img = Image.open(png_path)
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype("DejaVuSans.ttf", 20)
    color = "red"

    for row in bboxes:
        cxcywh = row[0:4]
        x1, y1 = cxcywh[0] - cxcywh[2] / 2, cxcywh[1] - cxcywh[3] / 2
        x2, y2 = cxcywh[0] + cxcywh[2] / 2, cxcywh[1] + cxcywh[3] / 2

        if row[4] > 0.15:
            label = class_names[np.argmax(row[5:])]
            txt_width, txt_height = font.getsize(label)
            draw.rectangle([x1, y1, x2, y2], width=1, outline=color)  # plot
            print("Found bbox:", cxcywh, label)

            draw.rectangle([x1, y1 - txt_height + 4, x1 + txt_width, y1], fill=color)
            draw.text((x1, y1 - txt_height + 1), label, fill=(255, 255, 255), font=font)
    return img


class TestRewards(unittest.TestCase):
    def setUp(self) -> None:
        self.sampleRewardConfig = {
            "type": "reward",
            "load_fn": "src.train.yolov7.load.load_yolov7",
            "input_format": "rgb",
            "checkpoint": "/home/jake/robotai/_checkpoints/yolov7.pt",
            "class_names": YOLOV7_CLASS_NAMES,

            # Input dimensions must be divisible by the stride
            # In current situations, the image will be cropped to the nearest multiple of the stride
            "dimension_stride": 32,
            
            "class_weights": {
                "person": 3,
                "spoon": 10,
            },
            "global_reward_scale": 0.10,

            "max_detections": 100,
            "detection_threshold": 0.50,
            "iou_threshold": 0.50,
        }

        # If the model is not cached, create and validate it
        self.model_fullname = model_fullname(self.sampleRewardConfig)

    def test_reward_inference(self):
        png_path = os.path.join(HOST_CONFIG.RECORD_DIR, "unittest", "horses.png")
        with open(png_path, "rb") as f1:
            y, uv = png_to_nv12m(f1)

        with load_vision_model(self.model_fullname) as engine:
             trt_outputs = engine.infer({"y": y, "uv": uv})

        img = draw_bboxes_pil(png_path, trt_outputs["bboxes"], self.sampleRewardConfig["class_names"])
        
        # SAVE PIL Image to file
        img.save(os.path.join(HOST_CONFIG.RECORD_DIR, "unittest", "horses_with_bboxes.png"))

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
         