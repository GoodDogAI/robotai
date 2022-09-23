import os
import torch
import tensorrt as trt
import unittest
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from src.config import HOST_CONFIG, BRAIN_CONFIGS
from src.train.modelloader import model_fullname, load_vision_model
from src.train.onnx_yuv import png_to_nv12m
from src.train.reward import yolov7_class_names

def draw_bboxes_pil(png_path, bboxes, class_names) -> Image:
    img = Image.open(png_path)
    draw = ImageDraw.Draw(img)
    fontsize = max(round(max(img.size) / 40), 12)
    font = ImageFont.truetype("DejaVuSans.ttf", fontsize)
    color = "red"

    for row in bboxes:
        cxcywh = row[0:4]

        if row[4] > 0.15:
            label = class_names[np.argmax(row[5:])]
            txt_width, txt_height = font.getsize(label)
            draw.rectangle([cxcywh[0] - cxcywh[2] / 2, cxcywh[1] - cxcywh[3] / 2, cxcywh[0] + cxcywh[2] / 2, cxcywh[1] + cxcywh[3] / 2], width=1, outline=color)  # plot
            print("Found bbox:", cxcywh, label)

            # draw.rectangle([box[0], box[1] - txt_height + 4, box[0] + txt_width, box[1]], fill=color)
            # draw.text((box[0], box[1] - txt_height + 1), label, fill=(255, 255, 255), font=font)
    return img


class TestRewards(unittest.TestCase):
    def setUp(self) -> None:
        self.sampleRewardConfig = {
            "type": "reward",
            "load_fn": "src.train.yolov7.load.load_yolov7",
            "input_format": "rgb",
            "checkpoint": "/home/jake/robotai/_checkpoints/yolov7.pt",

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

        img = draw_bboxes_pil(png_path, trt_outputs["bboxes"], yolov7_class_names)
        
        # SAVE PIL Image to file
        img.save(os.path.join(HOST_CONFIG.RECORD_DIR, "unittest", "horses_with_bboxes.png"))