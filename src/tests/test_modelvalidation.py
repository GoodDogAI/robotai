

import os
import tensorrt as trt
import numpy as np
import unittest
import png

from einops import rearrange
from skimage.color import ycbcr2rgb, rgb2ycbcr

from src.config import HOST_CONFIG, BRAIN_CONFIGS
from src.train.modelloader import load_vision_model
from src.train.log_validation import full_validate_log

class TestModelValidation(unittest.TestCase):
    def test_typical_model_deltas(self):
        # Load an image, and run it through the model
        img = png.Reader(filename = os.path.join(HOST_CONFIG.RECORD_DIR, "unittest", "215.png"))
        width, height, rowdata, info = img.asRGB8()

        # Convert to a numpy array
        img_data = np.vstack([np.uint8(row) for row in rowdata])
        img_data = np.reshape(img_data, (height, width, 3))

        yuv = rgb2ycbcr(img_data).astype(np.float32)
        y = yuv[:, :, 0]
        uv = yuv[0::2, 0::2, 1:]
        y = rearrange(y, "h w -> 1 1 h w")
        uv = rearrange(uv, "h w (c1 c2) -> 1 1 h (w c1 c2)", c1=1)

        with load_vision_model("yolov7-tiny-s53") as engine:
            trt_outputs = engine.infer({"y": np.copy(y), "uv": np.copy(uv)})
            trt_intermediate_orig = np.copy(trt_outputs["intermediate"])

            # Now, perturb the y input slightly
            offsets = np.random.uniform(0, 1, y.shape)
            y += offsets

            trt_outputs = engine.infer({"y": np.copy(y), "uv": np.copy(uv)})
            trt_intermediate_new = np.copy(trt_outputs["intermediate"])

            # Now, compute the difference between the two
            trt_intermediate_diff = trt_intermediate_new - trt_intermediate_orig
            num_diffs = (trt_intermediate_diff > 0.01).sum()
            print(trt_intermediate_diff)




    def test_vision_intermediate_video(self):
        #test_path = os.path.join(HOST_CONFIG.RECORD_DIR, "unittest", "alphalog-5205d621-2022-9-14-21_57.log")
        test_path = os.path.join(HOST_CONFIG.RECORD_DIR, "alphalog-6eb8d100-2022-9-16-16_21.log")
        #test_path = os.path.join(HOST_CONFIG.RECORD_DIR, "alphalog-6705c9d2-2022-9-16-16_47.log")
        
        with open(test_path, "rb") as f:
            full_validate_log(f)
