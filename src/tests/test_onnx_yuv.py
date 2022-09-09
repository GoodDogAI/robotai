from gzip import _GzipReader
import unittest
import os
import torch

from src.train.onnx_yuv import nv12m_to_rgb, get_onnx

class ONNXYUVTest(unittest.TestCase):
    def test_convert(self):
        y = torch.randint(low=0, high=255, size=(1, 1, 720, 1280), dtype=torch.uint8)
        uv = torch.randint(low=0, high=255, size=(1, 1, 720 // 2, 1280), dtype=torch.uint8)

        rgb = nv12m_to_rgb(y, uv)
        print(rgb.shape, rgb.dtype)

        onnx = get_onnx(nv12m_to_rgb, (y, uv))

        print(onnx)




