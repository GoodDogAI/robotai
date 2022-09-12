import unittest
from einops import rearrange
from skimage.color import ycbcr2rgb, rgb2ycbcr

import numpy as np
import torch
from skimage import data

from src.train.onnx_yuv import nv12m_to_rgb, get_onnx

img = data.astronaut()


# TODO Test that it exports on the device
# TODO, I just found out that TensorRT only supports INT8 and not UINT8, so we need to just cast it
# TODO Fix the rest of the unit tests

class ONNXYUVTest(unittest.TestCase):
    def test_convert(self):
        # Start with random RGB image
        rgb_input = torch.randint(low=0, high=255, size=(720 // 2, 1280 // 2, 3), dtype=torch.uint8)
        
        # Upsample it, into 2x2 squares, this is important, because it makes sure that the
        # image is consistent in the NV12M format which has less color data than a true random image
        # Otherwise, you'd get out-of-range colors with the artifically generated data here
        rgb_input = rgb_input.repeat_interleave(2, dim=0).repeat_interleave(2, dim=1)

        # Convert to YCbCr using a known reference implementation
        yuv_input = torch.tensor(rgb2ycbcr(rgb_input.numpy()))

        # Take the YUV image and split it into NV12M format
        y = yuv_input[:, :, 0]
        uv = yuv_input[0::2, 0::2, 1:]

        y = rearrange(y, "h w -> 1 1 h w")
        uv = rearrange(uv, "h w (c1 c2) -> 1 1 h (w c1 c2)", c1=1)
    
        # Run the conversion using our optimized method
        rgb = nv12m_to_rgb(y, uv)
        onnx = get_onnx(nv12m_to_rgb, (y, uv))

        # Compare the results, for this, we actually have to take the NV12M UV plane, and reupsample it
        u = uv[:, :, :, 0::2].repeat_interleave(2, dim=3).repeat_interleave(2, dim=2)
        v = uv[:, :, :, 1::2].repeat_interleave(2, dim=3).repeat_interleave(2, dim=2)
        yuv = torch.cat([y, u, v], dim=1).float() 
        yuv = rearrange(yuv, "b c h w -> b h w c")[0]
        back_rgb = (torch.tensor(ycbcr2rgb(yuv.numpy())) * 255).float()

        rgb = rearrange(rgb, "b c h w -> b h w c")[0] * 255

        diffs = torch.abs(rgb - back_rgb)
        assert torch.allclose(back_rgb, rgb, atol=1)
        print("Done")





