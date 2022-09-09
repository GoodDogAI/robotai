import unittest
from einops import rearrange
from skimage.color import ycbcr2rgb, rgb2ycbcr

import numpy as np
import torch
from skimage import data

from src.train.onnx_yuv import nv12m_to_rgb, get_onnx

img = data.astronaut()

class ONNXYUVTest(unittest.TestCase):
    def test_convert(self):
        # Start with random RGB image
        rgb_input = torch.randint(low=0, high=255, size=(720, 1280, 3), dtype=torch.uint8)

        # Convert to YCbCr and back using a known reference implementation
        yuv_input = torch.tensor(rgb2ycbcr(rgb_input.numpy()))
        back_rgb = (torch.tensor(ycbcr2rgb(yuv_input.numpy())) * 255).float()

        assert torch.equal(rgb_input, torch.round(back_rgb).to(torch.uint8))

        # Take the YUV image and split it into NV12M format
        y = yuv_input[:, :, 0]
        uv = yuv_input[0::2, 0::2, 1:]

        y = rearrange(y, "h w -> 1 1 h w")
        uv = rearrange(uv, "h w (c1 c2) -> 1 1 h (w c1 c2)", c1=1)
    
        # Run the conversion using our optimized method
        rgb = nv12m_to_rgb(y, uv)
        onnx = get_onnx(nv12m_to_rgb, (y, uv))

        # Compare the results
        rgb = rearrange(rgb, "b c h w -> b h w c")[0] * 255

        
        assert torch.allclose(back_rgb, rgb, atol=1)
        print("Done")
        # u = uv[:, :, :, 0::2].repeat_interleave(2, dim=3).repeat_interleave(2, dim=2)
        # v = uv[:, :, :, 0::2].repeat_interleave(2, dim=3).repeat_interleave(2, dim=2)
        # yuv = torch.cat([y, u, v], dim=1).float()
        # yuv = np.transpose(yuv, (0, 2, 3, 1)).squeeze(0)

        # reference = ycbcr2rgb(yuv.numpy())

        # print(reference)




