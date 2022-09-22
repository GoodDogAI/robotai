import torch
import onnx
import tempfile
import numpy as np
import png
import onnx_graphsurgeon

from typing import Callable, Tuple, BinaryIO

from einops import rearrange
from skimage.color import ycbcr2rgb, rgb2ycbcr

# Used because TensorRT only supports INT8, not UINT8 inputs, so we need to do a
# "reinterpret_cast" to convert the UINT8 to INT8
def int8_from_uint8(x: int) -> int:
    # Return two's complement representation of x
    return x - 256 if x > 127 else x

def png_to_nv12m(png_file: BinaryIO) -> Tuple[np.ndarray, np.ndarray]:
    img = png.Reader(png_file)
    width, height, rowdata, info = img.asRGB8()

    # Convert to a numpy array
    img_data = np.vstack([np.uint8(row) for row in rowdata])
    img_data = np.reshape(img_data, (height, width, 3))

    yuv = rgb2ycbcr(img_data).astype(np.float32)
    y = yuv[:, :, 0]
    uv = yuv[0::2, 0::2, 1:]
    y = rearrange(y, "h w -> 1 1 h w")
    uv = rearrange(uv, "h w (c1 c2) -> 1 1 h (w c1 c2)", c1=1)

    return y, uv

def expand_uv(uv: torch.Tensor) -> torch.Tensor:
    len = uv.numel()
    (batch, chan, height, width) = uv.shape
    uv = uv.reshape([len,1]).expand([len,2])
    uv = uv.reshape([batch*chan*height, 1, width*2]).expand([-1, 2, -1]).reshape([batch, chan, height*2, width*2])
    return uv

# Allows you to modify an ONNX model to use YUV input instead of RGB
def nv12m_to_rgb(y: torch.Tensor, uv: torch.Tensor) -> torch.Tensor:
    # y.shape = [batch, 1, height, width]
    # uv.shape = [batch, 1, height/2, width]
    assert len(y.shape) == 4
    assert len(uv.shape) == 4

    assert y.dtype == torch.float32, "Expecting float y in range of [16, 235]"
    assert uv.dtype == torch.float32, "Expecting float uv in range of [16, 240]"

    y -= 16.0
    uv -= 128.0
    
    u = expand_uv(uv[:, :, :, 0::2])
    v = expand_uv(uv[:, :, :, 1::2])

    mat = torch.tensor([[1.164, 0.0, 1.596],
                        [1.164, -0.392, -0.813],
                        [1.164, 2.017, 0.0]], dtype=torch.float32, device=y.device)

    yuv = torch.cat([y, u, v], dim=1)

    rgb = torch.conv2d(yuv, mat.reshape([3, 3, 1, 1]))
    
    rgb = (rgb/ 255.0).clamp(0.0, 1.0)
    
    return rgb


class NV12MToRGB(torch.nn.Module):
    def forward(self, y: torch.Tensor, uv: torch.Tensor) -> torch.Tensor:
        return nv12m_to_rgb(y, uv)


class CenterCrop(torch.nn.Module):
    def __init__(self, size):
        super().__init__()
        self.size = size

    def forward(self, image):
        (batch, chan, height, width) = image.shape
        (crop_height, crop_width) = self.size
        assert crop_height <= height
        assert crop_width <= width
        y = (height - crop_height) // 2
        x = (width - crop_width) // 2
        return image[:, :, y:y+crop_height, x:x+crop_width]


class ConvertCropVision(torch.nn.Module):
    def __init__(self, converter, cropper, vision_model):
        super().__init__()
        self.converter = converter
        self.cropper = cropper
        self.vision_model = vision_model

    def forward(self, y=None, uv=None, rgb=None):
        img = self.converter(y, uv) 
        img = self.cropper(img)
        return self.vision_model(img)


def get_onnx(func: Callable, args: Tuple) -> onnx.ModelProto:
    module = NV12MToRGB()

    with tempfile.NamedTemporaryFile() as f:
        torch.onnx.export(module, args, f)
        onnx_model = onnx.load(f)
        graph = onnx_graphsurgeon.import_onnx(onnx_model)
        graph.toposort()
        graph.fold_constants()
        graph.cleanup()
        onnx.save(onnx_graphsurgeon.export_onnx(graph), f)

        return graph