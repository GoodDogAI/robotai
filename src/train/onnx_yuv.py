import torch
import onnx
import tempfile
from typing import Callable, Tuple

def expand_uv(uv: torch.Tensor) -> torch.Tensor:
    len = uv.numel()
    (batch, chan, height, width) = uv.shape
    uv = uv.reshape([len,1]).expand([len,2]).reshape([batch, chan, height, width*2])
    uv = uv.reshape([batch*chan*height, 1, width*2]).expand([-1, 2, -1]).reshape([batch, chan, height*2, width*2])
    return uv

# Allows you to modify an ONNX model to use YUV input instead of RGB
def nv12m_to_rgb(y: torch.Tensor, uv: torch.Tensor) -> torch.Tensor:
    # y.shape = [batch, 1, height, width], dtype=int8
    # uv.shape = [batch, 1, height/2, width], dtype=int8

    # Convert to float
    y = y.float()
    uv = uv.float()

    y -= 16
    uv -= 128
    
    u = expand_uv(uv[:, :, :, 0::2])
    v = expand_uv(uv[:, :, :, 1::2])

    mat = torch.tensor([[1.164, 0.0, 1.596],[1.164, -0.392, -0.813],[1.164, 2.017, 0.0]], dtype=torch.float32)

    yuv = torch.cat([y, u, v], dim=1)
    rgb = torch.conv2d(yuv, mat.reshape([3, 3, 1, 1]))
    rgb = (rgb/ 255.0).clamp(0.0, 1.0)
    return rgb

def get_onnx(func: Callable, args: Tuple) -> onnx.ModelProto:
    jit = torch.jit.script(func)

    #with tempfile.NamedTemporaryFile() as f:
    f = "/home/jake/test.onnx"
    torch.onnx.export(jit, args, f)
    return onnx.load(f)