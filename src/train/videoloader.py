import torch
import numpy as np
import src.PyNvCodec as nvc
import PytorchNvCodec as pnvc

from typing import Iterator, List

from cereal import log
from src.video import V4L2_BUF_FLAG_KEYFRAME

import torchdata.datapipes as dp
from .config_train import RECORD_DIR
from src.include.config import load_realtime_config

CONFIG = load_realtime_config()
DECODE_WIDTH = int(CONFIG["CAMERA_WIDTH"])
DECODE_HEIGHT = int(CONFIG["CAMERA_HEIGHT"])

def surface_to_tensor(surface: nvc.Surface) -> torch.Tensor:
    """
    Converts an NV12 surface to cuda float Y,UV tensor.
    """
    if surface.Format() != nvc.PixelFormat.NV12:
        raise RuntimeError('Surface shall be of NV12 pixel format')

    surf_plane = surface.PlanePtr()
    height = surf_plane.Height()
    img_tensor = pnvc.DptrToTensor(surf_plane.GpuMem(),
                                   surf_plane.Width(),
                                   height,
                                   surf_plane.Pitch(),
                                   surf_plane.ElemSize())
    if img_tensor is None:
        raise RuntimeError('Can not export to tensor.')


    y_tensor = img_tensor[:height]
    uv_tensor = img_tensor[height:]
    uv_tensor = uv_tensor.repeat_interleave(2, dim=0)

    merged_tensor = torch.stack((y_tensor, uv_tensor)) 
    
    merged_tensor = merged_tensor.type(dtype=torch.cuda.FloatTensor)
    merged_tensor = torch.divide(merged_tensor, 255.0)
    merged_tensor = torch.clamp(merged_tensor, 0.0, 1.0)

    return merged_tensor


@dp.functional_datapipe("decode_log_frames")
class LogImageFrameDecoder(dp.iter.IterDataPipe):
    def __init__(self, source_datapipe, **kwargs) -> None:
        self.source_datapipe = source_datapipe
        self.kwargs = kwargs

        self.nv_dec = nvc.PyNvDecoder(
            DECODE_WIDTH,
            DECODE_HEIGHT,
            nvc.PixelFormat.NV12, # All actual decodes must be NV12 format
            nvc.CudaVideoCodec.HEVC,
            0, # TODO Set the GPU ID dynamically or something
        )
                                           

    def __iter__(self) -> Iterator[torch.Tensor]:
         for path, file in self.source_datapipe:
            # Read the events from the log file
            events = log.Event.read_multiple(file)
            first = True

            # Get the actual events, starting with a keyframe, which we will need
            for evt in events:
                if evt.which() == "headEncodeData":
                    if first:
                        assert(evt.headEncodeData.idx.flags & V4L2_BUF_FLAG_KEYFRAME)
                        first = False

                    packet = np.frombuffer(evt.headEncodeData.data, dtype=np.uint8)
                    surface = self.nv_dec.DecodeSurfaceFromPacket(packet)

                    if not surface.Empty():
                        yield surface_to_tensor(surface)

            while True:
                surface = self.nv_dec.FlushSingleSurface()

                if surface.Empty():
                    break
                else:
                    yield surface_to_tensor(surface)

def build_datapipe(root_dir=RECORD_DIR):
    datapipe = dp.iter.FileLister(root_dir)
    datapipe = datapipe.filter(filter_fn=lambda filename: filename.endswith('.log'))
    datapipe = datapipe.open_files(mode='rb')
    datapipe = datapipe.decode_log_frames()

    return datapipe