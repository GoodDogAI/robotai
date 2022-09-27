import torch
import numpy as np
import src.PyNvCodec as nvc
import src.PytorchNvCodec as pnvc

from typing import Iterator, List, Literal

from cereal import log
from src.video import V4L2_BUF_FLAG_KEYFRAME
from src.train.modelloader import load_vision_model
from polygraphy.cuda import DeviceView

import torchdata.datapipes as dp
from src.config import HOST_CONFIG, DEVICE_CONFIG


def surface_to_y_uv(surface: nvc.Surface) -> torch.Tensor:
    """
    Converts an NV12 surface to cuda float Y,UV tensor.
    """
    if surface.Format() != nvc.PixelFormat.NV12:
        raise RuntimeError('Surface shall be of NV12 pixel format')

    surf_plane = surface.PlanePtr()
    img_tensor = pnvc.DptrToTensor(surf_plane.GpuMem(),
                                   surf_plane.Width(),
                                   surf_plane.Height(),
                                   surf_plane.Pitch(),
                                   surf_plane.ElemSize())
    if img_tensor is None:
        raise RuntimeError('Can not export to tensor.')

    height = surf_plane.Height() * 2 // 3
    y_tensor = img_tensor[:height]
    uv_tensor = img_tensor[height:]
    
    # Convert to float in range [16, 235], [16, 240]
    y_tensor = torch.unsqueeze(y_tensor, 0).float()
    uv_tensor = torch.unsqueeze(uv_tensor, 0).float()

    return y_tensor, uv_tensor


@dp.functional_datapipe("decode_log_frames")
class LogImageFrameDecoder(dp.iter.IterDataPipe):
    def __init__(self, source_datapipe, **kwargs) -> None:
        self.source_datapipe = source_datapipe
        self.kwargs = kwargs

        self.nv_dec = nvc.PyNvDecoder(
            DEVICE_CONFIG.CAMERA_WIDTH,
            DEVICE_CONFIG.CAMERA_HEIGHT,
            nvc.PixelFormat.NV12, # All actual decodes must be NV12 format
            nvc.CudaVideoCodec.HEVC,
            HOST_CONFIG.DEFAULT_DECODE_GPU_ID, # TODO Set the GPU ID dynamically or something
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
                        assert evt.headEncodeData.idx.flags & V4L2_BUF_FLAG_KEYFRAME
                        first = False

                    packet = np.frombuffer(evt.headEncodeData.data, dtype=np.uint8)
                    surface = self.nv_dec.DecodeSurfaceFromPacket(packet)

                    if not surface.Empty():
                        yield surface_to_y_uv(surface)

            while True:
                surface = self.nv_dec.FlushSingleSurface()

                if surface.Empty():
                    break
                else:
                    yield surface_to_y_uv(surface)


@dp.functional_datapipe("calculate_intermediate_and_reward")
class IntermediateRewardCalculator(dp.iter.IterDataPipe):
    def __init__(self, source_datapipe, **kwargs) -> None:
        self.source_datapipe = source_datapipe
        self.kwargs = kwargs

        self.intermediate_engine = kwargs["intermediate_engine"]
        self.reward_engine = kwargs["reward_engine"]
                                           

    def __iter__(self) -> Iterator[torch.Tensor]:
         for y, uv in self.source_datapipe:
            intermediates = self.intermediate_engine.infer({"y": DeviceView(y.data_ptr(), y.shape, np.float32),
                                                            "uv": DeviceView(uv.data_ptr(), uv.shape, np.float32)}, copy_outputs_to_host=True)

            intermediate = torch.tensor(intermediates["intermediate"], dtype=torch.float32, device=y.device)                                                           

            rewards = self.reward_engine.infer({"y": DeviceView(y.data_ptr(), y.shape, np.float32),
                                            "uv": DeviceView(uv.data_ptr(), uv.shape, np.float32)}, copy_outputs_to_host=True)

            reward = torch.tensor(rewards["reward"], dtype=torch.float32, device=y.device)                   

            yield intermediate, reward


def build_datapipe(root_dir=HOST_CONFIG.RECORD_DIR, train_or_valid: Literal["train", "valid"]="train", split: float=0.90):
    datapipe = dp.iter.FileLister(root_dir)
    datapipe = datapipe.filter(filter_fn=lambda filename: filename.endswith('.log'))
    datapipe = datapipe.shuffle()

    train, valid = datapipe.demux(num_instances=2, classifier_fn=lambda x: 0 if torch.rand(1).item() < split else 1)

    if train_or_valid == "train":
        datapipe = train
    else:
        datapipe = valid

    datapipe = datapipe.sharding_filter()
    datapipe = datapipe.open_files(mode='rb')
    datapipe = datapipe.decode_log_frames()

    return datapipe