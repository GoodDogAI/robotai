import os
import numpy as np
import src.PyNvCodec as nvc

from cereal import log


def load_image(logpath: str, index: int) -> np.ndarray:
    nv_dec = nvc.PyNvDecoder(
        1280,
        720,
        nvc.PixelFormat.NV12,
        nvc.CudaVideoCodec.HEVC,
        0,
    )

    frame_nv12 = np.ndarray(shape=(0), dtype=np.uint8)


    with open(logpath, "rb") as f:
        events = log.Event.read_multiple(f)

        evt = next(events)
        packet = evt.headEncodeData.data
        print(len(packet), "bytes")

        packet = np.frombuffer(packet, dtype=np.uint8)

        frame_ready = nv_dec.DecodeFrameFromPacket(
            frame_nv12, packet)

        print(f"frame ready: {frame_ready}")
