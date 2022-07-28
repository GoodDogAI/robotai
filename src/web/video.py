import os
import numpy as np
import src.PyNvCodec as nvc
from src.include.config import load_realtime_config
from src.web.config_web import WEB_VIDEO_DECODE_GPU_ID

from cereal import log

NALU_TYPES = {
    0: 	"Unspecified 	non-VCL",
    1: 	"Coded slice of a non-IDR picture 	VCL",
    2: 	"Coded slice data partition A 	VCL",
    3: 	"Coded slice data partition B 	VCL",
    4: 	"Coded slice data partition C 	VCL",
    5: 	"Coded slice of an IDR picture 	VCL",
    6: 	"Supplemental enhancement information (SEI) 	non-VCL",
    7: 	"Sequence parameter set 	non-VCL",
    8: 	"Picture parameter set 	non-VCL",
    9: 	"Access unit delimiter 	non-VCL",
    10: 	"End of sequence 	non-VCL",
    11: 	"End of stream 	non-VCL",
    12: 	"Filler data 	non-VCL",
    13: 	"Sequence parameter set extension 	non-VCL",
    14: 	"Prefix NAL unit 	non-VCL",
    15: 	"Subset sequence parameter set 	non-VCL",
    16: 	"Reserved 	non-VCL",
    19: 	"Coded slice of an auxiliary coded picture without partitioning 	non-VCL",
    20: 	"Coded slice extension 	non-VCL",
    21: 	"Coded slice extension for depth view components 	non-VCL",
    22: 	"Reserved 	non-VCL",
    24: 	"Unspecified",
}
CONFIG = load_realtime_config()

DECODE_WIDTH = int(CONFIG["CAMERA_WIDTH"])
DECODE_HEIGHT = int(CONFIG["CAMERA_HEIGHT"])


def _rgb_from_surface(surface: "nvc.PySurface") -> np.ndarray:
    frame_rgb = np.ndarray(shape=(0), dtype=np.uint8)
    nv_cc = nvc.ColorspaceConversionContext(nvc.ColorSpace.BT_601, nvc.ColorRange.MPEG)

    nv_yuv = nvc.PySurfaceConverter(DECODE_WIDTH, DECODE_HEIGHT, nvc.PixelFormat.NV12, nvc.PixelFormat.YUV420, WEB_VIDEO_DECODE_GPU_ID)
    nv_rgb = nvc.PySurfaceConverter(DECODE_WIDTH, DECODE_HEIGHT, nvc.PixelFormat.YUV420, nvc.PixelFormat.RGB, WEB_VIDEO_DECODE_GPU_ID)

    nv_dl = nvc.PySurfaceDownloader(DECODE_WIDTH, DECODE_HEIGHT, nvc.PixelFormat.RGB, WEB_VIDEO_DECODE_GPU_ID)
    surface = nv_yuv.Execute(surface, nv_cc)
    surface = nv_rgb.Execute(surface, nv_cc)
    nv_dl.DownloadSingleSurface(surface, frame_rgb)

    return frame_rgb.reshape((DECODE_HEIGHT, -1))


def load_image(logpath: str, index: int) -> np.ndarray:
    nv_dec = nvc.PyNvDecoder(
        DECODE_WIDTH,
        DECODE_HEIGHT,
        nvc.PixelFormat.NV12,
        nvc.CudaVideoCodec.HEVC,
        WEB_VIDEO_DECODE_GPU_ID,
    )

  
    events_sent = 0
    events_recv = 0

    with open(logpath, "rb") as f:
        events = log.Event.read_multiple(f)

        for evt in events:
            packet = evt.headEncodeData.data
            assert packet[0] == 0 and packet[1] == 0 and packet[2] == 0 and packet[3] == 1
            nalu_type = (packet[4] & 0x1F)

            packet = np.frombuffer(packet, dtype=np.uint8)
            surface = nv_dec.DecodeSurfaceFromPacket(packet)
            events_sent += 1

            if not surface.Empty():
                if events_recv == index:
                    return _rgb_from_surface(surface)

                events_recv += 1

        while True:
            surface = nv_dec.FlushSingleSurface()

            if surface.Empty():
                break
            else:
                if events_recv == index:
                    _rgb_from_surface(surface)

                events_recv += 1

        raise ValueError("Unable to decode video stream to desired packet")


