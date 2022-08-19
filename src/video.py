from ast import Bytes
import os
import numpy as np

from typing import List
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
V4L2_BUF_FLAG_KEYFRAME = 0x8

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
    video_packets = []

    with open(logpath, "rb") as f:
        events = log.Event.read_multiple(f)

        # Get the actual events, starting with a keyframe, which we will need
        for i, evt in enumerate(events):
            if evt.which() == "headEncodeData":
                if evt.headEncodeData.idx.flags & V4L2_BUF_FLAG_KEYFRAME:
                    video_packets.clear()

                video_packets.append(evt.headEncodeData.data)

            if i == index:
                break
            
    return decode_last_frame(video_packets)
   

# Returns the last frame from a list of video packets as an image
def decode_last_frame(packets: List[bytes], pixel_format: nvc.PixelFormat=nvc.PixelFormat.RGB, width: int=DECODE_WIDTH, height: int=DECODE_HEIGHT) -> np.ndarray:
    nv_dec = nvc.PyNvDecoder(
        width,
        height,
        nvc.PixelFormat.NV12, # All actual decodes must be NV12 format
        nvc.CudaVideoCodec.HEVC,
        WEB_VIDEO_DECODE_GPU_ID,
    )
    packets_sent = 0
    packets_recv = 0

    assert len(packets) > 0, "Need to have some packets"
    assert pixel_format == nvc.PixelFormat.RGB, "Other formats Not implemented yet"

    # Workaround a bug in the nvidia library, where sending a single packet will never get decoded
    # That can only happen if we have a single iframe, so just send it twice
    if len(packets) == 1:
        packets.append(packets[0])

    for packet in packets:
        packet = np.frombuffer(packet, dtype=np.uint8)
        surface = nv_dec.DecodeSurfaceFromPacket(packet)
        packets_sent += 1

        if not surface.Empty():
            if packets_recv == len(packets) - 1:
                return _rgb_from_surface(surface)   

            packets_recv += 1

    
    while True:
        surface = nv_dec.FlushSingleSurface()

        if surface.Empty():
            break
        else:
            if packets_recv == len(packets) - 1:
                return _rgb_from_surface(surface)

            packets_recv += 1

    raise ValueError("Unable to decode video stream to desired packet")

# Takes in frames as (HEIGHT, RGBRGBRGB...) shape ndarrays and returns HEVC output packets
def create_video(frames: List[np.ndarray]) -> List[Bytes]:
    width: int=DECODE_WIDTH
    height: int=DECODE_HEIGHT

    result = []

    assert len(frames) > 0, "Need to send at least one frame"
    assert frames[0].shape[0] == height, "First dimension must be the height"
    assert frames[0].shape[1] == width * 3, "Second dimension must be width * 3 (RGBRGBRGB...)"

    nv_enc = nvc.PyNvEncoder({'preset': 'P5', 'tuning_info': 'high_quality', 'codec': 'hevc',
                                'fps': '15', 'rc': 'vbr', 'gop': '15', 'bf': '0',
                                 'profile': 'high', 's': f"{width}x{height}", 'bitrate': '10M', 'maxbitrate': '20M'}, format=nvc.PixelFormat.NV12, gpu_id=WEB_VIDEO_DECODE_GPU_ID)

    nv_cc = nvc.ColorspaceConversionContext(color_space=nvc.ColorSpace.BT_601,
                                                    color_range=nvc.ColorRange.MPEG)

    nv_ul = nvc.PyFrameUploader(width, height, nvc.PixelFormat.RGB, WEB_VIDEO_DECODE_GPU_ID)
    nv_rgb2yuv = nvc.PySurfaceConverter(width, height, nvc.PixelFormat.RGB, nvc.PixelFormat.YUV420, WEB_VIDEO_DECODE_GPU_ID)
    nv_yuv2n12 = nvc.PySurfaceConverter(width, height, nvc.PixelFormat.YUV420, nvc.PixelFormat.NV12, WEB_VIDEO_DECODE_GPU_ID)

    packet = np.ndarray(shape=(0), dtype=np.uint8)

    for frame in frames:
        surface = nv_ul.UploadSingleFrame(frame)
        surface = nv_rgb2yuv.Execute(surface, nv_cc)
        surface = nv_yuv2n12.Execute(surface, nv_cc)

        success = nv_enc.EncodeSingleSurface(surface, packet, sync=True)
        assert success

        result.append(packet.tobytes())

    return result

