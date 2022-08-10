import unittest
import os
from webbrowser import get
import numpy as np

from src.tests.utils import get_test_image
from src.web.video import load_image
from src.web.config_web import RECORD_DIR
import src.PyNvCodec as nvc
from src.web.config_web import WEB_VIDEO_DECODE_GPU_ID
from cereal import log

class VPFTest(unittest.TestCase):
    def test_load(self):
        test_path = os.path.join(RECORD_DIR, "unittest", "alphalog-2022-7-28-16_54.log")
        
        img = load_image(test_path, 0)
        self.assertEqual(img.shape, (720, 1280 * 3))

        img = load_image(test_path, 20)
        self.assertEqual(img.shape, (720, 1280 * 3))

    def test_decode_single_frame(self):
        test_path = os.path.join(RECORD_DIR, "unittest", "alphalog-2022-7-28-16_54.log")
        
        nv_dec = nvc.PyNvDecoder(
            1280,
            720,
            nvc.PixelFormat.NV12,
            nvc.CudaVideoCodec.HEVC,
            WEB_VIDEO_DECODE_GPU_ID,
        )

        with open(test_path, "rb") as f:
            events = log.Event.read_multiple(f)
            
            first = next(events)
            packet = first.headEncodeData.data

            packet = np.frombuffer(packet, dtype=np.uint8)

            # Important note, the library has a bug, where sending a single packet will
            # never cause the video frame to be decoded
            # The solution is to send a single IDR packet twice if that's the case
            surface = nv_dec.DecodeSurfaceFromPacket(packet)
            self.assertTrue(surface.Empty())

            surface = nv_dec.DecodeSurfaceFromPacket(packet)
            self.assertTrue(surface.Empty())
            
            surface = nv_dec.FlushSingleSurface()
            self.assertFalse(surface.Empty())

            surface = nv_dec.FlushSingleSurface()
            self.assertFalse(surface.Empty())
            
            surface = nv_dec.FlushSingleSurface()
            self.assertTrue(surface.Empty())

    def test_encode_decode(self):
        width, height = 1280, 720
        nv_enc = nvc.PyNvEncoder({'preset': 'P5', 'tuning_info': 'high_quality', 'codec': 'hevc',
                                 'profile': 'high', 's': f"{width}x{height}", 'bitrate': '10M'}, format=nvc.PixelFormat.NV12, gpu_id=WEB_VIDEO_DECODE_GPU_ID)

        nv_cc = nvc.ColorspaceConversionContext(color_space=nvc.ColorSpace.BT_601,
                                                      color_range=nvc.ColorRange.MPEG)

        nv_ul = nvc.PyFrameUploader(width, height, nvc.PixelFormat.RGB, WEB_VIDEO_DECODE_GPU_ID)
        nv_dl = nvc.PySurfaceDownloader(width, height, nvc.PixelFormat.RGB, WEB_VIDEO_DECODE_GPU_ID)
        nv_rgb2yuv = nvc.PySurfaceConverter(width, height, nvc.PixelFormat.RGB, nvc.PixelFormat.YUV420, WEB_VIDEO_DECODE_GPU_ID)
        nv_yuv2n12 = nvc.PySurfaceConverter(width, height, nvc.PixelFormat.YUV420, nvc.PixelFormat.NV12, WEB_VIDEO_DECODE_GPU_ID)
        nv_n122yuv = nvc.PySurfaceConverter(width, height, nvc.PixelFormat.NV12, nvc.PixelFormat.YUV420, WEB_VIDEO_DECODE_GPU_ID)
        nv_yuv2rgb = nvc.PySurfaceConverter(width, height, nvc.PixelFormat.YUV420, nvc.PixelFormat.RGB, WEB_VIDEO_DECODE_GPU_ID)

        red_img = get_test_image((255, 0, 0), 1280, 720)
        green_img = get_test_image((0, 255, 0), 1280, 720)
        blue_img = get_test_image((0, 0, 255), 1280, 720)

        packet = np.ndarray(shape=(0), dtype=np.uint8)

        surface = nv_ul.UploadSingleFrame(red_img)
        print(surface)

        surface = nv_rgb2yuv.Execute(surface, nv_cc)
        print(surface)

        surface = nv_yuv2n12.Execute(surface, nv_cc)
        print(surface)

        success = nv_enc.EncodeSingleSurface(surface, packet, sync=True)
        self.assertTrue(success)

        print(packet.shape)

        nv_dec = nvc.PyNvDecoder(
            1280,
            720,
            nvc.PixelFormat.NV12,
            nvc.CudaVideoCodec.HEVC,
            WEB_VIDEO_DECODE_GPU_ID,
        )

        result = nv_dec.DecodeSurfaceFromPacket(packet)
        result = nv_dec.DecodeSurfaceFromPacket(packet)
        print(result)
        result = nv_dec.FlushSingleSurface()
        print(result)
        result = nv_dec.FlushSingleSurface()
        print(result)

        result = nv_n122yuv.Execute(result, nv_cc)
        result = nv_yuv2rgb.Execute(result, nv_cc)
        frame_rgb = np.ndarray(shape=(0), dtype=np.uint8)
        nv_dl.DownloadSingleSurface(result, frame_rgb)

        frame_rgb = frame_rgb.reshape((height, -1))

        np.testing.assert_array_almost_equal(frame_rgb / 255.0, red_img / 255.0, decimal=1)



           


