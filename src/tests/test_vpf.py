from struct import pack
import unittest
import os
from webbrowser import get
import numpy as np

from src.tests.utils import get_test_image
from src.web.video import load_image, create_video, decode_last_frame
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

        red_img = get_test_image((255, 0, 0), width, height)
        green_img = get_test_image((0, 255, 0), width, height)
        blue_img = get_test_image((0, 0, 255), width, height)

        packets = create_video([red_img, green_img, blue_img], width, height)
        self.assertEqual(len(packets), 3)

        frame_rgb = decode_last_frame([packets[0]], nvc.PixelFormat.RGB, width, height)
        np.testing.assert_array_almost_equal(frame_rgb / 255.0, red_img / 255.0, decimal=1)

        frame_rgb = decode_last_frame([packets[0], packets[1]], nvc.PixelFormat.RGB, width, height)
        np.testing.assert_array_almost_equal(frame_rgb / 255.0, green_img / 255.0, decimal=1)

        frame_rgb = decode_last_frame([packets[0], packets[1], packets[2]], nvc.PixelFormat.RGB, width, height)
        np.testing.assert_array_almost_equal(frame_rgb / 255.0, blue_img / 255.0, decimal=1)


           


