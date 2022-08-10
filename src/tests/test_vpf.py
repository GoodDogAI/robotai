import unittest
import os
import numpy as np

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


           


