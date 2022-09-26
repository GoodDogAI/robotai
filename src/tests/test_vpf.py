import unittest
import os
import torch
import numpy as np
from einops import rearrange

from src.tests.utils import get_test_image
from src.train.onnx_yuv import nv12m_to_rgb
from src.video import get_image_packets, create_video, decode_last_frame
from src.config import HOST_CONFIG
import src.PyNvCodec as nvc

from cereal import log

class VPFTest(unittest.TestCase):
    def test_load(self):
        test_path = os.path.join(HOST_CONFIG.RECORD_DIR, "unittest", "alphalog-22c37d10-2022-9-16-21_21.log")
        
        packets = get_image_packets(test_path, 100)
        y, uv = decode_last_frame(packets)
        self.assertEqual(y.shape, (720, 1280))
        self.assertEqual(uv.shape, (360, 1280))

    def test_decode_single_frame(self):
        test_path = os.path.join(HOST_CONFIG.RECORD_DIR, "unittest", "alphalog-22c37d10-2022-9-16-21_21.log")
        
        nv_dec = nvc.PyNvDecoder(
            1280,
            720,
            nvc.PixelFormat.NV12,
            nvc.CudaVideoCodec.HEVC,
            HOST_CONFIG.DEFAULT_DECODE_GPU_ID,
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

        packets = create_video([red_img, green_img, blue_img])
        self.assertEqual(len(packets), 3)

        frame_rgb = decode_last_frame([packets[0]], nvc.PixelFormat.RGB, width, height)
        np.testing.assert_array_almost_equal(frame_rgb / 255.0, red_img / 255.0, decimal=1)

        frame_rgb = decode_last_frame([packets[0], packets[1]], nvc.PixelFormat.RGB, width, height)
        np.testing.assert_array_almost_equal(frame_rgb / 255.0, green_img / 255.0, decimal=1)

        frame_rgb = decode_last_frame([packets[0], packets[1], packets[2]], nvc.PixelFormat.RGB, width, height)
        np.testing.assert_array_almost_equal(frame_rgb / 255.0, blue_img / 255.0, decimal=1)

    def _decode_assert_color(self, width, height, packets, target_color, tolerance=0.01):
        frame_y, frame_uv = decode_last_frame(packets, nvc.PixelFormat.NV12, width, height)
        self.assertEqual(frame_y.shape, (height, width))
        self.assertEqual(frame_uv.shape, (height // 2, width))

        y = rearrange(torch.from_numpy(frame_y).to(torch.float32), "h w -> 1 1 h w")
        uv = rearrange(torch.from_numpy(frame_uv).to(torch.float32), "h w -> 1 1 h w")
        rgb = nv12m_to_rgb(y, uv)

        self.assertEqual(rgb.shape, (1, 3, height, width))
        
        np.testing.assert_allclose(rgb[0, :, 0, 0].numpy(), target_color, atol=tolerance)

    def test_encode_decode_nv12(self):
        width, height = 1280, 720

        red_img = get_test_image((255, 0, 0), width, height)
        green_img = get_test_image((0, 255, 0), width, height)
        blue_img = get_test_image((0, 0, 255), width, height)
        white_img = get_test_image((255, 255, 255), width, height)
        black_img = get_test_image((0, 0, 0), width, height)

        packets = create_video([red_img, green_img, blue_img, white_img, black_img])
        self.assertEqual(len(packets), 5)

        # Red packet
        self._decode_assert_color(width, height, [packets[0]], [1.0, 0.0, 0.0])

        # Green packet
        self._decode_assert_color(width, height, [packets[0], packets[1]], [0.0, 1.0, 0.0])

        # Blue packet
        self._decode_assert_color(width, height, [packets[0], packets[1], packets[2]], [0.0, 0.0, 1.0], tolerance=0.02)

        # White packet
        self._decode_assert_color(width, height, [packets[0], packets[1], packets[2], packets[3]], [1.0, 1.0, 1.0])

        # Black packet
        self._decode_assert_color(width, height, [packets[0], packets[1], packets[2], packets[3], packets[4]], [0.0, 0.0, 0.0])



           


