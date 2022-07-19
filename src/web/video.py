import os
import numpy as np
import src.PyNvCodec as nvc

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

GPU_ID = 0

def load_image(logpath: str, index: int) -> np.ndarray:
    width, height = 1280, 720
    nv_dec = nvc.PyNvDecoder(
        width,
        height,
        nvc.PixelFormat.RGB,
        nvc.CudaVideoCodec.HEVC,
        GPU_ID,
    )

    nv_dl = nvc.PySurfaceDownloader(width, height, nv_dec.Format(), GPU_ID)

    frame_rgb = np.ndarray(shape=(0), dtype=np.uint8)
    packet_data = nvc.PacketData()

    events_sent = 0
    events_recv = 0

    with open(logpath, "rb") as f:
        events = log.Event.read_multiple(f)

        for evt in events:
            packet = evt.headEncodeData.data
            print(len(packet), "bytes")
            assert packet[0] == 0 and packet[1] == 0 and packet[2] == 0 and packet[3] == 1
            nalu_type = (packet[4] & 0x1F)
            print(f"{nalu_type} = {NALU_TYPES[nalu_type]}")

            packet = np.frombuffer(packet, dtype=np.uint8)
            surface = nv_dec.DecodeSurfaceFromPacket(packet)
            events_sent += 1

            print(f"surface empty: {surface.Empty()}")

            if not surface.Empty():
                if events_recv == index:
                    print(surface)
                    nv_dl.DownloadSingleSurface(surface, frame_rgb)
                    return frame_rgb.reshape((720, -1))

                events_recv += 1

        while True:
            surface = nv_dec.FlushSingleSurface()

            if surface.Empty():
                break
            else:
                events_recv += 1

        print(events_sent, events_recv)


