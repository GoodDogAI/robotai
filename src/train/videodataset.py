import datasets
import glob
import torch
import os
import random
import numpy as np
import src.PyNvCodec as nvc
import src.PytorchNvCodec as pnvc

from src.video import V4L2_BUF_FLAG_KEYFRAME
from src.config import DEVICE_CONFIG, HOST_CONFIG
from src.train.videoloader import surface_to_y_uv

from cereal import log

class VideoFrameDataset(datasets.GeneratorBasedBuilder):
    DEFAULT_WRITER_BATCH_SIZE = 32

    def _info(self) -> datasets.DatasetInfo:
        return datasets.DatasetInfo(
            description="Extract intermediate and reward from video frames",
            version="0.0.1",
            features=datasets.Features(
                {
                    "y": datasets.Array3D(shape=(1, DEVICE_CONFIG.CAMERA_HEIGHT, DEVICE_CONFIG.CAMERA_WIDTH), dtype="float32"),
                    "uv": datasets.Array3D(shape=(1, DEVICE_CONFIG.CAMERA_HEIGHT // 2, DEVICE_CONFIG.CAMERA_WIDTH), dtype="float32"),
                }
            ),
        )

    def _split_generators(self, dl_manager: datasets.DownloadManager):
        # List all files in self.bash_path 
        files = glob.glob(f"{self.base_path}/*.log")

        train_count = min(int(len(files) * 0.9), len(files) - 1)
        valid_count = len(files) - train_count

        train_files = files[:train_count]
        valid_files = files[train_count:]

        return [
            datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={"files": train_files}),
            datasets.SplitGenerator(name=datasets.Split.VALIDATION, gen_kwargs={"files": valid_files}),
        ]

    def _generate_examples(self, **kwargs):
        files = kwargs["files"]

        nv_dec = nvc.PyNvDecoder(
            DEVICE_CONFIG.CAMERA_WIDTH,
            DEVICE_CONFIG.CAMERA_HEIGHT,
            nvc.PixelFormat.NV12, # All actual decodes must be NV12 format
            nvc.CudaVideoCodec.HEVC,
            HOST_CONFIG.DEFAULT_DECODE_GPU_ID, # TODO Set the GPU ID dynamically or something
        )

        key_queue = []

        for file_path in files:
            with open(file_path, "rb") as f:
                # Read the events from the log file
                events = log.Event.read_multiple(f)
                first = True

                # Get the actual events, starting with a keyframe, which we will need
                for evt in events:
                    if evt.which() == "headEncodeData":
                        if first:
                            assert evt.headEncodeData.idx.flags & V4L2_BUF_FLAG_KEYFRAME
                            first = False

                        packet = np.frombuffer(evt.headEncodeData.data, dtype=np.uint8)
                        surface = nv_dec.DecodeSurfaceFromPacket(packet)
                        key_queue.append(f"{os.path.basename(file_path)}-{evt.headEncodeData.idx.frameId}")

                        if not surface.Empty():
                            y, uv = surface_to_y_uv(surface)
                            yield key_queue.pop(0), {"y": y.cpu().numpy(), "uv": uv.cpu().numpy()}

                while True:
                    surface = nv_dec.FlushSingleSurface()

                    if surface.Empty():
                        break
                    else:
                        y, uv = surface_to_y_uv(surface)
                        yield key_queue.pop(0), {"y": y.cpu().numpy(), "uv": uv.cpu().numpy()}