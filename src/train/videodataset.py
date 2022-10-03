import datasets
import glob
import torch
import os
import random
import numpy as np
import src.PyNvCodec as nvc
import src.PytorchNvCodec as pnvc
from polygraphy.cuda import DeviceView

from src.video import V4L2_BUF_FLAG_KEYFRAME
from src.config import DEVICE_CONFIG, HOST_CONFIG, MODEL_CONFIGS
from src.train.modelloader import load_vision_model, model_fullname
from src.train.videoloader import surface_to_y_uv

from cereal import log

class IntermediateRewardDataset(datasets.GeneratorBasedBuilder):
    DEFAULT_WRITER_BATCH_SIZE = 32

    def _info(self) -> datasets.DatasetInfo:
        return datasets.DatasetInfo(
            description="Extract intermediate and reward from video frames",
            version="0.0.1",
            features=datasets.Features(
                {
                    #"intermediate": datasets.Array2D(shape=(1, 17003), dtype="float32"),
                    "intermediate": datasets.Array2D(shape=(1, 8502), dtype="float32"),
                    "reward": datasets.Value("float32"),
                }
            ),
        )

    def _split_generators(self, dl_manager: datasets.DownloadManager):
        # List all files in self.bash_path 
        files = glob.glob(f"{self.base_path}/*.log")
        random.seed(1)
        random.shuffle(files)

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

        with load_vision_model(model_fullname(MODEL_CONFIGS["yolov7-tiny-s53-deeper"])) as intermediate_engine, \
             load_vision_model(model_fullname(MODEL_CONFIGS["yolov7-tiny-prioritize_centered_nms"])) as reward_engine:

            def process_frame(y, uv):
                y = torch.unsqueeze(y, 0)
                uv = torch.unsqueeze(uv, 0)

                intermediates = intermediate_engine.infer({"y": DeviceView(y.data_ptr(), y.shape, np.float32),
                                                           "uv": DeviceView(uv.data_ptr(), uv.shape, np.float32)}, copy_outputs_to_host=True)

                intermediate = intermediates["intermediate"]                                                     

                rewards = reward_engine.infer({"y": DeviceView(y.data_ptr(), y.shape, np.float32),
                                               "uv": DeviceView(uv.data_ptr(), uv.shape, np.float32)}, copy_outputs_to_host=True)

                reward = rewards["reward"].item()

                return {"intermediate": intermediate, "reward": reward}                                                   


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
                                yield key_queue.pop(0), process_frame(y, uv)

                    while True:
                        surface = nv_dec.FlushSingleSurface()

                        if surface.Empty():
                            break
                        else:
                            y, uv = surface_to_y_uv(surface)
                            yield key_queue.pop(0), process_frame(y, uv)