import os
import torch
import pyarrow
import json
import pandas as pd
import numpy as np

from pathlib import Path
from typing import Dict
from cereal import log
import src.PyNvCodec as nvc
import src.PytorchNvCodec as pnvc
from polygraphy.cuda import DeviceView

from config import HOST_CONFIG, DEVICE_CONFIG

from src.video import V4L2_BUF_FLAG_KEYFRAME
from src.logutil import LogHashes, LogSummary
from src.train.videoloader import surface_to_y_uv
from src.train.modelloader import load_vision_model, model_fullname
from src.msgvec.pymsgvec import PyMsgVec, PyTimeoutResult


# This class takes in a directory, and runs models on all of the video frames, saving
# the results to an arrow cache file. The keys are the videofilename-frameindex.
class ArrowModelCache():
    def __init__(self, dir: str, model_config: Dict):
        self.lh = LogHashes(dir)
        self.model_fullname = model_fullname(model_config)
        self.model_config = model_config
        Path(HOST_CONFIG.CACHE_DIR, "arrow", self.model_fullname).mkdir(parents=True, exist_ok=True)

    def get_cache_path(self, log: LogSummary):
        return os.path.join(HOST_CONFIG.CACHE_DIR, "arrow", self.model_fullname, log.get_runname() + ".arrow")

    def _process_frame(self, engine, y, uv):
        y = torch.unsqueeze(y, 0)
        uv = torch.unsqueeze(uv, 0)

        result = engine.infer({"y": DeviceView(y.data_ptr(), y.shape, np.float32),
                               "uv": DeviceView(uv.data_ptr(), uv.shape, np.float32)}, copy_outputs_to_host=True)

        if self.model_config["type"] == "vision":
            return result["intermediate"]
        elif self.model_config["type"] == "reward":
            return result["reward"]
        else:
            raise NotImplementedError()

    def _build_for_filegroup(self, group, engine):
        nv_dec = nvc.PyNvDecoder(
                    DEVICE_CONFIG.CAMERA_WIDTH,
                    DEVICE_CONFIG.CAMERA_HEIGHT,
                    nvc.PixelFormat.NV12, # All actual decodes must be NV12 format
                    nvc.CudaVideoCodec.HEVC,
                    HOST_CONFIG.DEFAULT_DECODE_GPU_ID, # TODO Set the GPU ID dynamically or something
                )

        key_queue = []

        for logfile in group:
            with open(os.path.join(self.lh.dir, logfile.filename), "rb") as f:
                print("Processing", logfile.filename)
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
                        key_queue.append(f"{logfile.get_runname()}-{evt.headEncodeData.idx.frameId}")

                        if not surface.Empty():
                            y, uv = surface_to_y_uv(surface)
                            yield key_queue.pop(0), self._process_frame(engine, y, uv)

                while True:
                    surface = nv_dec.FlushSingleSurface()

                    if surface.Empty():
                        break
                    else:
                        y, uv = surface_to_y_uv(surface)
                        yield key_queue.pop(0), self._process_frame(engine, y, uv)
                            

    def build_cache(self, force_rebuild=False):
        with load_vision_model(self.model_fullname) as engine:
            for group in self.lh.group_logs():
                cache_path = self.get_cache_path(group[0])

                if os.path.exists(cache_path) and not force_rebuild:
                    continue

                print(f"Building cache {cache_path}")

                raw_data = []
                all_keys = set()

                for key, vision_value in self._build_for_filegroup(group, engine):
                    raw_data.append({"key": key, "shape": vision_value.shape, "value": vision_value.flatten()})

                    assert key not in all_keys
                    all_keys.add(key)

                # Save those into a Dataframe
                df = pd.DataFrame.from_records(raw_data,
                                               columns=["key", "shape", "value"], index="key")                                      

                # Convert from pandas to Arrow
                table = pyarrow.Table.from_pandas(df)

                # Write out to file
                with pyarrow.OSFile(cache_path, 'wb') as sink:
                    with pyarrow.RecordBatchFileWriter(sink, table.schema) as writer:
                        writer.write_table(table)
                            

# This class is similar to ArrowModelCache, but it will read in log entries and actually create
# the obs, act, reward, done tuples that will be used for RL training.
# It relies on the ArrowModelCache for filling in vision intermediates and rewards
class ArrowRLCache():
    def __init__(self, dir: str, brain_model_config: Dict) -> None:
        self.lh = LogHashes(dir)
        self.brain_config = brain_model_config
        self.brain_fullname = model_fullname(brain_model_config)
        Path(HOST_CONFIG.CACHE_DIR, "arrow", self.brain_fullname).mkdir(parents=True, exist_ok=True)

    def get_cache_path(self, log: LogSummary):
        return os.path.join(HOST_CONFIG.CACHE_DIR, "arrow", self.brain_fullname, log.get_runname() + ".arrow")


    def build_cache(self, force_rebuild=False):
         for group in self.lh.group_logs():
            cache_path = self.get_cache_path(group[0])

            if os.path.exists(cache_path) and not force_rebuild:
                continue

            msgvec = PyMsgVec(json.dumps(self.brain_config["msgvec"]).encode("utf-8"))
            raw_data = []

            cur_inference = None
            cur_packet = {}

            for logfile in group:
                with open(os.path.join(self.lh.dir, logfile.filename), "rb") as f:
                    print("Processing", logfile.filename)
                    events = log.Event.read_multiple(f)

                    # Get the actual events, starting with a keyframe, which we will need
                    for evt in events:
                        status = msgvec.input(evt.as_builder().to_bytes())

                        if status["act_ready"]:
                            cur_packet["act"] = msgvec.get_act_vector()
                            raw_data.append(cur_packet)
                            cur_packet = {}

                        if evt.which() == "modelInference":
                            cur_inference = evt
                            timeout, cur_packet["obs"] = msgvec.get_obs_vector()

                        
