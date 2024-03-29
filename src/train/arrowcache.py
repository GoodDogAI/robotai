import os
import torch
import pyarrow
import time
import functools
import pandas as pd
import numpy as np

from pathlib import Path
from typing import Dict, List
from cereal import log
import PyNvCodec as nvc
from polygraphy.cuda import DeviceView
from tinydb import TinyDB, Query

from src.config import MODEL_CONFIGS, HOST_CONFIG, DEVICE_CONFIG

from src.video import V4L2_BUF_FLAG_KEYFRAME
from src.logutil import LogHashes, LogSummary, get_runname
from src.train.videoloader import surface_to_y_uv
from src.train.modelloader import load_vision_model, model_fullname


# This class takes in a directory, and runs models on all of the video frames, saving
# the results to an arrow cache file. The keys are the videofilename-frameindex.
class ArrowModelCache():
    def __init__(self, dir: str, model_config: Dict, force_rebuild: bool=False):
        self.dir = dir
        self.model_fullname = model_fullname(model_config)
        self.model_config = model_config
        Path(HOST_CONFIG.CACHE_DIR, "arrow", self.model_fullname).mkdir(parents=True, exist_ok=True)

        self._build_cache(force_rebuild)

    def get_cache_path(self, run_name: str):
        return os.path.join(HOST_CONFIG.CACHE_DIR, "arrow", self.model_fullname, run_name + ".arrow")

    def _get_tinydb(self):
        return TinyDB(os.path.join(HOST_CONFIG.CACHE_DIR, "arrow", self.model_fullname, "db.json"))

    def _process_frame(self, engine, y, uv):
        # Unsqueeze to batch size 1
        y = torch.unsqueeze(y, 0)
        uv = torch.unsqueeze(uv, 0)

        result = engine.infer({"y": DeviceView(y.data_ptr(), y.shape, np.float32),
                               "uv": DeviceView(uv.data_ptr(), uv.shape, np.float32)}, copy_outputs_to_host=True)


        if self.model_config["type"] == "vision":
            # Remember to resqueeze to remove the batch dimension
            # Later, this can be optimized to process frames in bigger batches
            return result["intermediate"][0]
        elif self.model_config["type"] == "reward":
            # Reward does not need to be resqueezed
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
            with open(os.path.join(self.dir, logfile.filename), "rb") as f:
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


    def _cache_needs_rebuild(self):
        lh = LogHashes(self.dir)
        db = self._get_tinydb()

        for group in lh.group_logs():
            cache_path = self.get_cache_path(group[0].get_runname())

            if not os.path.exists(cache_path):
                return True

            for log in group:
                if not db.contains(Query().filename == log.filename):
                    return True

        return False                        

    def _build_cache(self, force_rebuild=False):
        # Quick exit if we already have the cache and it doesn't need to be added to or rebuilt
        if not force_rebuild and not self._cache_needs_rebuild():
            return

        lh = LogHashes(self.dir)
        db = self._get_tinydb()

        with load_vision_model(self.model_fullname) as engine:
            for group in lh.group_logs():
                cache_path = self.get_cache_path(group[0].get_runname())

                if db.count(Query().filename.one_of([x.filename for x in group])) == len(group) and not force_rebuild:
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
                table = pyarrow.Table.from_pandas(df, preserve_index=True)

                # Write out to file
                with pyarrow.OSFile(cache_path, 'wb') as sink:
                    with pyarrow.RecordBatchFileWriter(sink, table.schema) as writer:
                        writer.write_table(table)

                # Update the cache of which documents we have processed
                for log in group:
                    db.insert({"filename": log.filename})
                        
    
    @functools.lru_cache(maxsize=16)
    def get_dataframe(self, run_name: str):
        bag_cache_name = self.get_cache_path(run_name)
        source = pyarrow.memory_map(bag_cache_name, "r")
        table = pyarrow.ipc.RecordBatchFileReader(source).read_all()

        pd = table.to_pandas()
        return pd

    def get(self, key, default=None):
        start = time.perf_counter()

        run_name = "-".join(key.split("-")[0:-1])

        try:
            pd = self.get_dataframe(run_name)
            result = pd.loc[key]
        except KeyError:
            return default
        except FileNotFoundError:
            print(f"Warning: Cache file {run_name} not found")
            return default

        #print(f"Took {time.perf_counter() - start} to load {key}")

        if len(result["shape"]) == 0:
            return result["value"].item()
        else:
            return result["value"]

                        