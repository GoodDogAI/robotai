import os
import torch
import pyarrow
import time
import random
import functools
import pandas as pd
import numpy as np

from pathlib import Path
from typing import Dict
from cereal import log
import src.PyNvCodec as nvc
import src.PytorchNvCodec as pnvc
from polygraphy.cuda import DeviceView

from src.config import MODEL_CONFIGS, HOST_CONFIG, DEVICE_CONFIG

from src.video import V4L2_BUF_FLAG_KEYFRAME
from src.logutil import LogHashes, LogSummary, get_runname
from src.train.videoloader import surface_to_y_uv
from src.train.modelloader import load_vision_model, model_fullname
from src.msgvec.pymsgvec import PyMsgVec, PyTimeoutResult, PyMessageTimingMode


# This class takes in a directory, and runs models on all of the video frames, saving
# the results to an arrow cache file. The keys are the videofilename-frameindex.
class ArrowModelCache():
    def __init__(self, dir: str, model_config: Dict, force_rebuild: bool=False):
        self.lh = LogHashes(dir)
        self.model_fullname = model_fullname(model_config)
        self.model_config = model_config
        Path(HOST_CONFIG.CACHE_DIR, "arrow", self.model_fullname).mkdir(parents=True, exist_ok=True)

        self._build_cache(force_rebuild)

    def get_cache_path(self, run_name: str):
        return os.path.join(HOST_CONFIG.CACHE_DIR, "arrow", self.model_fullname, run_name + ".arrow")

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

    def _cache_needs_rebuild(self):
        for group in self.lh.group_logs():
            cache_path = self.get_cache_path(group[0].get_runname())

            if not os.path.exists(cache_path):
                return True

        return False                        

    def _build_cache(self, force_rebuild=False):
        # Quick exit if we already have the cache and it doesn't need to be added to or rebuilt
        if not force_rebuild and not self._cache_needs_rebuild():
            return

        with load_vision_model(self.model_fullname) as engine:
            for group in self.lh.group_logs():
                cache_path = self.get_cache_path(group[0].get_runname())

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
                table = pyarrow.Table.from_pandas(df, preserve_index=True)

                # Write out to file
                with pyarrow.OSFile(cache_path, 'wb') as sink:
                    with pyarrow.RecordBatchFileWriter(sink, table.schema) as writer:
                        writer.write_table(table)
    
    @functools.lru_cache(maxsize=16)
    def get_dataframe(self, run_name: str):
        print("Loading dataframe", run_name)
        bag_cache_name = self.get_cache_path(run_name)
        source = pyarrow.memory_map(bag_cache_name, "r")
        table = pyarrow.ipc.RecordBatchFileReader(source).read_all()

        pd = table.to_pandas()
        return pd

    def get(self, key, default=None):
        start = time.perf_counter()

        run_name = "-".join(key.split("-")[0:-1])
        pd = self.get_dataframe(run_name)

        try:
            result = pd.loc[key]
        except KeyError:
            return default

        #print(f"Took {time.perf_counter() - start} to load {key}")

        if len(result["shape"]) == 0:
            return result["value"].item()
        else:
            return result["value"]

                            

# This class will read in log entries and actually create
# the obs, act, reward, done tuples that will be used for RL training.
# It relies on the ArrowModelCache for filling in vision intermediates and rewards
class ArrowRLDataset():
    def __init__(self, dir: str, brain_model_config: Dict) -> None:
        self.lh = LogHashes(dir)
        self.brain_config = brain_model_config
        self.brain_fullname = model_fullname(brain_model_config)

        self.vision_cache = ArrowModelCache(dir, MODEL_CONFIGS[self.brain_config["models"]["vision"]])
        self.reward_cache = ArrowModelCache(dir, MODEL_CONFIGS[self.brain_config["models"]["reward"]])

    def _generate_log_group(self, log_group):
        msgvec = PyMsgVec(self.brain_config["msgvec"], PyMessageTimingMode.REPLAY)

        assert self.brain_config["msgvec"]["done"]["mode"] == "on_reward_override"

        raw_data = []

        last_reward_was_override = False
        continue_processing_group = True
        cur_packet = {}


        for logfile in log_group:
            if not continue_processing_group:
                break

            with open(os.path.join(self.lh.dir, logfile.filename), "rb") as f:
                print("Processing", logfile.filename)
                events = log.Event.read_multiple(f)

                # Get the actual events, starting with a keyframe, which we will need
                for evt in events:
                    status = msgvec.input(evt.as_builder())

                    if status["act_ready"]:
                        cur_packet["act"] = msgvec.get_act_vector()

                        if "obs" in cur_packet and "act" in cur_packet and "reward" in cur_packet and "done" in cur_packet:
                            raw_data.append(cur_packet)
                            cur_packet = {}

                    if evt.which() == "modelInference":
                        key = f"{logfile.get_runname()}-{evt.modelInference.frameId}"

                        vision_vec = self.vision_cache.get(key, None)

                        if vision_vec is None:
                            continue_processing_group = False
                            break

                        msgvec.input_vision(vision_vec, evt.modelInference.frameId)
                        timeout, cur_packet["obs"] = msgvec.get_obs_vector()
                        reward_valid, reward_value = msgvec.get_reward()

                        if timeout == PyTimeoutResult.MESSAGES_NOT_READY:
                            continue

                        if reward_valid:
                            cur_packet["reward"] = reward_value
                            cur_packet["reward_override"] = True
                        else:
                            reward = self.reward_cache.get(key, None)
                            if reward is None:
                                continue_processing_group = False
                                break
                            cur_packet["reward"] = reward
                            cur_packet["reward_override"] = False

                        cur_packet["key"] = key
                        cur_packet["done"] = False

                        if not reward_valid and last_reward_was_override:
                            cur_packet["done"] = True

                        last_reward_was_override = reward_valid


        # The last packet is always done
        if len(raw_data) > 0:
            raw_data[-1]["done"] = True

        # You're going to want to reprocess the dictionary to create the next_obs datapoints
        # Note: One day this could be better optimized
        final_data = []

        for index, data in enumerate(raw_data[:-1]):
            if data["done"]:
                continue

            data["next_obs"] = raw_data[index + 1]["obs"]
            data["done"] = raw_data[index + 1]["done"]

            final_data.append(data)

        # Once you process a whole group, you can yield the results
        yield from final_data        

    def generate_samples(self):
        # Each grouped log is handled separately, but the root-level groups are shuffled
        groups = self.lh.group_logs()
        random.shuffle(groups)

        for group in groups:
            yield from self._generate_log_group(group)
                        
