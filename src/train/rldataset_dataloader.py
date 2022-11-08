import os
import gym
import time
import torch
import math
import itertools
import numpy as np

from gym import spaces
from tqdm import tqdm

from stable_baselines3 import SAC
from stable_baselines3.common.logger import configure, HParam
from torch.profiler import profile, record_function, ProfilerActivity

from src.config import HOST_CONFIG, MODEL_CONFIGS
from src.msgvec.pymsgvec import PyMsgVec, PyTimeoutResult, PyMessageTimingMode
from src.train.rldataset import MsgVecDataset
from stable_baselines3.common.buffers import ReplayBuffer
from src.train.stable_baselines_buffers import HostReplayBuffer



brain_config = MODEL_CONFIGS["basic-brain-test1"]
msgvec = PyMsgVec(brain_config["msgvec"], PyMessageTimingMode.REPLAY)
cache = MsgVecDataset(os.path.join(HOST_CONFIG.RECORD_DIR), brain_config)

class PTMsgVec(torch.utils.data.IterableDataset):
    def __init__(self, msgvec, cache):
        self.msgvec = msgvec
        self.cache = cache

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        groups = self.cache.lh.group_logs()
        per_worker = int(math.ceil(len(groups)) / float(worker_info.num_workers))
        worker_id = worker_info.id
        iter_start = worker_id * per_worker
        iter_end = min(iter_start + per_worker, len(groups))

        print(f"Total groups: {len(groups)}")
        print(f"Worker {worker_id} has {iter_start} to {iter_end}")

        local_groups = groups[iter_start:iter_end]
        for group in local_groups:
            # Take only up to the first 4 logs in each group
            group = group[:4]

            result = self.cache.generate_log_group(group, shuffle_within_group=True)
            print(f"Worker {worker_id} has {len(result)} logs")
            yield from result

ds = PTMsgVec(msgvec, cache)
batch_size = 128
dl = torch.utils.data.DataLoader(ds, batch_size=batch_size, num_workers=2)

count = 0
max_count = 50000
start = time.perf_counter()
for index, data in enumerate(dl):
    # if count % 100 == 0:
    #     print(f"Count: {count}, Time: {time.perf_counter() - start}")

    count += batch_size

    if count > max_count:
        break

end = time.perf_counter()
print(data)
print(f"Took: {end - start} to load {count} samples")
print(f"Samples per second: {(count) / (end - start)}")
