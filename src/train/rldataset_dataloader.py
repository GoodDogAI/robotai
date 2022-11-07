import os
import gym
import time
import torch
import glob
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

