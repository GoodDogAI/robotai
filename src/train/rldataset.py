import os
import random
import pandas as pd
import numpy as np


from typing import Dict, List
from cereal import log

from src.config import MODEL_CONFIGS

from src.logutil import LogHashes, LogSummary
from src.train.modelloader import model_fullname
from src.train.arrowcache import ArrowModelCache
from src.msgvec.pymsgvec import PyMsgVec, PyTimeoutResult, PyMessageTimingMode


# This class will read in log entries and actually create
# the obs, act, reward, done tuples that will be used for RL training.
# It relies on the ArrowModelCache for filling in vision intermediates and rewards
class MsgVecDataset():
    def __init__(self, dir: str, brain_model_config: Dict) -> None:
        self.lh = LogHashes(dir)
        self.brain_config = brain_model_config
        self.brain_fullname = model_fullname(brain_model_config)

        self.vision_cache = ArrowModelCache(dir, MODEL_CONFIGS[self.brain_config["models"]["vision"]])
        self.reward_cache = ArrowModelCache(dir, MODEL_CONFIGS[self.brain_config["models"]["reward"]])

    def generate_log_group(self, log_group: List[LogSummary], shuffle_within_group: bool = True): 
        msgvec = PyMsgVec(self.brain_config["msgvec"], PyMessageTimingMode.REPLAY)

        assert self.brain_config["msgvec"]["done"]["mode"] == "on_reward_override"

        raw_data = []

        last_log_mono_time = None
        last_reward_was_override = False
        continue_processing_group = True
        cur_packet = {}


        for logfile in log_group:
            if not continue_processing_group:
                break

            with open(os.path.join(self.lh.dir, logfile.filename), "rb") as f:
                events = log.Event.read_multiple(f)

                # Get the actual events, starting with a keyframe, which we will need
                for evt in events:
                    if last_log_mono_time is not None and evt.logMonoTime < last_log_mono_time:
                        raise RuntimeError("Log files are not in order")

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
                            msgvec._debug_print_timing()
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

                    last_log_mono_time = evt.logMonoTime


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

        if shuffle_within_group:
            random.shuffle(final_data)

        # Once you process a whole group, you can yield the results
        yield from final_data        

    def generate_samples(self, shuffle_groups: bool = True, shuffle_within_group: bool = True):
        # Each grouped log is handled separately, but the root-level groups are shuffled
        groups = self.lh.group_logs()

        if shuffle_groups:
            random.shuffle(groups)

        for group in groups:
            yield from self.generate_log_group(group, shuffle_within_group)

        # You could enable multiprocessing, but the memory usage is very high unfortunately
        # pool = multiprocessing.Pool()

        # for grp_result in pool.imap(functools.partial(self._generate_log_group, shuffle_within_group=shuffle_within_group), groups, chunksize=1):
        #     yield from grp_result
                        
