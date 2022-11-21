import os
import contextlib
import random
import pandas as pd
import numpy as np


from typing import Dict, List, NamedTuple
from cereal import log

from src.config import MODEL_CONFIGS

from src.logutil import LogHashes, LogSummary
from src.train.modelloader import model_fullname
from src.train.arrowcache import ArrowModelCache
from src.msgvec.pymsgvec import PyMsgVec, PyTimeoutResult, PyMessageTimingMode
from src.train.reward_modifiers import RewardModifierFunc, default_reward_modifier


# This class will read in log entries and actually create
# the obs, act, reward, done tuples that will be used for RL training.
# It relies on the ArrowModelCache for filling in vision intermediates and rewards
class MsgVecDataset():
    def __init__(self, dir: str, brain_model_config: Dict, reward_modifer: RewardModifierFunc=default_reward_modifier) -> None:
        self.dir = dir
        self.brain_config = brain_model_config
        self.reward_modifier = reward_modifer

        self.brain_fullname = model_fullname(brain_model_config)

        self.vision_cache = ArrowModelCache(dir, MODEL_CONFIGS[self.brain_config["models"]["vision"]])
        self.reward_cache = ArrowModelCache(dir, MODEL_CONFIGS[self.brain_config["models"]["reward"]])

        lh = LogHashes(dir)
        self.groups = lh.group_logs()

    def estimated_size(self) -> int:
        est = 0
        logs_per_group = 800

        for group in self.groups:
            for log in group:
                est += logs_per_group

        return est

    def _sort_log_group_messages(self, log_group: List[LogSummary]) -> List[log.Event]:
        # Sorts and returns the messages within the log group
        # Occasionally, on log rotation, you may have some messages out of order
        # So, we say that the first message in each log is the start time for that log, and we read ahead by
        # 1 log to make sure that the messages are in order
        all_events = []
        to_yield = 0

        with contextlib.ExitStack() as stack:
            # Read in the first log fully to populate the buffer
            f = stack.enter_context(open(os.path.join(self.dir, log_group[0].filename), "rb"))
            all_events += [evt for evt in log.Event.read_multiple(f)]
            to_yield += len(all_events)

            # Now for the remaining logs, read in, resort then yield up to the previous amount
            for logfile in log_group[1:]:
                f = stack.enter_context(open(os.path.join(self.dir, logfile.filename), "rb"))
                all_events += [evt for evt in log.Event.read_multiple(f)]
                all_events.sort(key=lambda evt: evt.logMonoTime)

                for evt in all_events[:to_yield]:
                    yield evt

                all_events = all_events[to_yield:]
                to_yield = len(all_events)

            # Yield the remaining events
            yield from all_events


    def generate_log_group(self, log_group: List[LogSummary], shuffle_within_group: bool = True): 
        msgvec = PyMsgVec(self.brain_config["msgvec"], PyMessageTimingMode.REPLAY)

        assert self.brain_config["msgvec"]["done"]["mode"] == "on_reward_override"

        raw_data = []

        group_runname = log_group[0].get_runname()
        last_timeout = PyTimeoutResult.MESSAGES_NOT_READY
        last_log_mono_time = None
        last_reward_was_override = False
        last_reward_modifier = 0.0
        last_reward_modifier_state = {}

        cur_packet = {}

        for evt in self._sort_log_group_messages(log_group):
            if last_log_mono_time is not None and evt.logMonoTime < last_log_mono_time:
                raise RuntimeError(f"Log files are not in order in {group_runname}")

            status = msgvec.input(evt.as_builder())
            last_reward_modifier, last_reward_modifier_state = self.reward_modifier(evt, last_reward_modifier_state) 

            if status["act_ready"]:
                cur_packet["act"] = msgvec.get_act_vector()

                if "obs" in cur_packet and "act" in cur_packet and "reward" in cur_packet and "done" in cur_packet:
                    cur_packet["reward"] += last_reward_modifier
                    raw_data.append(cur_packet)
                    cur_packet = {}

            if evt.which() == "modelInference":
                key = f"{group_runname}-{evt.modelInference.frameId}"

                vision_vec = self.vision_cache.get(key, None)

                # TODO Throw exception if the vision vector is not found and it should have been
                # Ex. if it's not one of the last 5 frames in the log
                if vision_vec is None:
                    break

                msgvec.input_vision(vision_vec, evt.modelInference.frameId)
                timeout, cur_packet["obs"] = msgvec.get_obs_vector()
                reward_valid, reward_value = msgvec.get_reward()

                if timeout == PyTimeoutResult.MESSAGES_NOT_READY:
                    if last_timeout == PyTimeoutResult.MESSAGES_PARTIALLY_READY or last_timeout == PyTimeoutResult.MESSAGES_ALL_READY:
                        # We got a timeout after messages were ready, so probably this batch is not good anymore
                        break
                    else:
                        msgvec._debug_print_timing()
                        continue
                elif timeout == PyTimeoutResult.MESSAGES_PARTIALLY_READY and last_timeout == PyTimeoutResult.MESSAGES_ALL_READY:
                    # We got a timeout after messages were ready, so probably this batch is not good anymore
                    break

                if reward_valid:
                    cur_packet["reward"] = reward_value
                    cur_packet["reward_override"] = True
                else:
                    reward = self.reward_cache.get(key, None)
                    if reward is None:
                        break

                    cur_packet["reward"] = reward
                    cur_packet["reward_override"] = False

                cur_packet["key"] = key
                cur_packet["done"] = False

                if not reward_valid and last_reward_was_override:
                    cur_packet["done"] = True

                last_reward_was_override = reward_valid
                last_timeout = timeout

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
        return final_data        

    # Use this method to return the full dataset, it returns each valid log entry exactly once
    def generate_dataset(self, shuffle_within_group: bool = True):
        for group in self.groups:
            yield from self.generate_log_group(group, shuffle_within_group)

    # Use this method estimate a randoml sample of a subset of the dataset
    # It samples log groups, then returns shuffled samples from within that loggroup
    def sample_dataset(self):
        # Each grouped log is handled separately, but the root-level groups are shuffled
        while True:
            group = random.choices(self.groups, weights=[len(g) for g in self.groups])[0]
            yield from self.generate_log_group(group, shuffle_within_group=True)

                        
