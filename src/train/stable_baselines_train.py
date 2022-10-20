import os
import gym
import tempfile
import torch
import wandb
import itertools
import numpy as np

from gym import spaces

from stable_baselines3 import SAC
from stable_baselines3.common.logger import configure

from wandb.integration.sb3 import WandbCallback

from src.config import HOST_CONFIG, MODEL_CONFIGS
from src.msgvec.pymsgvec import PyMsgVec, PyTimeoutResult, PyMessageTimingMode
from src.train.arrowcache import ArrowRLDataset

class MsgVecEnv(gym.Env):
    def __init__(self, msgvec) -> None:
        super().__init__()
        self.action_space = spaces.Box(low=-1, high=1, shape=(msgvec.act_size(),))
        self.observation_space = spaces.Box(low=-1, high=1, shape=(msgvec.obs_size(),))



if __name__ == "__main__":
    brain_config = MODEL_CONFIGS["basic-brain-test1"]
    msgvec = PyMsgVec(brain_config["msgvec"], PyMessageTimingMode.REPLAY)
    cache = ArrowRLDataset(os.path.join(HOST_CONFIG.RECORD_DIR), brain_config)
      
    buffer_size = 50_000

    env = MsgVecEnv(msgvec)
    model = SAC("MlpPolicy", env, buffer_size=buffer_size, verbose=1, 
                replay_buffer_kwargs={"handle_timeout_termination": False})
 
    # Setup the logger
    logger = configure("/home/jake/robotai/_sb3_logs", ["stdout", "tensorboard"])
    model.set_logger(logger)

    # Fill the replay buffer
    buffer = model.replay_buffer
    samples_added = 0
      
    for entry in itertools.islice(cache.generate_samples(), buffer_size):
        print(entry)
        buffer.add(obs=entry["obs"], action=entry["act"], reward=entry["reward"], next_obs=entry["next_obs"], done=entry["done"], infos=None)
        samples_added += 1

    print(f"Added {samples_added} samples to the replay buffer")

    for i in range(1000*1000):
        model.train(gradient_steps=1000, batch_size=512)
        print("Trained 1000 steps")
        logger.dump(step=i)

        if i % 20 == 0:
            model.save("/home/jake/robotai/_checkpoints/basic-brain-test1-sb3-0.zip")
            print("Model saved")