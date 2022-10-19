import os
import gym
import tempfile
import torch
import numpy as np

from gym import spaces

from stable_baselines3 import SAC
from stable_baselines3.common.logger import configure

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
    cache = ArrowRLDataset(os.path.join(HOST_CONFIG.RECORD_DIR, "unittest"), brain_config)
      
    env = MsgVecEnv(msgvec)
    model = SAC("MlpPolicy", env, buffer_size=50000, verbose=1, 
                replay_buffer_kwargs={"handle_timeout_termination": False})
 
    # Setup the logger
    logger = configure(tempfile.TemporaryDirectory().name, ["stdout"])
    model.set_logger(logger)

    # Fill the replay buffer
    buffer = model.replay_buffer
      
    for entry in cache.generate_samples():
        # TODO NEXT OBS is not correct technically
        buffer.add(obs=entry["obs"], action=entry["act"], reward=entry["reward"], next_obs=entry["obs"], done=entry["done"], infos=None)

    model.train(gradient_steps=1, batch_size=64)

    model.save("/home/jake/robotai/_checkpoints/basic-brain-test1-sb3-0.zip")

    
    # Test exporting to onnx
    observation_size = model.observation_space.shape
    dummy_input = torch.randn(1, *observation_size).to(model.device)
    torch.onnx.export(
        model.actor,
        dummy_input,
        "my_sac_actor.onnx",
        opset_version=12,
        input_names=["input"])

    print("Exported ONNX")