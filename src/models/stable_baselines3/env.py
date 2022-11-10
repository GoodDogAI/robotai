import gym

from gym import spaces
from tqdm import tqdm

from src.msgvec.pymsgvec import PyMsgVec

# Basic environment that gets its shape from a msgvec configuration
class MsgVecEnv(gym.Env):
    def __init__(self, msgvec: PyMsgVec) -> None:
        super().__init__()
        self.action_space = spaces.Box(low=-1, high=1, shape=(msgvec.act_size(),))
        self.observation_space = spaces.Box(low=-1, high=1, shape=(msgvec.obs_size(),))
