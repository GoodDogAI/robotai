import warnings
from abc import ABC, abstractmethod
from typing import Any, Dict, Generator, List, Optional, Union

import numpy as np
import torch
from gym import spaces

from stable_baselines3.common.preprocessing import get_action_dim, get_obs_shape
from stable_baselines3.common.type_aliases import (
    DictReplayBufferSamples,
    DictRolloutBufferSamples,
    ReplayBufferSamples,
    RolloutBufferSamples,
)
from stable_baselines3.common.utils import get_device
from stable_baselines3.common.vec_env import VecNormalize


from stable_baselines3.common.buffers import ReplayBuffer

# Base replay buffer code
# https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/common/buffers.py

# This custom version of the replay buffer is very similar to the original, except the first time that you call
# to_torch, it will convert the internal buffer to a device-base tensor, and then sample directly from those
class HostReplayBuffer(ReplayBuffer):

    def __init__(self, buffer_size: int, observation_space: spaces.Space, action_space: spaces.Space, device: Union[torch.device, str] = "auto", n_envs: int = 1, optimize_memory_usage: bool = False, handle_timeout_termination: bool = True):
        super().__init__(buffer_size, observation_space, action_space, device, n_envs, optimize_memory_usage, handle_timeout_termination)

        self.device_cache_ready = False
        assert handle_timeout_termination == False, "Timeouts not supported"
        assert optimize_memory_usage == False, "Optimize memory usage not supported"

        self.do_normalize_reward = False

    def add(self, obs: np.ndarray, next_obs: np.ndarray, action: np.ndarray, reward: np.ndarray, done: np.ndarray, infos: List[Dict[str, Any]]) -> None:
        self.device_cache_ready = False
        return super().add(obs, next_obs, action, reward, done, infos)
        
    def reset(self) -> None:
        self.device_cache_ready = False
        return super().reset()

    def normalize_reward(self, reward_mean, reward_std):
        self.do_normalize_reward = True
        self.reward_mean = torch.tensor([reward_mean], dtype=torch.float32, device=self.device)
        self.reward_std = torch.tensor([reward_std], dtype=torch.float32, device=self.device)
        
    def sample(self, batch_size: int, env: Optional[VecNormalize] = None) -> ReplayBufferSamples:
        if not self.device_cache_ready:
            self.device_observations = torch.from_numpy(self.observations).to(self.device)
            self.device_next_observations = torch.from_numpy(self.next_observations).to(self.device)
            self.device_actions = torch.from_numpy(self.actions).to(self.device)
            self.device_rewards = torch.from_numpy(self.rewards).to(self.device)
            self.device_dones = torch.from_numpy(self.dones).to(self.device)
            self.device_cache_ready = True

        upper_bound = self.buffer_size if self.full else self.pos
        batch_inds = torch.randint(0, upper_bound, size=(batch_size,), device=self.device)
        env_indices = torch.randint(0, high=self.n_envs, size=(len(batch_inds),), device=self.device)

        rewards = self.device_rewards[batch_inds, env_indices].reshape(-1, 1)

        if self.do_normalize_reward:
            rewards = (rewards - self.reward_mean) / self.reward_std

        data = (
            self._normalize_obs(self.device_observations[batch_inds, env_indices, :], env),
            self.device_actions[batch_inds, env_indices, :],
            self._normalize_obs(self.device_next_observations[batch_inds, env_indices, :], env),
            self.device_dones[batch_inds, env_indices].reshape(-1, 1),
            self._normalize_reward(rewards, env),
        )
        return ReplayBufferSamples(*data)
