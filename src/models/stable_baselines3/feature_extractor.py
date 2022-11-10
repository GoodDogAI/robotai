
import gym
import torch
import numpy as np
import matplotlib.pyplot as plt


from stable_baselines3 import SAC
from stable_baselines3.common.preprocessing import get_flattened_obs_dim
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

# This feature extractor allows you to normalize the observation vector, based on the mean and std of the training data
# You should pass in obs and means tensors as such:
#  obs_means = torch.zeros(env.observation_space.shape, dtype=torch.float32, requires_grad=False).to("cuda")
#  obs_stds = torch.zeros(env.observation_space.shape, dtype=torch.float32, requires_grad=False).to("cuda")
# Then, before training begins, update them in-place, by iterating through your dataset
class MsgVecNormalizeFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.Space, obs_means: torch.Tensor, obs_stds: torch.Tensor):
        super().__init__(observation_space, get_flattened_obs_dim(observation_space))
        self.flatten = torch.nn.Flatten()
        self.obs_means = obs_means
        self.obs_stds = obs_stds

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return (self.flatten(observations) - self.obs_means) / self.obs_stds
