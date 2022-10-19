import torch
from stable_baselines3.sac import SAC

def load_stable_baselines3_actor(checkpoint_zip: str, device=None) -> torch.nn.Module:
    sac = SAC.load(checkpoint_zip)
    return sac.actor