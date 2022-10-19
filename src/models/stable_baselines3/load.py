import torch
from stable_baselines3.sac import SAC

class OnnxableActor(torch.nn.Module):
    def __init__(self, actor: torch.nn.Module):
        super().__init__()
        self.actor = actor

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        # NOTE: You may have to process (normalize) observation in the correct
        #       way before using this. See `common.preprocessing.preprocess_obs`
        return self.actor(observation, deterministic=True)

def load_stable_baselines3_actor(checkpoint_zip: str, device=None) -> torch.nn.Module:
    sac = SAC.load(checkpoint_zip)
    return OnnxableActor(sac.actor)