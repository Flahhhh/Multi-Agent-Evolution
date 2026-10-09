import gymnasium as gym
import torch
from gymnasium.core import WrapperObsType, ObsType


class TorchWrapper(gym.ObservationWrapper):
    def observation(self, observation: ObsType) -> WrapperObsType:
        return torch.from_numpy(observation)


def make_env(env_cfg=None):
    env = TorchWrapper(gym.make("Flah/MABattle-v0", env_cfg=env_cfg))

    return env
