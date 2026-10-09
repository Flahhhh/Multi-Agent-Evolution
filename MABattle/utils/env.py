import gymnasium as gym
import torch
from gymnasium.core import RenderFrame, WrapperObsType, WrapperActType, ObsType

from const import env_name


class TorchWrapper(gym.ObservationWrapper):
    def observation(self, observation: ObsType) -> WrapperObsType:
        return torch.from_numpy(observation)


def make_env(env_cfg=None):
    env = TorchWrapper(gym.make(env_name, env_cfg=env_cfg))
    #env = NumpyToTorch(gym.make(env_name))

    return env
