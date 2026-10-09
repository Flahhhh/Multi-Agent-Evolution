import random

from MABattle.utils import make_env, play_game
from .agent import RandomAgent


def ma_fit_randomizer(idx, population, model_cls, env_cfg):
    model = model_cls(env_cfg.flatten_state_shape, env_cfg.num_agents, env_cfg.action_space)
    opponent = RandomAgent(env_cfg.num_agents)

    model.load_state_dict(population[idx])

    return ma_fit_base(model, opponent, env_cfg)

def ma_fit_competitive_coevolution(idx, population, model_cls, env_cfg):
    model = model_cls(env_cfg.flatten_state_shape, env_cfg.num_agents, env_cfg.action_space)
    opponent = model_cls(env_cfg.flatten_state_shape, env_cfg.num_agents, env_cfg.action_space)

    model.load_state_dict(population[idx])
    opponent.load_state_dict(random.choice(population))

    return ma_fit_base(model, opponent, env_cfg)

def ma_fit_base(model, opponent, env_cfg):
    env = make_env(env_cfg)

    logs, fit, _, _, _, _, _ = play_game(model, opponent, env)
    env.close()

    return logs, fit