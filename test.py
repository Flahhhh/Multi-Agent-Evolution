import torch
from torch.multiprocessing import get_context

from MABattle.MABattleV0 import EnvCfg
from MABattle.utils.game import test_play_game

from utils import RandomAgent
from net import MAFCQNet, MAFCQNoisyNet

num_tests = 10000
num_processes = 8


def run_test():
    ctx = get_context('spawn')
    pool = ctx.Pool(num_processes)

    state_dict = torch.load(
        r"results/DQN/2026-08-07 01-30/Models/gradient/DQN_24500.pt",
        weights_only=False, map_location=torch.device("cpu"))["model"]
    env_cfg = EnvCfg()

    opponent = RandomAgent(env_cfg.num_agents)

    results = pool.starmap(test_play_game, [(env_cfg, state_dict, MAFCQNoisyNet, opponent) for _ in range(num_tests)])
    all_rewards, all_images = zip(*results)

    _, idx = torch.max(torch.tensor(all_rewards), dim=0)
    images = all_images[idx]

    images[0].save(f"output.gif", save_all=True, append_images=images[1:], duration=500, loop=0)
    print(all_rewards[idx])


if __name__ == "__main__":
    run_test()
