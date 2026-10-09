import torch
from torch.multiprocessing import get_context
import gymnasium as gym

from MABattle.utils.env import make_env
from MABattle.utils.game import test_play_game

from const import env_name, device_name
from utils import RandomAgent
from net import MADeterministicFCNet, MAFCQNet, MAFCQNoisyNet

num_tests = 10000
num_processes = 8


def run_test():
    ctx = get_context('spawn')  # или 'forkserver'
    pool = ctx.Pool(num_processes)

    device = torch.device(device_name)
    # model = MAFCQNoisyNet().to(device).eval()

    # state_dict = torch.load(r"C:\Users\YOU-LA\Desktop\apple\genetic_algorithms\nne_v2(QZero)\logs\2026-05-06 20-04\Models\nne-opponent_actions_alldata_buffer_random_3_step_bagging_joint-1-I-DEC-295.pt",
    #                        weights_only=False)["model"]
    state_dict = torch.load(
        r"C:\Users\YOU-LA\Desktop\apple\genetic_algorithms\nne_v2(QZero)\logs_dqn_ind\2026-06-29 14-27\Models\gradient\nne-random-1_gradient.pt",
        weights_only=False, map_location=torch.device("cpu"))["model"]
    #print(state_dict)

    # model_ = MAFCQNet().to(device).eval()
    # state_dict = torch.load(r"C:\Users\YOU-LA\Desktop\apple\genetic_algorithms\nne_v2(QZero)\logs\2026-03-31 10-43\Models\nne-opponent_approx_gradient_buffer_random_3_step_bagging-6-295.pt",
    #                        weights_only=False)["model"]
    # model_.load_state_dict(state_dict)

    opponent = RandomAgent()

    results = pool.starmap(test_play_game, [(state_dict, MAFCQNoisyNet, opponent, device) for _ in range(num_tests)])
    all_rewards, all_images = zip(*results)

    _, idx = torch.median(torch.tensor(all_rewards), dim=0)
    images = all_images[idx]

    #print(images)
    images[0].save(f"output.gif", save_all=True, append_images=images[1:], duration=500, loop=0)
    print(all_rewards[idx])


if __name__ == "__main__":
    run_test()
