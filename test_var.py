import os
import json
from torch.multiprocessing import get_context
from tqdm import tqdm
import torch
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab

from MABattle.MABattleV0 import EnvCfg
from MABattle.utils import play_game, test_play_game, make_env
from net import MAFCQNet, MAFCQNoisyNet
from utils import RandomAgent

def play(env_cfg, model_state, model_cls, opponent):
    env = make_env(env_cfg)

    try:

        model = model_cls(env_cfg.flatten_state_shape, env_cfg.num_agents, env_cfg.action_space)
        model.load_state_dict(model_state)

        _, global_reward, rewards, zero_rewards, step_discounted_rewards, pdv_rewards, c = play_game(model, opponent,
                                                                                                     env)
    finally:
        env.close()

    return zero_rewards, c

if __name__ == "__main__":
    def eval_stat(pool, model_state, model_cls, opponent, epochs=100):
        all_data = pool.starmap(play, [(env_cfg, model_state, model_cls, opponent) for _ in range(epochs)])
        zero_rewards, steps = zip(*all_data)

        zero_rewards = torch.tensor(zero_rewards)
        steps = torch.tensor(steps, dtype=torch.float)

        return torch.mean(steps), torch.mean(zero_rewards), torch.std(zero_rewards, correction=1), torch.var(zero_rewards, correction=1)

    test_dir = "DQN_PDV"
    model_name_format = "{name}_{num}.pt"
    model_cls = MAFCQNoisyNet

    env_cfg = EnvCfg()

    opponent = RandomAgent(env_cfg.num_agents)

    ctx = get_context('spawn')
    pool = ctx.Pool(8)

    evol_epochs = 24500
    save_freq = 500

    model_count = evol_epochs // save_freq - 1
    runs = 10
    model_test_iterations = 100

    mean_steps_mat = torch.zeros([model_count, runs])
    mean_rewards_mat = torch.zeros([model_count, runs])
    stds_mat = torch.zeros([model_count, runs])
    vars_mat = torch.zeros([model_count, runs])

    for i in tqdm(range(model_count)):
        for j in range(runs):
            state_dict = torch.load(os.path.join(test_dir, os.listdir(test_dir)[j], "Models", "gradient", model_name_format.format(name=test_dir, num=i*save_freq)),
                                    weights_only=False)["model"]

            steps, reward, std, var = eval_stat(pool, state_dict, model_cls, opponent, epochs=model_test_iterations)

            mean_steps_mat[i, j] = steps
            mean_rewards_mat[i, j] = reward
            stds_mat[i, j] = std
            vars_mat[i, j] = var

    pool.terminate()
    pool.join()
    pool.close()
    del pool

    steps = mean_steps_mat.mean(-1).squeeze(-1)
    rewards = mean_rewards_mat.mean(-1).squeeze(-1)
    vars = vars_mat.mean(-1).squeeze(-1)
    stds = vars.sqrt()
    sem = stds / (runs**0.5)


    algo_stds = mean_rewards_mat.std(-1, correction=1)
    algo_vars = mean_rewards_mat.var(-1, correction=1)
    algo_sem = algo_stds / (runs**0.5)

    plt.style.use('fivethirtyeight')
    params = {
        'figure.figsize': (15, 8),
        'font.size': 24,
        'legend.fontsize': 20,
        'axes.titlesize': 28,
        'axes.labelsize': 24,
        'xtick.labelsize': 20,
        'ytick.labelsize': 20
    }
    pylab.rcParams.update(params)

    fig, axs = plt.subplots(2, 2, figsize=(20,30), sharey=False, sharex=True)
    X = list(range(0, evol_epochs, save_freq))[:-1]
    axs[0][0].plot(X, rewards, label='Мат. ожидание награды (Mean)', color='blue', linewidth=2)
    axs[0][0].set_title("Награда с учётом дисперсии среды и алгоритма")
    axs[0][0].legend(loc='best')

    axs[0][0].fill_between(X,
                     rewards - sem,
                     rewards + sem,
                     color='blue', alpha=0.2, label='Стандартное отклонение среды (1 std)')

    axs[0][1].plot(X, vars, label='Дисперсия среды (Variance)', color='red', linewidth=2)
    axs[0][1].set_title("Дисперсия с учётом стохастичности среды")
    axs[0][1].legend(loc='best')


    axs[1][0].plot(X, rewards, label='Мат. ожидание награды (Mean)', color='blue', linewidth=2)
    axs[1][0].set_title("Награда с учётом дисперсии алгоритма")
    axs[1][0].legend(loc='best')

    axs[1][0].fill_between(X,
                     rewards - algo_sem,
                     rewards + algo_sem,
                     color='blue', alpha=0.2, label='Стандартное отклонение алгоритма (1 std)')

    axs[1][1].plot(X, algo_vars, label='Дисперсия алгоритма (Variance)', color='red', linewidth=2)
    axs[1][1].set_title("Дисперсия алгоритма")
    axs[1][1].legend(loc='best')

    for row in axs:
        for ax in row:
            ax.set_xlabel('Episodes')

    plt.tight_layout()

    plt.savefig('result.png', dpi=300, bbox_inches='tight')
    plt.show()

    data = {
        "mean_steps": steps.tolist(),
        "mean_reward": rewards.tolist(),
        "env_std": stds.tolist(),
        "env_var": vars.tolist(),

        "algo_std": algo_stds.tolist(),
        "algo_var": algo_vars.tolist(),
    }
    with open("results.json", "w") as file:
        json.dump(data, file, indent=4)


