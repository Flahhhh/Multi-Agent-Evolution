import os
import json

import torch
from matplotlib import pyplot as plt, pylab

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

results = {
    #"dqn_decomposition_evolution": "results/dqn_decomposition/evolution",
    #"dqn_decomposition_gradient": ("results/dqn_decomposition/gradient", "yellow"),

    #"dqn_independent_evolution": "results/dqn_independent/evolution",
    "DQN": (os.path.join("results", "DQN"), "red"),

    #"dqn_pdv_decomposition_evolution(pdv_lr_0001)": "results/dqn_pdv_decomposition/pdv_lr_0001/evolution",
    #"dqn_pdv_decomposition_gradient(pdv_lr_0001)": ("results/dqn_pdv_decomposition/pdv_lr_0001/gradient", "lightblue"),
    #"dqn_pdv_decomposition_evolution(pdv_lr_005)": "results/dqn_pdv_decomposition/pdv_lr_005/evolution",
    #"dqn_pdv_decomposition_gradient(pdv_lr_005)": ("results/dqn_pdv_decomposition/pdv_lr_005/gradient", "blue"),

    #"dqn_pdv_independent_evolution(pdv_lr_0001)": "results/dqn_pdv_independent/pdv_lr_0001/evolution",
    #"dqn_pdv_independent_gradient(pdv_lr_0001)": ("results/dqn_pdv_independent/pdv_lr_0001/gradient", "lightblue"),
    #"dqn_pdv_independent_evolution(pdv_lr_005)": "results/dqn_pdv_independent/pdv_lr_005/evolution",
    "DQN+PDV": (os.path.join("results", "DQN_PDV"), "blue"),
}
rr = {}

fig, axs = plt.subplots(2, 1, figsize=(20, 30), sharey=False, sharex=True)
runs = 10
save_freq = 500

X = None
for name, (path, color) in results.items():
    with open(os.path.join(path, 'results.json'), 'r', encoding='utf-8') as file:
        data = json.load(file)

        if X is None:
            X = torch.tensor(list(range(len(data["mean_reward"]))))*save_freq

    rr[name] = data

    sem = torch.tensor(data["env_std"]) / (runs**0.5)
    rewards = torch.tensor(data["mean_reward"])

    axs[0].fill_between(X,
                     rewards - sem,
                     rewards + sem,
                     color=color, alpha=0.2)
    axs[0].plot(X, rewards, label=f'Мат. ожидание награды ({name})', color=color, linewidth=2)
    axs[0].set_title("Награда с учётом дисперсии среды и алгоритма")
    axs[0].legend(loc='best')

    axs[1].plot(X, data["mean_steps"], label=f'Число шагов ({name})', color=color, linewidth=2)
    axs[1].set_title("Число шагов")
    axs[1].legend(loc='best')

    """
    axs[1][0].plot(X, data["env_var"], label=f'Дисперсия с учётом среды ({name})', color=color, linewidth=2)
    axs[1][0].set_title("Дисперсия")
    axs[1][0].legend(loc='best')

    axs[1][1].plot(X, data["algo_var"], label=f'Дисперсия обучения ({name})', color=color, linewidth=2)
    axs[1][1].set_title("Дисперсия")
    axs[1][1].legend(loc='best')
    """

    #plt.plot(X, data["env_std"], label=name, color=color)


plt.legend(loc="best")
plt.show()

print(rr)
