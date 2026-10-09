import os
import json

from matplotlib import pyplot as plt

results = {
    #"dqn_decomposition_evolution": "results/dqn_decomposition/evolution",
    "dqn_decomposition_gradient": ("results/dqn_decomposition/gradient", "yellow"),

    #"dqn_independent_evolution": "results/dqn_independent/evolution",
    "dqn_independent_gradient": ("results/dqn_independent/gradient", "red"),

    #"dqn_pdv_decomposition_evolution(pdv_lr_0001)": "results/dqn_pdv_decomposition/pdv_lr_0001/evolution",
    "dqn_pdv_decomposition_gradient(pdv_lr_0001)": ("results/dqn_pdv_decomposition/pdv_lr_0001/gradient", "lightblue"),
    #"dqn_pdv_decomposition_evolution(pdv_lr_005)": "results/dqn_pdv_decomposition/pdv_lr_005/evolution",
    "dqn_pdv_decomposition_gradient(pdv_lr_005)": ("results/dqn_pdv_decomposition/pdv_lr_005/gradient", "blue"),

    #"dqn_pdv_independent_evolution(pdv_lr_0001)": "results/dqn_pdv_independent/pdv_lr_0001/evolution",
    "dqn_pdv_independent_gradient(pdv_lr_0001)": ("results/dqn_pdv_independent/pdv_lr_0001/gradient", "lightblue"),
    #"dqn_pdv_independent_evolution(pdv_lr_005)": "results/dqn_pdv_independent/pdv_lr_005/evolution",
    "dqn_pdv_independent_gradient(pdv_lr_005)": ("results/dqn_pdv_independent/pdv_lr_005/gradient", "blue"),
}
rr = {}

X = None
for name, (path, color) in results.items():
    with open(os.path.join(path, 'results.json'), 'r', encoding='utf-8') as file:
        data = json.load(file)

        if X is None:
            X = list(range(len(data["mean_reward"])))

    rr[name] = data
    plt.plot(X, data["env_std"], label=name, color=color)

plt.legend(loc="best")
plt.show()

print(rr)
