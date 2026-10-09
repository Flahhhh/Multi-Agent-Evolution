import datetime
import os
from copy import deepcopy
from dataclasses import dataclass, asdict

import numpy as np
import torch
from torch import softmax

from agents.evolution.base import BaseEvolutionAgent
from agents import DQN, DQNCfg, DQN_PDV, DQN_PDVCfg
from utils import save_json


class MERL(BaseEvolutionAgent):
    def __init__(self, cfg: dataclass, env_cfg: dataclass):
        super().__init__(cfg, env_cfg)

        if cfg.gradient_alg == "DQN":
            gradient_cfg = DQNCfg()
            self.gradient_alg = DQN(gradient_cfg, env_cfg)
        elif cfg.gradient_alg == "DQN_PDV":
            gradient_cfg = DQN_PDVCfg()
            self.gradient_alg = DQN_PDV(gradient_cfg, env_cfg)
        else:
            raise ValueError("Invalid gradient algorithm")

    def fit(self):
        fitness, logs = super().fit()

        for log in logs:
            self.gradient_alg.buffer.update_median_value()
            self.gradient_alg.buffer.add_samples(*log, use_median_weight=True)

        return fitness, logs

    def train_epoch(self):
        evolution_metrics = super().train_epoch()
        gradient_metrics = self.gradient_alg.train_epoch()

        return {**evolution_metrics, **gradient_metrics}

    def generate_new_population(self):
        new_population = np.array([None] * self.population_size)

        elite_ids = torch.topk(self.fitness, self.elite_amount).indices

        new_population[:self.elite_amount] = self.population[elite_ids]
        gradient_alg_params = self.gradient_alg.model.state_dict()

        for idx in range(self.elite_amount, self.population_size):
            solution = deepcopy(self.population[idx])
            relatives_idx = self._select_relatives_idx()
            relatives = self.population[relatives_idx]

            relatives_fitness = self.fitness[relatives_idx]
            weights = softmax(relatives_fitness, dim=0)
            gradient_param_weight = torch.rand(1) * self.gradient_param_scale

            with torch.no_grad():
                for key, p in solution.items():
                    params_stack = torch.stack([r[key] for r in relatives])
                    weights_expanded = weights.view(-1, *([1] * (params_stack.dim() - 1)))

                    new_p = torch.sum(weights_expanded * params_stack, dim=0) + torch.randn_like(p) * self.noise_scale
                    new_p = (1-gradient_param_weight) * new_p + gradient_alg_params[key] * gradient_param_weight

                    solution[key].copy_(
                        new_p
                    )

            new_population[idx] = solution

        self.population = new_population
        return new_population

    def callback(self, metrics, epoch):
        super().callback(metrics, epoch)

        if epoch % 5 == 0:
            state = {'info': "NNE-V1",
                     'date': datetime.datetime.now(),
                     'epochs': epoch,
                     'agent': "gradient",
                     'model': self.gradient_alg.model.state_dict(),
                     }
            str_dir = os.path.join(self.root_dir, f'Models/gradient/{self.cfg.name}-{epoch}.pt')
            torch.save(state, str_dir)

    def train(self, epochs: int):
        gradient_model_dir = os.path.join(self.root_dir, "Models/gradient")
        if not os.path.isdir(gradient_model_dir):
            os.makedirs(gradient_model_dir)

        save_json(asdict(self.gradient_alg.cfg), os.path.join(self.root_dir, "gradient_config.json"))

        super().train(epochs)
