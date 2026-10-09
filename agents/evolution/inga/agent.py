from copy import deepcopy

import numpy as np
import torch
from torch import softmax

from agents.evolution.base import BaseEvolutionAgent


class InGA(BaseEvolutionAgent):
    def generate_new_population(self):
        new_population = np.array([None] * self.population_size)

        elite_ids = torch.topk(self.fitness, self.elite_amount).indices

        new_population[:self.elite_amount] = self.population[elite_ids]

        for idx in range(self.elite_amount, self.population_size):
            solution = deepcopy(self.population[idx])
            relatives_idx = self._select_relatives_idx()
            relatives = self.population[relatives_idx]

            relatives_fitness = self.fitness[relatives_idx]
            weights = softmax(relatives_fitness, dim=0)

            with torch.no_grad():
                for key, p in solution.items():
                    params_stack = torch.stack([r[key] for r in relatives])
                    weights_expanded = weights.view(-1, *([1] * (params_stack.dim() - 1)))

                    new_p = torch.sum(weights_expanded * params_stack, dim=0) + torch.randn_like(p) * self.noise_scale

                    solution[key].copy_(
                        new_p
                    )

            new_population[idx] = solution

        self.population = new_population
        return new_population