from dataclasses import dataclass
import numpy as np
import torch

from agents.base import BaseAgent


class BaseEvolutionAgent(BaseAgent):
    population: np.array()

    def __init__(self, cfg: dataclass):
        super().__init__(cfg)
        self.generate_initial_population()

    def generate_initial_population(self):
        population = []
        for _ in range(self.population_size):
            solution = self.model_cls().state_dict()
            with torch.no_grad():
                for key, param in solution.items():
                    solution[key].add_(torch.randn_like(param) * 0.1)
            population.append(solution)
        return np.array(population)

    def fit(self):
        ...

    def generate_new_population(self):
        ...

    def train_epoch(self):
        self.fit()
        self.generate_new_population()
