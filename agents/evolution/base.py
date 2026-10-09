import datetime
import os
from abc import abstractmethod
from dataclasses import dataclass, asdict

import numpy as np
import torch
from torch.multiprocessing import get_context

from agents.base import BaseAgent
from net import MAFCQNet, MAFCQNoisyNet
from utils import ma_fit_randomizer, ma_fit_competitive_coevolution, save_json


class BaseEvolutionAgent(BaseAgent):
    population: np.ndarray
    fitness: torch.FloatTensor

    def __init__(self, cfg: dataclass, env_cfg: dataclass):
        super().__init__(cfg, env_cfg)

        if cfg.fitness_type == "Randomizer":
            self.fit_fn = ma_fit_randomizer
        elif cfg.fitness_type == "Competitive-Coevolution":
            self.fit_fn = ma_fit_competitive_coevolution
        elif cfg.fitness_type == "Greedy":
            raise NotImplementedError("This type of opponent is not implemented")
        else:
            raise ValueError("Invalid fitness type")

        if cfg.model_type == "MAFCQNet":
            self.model_cls = MAFCQNet
        elif cfg.model_type == "MAFCQNoisyNet":
            self.model_cls = MAFCQNoisyNet
        else:
            raise ValueError("Invalid model type")

        self.ctx = get_context('spawn')
        self.pool = self.ctx.Pool(cfg.num_processes)
        self.population = self.generate_initial_population()

    def generate_initial_population(self):
        population = []
        for _ in range(self.cfg.population_size):
            solution = self.model_cls(self.env_cfg.flatten_state_shape, self.env_cfg.num_agents, self.env_cfg.action_space).state_dict()
            population.append(solution)
        return np.array(population)

    def fit(self):
        results = self.pool.starmap(self.fit_fn, [(idx, self.population, self.model_cls, self.env_cfg) for idx in range(len(self.population))])
        logs, fitness = zip(*results)
        self.fitness = torch.FloatTensor(fitness)
        all_logs = []
        for log in logs:
            all_logs += log

        return self.fitness, all_logs

    def train_epoch(self):
        metrics = {}

        self.fit()
        self.generate_new_population()

        self.best = self.population[self.fitness.argmax()]

        metrics['evolution/best_fitness'] = self.fitness.max().item()
        metrics['evolution/mean_fitness'] = self.fitness.mean().item()

        return metrics

    def callback(self, metrics, epoch):
        for metric in metrics.items():
            self.writer.add_scalar(*metric, epoch)

        if epoch % 5 == 0:
            state = {'info': "NNE-V1",
                     'date': datetime.datetime.now(),
                     'epochs': epoch,
                     'agent': "evolution",
                     'model': self.best,
                     }
            str_dir = os.path.join(self.root_dir, f'Models/evolution/{self.cfg.name}-{epoch}.pt')
            torch.save(state, str_dir)

    def train(self, epochs: int):
        save_json(asdict(self.cfg), os.path.join(self.root_dir, "evolution_config.json"))

        evolution_model_dir = os.path.join(self.root_dir, "Models/evolution")
        if not os.path.isdir(evolution_model_dir):
            os.makedirs(evolution_model_dir)

        super().train(epochs)

    @abstractmethod
    def generate_new_population(self):
        pass
