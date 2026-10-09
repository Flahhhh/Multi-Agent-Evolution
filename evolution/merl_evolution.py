import random
from copy import deepcopy
#from multiprocessing import get_context

from torch.multiprocessing import get_context

import numpy as np
from tqdm import tqdm

import torch
from torch.nn.functional import softmax

from evolution.config import MERLCfg
from rl.agents import DQN, DQNCfg, DQN_PDV, DQN_PDVCfg
from utils import RandomAgent, process_log_lists
from const import device_name


base_cfg = MERLCfg()

class Evolution:
    def load_config(self, cfg):
        self.cfg = cfg

        for attr, value in vars(cfg).items():
            setattr(self, attr, value)

    def __init__(self, epochs, model_cls, fitness_fn, population_size, device, cfg=base_cfg, callback_fn=None):
        if device is None:
            self.device = torch.device(device_name)
        else:
            self.device = device

        self.load_config(cfg)

        #self.num_processes = 8
        #self.pool = Pool(self.num_processes)

        self.population_size = population_size
        self.model_cls = model_cls
        self.device = device
        self.fitness_fn = fitness_fn
        self.callback_fn = callback_fn

        self.population = self.generate_initial_population()
        self.fitness = torch.FloatTensor([0] * self.population_size)

        self.epoch = 0
        #self.elite_amount = 16
        #self.ddpg_amount = 1

        #self.num_relatives = 4
        #self.init_noise_scale = 0.3
        #self.gradient_param_scale = 0.005

        self.noise_scale = self.init_noise_scale
        #self.min_noise_scale = 0.015
        self.mutation_amount = self.population_size - self.elite_amount
        self.best = RandomAgent()

        self.gradient_alg = DQN_PDV(DQN_PDVCfg(), epochs, device=self.device)

        ctx = get_context('spawn')  # или 'forkserver'
        self.pool = ctx.Pool(self.num_processes)

    def _update_noise_scale(self):
        self.noise_scale -= (self.init_noise_scale - self.min_noise_scale) * self.epoch

    def generate_initial_population(self):
        population = []
        for _ in range(self.population_size):
            solution = self.model_cls().state_dict() #.to(self.device)
            # Добавьте небольшой случайный шум к весам
            with torch.no_grad():
                for key, param in solution.items():
                    solution[key].add_(torch.randn_like(param) * 0.1)
            population.append(solution)
        return np.array(population)

    # def generate_initial_population(self):
    #    return np.array([self.model_cls().to(self.device) for _ in range(self.population_size)])

    def fit(self):

        try:
            self.gradient_alg.buffer.update_median_value()
            #print(self.population[0])
            #print(random.choice(self.population))
            results = self.pool.starmap(self.fitness_fn, [(solution, self.best, self.model_cls) for solution in self.population])
            #results = pool.starmap(self.fitness_fn, [(model, random.choice(self.population), self.device) for model in self.population])


            logs_ = [r[0] for r in results]
            self.fitness = torch.tensor([r[1] for r in results])
            #print(logs_)
            logs = []
            #print("AAAAAA")
            for log in logs_:
                logs += log
            for log in logs:
                self.gradient_alg.buffer.add_samples(*log)

        finally:
            pass
            #self.pool.terminate()  # принудительно убить всех воркеров
            #self.pool.join()
            #self.pool.close()

        #with Pool(self.num_processes) as pool:
        #    self.fitness = torch.tensor(
        #        pool.starmap(self.fitness_fn, [(model, random.choice(self.population), self.gradient_alg.buffer) for model in self.population]))
        #        #pool.starmap(self.fitness_fn, [(model, self.gradient_alg.buffer, self.best) for model in self.population]))

        # print(self.fitness)
        # self.update_buffer(logs)
        self.best = self.population[self.fitness.argmax()]

    def update_buffer(self, logs):
        states, actions, next_states, rewards, dones = logs["states"], logs["actions"], logs["next_states"], logs[
            "rewards"], logs["dones"]
        self.gradient_alg.buffer.add_samples(
            process_log_lists(states), process_log_lists(actions), process_log_lists(next_states),
            process_log_lists(rewards), process_log_lists(dones)
        )

    def get_best(self):
        return self.best

    def generate_new_population(self):
        # new_population = self.population
        new_population = np.array([None] * self.population_size)

        elite_ids = torch.topk(self.fitness, self.elite_amount).indices

        new_population[:self.elite_amount] = self.population[elite_ids]
        #new_population[self.elite_amount:self.elite_amount + self.ddpg_amount] = self.gradient_alg.get_solutions(
        #    self.ddpg_amount)

        gradient_alg_params = self.gradient_alg.get_solution()

        for idx in range(self.elite_amount, self.population_size):
            solution = deepcopy(self.population[idx])
            relatives_idx = self._select_relatives_idx()
            relatives = self.population[relatives_idx]

            relatives_fitness = self.fitness[relatives_idx]
            weights = softmax(relatives_fitness, dim=0)#.to(self.device)
            gradient_param_weight = torch.rand(1) * self.gradient_param_scale

            with torch.no_grad():
                for key, p in solution.items():
                    #print(key, p)
                    params_stack = torch.stack([r[key] for r in relatives])
                    weights_expanded = weights.view(-1, *([1] * (params_stack.dim() - 1)))

                    new_p = torch.sum(weights_expanded * params_stack, dim=0) + torch.randn_like(p) * self.noise_scale
                    new_p = (1-gradient_param_weight) * new_p + gradient_alg_params[key] * gradient_param_weight

                    solution[key].copy_(
                        new_p
                        #torch.sum(weights_expanded * params_stack, dim=0) +
                        #torch.randn_like(p) * self.noise_scale
                    )
                    # p.copy_(
                    #    torch.mean(torch.stack([list(r.parameters())[i] for r in relatives]), dim=0) +
                    #    torch.randn_like(p) * self.noise_scale
                    # )

            new_population[idx] = solution

        self.population = new_population
        return new_population

    def _get_probs(self):
        return softmax(self.fitness, dim=0)

    def _select_relatives_idx(self):
        return torch.multinomial(self._get_probs(), self.num_relatives, replacement=False)

    def train(self, epochs):
        self.epochs = epochs
        pbar = tqdm(range(epochs))

        try:
            for epoch in pbar:
                self.epoch = epoch
                self.fit()
                self.metrics = self.gradient_alg.train_epoch()
                #self.ddpg_sum_reward, self.ddpg_mean_reward, self.ddpg_sum_zero_reward, self.ddpg_mean_zero_reward, self.ddpg_loss, self.ddpg_pdv_loss, self.ddpg_c, \
                #    self.snrs, self.mean_snr, self.snrs_twin, self.mean_snr_twin, self.ddpg_step_discounted_reward = self.gradient_alg.train_epoch()
                self.generate_new_population()

                best_fitness, mean_fitness = float(self.fitness.max()), float(self.fitness.mean())
                self.metrics["evolution/best_fitness"] = best_fitness
                self.metrics["evolution/mean_fitness"] = mean_fitness
                self.metrics["evolution/noise_scale"] = self.noise_scale

                if self.callback_fn is not None:
                    self.callback_fn(self)

                pbar.set_postfix({"BEST": best_fitness, "MEAN": mean_fitness})
                # f"BEST: {self.fitness.max()} | MEAN: {self.fitness.mean()}")
                # print(f"[INFO] Epoch {epoch}: {self.fitness.max()}")
        finally:
            self.pool.terminate()  # принудительно убить всех воркеров
            self.pool.join()
            self.pool.close()

            del self.pool
