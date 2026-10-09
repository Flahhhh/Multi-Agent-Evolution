import datetime
import os
from abc import abstractmethod
from dataclasses import dataclass, asdict
from copy import deepcopy

import torch
from torch.nn import HuberLoss, MSELoss
from torch.optim import AdamW, Adam, RMSprop

from MABattle.utils import play_game
from agents.base import BaseAgent
from buffer import ReplayBuffer
from utils import RandomAgent, avg, save_json
from net import MAFCQNoisyNet


class BaseGradientAgent(BaseAgent):
    def __init__(self, cfg: dataclass, env_cfg: dataclass, use_pdv_buffer: bool = False):
        super().__init__(cfg, env_cfg)

        if self.cfg.optimizer_name == "Adam":
            self.optimizer_cls = lambda p, lr: Adam(p, lr=lr, amsgrad=True)
        elif self.cfg.optimizer_name == "AdamW":
            self.optimizer_cls = lambda p, lr: AdamW(p, lr=lr, amsgrad=True)
        elif self.cfg.optimizer_name == "RMSprop":
            self.optimizer_cls = RMSprop
        else:
            raise ValueError("Invalid optimizer type")

        if self.cfg.opponent_type == "Random":
            self.opponent_cls = lambda: RandomAgent(self.env_cfg.num_agents)
        elif self.opponent_cls == "Greedy":
            raise NotImplementedError("This type of opponent is not implemented")
        else:
            raise ValueError("Invalid opponent type")

        if self.cfg.loss == "HuberLoss":
            self.loss_fn = HuberLoss()
        elif self.cfg.loss == "MSELoss":
            self.loss_fn = MSELoss()
        else:
            raise ValueError("Invalid loss function type")

        self.model = MAFCQNoisyNet(env_cfg.flatten_state_shape, env_cfg.num_agents, env_cfg.action_space,
                                   sigma_init=self.cfg.sigma_init)
        self.model_twin = MAFCQNoisyNet(env_cfg.flatten_state_shape, env_cfg.num_agents, env_cfg.action_space,
                                        sigma_init=self.cfg.sigma_init)

        self.model_target = deepcopy(self.model)
        self.model_target_twin = deepcopy(self.model_twin)

        self.model_optimizer = self.optimizer_cls(self.model.parameters(), self.cfg.lr)
        self.model_optimizer_twin = self.optimizer_cls(self.model_twin.parameters(), self.cfg.lr)

        self.buffer = ReplayBuffer(alpha=self.cfg.init_alpha, size=self.cfg.buffer_size, num_agents=self.env_cfg.num_agents,
                                   flatten_state_shape=self.env_cfg.flatten_state_shape, batch_size=self.cfg.batch_size,
                                   device=self.cfg.device, is_pdv=use_pdv_buffer)

        self.gamma_2 = self.gamma ** 2

    def train_epoch(self) -> dict:
        metrics = {}

        for _ in range(self.cfg.num_epoch_steps):
            for _ in range(self.cfg.num_step_game):
                game_metrics = self.play_game()
                for metric, value in game_metrics.items():
                    if metric not in metrics:
                        metrics[metric] = []
                    metrics[metric].append(value)

            if self.cfg.learning_starts < self.buffer.entries:
                for _ in range(self.cfg.num_step_train):
                    train_metrics = self.update_network()
                    for metric, value in train_metrics.items():
                        if metric not in metrics:
                            metrics[metric] = []
                        metrics[metric].append(value)

                    self.update_targets()

        metrics = {metric: avg(values) for metric, values in metrics.items()}

        snrs, mean_snr = self.model.get_snr()
        snrs_twin, mean_snr_twin = self.model.get_snr()
        for i in range(len(snrs)):
            metrics[f"gradient_snr/snr_layer_{i}"] = snrs[i]
            metrics[f"gradient_snr/snr_layer_{i}_twin"] = snrs_twin[i]
        metrics["gradient_snr/mean_snr"] = mean_snr
        metrics["gradient_snr/mean_snr_twin"] = mean_snr_twin

        return metrics

    def _play_single_game(self, model, opponent) -> tuple[float, float, float, float, float, float, int]:
        logs, _, rewards, zero_rewards, step_discounted_reward, pdv_reward, c = play_game(model, opponent,
                                                                                          self.env)
        for log in logs:
            self.buffer.add_samples(*log, use_median_weight=False)

        return rewards, rewards / c, zero_rewards, zero_rewards / c, step_discounted_reward, pdv_reward, c

    """reward, mean_reward, zero_reward, mean_zero_reward, step_discounted_reward, pdv_reward, steps"""

    def play_game(self) -> dict:
        opponent = self.opponent_cls()

        reward, mean_reward, zero_reward, mean_zero_reward, step_discounted_reward, pdv_reward, steps = \
            map(avg, zip(self._play_single_game(self.model, opponent),
                         self._play_single_game(self.model_twin, opponent))
                )

        return {
            "gradient/sum_reward": reward, "gradient/mean_reward": mean_reward,
            "gradient/sum_zero_reward": zero_reward, "gradient/mean_zero_reward": mean_zero_reward,
            "gradient/step_discounted_reward": step_discounted_reward, "gradient/pdv_reward": pdv_reward, "gradient/epoch_steps": steps
        }

    def update_targets(self):
        with torch.no_grad():
            for p, p_ in zip(self.model.parameters(), self.model_target.parameters()):
                p.copy_((1 - self.cfg.tau) * p + self.cfg.tau * p_)

            for p, p_ in zip(self.model_twin.parameters(), self.model_target_twin.parameters()):
                p.copy_((1 - self.cfg.tau) * p + self.cfg.tau * p_)

    def callback(self, metrics, epoch):
        for metric in metrics.items():
            self.writer.add_scalar(*metric, epoch)

        if epoch % 500 == 0:
            state = {'info': "NNE-V1",  # описание
                     'date': datetime.datetime.now(),  # дата и время
                     'epochs': epoch,
                     'agent': "gradient",
                     'model': self.model.state_dict(),
                     }
            str_dir = os.path.join(self.root_dir, f'Models/gradient/{self.cfg.name}_{epoch}.pt')
            torch.save(state, str_dir)

    def train(self, epochs: int):
        save_json(asdict(self.cfg), os.path.join(self.root_dir, "gradient_config.json"))

        gradient_model_dir = os.path.join(self.root_dir, "Models/gradient")
        if not os.path.isdir(gradient_model_dir):
            os.makedirs(gradient_model_dir)

        super().train(epochs)

    @abstractmethod
    def _update_single_network(self, *args, **kwargs) -> tuple:
        pass

    @abstractmethod
    def update_network(self) -> dict:
        pass
