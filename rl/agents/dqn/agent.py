import copy
import random
from copy import deepcopy
import torch
from torch.nn import HuberLoss, MSELoss
from torch.optim import AdamW, Adam

from MABattle.utils import make_env, play_game
from net import MAFCQNet, MAFCVNet, MAFCQNoisyNet, MAFCVNoisyNet
from const import device_name
from rl.agents.dqn.config import DQNCfg
from rl.base import BaseAgent
from utils import RandomAgent

from rl.buffer import ReplayBuffer


class DQN(BaseAgent):
    def __init__(self, cfg: DQNCfg, epochs: int, env=None, device=None):
        self.load_config(cfg)

        if device is None:
            self.device = torch.device(device_name)
        else:
            self.device = device

        if env is None:
            self.env = make_env()
        else:
            self.env = env

        self.model = MAFCQNoisyNet(sigma_init=self.sigma_init).to(self.device)
        self.model_twin = MAFCQNoisyNet(sigma_init=self.sigma_init).to(self.device)

        self.model_target = deepcopy(self.model).to(self.device)
        self.optimizer = AdamW(self.model.parameters(), lr=self.lr, amsgrad=True)

        self.model_target_twin = deepcopy(self.model_twin).to(self.device)
        self.optimizer_twin = AdamW(self.model_twin.parameters(), lr=self.lr, amsgrad=True)

        # self.opponent_model = MAFCQNet().to(self.device)
        # self.opponent_optimizer = AdamW(self.opponent_model.parameters(), lr=0.0001, amsgrad=True)

        self.buffer = ReplayBuffer(alpha=self.init_alpha, size=self.buffer_size, batch_size=self.batch_size,
                                   device=self.device)

        # self.gamma = 0.99
        self.gamma_2 = self.gamma ** 2

        # self.tau = 0.0001
        # self.target_update_freq = 1024

        if self.loss == "HuberLoss":
            self.loss_fn = HuberLoss()
        elif self.loss == "MSELoss":
            self.loss_fn = MSELoss()

        # self.num_epoch_steps = 1
        # self.num_step_game = 32
        # self.num_step_train = 16

        self.epochs = epochs
        self.agent_train_epoch = 0
        self.agent_play_epoch = 0

        self.value_weight = 1

    def _update_targets(self):
        with torch.no_grad():
            for p, p_ in zip(self.model.parameters(), self.model_target.parameters()):
                p_.copy_(self.tau * p + (1 - self.tau) * p_)
        with torch.no_grad():
            for p, p_ in zip(self.model_twin.parameters(), self.model_target_twin.parameters()):
                p_.copy_(self.tau * p + (1 - self.tau) * p_)


    def _update_networks(self, agent_id, model, model_optimizer, model_target):
        # ids, states, actions, next_states, rewards, finishes, probs = self.buffer.sample()
        ids, states, actions, rewards, pre_finishes, post_rewards, finishes, next_states = self.buffer.sample(agent_id)

        # print("PREV:",states.shape, actions.shape, rewards.shape)
        # print("OPPONENT:",opponent_finishes.shape, opponent_next_states.shape, opponent_actions.shape)
        # print("UNIT:",unit_finishes.shape, unit_next_states.shape)

        # print(states.shape, actions.shape, rewards.shape,
        #      opponent_finishes.shape, opponent_next_states.shape, opponent_actions.shape,
        #      unit_finishes.shape, unit_next_states.shape)
        # q_values = torch.gather(self.model(states), dim=1, index=actions).squeeze()
        # print("QVVV", q_values.shape)

        # opponent_predictions = self.opponent_model(opponent_next_states)

        # opponent_actions = self.model(opponent_next_states).argmax(dim=-1)
        # opponent_prediction_values = self.model_target(opponent_next_states)
        # opponent_values = torch.gather(opponent_prediction_values, dim=-1,
        #                                index=opponent_actions.unsqueeze(-1)).squeeze(-1).detach()
        # opponent_actions = self.model(opponent_next_states).argmax(-1)

        # q_target = rewards + self.gamma_2 * ((1 - unit_finishes) * unit_values.detach())
        # q_target = rewards - self.gamma * ((1 - opponent_finishes) * opponent_rewards) + self.gamma_2 * (
        #        (1 - unit_finishes) * unit_values.detach())  # .sum(-1)
        # q_target = rewards - \
        #           self.gamma * self.value_weight * ((1 - opponent_finishes) * opponent_rewards) - \
        #           self.gamma * (1-self.value_weight) * ((1 - opponent_finishes) * opponent_values) + \
        #           self.gamma_2 * ((1 - unit_finishes) * unit_values.detach())

        # post_decision_values = self.post_decision_model(opponent_next_states).squeeze(-1)

        #post_decision_values_other = pdv_model_other(opponent_next_states).squeeze(-1)


        next_actions = model(next_states).argmax(-1)
        next_predictions = model_target(next_states)
        next_values = torch.gather(next_predictions, dim=-1, index=next_actions.unsqueeze(1)).squeeze()
        # q_target = rewards - \
        #            self.gamma * ((1 - opponent_finishes) * opponent_values.detach())
        # print("III",rewards.shape, opponent_finishes.shape, post_decision_values.shape)
        q_target = rewards + self.gamma * (1 - pre_finishes) * post_rewards + self.gamma_2 * (1-finishes) * next_values.detach()
        q_values = torch.gather(model(states), dim=-1, index=actions.unsqueeze(-1)).squeeze(-1)

        if self.multi_agent_mode == "Independent":
            loss = self.loss_fn(q_values, q_target)
        elif self.multi_agent_mode == "Decomposition":
            loss = self.loss_fn(q_values.sum(-1), q_target.sum(-1))
        else:
            raise ValueError()

        """
        if self.summarization_mode == "CEN":
            q_values = torch.gather(self.model(states), dim=-1, index=actions.unsqueeze(-1)).squeeze(-1).sum(-1)
            if self.target == "I":
                q_target = rewards.sum(-1) + \
                           self.gamma * ((1 - opponent_finishes) * opponent_rewards).sum(-1) + \
                           self.gamma_2 * ((1 - unit_finishes) * unit_values.detach()).sum(-1)
            elif self.target == "II":
                opponent_prediction_values = self.model_target(opponent_next_states)
                opponent_values = torch.gather(opponent_prediction_values, dim=-1,
                                               index=opponent_actions.unsqueeze(-1)).squeeze(-1).detach()

                q_target = rewards.sum(-1) + \
                           self.gamma * self.value_weight * ((1 - opponent_finishes) * opponent_rewards).sum(-1) + \
                           self.gamma * (1-self.value_weight) * ((1 - opponent_finishes) * opponent_values).sum(-1) + \
                           self.gamma_2 * ((1 - unit_finishes) * unit_values.detach()).sum(-1)
            elif self.target == "III":
                opponent_prediction_values = self.model_target(opponent_next_states)
                opponent_values = torch.gather(opponent_prediction_values, dim=-1,
                                               index=opponent_actions.unsqueeze(-1)).squeeze(-1).detach()

                q_target = rewards.sum(-1) + \
                           self.gamma * ((1 - opponent_finishes) * opponent_values).sum(-1) + \
                           self.gamma_2 * ((1 - unit_finishes) * unit_values.detach()).sum(-1)
            loss = self.loss_fn(q_values, q_target)

            td_errors = (q_target-q_values).abs()
        elif self.summarization_mode == "DEC":
            q_values = torch.gather(self.model(states), dim=-1, index=actions.unsqueeze(-1)).squeeze(-1)

            if self.target == "I":
                q_target = rewards + \
                           self.gamma * ((1 - opponent_finishes) * opponent_rewards) + \
                           self.gamma_2 * ((1 - unit_finishes) * unit_values.detach())
            elif self.target == "II":
                opponent_prediction_values = self.model_target(opponent_next_states)
                opponent_values = torch.gather(opponent_prediction_values, dim=-1,
                                               index=opponent_actions.unsqueeze(-1)).squeeze(-1).detach()

                q_target = rewards + \
                           self.gamma * self.value_weight * ((1 - opponent_finishes) * opponent_rewards) + \
                           self.gamma * (1-self.value_weight) * ((1 - opponent_finishes) * opponent_values) + \
                           self.gamma_2 * ((1 - unit_finishes) * unit_values.detach())
            elif self.target == "III":
                opponent_prediction_values = self.model_target(opponent_next_states)
                opponent_values = torch.gather(opponent_prediction_values, dim=-1,
                                               index=opponent_actions.unsqueeze(-1)).squeeze(-1).detach()

                q_target = rewards + \
                           self.gamma * ((1 - opponent_finishes) * opponent_values) + \
                           self.gamma_2 * ((1 - unit_finishes) * unit_values.detach())
            loss = self.loss_fn(q_values, q_target)

            td_errors = (q_target.sum(-1)-q_values.sum(-1)).abs()
        """

        # q_target = rewards - self.gamma * ((1 - opponent_finishes) * opponent_values.detach()) + self.gamma_2 * (
        #            (1 - unit_finishes) * unit_values.detach())  # .sum(-1)
        # q_target = rewards - self.gamma * ((1-opponent_finishes.float()) * opponent_values.detach()).sum(-1) + self.gamma_2 * ((1-unit_finishes.float()) * unit_values.detach()).sum(-1)
        # print("QTTT", q_target.shape)

        # print(q_values.shape, opponent_prediction_values.shape, opponent_values.shape, next_actions.shape, unit_predictions.shape, unit_values.shape, q_values.shape)


        model_optimizer.zero_grad()
        loss.backward()
        model_optimizer.step()

        td_errors = (q_target.sum(-1) - q_values.sum(-1)).abs().detach()
        self.buffer.set_weights(agent_id, ids, td_errors)

        """
        opponent_predictions = self.opponent_model(opponent_next_states)
        opponent_targets = torch.zeros_like(opponent_predictions, device=self.device)
        ones_for_scatter = torch.ones_like(opponent_actions, dtype=opponent_predictions.dtype,
                                           device=self.device)  # [num_selected_opp_alive]

        #opponent_targets[opponent_actions.unsqueeze(0)] = 1
        #print(opponent_targets.shape, opponent_actions.shape, ones_for_scatter.shape)
        opponent_targets = opponent_targets.scatter_(dim=-1, index=opponent_actions.unsqueeze(-1),
                                                                   src=ones_for_scatter.unsqueeze(-1))
        #print(opponent_targets)

        #opponent_targets = torch.gather(opponent_targets, dim=1, index=opponent_actions.unsqueeze(1)).squeeze()
        opponent_loss = self.loss_fn(opponent_predictions, opponent_targets)

        self.opponent_optimizer.zero_grad()
        opponent_loss.backward()
        self.opponent_optimizer.step()
        """

        return loss.abs()

    def _play_game(self, model):
        logs, _, rewards, zero_rewards, step_discounted_reward, pdv_reward, c = play_game(model, RandomAgent(), self.env)
        for log in logs:
            self.buffer.add_samples(*log, use_median_weight=False)
        """
        env = self.env
        agent_turn = random.choice((-1, 1))

        models = {
            agent_turn: self.actor,
            -agent_turn: RandomAgent()
        }

        finished = False
        state, info = env.reset()
        rewards = 0.
        c = 0

        while not finished:
            legals = info["legals"]
            to_play = env.unwrapped.to_play()
            actions, logits = models[to_play].get_actions(state.to(self.device), legals.to(self.device))

            # print(c, "SSSSSS")
            next_state, reward, finished, truncated, info = env.step(actions)
            finished = finished or truncated

            alive_agents = info["alive_agents"]
            #print(logits, logits[alive_agents], alive_agents)
            state = next_state

            if to_play==agent_turn:
                self.buffer.add_samples(state[alive_agents], logits[alive_agents], next_state[alive_agents], reward, torch.FloatTensor([finished]*len(reward)))
                rewards += sum(reward)
                c += 1
        """

        return rewards, rewards / c, zero_rewards, zero_rewards / c, step_discounted_reward, pdv_reward, c

    def train_epoch(self):
        metrics = {}

        sr, mr, szr, mzr, cs = [], [], [], [], []
        pdvr = []
        sdr = []
        loss = []

        for i in range(self.num_epoch_steps):
            for j in range(self.num_step_game):
                sum_reward_1, mean_reward_1, sum_zero_rewards_1, mean_zero_rewards_1, step_discounted_reward_1, pdv_reward_1, c_1 = self._play_game(self.model)
                sum_reward_2, mean_reward_2, sum_zero_rewards_2, mean_zero_rewards_2, step_discounted_reward_2, pdv_reward_2, c_2 = self._play_game(
                    self.model_twin)
                sr.append((sum_reward_1 + sum_reward_2) / 2)
                mr.append((mean_reward_1 + mean_reward_2) / 2)
                szr.append((sum_zero_rewards_1 + sum_zero_rewards_2) / 2)
                mzr.append((mean_zero_rewards_1 + mean_zero_rewards_2) / 2)
                pdvr.append((pdv_reward_1+pdv_reward_2)/2)
                cs.append((c_1 + c_2) / 2)
                sdr.append((step_discounted_reward_1 + step_discounted_reward_2) / 2)
                self.agent_play_epoch += 1
                self._update_eps()

            for j in range(self.num_step_train):
                loss_1 = self._update_networks(1, self.model, self.optimizer, self.model_target_twin)
                loss_2 = self._update_networks(2, self.model_twin, self.optimizer_twin, self.model_target)

                if self.grow_alpha:
                    self.buffer.alpha = min(self.max_alpha,
                                            self.init_alpha + (1 - self.init_alpha) * self.agent_train_epoch / (
                                                    self.epochs * self.num_epoch_steps * self.num_step_train * self.alpha_grow_ratio))

                loss.append((loss_1 + loss_2) / 2)

                self.agent_train_epoch += 1

                self._update_value_weight()
                if self.agent_train_epoch % self.target_update_freq == 0:
                    self._update_targets()

        snrs, mean_snr = self.model.get_snr()
        snrs_twin, mean_snr_twin = self.model_twin.get_snr()

        for i in range(len(snrs)):
            metrics[f"gradient_snr/snr_layer_{i}"] = snrs[i]
            metrics[f"gradient_snr/snr_layer_{i}_twin"] = snrs_twin[i]
        metrics["gradient_snr/mean_snr"] = mean_snr
        metrics["gradient_snr/mean_snr_twin"] = mean_snr_twin
        metrics["gradient/sum_reward"] = sum(sr) / len(sr)
        metrics["gradient/mean_reward"] = sum(mr) / len(mr)
        metrics["gradient/sum_zero_reward"] = sum(szr) / len(szr)
        metrics["gradient/mean_zero_reward"] = sum(mzr) / len(mzr)
        metrics["gradient/step_discounted_reward"] = sum(sdr) / len(sdr)
        metrics["gradient/pdv_reward"] = sum(pdvr) / len(pdvr)
        metrics["gradient/epoch_steps"] = sum(cs) / len(cs)
        metrics["gradient/loss"] = sum(loss) / len(loss)
        metrics["gradient/epsilon"] = self.model.eps
        metrics["gradient/buffer_alpha"] = self.buffer.alpha

        return metrics

    def _update_value_weight(self):
        self.value_weight -= 1 / (self.epochs * self.num_epoch_steps * self.num_step_train)

    def _update_eps(self):
        """
        self.model.eps = max(self.init_eps * (1 - self.agent_play_epoch / (
                    self.epochs * self.num_epoch_steps * self.num_step_game * self.decay_ratio)), self.min_eps)
        self.model_twin.eps = max(self.init_eps * (1 - self.agent_play_epoch / (
                    self.epochs * self.num_epoch_steps * self.num_step_game * self.decay_ratio)), self.min_eps)
        """

    def get_solutions(self, amount: int):
        return [copy.deepcopy(self.model).cpu().state_dict() for _ in range(amount)]

    def get_solution(self):
        return copy.deepcopy(self.model).cpu().state_dict()
