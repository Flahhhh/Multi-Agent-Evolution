import copy

import torch
from torch.optim import AdamW, Adam, RMSprop

from MABattle.utils import play_game
from net import MAFCQNet, MAPolicyNet
from rl.buffer import ReplayBuffer
from utils import RandomAgent


class SAC:
    def __init__(self):
        if self.device_name == "cpu":
            self.device = torch.device("cpu")
        elif self.device_name == "cuda":
            self.device = torch.device("cuda:0")

        self.q1_net = MAFCQNet()
        self.q1_net_target = copy.deepcopy(self.q1_net)

        self.q2_net = MAFCQNet()
        self.q2_net_target = copy.deepcopy(self.q2_net)

        self.p_net = MAPolicyNet()

        self.alpha = torch.tensor(self.default_alpha, dtype=torch.float32, requires_grad=True, device=self.device)
        self.alpha_optimizer = AdamW([self.alpha], lr=self.alpha_lr, amsgrad=True)

        if self.q_optimizer == "Adam":
            q_optimizer = lambda p, lr: Adam(p, lr=lr, amsgrad=True)
        elif self.q_optimizer == "AdamW":
            q_optimizer = lambda p, lr: AdamW(p, lr=lr, amsgrad=True)
        elif self.q_optimizer == "RMSprop":
            q_optimizer = RMSprop
        else:
            raise ValueError("Неизвестный оптимизатор функции ценности")

        self.q1_optimizer = q_optimizer(self.q1_net.parameters(), lr=self.q_lr)
        self.q2_optimizer = q_optimizer(self.q2_net.parameters(), lr=self.q_lr)

        if self.policy_optimizer == "Adam":
            policy_optimizer = lambda p, lr: Adam(p, lr=lr, amsgrad=True)
        elif self.policy_optimizer == "AdamW":
            policy_optimizer = lambda p, lr: AdamW(p, lr=lr, amsgrad=True)
        elif self.policy_optimizer == "RMSprop":
            policy_optimizer = RMSprop
        else:
            raise ValueError("Неизвестный оптимизатор политики")

        self.p_optimizer = policy_optimizer(self.p_net.parameters(), lr=self.policy_lr)

        self.buffer = ReplayBuffer(self.buffer_alpha, self.buffer_size, self.batch_size, device=self.device)

    def train_epoch(self):
        sr, mr, szr, mzr, cs = [], [], [], [], []
        sdr = []
        q1_loss, q2_loss, p_loss, alpha_loss = [], [], [], []

        for _ in range(self.num_epoch_steps):
            for _ in range(self.num_step_game):
                sum_reward, mean_reward, sum_zero_reward, mean_zero_reward, sum_step_discount_reward, steps_count = self._play_game()
                sr.append(sum_reward)
                mr.append(mean_reward)
                szr.append(sum_zero_reward)
                mzr.append(mean_zero_reward)
                cs.append(steps_count)
                sdr.append(sum_step_discount_reward)

            for _ in range(self.num_step_train):
                q1_loss_, q2_loss_, p_loss_, alpha_loss_ = self._update_networks()
                self.update_targets()

                q1_loss.append(q1_loss_)
                q2_loss.append(q2_loss_)
                p_loss.append(p_loss_)
                alpha_loss.append(alpha_loss_)

        return sum(sr)/len(sr), sum(mr)/len(mr), sum(szr)/len(szr), sum(mzr)/len(mzr), sum(cs)/len(cs), sum(sdr)/len(sdr), \
            sum(q1_loss)/len(q1_loss), sum(q2_loss)/len(q2_loss), sum(p_loss)/len(p_loss), sum(alpha_loss)/len(alpha_loss)

    def update_targets(self):
        with torch.no_grad():
            for p, p_ in zip(self.q1_net.parameters(), self.q1_net_target.parameters()):
                p_.copy_(self.tau * p + (1 - self.tau) * p_)
        with torch.no_grad():
            for p, p_ in zip(self.q2_net.parameters(), self.q2_net_target.parameters()):
                p_.copy_(self.tau * p + (1 - self.tau) * p_)

    def _play_game(self):
        logs, _, rewards, zero_rewards, step_discounted_reward, c = play_game(self.p_net, RandomAgent(), self.env)
        for log in logs:
            self.buffer.add_samples(*log, use_median_weight=False)

        return sum(rewards), sum(rewards) / len(rewards), sum(zero_rewards), sum(zero_rewards) / len(zero_rewards), sum(step_discounted_reward), c

    def _update_networks(self):
        states, actions, rewards, next_states, dones = self.buffer.sample()
        policy_actions, policy_probs = self.p_net.process(states)
        policy_entropy = -torch.sum(policy_probs*torch.log(policy_probs)).detach()

        with torch.no_grad():
            policy_next_actions, policy_next_probs = self.p_net.process(next_states)
            policy_next_entropy = -torch.sum(policy_next_probs*torch.log(policy_next_probs))

            next_values_1 = self.q1_net(next_states).gather(-1, policy_next_actions)
            next_values_2 = self.q2_net(next_states).gather(-1, policy_next_actions)
            min_next_values = torch.min(next_values_1, next_values_2)

            next_values = torch.sum(policy_next_probs * min_next_values)
            targets = rewards + self.gamma * (1 - dones) * (next_values + self.alpha * policy_entropy)

        values1 = torch.gather(self.q1_net(states), dim=-1, index=actions.unsqueeze(-1)).squeeze(-1)
        loss1 = self.loss_fn(values1, targets)
        self.q1_optimizer.zero_grad()
        loss1.backward()
        self.q1_optimizer.step()

        values2 = torch.gather(self.q2_net(states), dim=-1, index=actions.unsqueeze(-1)).squeeze(-1)
        loss2 = self.loss_fn(values2, targets)
        self.q2_optimizer.zero_grad()
        loss2.backward()
        self.q2_optimizer.step()

        with torch.no_grad():
            q1_values = torch.gather(self.q1_net(states), dim=-1, index=actions.unsqueeze(-1)).squeeze(-1)
            q2_values = torch.gather(self.q2_net(states), dim=-1, index=actions.unsqueeze(-1)).squeeze(-1)
            min_values = torch.min(q1_values, q2_values)

        p_loss = torch.sum(policy_probs * (self.alpha * torch.log(policy_probs) - min_values))

        self.p_optimizer.zero_grad()
        p_loss.backward()
        self.p_optimizer.step()

        log_probs = torch.log(policy_probs).gather(-1, policy_actions)
        alpha_loss = torch.log(self.alpha) * (policy_entropy + log_probs)

        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        return loss1.abs().detach(), loss2.abs().detach(), p_loss.abs().detach(), alpha_loss.abs().detach()
