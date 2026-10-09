import copy
from copy import deepcopy
import torch
from torch.nn import HuberLoss
from torch.optim import Adam

from MABattle.utils import make_env, play_game
from net import MADeterministicFCNet, MACriticNet
from const import device_name
from utils import RandomAgent

from rl.buffer import ReplayBuffer

class DDPG:
    def __init__(self, env=None, device=None):
        if device is None:
            self.device = torch.device(device_name)
        else:
            self.device = device

        if env is None:
            self.env = make_env()
        else:
            self.env = env

        self.actor = MADeterministicFCNet().to(self.device)
        self.actor_target = deepcopy(self.actor).to(self.device)
        self.actor_optimizer = Adam(self.actor.parameters())

        self.critic = MACriticNet().to(self.device)
        self.critic_target = deepcopy(self.critic).to(self.device)
        self.critic_optimizer = Adam(self.critic.parameters())

        self.buffer = ReplayBuffer(size=1024, batch_size=32, device=self.device)

        self.gamma = 0.99
        self.tau = 0.001
        self.target_update_freq = 1

        self.loss_fn = HuberLoss()

    def _update_targets(self):
        with torch.no_grad():
            for p, p_ in zip(self.actor.parameters(), self.actor_target.parameters()):
                p_.copy_((1-self.tau) * p + self.tau * p_)

            for p, p_ in zip(self.critic.parameters(), self.critic_target.parameters()):
                p_.copy_((1-self.tau) * p + self.tau * p_)

    def _update_networks(self):
        #ids, states, actions, next_states, rewards, finishes, probs = self.buffer.sample()
        states, actions, next_states, rewards, finishes = self.buffer.sample()

        critic_value = self.critic(states, actions)
        critic_target = rewards.unsqueeze(1) + self.gamma * (1-finishes).unsqueeze(1) * self.critic_target(next_states, self.actor_target(next_states))
        #critic_target = rewards.unsqueeze(1) + self.gamma * (1-finishes).unsqueeze(1) * self.critic_target(next_states, self.actor_target(next_states))
        #print("Value:", critic_value.shape, "Target:", critic_target.shape, rewards.unsqueeze(1).shape, self.critic_target(next_states, self.actor_target(next_states)).shape)
        critic_loss = self.loss_fn(critic_value, critic_target)
        #critic_losses = critic_target - critic_value
        #critic_loss = (critic_losses**2/2).mean()

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        actor_loss = -self.critic(states, self.actor(states)).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        #self.buffer.update_probs(ids, critic_losses.abs())
        return actor_loss.abs(), critic_loss

    def _play_game(self):
        _, rewards, c = play_game(self.actor, RandomAgent(), self.buffer, self.env, self.device, zero_sum=True)
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

        return rewards, rewards/c

    def train_epoch(self, epoch):
        sum_reward, mean_reward = self._play_game()
        actor_loss, critic_loss = self._update_networks()

        if epoch % self.target_update_freq == 0:
            self._update_targets()

        return sum_reward, mean_reward, actor_loss, critic_loss

    def get_solutions(self, amount: int):
        return [copy.deepcopy(self.actor) for _ in range(amount)]
