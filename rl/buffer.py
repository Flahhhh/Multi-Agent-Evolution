from multiprocessing import Lock

import numpy as np
import torch

EPS = 1e-10

class ReplayBuffer:
    def __init__(self, alpha, size, num_agents, flatten_state_shape, batch_size=32, device=None, is_pdv=False):
        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = device

        self.is_pdv = is_pdv
        #self.lock = Lock()

        self.size = torch.tensor([size])#.share_memory_()
        self.batch_size = batch_size
        self.entries = torch.tensor([0])#.share_memory_()

        self.prev_states = torch.zeros((size, num_agents, flatten_state_shape), dtype=torch.float32)#.share_memory_()
        self.prev_actions = torch.zeros((size, num_agents), dtype=torch.long)#.share_memory_()
        self.rewards = torch.zeros(size, num_agents, dtype=torch.float32)#.share_memory_()

        if is_pdv:
            self.pdv_states = torch.zeros((size, num_agents, flatten_state_shape), dtype=torch.float32)#.share_memory_()
            self.opponent_actions = torch.zeros((size, num_agents), dtype=torch.long)#.share_memory_()
        self.pdv_dones = torch.zeros(size, num_agents, dtype=torch.float32)#.share_memory_()
        self.opponent_rewards = torch.zeros(size, num_agents, dtype=torch.float32)#.share_memory_()
            
        self.unit_next_states = torch.zeros((size, num_agents, flatten_state_shape), dtype=torch.float32)#.share_memory_()
        self.unit_dones = torch.zeros(size, num_agents, dtype=torch.float32)#.share_memory_()
        self.weights = torch.ones(size, dtype=torch.float32)
        self.weights_twin = torch.ones(size, dtype=torch.float32)

        self.all_ids = torch.arange(0, size, dtype=torch.int)
        #self.mem = np.empty((size, 8), dtype=object)
        self.alpha = alpha

    def __len__(self):
        return self.entries

    def update_median_value(self):
        self.median_value = self.weights.median()
        self.median_value_twin = self.weights_twin.median()

    def add_samples(self, prev_state, prev_action, reward,
                    pdv_dones, pdv_states, opponent_actions, opponent_rewards,
                    unit_dones, unit_next_states, use_median_weight=True):
        i = self.__len__() % self.size
        #max_weight = self.weights.max()
        #self.weights.median()
        #self.weights.median()

        if use_median_weight:
            self.weights[i] = self.median_value #self.weights.median()
            self.weights_twin[i] = self.median_value_twin #self.weights_twin.median()
        else:
            self.weights[i] = self.weights.max()
            self.weights_twin[i] = self.weights_twin.max()

        self.prev_states[i] = prev_state
        self.prev_actions[i] = prev_action
        self.rewards[i] = reward

        if self.is_pdv:
            self.pdv_states[i] = pdv_states
            self.opponent_actions[i] = opponent_actions
        self.pdv_dones[i] = pdv_dones.float()
        self.opponent_rewards[i] = opponent_rewards

        self.unit_dones[i] = unit_dones.float()
        self.unit_next_states[i] = unit_next_states
        #self.unit_dones[i] = unit_dones.float()


        #self.mem[i] = (
        #    prev_state, prev_action, reward,
        #    opponent_dones, opponent_next_states, opponent_actions,
        #    unit_dones, unit_next_states
        #)
        """
        self.states[i] = state
        self.actions[i] = action
        self.opponent_next_states[i] = opponent_next_states
        self.unit_next_states[i] = unit_next_states
        self.rewards[i] = reward
        self.opponent_dones[i] = opponent_dones
        self.unit_dones[i] = unit_dones
        self.opponent_actions[i] = opponent_actions
        """

        self.entries = min(self.size, self.entries + 1)

    def set_weights(self, agent_id, ids, weights):
        if agent_id == 1:
            self.weights[ids] = weights.cpu()
        elif agent_id == 2:
            self.weights_twin[ids] = weights.cpu()

    def get_probs(self, weights):
        probs = (weights[:self.__len__()] + EPS) ** self.alpha
        return probs / probs.sum()

    def sample(self, agent_id):
        if agent_id == 1:
            probs = self.get_probs(self.weights)
        elif agent_id == 2:
            probs = self.get_probs(self.weights_twin)

        ids = torch.multinomial(probs, self.batch_size, replacement=False)
        #print(probs, len(probs), sum(probs))
        #ids = torch.randint(0, self.__len__().item(), size=[self.batch_size])
        #ids = np.random.choice(self.all_ids[:self.__len__()], size=self.batch_size, p=self.get_probs())

        prev_states = self.prev_states[ids].to(self.device)
        prev_actions = self.prev_actions[ids].to(self.device)
        rewards = self.rewards[ids].to(self.device)

        if self.is_pdv:
            pdv_states = self.pdv_states[ids].to(self.device)
            opponent_actions = self.opponent_actions[ids].to(self.device)
        pdv_dones = self.pdv_dones[ids].to(self.device)
        opponent_rewards = self.opponent_rewards[ids].to(self.device)

        unit_dones = self.unit_dones[ids].to(self.device)
        unit_next_states = self.unit_next_states[ids].to(self.device)

        #samples = self.mem[ids]
        #prev_states, prev_actions, reward, \
        #    opponent_dones, opponent_states, opponent_actions, \
        #    unit_dones, unit_next_states = zip(*samples)

        #print(torch.stack(prev_states).shape, torch.stack(prev_actions).shape, prev_alive_agents.shape)
        #print(torch.stack(opponent_dones).shape)
        
        #st = torch.stack(prev_states)
        #num_alive_per_batch = prev_alive_agents.sum(dim=1)
        #split_selected_states = torch.split(st, num_alive_per_batch.tolist(), dim=0)
        #print(split_selected_states, len(split_selected_states))

        #prev_states, prev_actions = \
        #    torch.stack(prev_states).to(self.device), \
        #    torch.stack(prev_actions).to(self.device)
        #print(prev_states.shape)

        #reward = torch.tensor([r.sum() for r in reward]).to(self.device)

        #opponent_dones, opponent_states, opponent_actions = \
        #    torch.stack(opponent_dones).to(self.device), \
        #    torch.stack(opponent_states).to(self.device), \
        #    torch.stack(opponent_actions).to(self.device)

        #unit_dones, unit_next_states = \
        #    torch.stack(unit_dones).to(self.device), \
        #    torch.stack(unit_next_states).to(self.device)

        if self.is_pdv:
            return ids, prev_states, prev_actions, rewards, \
                pdv_dones, pdv_states, opponent_actions, opponent_rewards, \
                unit_dones, unit_next_states
        else:
            return ids, prev_states, prev_actions, rewards, pdv_dones, opponent_rewards, \
                unit_dones, unit_next_states
        """
        states = self.states[ids].to(self.device)
        actions = self.actions[ids].to(self.device)
        opponent_next_states = self.opponent_next_states[ids].to(self.device)
        unit_next_states = self.unit_next_states[ids].to(self.device)
        rewards = self.rewards[ids].to(self.device)
        opponent_dones = self.opponent_dones[ids].to(self.device)
        unit_dones = self.unit_dones[ids].to(self.device)
        opponent_actions = self.opponent_actions[ids].to(self.device)

        return states, actions, rewards, \
            opponent_next_states, opponent_actions, opponent_dones, unit_next_states, unit_dones
        """
