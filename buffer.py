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

        self.size = torch.tensor([size])
        self.batch_size = batch_size
        self.entries = torch.tensor([0])

        self.prev_states = torch.zeros((size, num_agents, flatten_state_shape), dtype=torch.float32)
        self.prev_actions = torch.zeros((size, num_agents), dtype=torch.long)
        self.rewards = torch.zeros(size, num_agents, dtype=torch.float32)

        if is_pdv:
            self.pdv_states = torch.zeros((size, num_agents, flatten_state_shape),
                                          dtype=torch.float32)
            self.opponent_actions = torch.zeros((size, num_agents), dtype=torch.long)
        self.pdv_dones = torch.zeros(size, num_agents, dtype=torch.float32)
        self.opponent_rewards = torch.zeros(size, num_agents, dtype=torch.float32)

        self.unit_next_states = torch.zeros((size, num_agents, flatten_state_shape),
                                            dtype=torch.float32)
        self.unit_dones = torch.zeros(size, num_agents, dtype=torch.float32)
        self.weights = torch.ones(size, dtype=torch.float32)
        self.weights_twin = torch.ones(size, dtype=torch.float32)

        self.all_ids = torch.arange(0, size, dtype=torch.int)
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

        if use_median_weight:
            self.weights[i] = self.median_value
            self.weights_twin[i] = self.median_value_twin
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

        if self.is_pdv:
            return ids, prev_states, prev_actions, rewards, \
                pdv_dones, pdv_states, opponent_actions, opponent_rewards, \
                unit_dones, unit_next_states
        else:
            return ids, prev_states, prev_actions, rewards, pdv_dones, opponent_rewards, \
                unit_dones, unit_next_states
