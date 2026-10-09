import math
import random

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.functional import relu, softmax


class MACriticNet(nn.Module):
    def __init__(self):
        super(MACriticNet, self).__init__()
        self.fc1 = nn.Linear(16+5, 64, bias=True)
        self.fc2 = nn.Linear(64, 128, bias=True)
        self.fc3 = nn.Linear(128, 128, bias=True)
        self.fc4 = nn.Linear(128, 64, bias=True)
        self.fc5 = nn.Linear(64, 1, bias=True)

    def forward(self, x, actions):
        x = torch.cat((x, actions), dim=-1)

        x = relu(self.fc1(x))
        x = relu(self.fc2(x))
        x = relu(self.fc3(x))
        x = relu(self.fc4(x))
        x = self.fc5(x)

        return x


class NoisyLinear(nn.Linear):
    def __init__(self, in_features, out_features, sigma_init=0.017, bias=True):
        super(NoisyLinear, self).__init__(in_features,out_features, bias=bias)
        self.sigma_weight = nn.Parameter(torch.full((out_features, in_features), sigma_init))
        self.register_buffer("epsilon_weight", torch.zeros(out_features, in_features))

        if bias:
            self.sigma_bias =  nn.Parameter(torch.full((out_features,), sigma_init))
            self.register_buffer("epsilon_bias", torch.zeros(out_features))

        self.reset_parameters()
    def reset_parameters(self):
        std = math.sqrt(3/self.in_features)
        self.weight.data.uniform_(-std, std)
        self.bias.data.uniform_(-std, std)
    def forward(self, input_data):
        self.epsilon_weight.normal_()
        bias = self.bias

        if bias is not None:
            self.epsilon_bias.normal_()
            bias = bias + self.epsilon_bias * self.sigma_bias

        weight = self.weight  + self.epsilon_weight * self.sigma_weight
        return F.linear(input_data, weight, bias)

    def calc_snr(self):
        rms_data = (self.weight ** 2).mean().sqrt()
        rms_sigma = (self.sigma_weight ** 2).mean().sqrt()
        snr = (rms_data / rms_sigma).data.cpu().numpy()

        return snr


class MAFCQNoisyNet(nn.Module):
    def __init__(self, in_shape, num_agents, num_actions, sigma_init=0.017):
        super(MAFCQNoisyNet, self).__init__()
        self.in_shape = in_shape
        self.num_agents = num_agents
        self.num_actions = num_actions

        self.fc1 = NoisyLinear(in_shape, 64, sigma_init=sigma_init, bias=True)
        self.fc2 = NoisyLinear(64, 128, sigma_init=sigma_init, bias=True)
        self.fc3 = NoisyLinear(128, 128, sigma_init=sigma_init, bias=True)
        self.fc4 = NoisyLinear(128, 64, sigma_init=sigma_init, bias=True)
        self.fc5_adv = NoisyLinear(64, num_actions, sigma_init=sigma_init)
        self.fc5_v = NoisyLinear(64, 1, sigma_init=sigma_init)
        self.noisy_layers = [self.fc1, self.fc2, self.fc3, self.fc4, self.fc5_adv, self.fc5_v]

    def get_snr(self):
        snrs = [layer.calc_snr() for layer in self.noisy_layers]

        return snrs, sum(snrs)/len(snrs)
    @property
    def device(self):
        return next(self.parameters()).device


    def _get_random(self, legals):
        return torch.tensor([random.choice(legals[i]) for i in range(self.num_agents)])

    def get_actions(self, x, legals, **kwargs):

        logits = self(x)
        legal_mask = torch.zeros((self.num_agents, self.num_actions), dtype=torch.bool, device=x.device)

        for i, legal_list in enumerate(legals):
            if legal_list:
                legal_mask[i, legal_list] = True

        masked_logits = logits.masked_fill(~legal_mask, -1e9)
        actions = masked_logits.argmax(dim=-1).tolist()

        return actions

    def forward(self, x):
        x = relu(self.fc1(x))
        x = relu(self.fc2(x))
        x = relu(self.fc3(x))
        x = relu(self.fc4(x))
        adv = self.fc5_adv(x)
        v = self.fc5_v(x)

        return v + (adv - adv.mean(-1, keepdim=True))

class MAFCQNet(nn.Module):
    def __init__(self, in_shape, num_agents, num_actions):
        super(MAFCQNet, self).__init__()
        self.in_shape = in_shape
        self.num_agents = num_agents
        self.num_actions = num_actions

        self.fc1 = nn.Linear(in_shape, 64, bias=True)
        self.fc2 = nn.Linear(64, 128, bias=True)
        self.fc3 = nn.Linear(128, 128, bias=True)
        self.fc4 = nn.Linear(128, 64, bias=True)
        self.fc5_adv = nn.Linear(64, num_actions)
        self.fc5_v = nn.Linear(64, 1)
    @property
    def device(self):
        return next(self.parameters()).device

    def _get_random(self, legals):
        return torch.tensor([random.choice(legals[i]) for i in range(self.num_agents)])

    def get_actions(self, x, legals, eps=0):
        if random.randint(1, 100) / 100 <= eps:
            return self._get_random(legals).tolist()
        else:
            actions = self(x)[1].argmax(-1).tolist()

            if not all(actions[i] in legals[i] for i in range(len(legals))):
                return self._get_random(legals).tolist()

        return actions
    def forward(self, x):
        x = relu(self.fc1(x))
        x = relu(self.fc2(x))
        x = relu(self.fc3(x))
        x = relu(self.fc4(x))
        adv = self.fc5_adv(x)
        v = self.fc5_v(x)

        return v.squeeze(-1), v + (adv - adv.mean(-1, keepdim=True))


class MAFCVNoisyNet(nn.Module):
    def __init__(self, in_shape, sigma_init=0.017):
        super(MAFCVNoisyNet, self).__init__()
        self.in_shape = in_shape

        self.fc1 = NoisyLinear(in_shape, 64, bias=True, sigma_init=sigma_init)
        self.fc2 = NoisyLinear(64, 128, bias=True, sigma_init=sigma_init)
        self.fc3 = NoisyLinear(128, 128, bias=True, sigma_init=sigma_init)
        self.fc4 = NoisyLinear(128, 64, bias=True, sigma_init=sigma_init)
        self.fc5 = NoisyLinear(64, 1, sigma_init=sigma_init)

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, x):
        x = relu(self.fc1(x))
        x = relu(self.fc2(x))
        x = relu(self.fc3(x))
        x = relu(self.fc4(x))
        x = self.fc5(x)

        return x


class MAFCVNet(nn.Module):
    def __init__(self, in_shape):
        super(MAFCVNet, self).__init__()
        self.in_shape = in_shape

        self.fc1 = nn.Linear(in_shape, 64, bias=True)
        self.fc2 = nn.Linear(64, 128, bias=True)
        self.fc3 = nn.Linear(128, 128, bias=True)
        self.fc4 = nn.Linear(128, 64, bias=True)
        self.fc5 = nn.Linear(64, 1)

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, x):
        x = relu(self.fc1(x))
        x = relu(self.fc2(x))
        x = relu(self.fc3(x))
        x = relu(self.fc4(x))
        x = self.fc5(x)

        return x
