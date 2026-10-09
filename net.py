import math
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torchrl.modules.distributions import MaskedCategorical
from torch.nn.functional import relu, gumbel_softmax, softmax

from MABattle.MABattleV0 import NUM_AGENTS, ACTION_SPACE, FLATTEN_STATE_SHAPE
#from const import device


class MNISTNet(nn.Module):
    def __init__(self):
        super(MNISTNet, self).__init__()

        self.conv1 = nn.Conv2d(1, 2, kernel_size=3, stride=1, padding="same", bias=True)
        self.conv2 = nn.Conv2d(2, 4, kernel_size=3, stride=1, padding="same", bias=True)
        self.conv3 = nn.Conv2d(4, 8, kernel_size=3, stride=1, padding="same", bias=True)
        self.conv4 = nn.Conv2d(8, 16, kernel_size=3, stride=1, padding="same", bias=True)

        self.max_pool = nn.MaxPool2d(2, 2)
        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(16, 32, bias=True)
        self.fc2 = nn.Linear(32, 10, bias=True)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.max_pool(self.relu(self.conv1(x)))
        x = self.max_pool(self.relu(self.conv2(x)))
        x = self.max_pool(self.relu(self.conv3(x)))
        x = self.max_pool(self.relu(self.conv4(x)))
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)

        return x


class Logic2Net(nn.Module):
    def __init__(self):
        super(Logic2Net, self).__init__()

        self.fc1 = nn.Linear(2, 4, bias=True)
        self.fc2 = nn.Linear(4, 1, bias=True)

    def forward(self, x):
        x = relu(self.fc1(x))
        x = relu(self.fc2(x))

        return x


class MaDeterministicNetBase(nn.Module):
    def __init__(self):
        super(MaDeterministicNetBase, self).__init__()
        self.agents = []

    def forward(self, x):
        raise NotImplementedError()

    def get_actions(self, x, legals):
        # actions = torch.full([NUM_AGENTS], 0, device=device)

        x = self.forward(x)
        x = x.masked_fill(~legals.bool(), -1e9)
        logits = gumbel_softmax(x, tau=1, hard=True)

        # actions = actions_one_hot.argmax(dim=-1)
        actions = logits.argmax(dim=-1)
        # mask = legals.any(1)
        # torch.tensor().masked_fill()

        # dist = MaskedCategorical(logits=x[mask], mask=legals[mask])
        # sample = dist.sample()

        # actions[mask] = sample

        #print(actions, actions_one_hot, len(actions))
        return actions, logits


class MAStochasticNetBase(nn.Module):
    def __init__(self):
        super(MAStochasticNetBase, self).__init__()
        self.agents = []

    def forward(self, x):
        raise NotImplementedError()

    def get_actions(self, x, legals):
        actions = torch.full([NUM_AGENTS], 0)#, device=device)

        x = self.forward(x)
        mask = legals.any(1)

        dist = MaskedCategorical(logits=x[mask], mask=legals[mask])
        sample = dist.sample()

        actions[mask] = sample

        return actions


class MAStochasticConvNet(MAStochasticNetBase):
    def __init__(self):
        super(MAStochasticConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 2, kernel_size=3, stride=1, padding="same", bias=True)
        self.conv2 = nn.Conv2d(2, 4, kernel_size=3, stride=1, padding="same", bias=True)
        self.flatten = nn.Flatten(start_dim=-3, end_dim=-1)
        self.fc1 = nn.Linear(64, 128)
        self.fc2 = nn.Linear(128, ACTION_SPACE * NUM_AGENTS)

        self.softmax = nn.Softmax(dim=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.flatten(x)

        x = self.relu(self.fc1(x))
        x = self.fc2(x)

        return x


class MAStochasticFCNet(MAStochasticNetBase):
    def __init__(self):
        super(MAStochasticFCNet, self).__init__()
        self.fc1 = nn.Linear(16, 32, bias=True)
        self.fc2 = nn.Linear(32, ACTION_SPACE)

    def forward(self, x):
        x = relu(self.fc1(x))
        x = self.fc2(x)

        return x


class MADeterministicFCNet(MaDeterministicNetBase):
    def __init__(self):
        super(MADeterministicFCNet, self).__init__()
        self.fc1 = nn.Linear(16, 32, bias=True)
        self.fc2 = nn.Linear(32, 64, bias=True)
        self.fc3 = nn.Linear(64, 32, bias=True)
        self.fc4 = nn.Linear(32, ACTION_SPACE)

    def forward(self, x, use_softmax=False):
        x = relu(self.fc1(x))
        x = relu(self.fc2(x))
        x = relu(self.fc3(x))
        x = self.fc4(x)

        if use_softmax:
            x = gumbel_softmax(x, tau=1, hard=True)

        return x


class MACriticNet(nn.Module):
    def __init__(self):
        super(MACriticNet, self).__init__()
        self.fc1 = nn.Linear(16+5, 64, bias=True)
        self.fc2 = nn.Linear(64, 128, bias=True)
        self.fc3 = nn.Linear(128, 128, bias=True)
        self.fc4 = nn.Linear(128, 64, bias=True)
        self.fc5 = nn.Linear(64, 1, bias=True)

    def forward(self, x, actions):
        #print(x.shape, actions.shape)
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
        #print((rms_data / rms_sigma).data.cpu().numpy())
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
        # Возвращает устройство первого параметра модели
        return next(self.parameters()).device


    def _get_random(self, legals):
        return torch.tensor([random.choice(legals[i]) for i in range(NUM_AGENTS)])

    def get_actions(self, x, legals, **kwargs):
        #actions = self(x).argmax(-1).tolist()
        #
        #if not all(actions[i] in legals[i] for i in range(len(legals))):
        #    return self._get_random(legals).tolist()

        logits = self(x)  # shape: (NUM_AGENTS, ACTION_SPACE)

        # Преобразуем legals в тензор маски
        legal_mask = torch.zeros((self.num_agents, self.num_actions), dtype=torch.bool, device=x.device)
        #print(legals)
        for i, legal_list in enumerate(legals):
            if legal_list:  # не пустой список
                legal_mask[i, legal_list] = True

        # Маскируем нелегальные логиты
        masked_logits = logits.masked_fill(~legal_mask, -1e9)
        #print(masked_logits)

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
        # Возвращает устройство первого параметра модели
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
        # Возвращает устройство первого параметра модели
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
        # Возвращает устройство первого параметра модели
        return next(self.parameters()).device

    def forward(self, x):
        x = relu(self.fc1(x))
        x = relu(self.fc2(x))
        x = relu(self.fc3(x))
        x = relu(self.fc4(x))
        x = self.fc5(x)

        return x

class MAPolicyNet(nn.Module):
    def __init__(self):
        super().__init__(MAPolicyNet, self).__init__()
        self.fc1 = nn.Linear(FLATTEN_STATE_SHAPE, 64, bias=True)
        self.fc2 = nn.Linear(64, 128, bias=True)
        self.fc3 = nn.Linear(128, 128, bias=True)
        self.fc4 = nn.Linear(128, 64, bias=True)
        self.fc5 = nn.Linear(64, ACTION_SPACE, bias=True)

    def forward(self, x):
        x = relu(self.fc1(x))
        x = relu(self.fc2(x))
        x = relu(self.fc3(x))
        x = relu(self.fc4(x))
        x = self.fc5(x)

        return x

    def get_action(self, x, legals):
        logits = self(x)  # shape: (NUM_AGENTS, ACTION_SPACE)

        # Преобразуем legals в тензор маски
        legal_mask = torch.zeros((NUM_AGENTS, ACTION_SPACE), dtype=torch.bool, device=x.device)
        # print(legals)
        for i, legal_list in enumerate(legals):
            if legal_list:  # не пустой список
                legal_mask[i, legal_list] = True

        # Маскируем нелегальные логиты
        masked_logits = logits.masked_fill(~legal_mask, -1e9)
        #probs = softmax(masked_logits, dim=-1)

        actions = masked_logits.argmax(dim=-1).tolist()

        return actions

    def process(self, x):
        logits = self(x)

        return logits.argmax(-1), softmax(logits, dim=-1)
