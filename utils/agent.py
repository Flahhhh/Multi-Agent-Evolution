import torch
import random

from MABattle.MABattleV0 import UNIT_POSSIBLE_ACTIONS

class RandomAgent:
    def __init__(self, num_agents):
        self.num_agents = num_agents

    def get_actions(self, _, legals):
        #actions = torch.full([NUM_AGENTS], 0)
        actions = [random.choice(legals[i]) for i in range(self.num_agents)]

        #mask = legals.sum(1)
        #for idx in range(NUM_AGENTS):
        #    if not mask[idx]: continue
        #    actions[idx] = random.choice(torch.argwhere(legals[idx]))

        return actions

class GreedyAgent:
    def __init__(self, state, legals):

        actions = []


        for i in range(...):
            max_j = 0
            rewards = []

            for j in range(len(legals[i])):
                 shift_i, shift_j = UNIT_POSSIBLE_ACTIONS[legals[i][j]]