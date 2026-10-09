import torch
import random


class RandomAgent:
    def __init__(self, num_agents):
        self.num_agents = num_agents

    def get_actions(self, _, legals):
        actions = [random.choice(legals[i]) for i in range(self.num_agents)]

        return actions


class GreedyAgent:
    def __init__(self, num_agents, state_shape, unit_possible_actions, team):
        self.num_agents = num_agents
        self.state_shape = state_shape
        self.unit_possible_actions = unit_possible_actions

        self.team = team

    def get_actions(self, state, legals, board_dict):

        state = state.reshape(self.state_shape)
        actions = [0] * self.num_agents

        for agent_idx, pos in board_dict[self.team].items():
            for action in legals[agent_idx]:
                step = self.unit_possible_actions[action]
                new_pos = (pos[0] + step[0], pos[1] + step[1])
                if new_pos in board_dict[-self.team]:
                    actions[agent_idx] = action
                    break
            else:
                actions[agent_idx] = random.choice(legals[agent_idx])
