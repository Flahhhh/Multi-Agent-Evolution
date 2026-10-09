from dataclasses import dataclass
from typing import List, Optional, Any
from PIL import Image
import torch
import numpy as np

import gymnasium as gym
from gymnasium.core import ObsType

from .board import Board, GameResults
from .config import EnvCfg


class MABattleEnv(gym.Env):
    metadata = {"render_modes": ["human", "ansi"], "render_fps": 4}

    def __init__(self, env_cfg: EnvCfg = None, render_mode: Optional[str] = None):
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.cfg = EnvCfg if env_cfg is None else env_cfg

        self.flatten_observations = True
        self.is_bool_legals = False
        self.is_log_win = True
        self.shape = (self.cfg.num_agents, self.cfg.board_size[0] * self.cfg.board_size[1]) if self.flatten_observations else (
            self.cfg.num_agents, *self.cfg.board_size)

        self.observation_space = gym.spaces.Box(
            low=-1,
            high=self.cfg.obs_high,
            shape=self.shape,
            dtype=np.float32
        )
        self.action_space = gym.spaces.Discrete(self.cfg.action_space)
        self.render_mode = render_mode
        self._board = Board(env_cfg)

        self.reward_type = "raw"
        self.max_c = 100

    def reset(self, *, seed: int | None = None,
              options: dict[str, Any] | None = None, ):
        super().reset(seed=seed)
        self._board.reset()
        self.c = 0

        state = self.get_state()
        legals = self.get_legals()
        alive_agents = self._board.get_alive()

        info = {
            "local_dones": self._get_dones(),
            "legals": legals,
            "alive_agents": alive_agents
        }

        return state, info

    def step(self, actions) -> tuple:
        alive_agents = self._board.get_alive()
        turn = self.to_play()

        # print(alive_agents, actions)
        rewards, done, captured, pdv_units = self._board.step({idx: actions[idx] for idx in alive_agents})
        if self.reward_type == "mean":
            reward = sum(rewards) / len(rewards)
        elif self.reward_type == "sum":
            reward = sum(rewards)
        elif self.reward_type == "raw":
            reward = torch.full((self.cfg.num_agents,), -1, dtype=torch.float32)
            reward[alive_agents] = torch.tensor(rewards, dtype=torch.float32)

        opponent_reward = torch.zeros(self.cfg.num_agents, dtype=torch.float32)
        opponent_reward[captured] = -100.0

        state = self.get_state()

        self.c += 1
        truncated = self.c >= self.max_c
        info = {
            "local_dones": self._get_dones(),
            "legals": self.get_legals(),
            "alive_agents": self._board.get_alive(),
            "opponent_reward": opponent_reward,
            "raw_reward": torch.tensor(rewards),
            "pdv_state": torch.FloatTensor(self.get_state(units=pdv_units))
        }
        if self.is_log_win:
            info["is_win"] = GameResults.Win if done else GameResults.Draw if truncated else GameResults.Continue

        return state, reward, done, truncated, info

    # def is_win(self):
    #    self._board.get_alive()

    def _get_dones(self):
        local_dones = torch.full(size=[self.cfg.num_agents], fill_value=True, dtype=torch.bool)
        local_dones[self._board.get_alive()] = False

        return local_dones

    def get_legals(self):
        legal_actions = self._board.get_possible_actions()

        if self.is_bool_legals:
            legals = torch.full((self.cfg.num_agents, self.cfg.action_space), False, dtype=torch.bool)

            for idx, move_idxs in legal_actions.items():
                for move_idx in move_idxs:
                    legals[idx, move_idx] = True
            return legals
        else:
            legals = [[0]] * self.cfg.num_agents
            for idx, move_idxs in legal_actions.items():
                legals[idx] = move_idxs

            return legals

    def render(self):
        return self.get_state()

    def get_image(self):
        if self._board.turn == -1:
            self._board.flip()

        image = Image.new("RGB", (self.cfg.board_size[0] * self.cfg.unit_image_size[0], self.cfg.board_size[1] * self.cfg.unit_image_size[1]), "black")
        blue_image = Image.open(self.cfg.blue_unit_image).convert("RGBA")
        red_image = Image.open(self.cfg.red_unit_image).convert("RGBA")
        cell_image = Image.open(self.cfg.cell_image).convert("RGBA")
        mask = red_image.split()[3]

        for i in range(self.cfg.board_size[0]):
            for j in range(self.cfg.board_size[1]):
                image.paste(cell_image, (j * self.cfg.unit_image_size[0], i * self.cfg.unit_image_size[1]))

                if (i, j) in self._board.units[1]:
                    image.paste(blue_image, (j * self.cfg.unit_image_size[0], i * self.cfg.unit_image_size[1]), mask=mask)
                elif (i, j) in self._board.units[-1]:
                    image.paste(red_image, (j * self.cfg.unit_image_size[0], i * self.cfg.unit_image_size[1]), mask=mask)

        if self._board.turn == -1:
            self._board.flip()

        return image

    def close(self):
        self._board = None
        return

    def to_play(self):
        return self._board.turn

    def get_state(self, units=None):
        state = np.zeros(shape=(self.cfg.num_agents, *self.cfg.board_size), dtype=np.float32)
        if units is None:
            units = self._board.units

        for pos, unit in units[self._board.turn].items():
            state[:, pos[0], pos[1]] = 1  # unit.idx
            state[unit.idx, pos[0], pos[1]] = 2

        for pos, unit in units[-self._board.turn].items():
            state[:, pos[0], pos[1]] = -1

        if self.flatten_observations:
            state = np.reshape(state, shape=[self.cfg.num_agents, self.cfg.board_size[0] * self.cfg.board_size[1]])

        return state

    def __getstate__(self):
        return self.get_state()
