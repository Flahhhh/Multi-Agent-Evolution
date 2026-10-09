from enum import Enum

from .config import EnvCfg
from .units import Unit


class GameResults(Enum):
    Continue = 0.0
    Draw = 0.0
    Win = 100.0
    Lose = -100.0


class Board:
    units: dict[int: dict[tuple[int, int]: Unit]]
    turn: int

    def __init__(self, env_cfg: EnvCfg, use_flip=True):
        self.cfg = env_cfg
        self.reset()

        self.use_flip = use_flip
        self.units = {1: {}, -1: {}}

    def reset(self):
        self.turn = 1
        self._spawn_units()

    def flip(self):
        for p, units in self.units.items():
            units_ = {}
            for pos, unit in units.items():
                new_pos = (self.cfg.board_size[0] - pos[0] - 1, self.cfg.board_size[1] - pos[1] - 1)
                unit.pos = new_pos
                units_[new_pos] = unit
            self.units[p] = units_

    def _spawn_units(self):
        self.units = {1: {}, -1: {}}
        line_idxs = list(range(self.cfg.board_size[0]))
        row_idxs = list(range(self.cfg.board_size[1]))

        for i in range(self.cfg.num_lines):
            for j in range(self.cfg.board_size[1]):
                pos = (line_idxs[i], j)
                idx = i * self.cfg.board_size[1] + j  # + 1
                unit = Unit(pos, 1, idx)
                self.units[1][pos] = unit

        for i in range(self.cfg.num_lines):
            for j in range(self.cfg.board_size[1]):
                pos = (line_idxs[::-1][i], row_idxs[::-1][j])
                idx = i * self.cfg.board_size[1] + j  # + 1
                # print(-1, idx)
                unit = Unit(pos, -1, idx)
                self.units[-1][pos] = unit

    def _validate_move(self, move: list, unit: Unit) -> bool:
        new_pos = (unit.pos[0] + move[0], unit.pos[1] + move[1])
        #new_pos = unit.pos + move

        return new_pos[0] in self.cfg.row_range and new_pos[1] in self.cfg.col_range

    def get_possible_actions(self) -> dict[int: list[int]]:
        moves = {}
        for _, unit in self.units[self.turn].items():
            moves[unit.idx] = []
            for move_idx in range(len(self.cfg.unit_possible_actions)):
                if self._validate_move(self.cfg.unit_possible_actions[move_idx], unit):
                    moves[unit.idx].append(move_idx)

        return moves

    def get_alive(self) -> list[int]:
        r = []
        for _, unit in self.units[self.turn].items():
            r.append(unit.idx)

        return r

    def get_num_opponents(self):
        return len(self.units[-self.turn])

    def _apply_action(self, action: int, unit: Unit) -> tuple[int, int | None]:
        c = None

        pos = unit.pos
        new_pos = (unit.pos[0] + self.cfg.unit_possible_actions[action][0], unit.pos[1] + self.cfg.unit_possible_actions[action][1])

        cur_units = self.units[self.turn]
        opp_units = self.units[-self.turn]

        if action == 0 or cur_units.get(tuple(new_pos), None) is not None:
            return 0, c

        r = 0
        if opp_units.get(tuple(new_pos), None) is not None:
            c = opp_units[tuple(new_pos)].idx
            opp_units.__delitem__(tuple(new_pos))
            r = self.cfg.capture_reward

        cur_units.__delitem__(tuple(pos))
        cur_units[tuple(new_pos)] = unit
        unit.pos = new_pos

        return r, c

    def _get_done(self):
        return len(self.units[-self.turn]) == 0

    def step(self, actions: dict[int: int]):
        possible_move_sets = self.get_possible_actions()
        units = self.units[self.turn]
        r = []
        c = []

        for pos in list(units.keys()):
            unit = units[pos]
            idx = unit.idx
            action = actions[idx]
            if action not in possible_move_sets[idx]:
                raise Exception(
                    f"MOVE IDX: {action} | UNIT IDX: {unit.idx} | NEW POS: {unit.pos + self.cfg.unit_possible_actions[action]} | LEGALS: {possible_move_sets[idx]} | VALIDATION: {action in possible_move_sets[idx]} | UNITS: {self.units[1]} | [ERROR]: Invalid move({idx, action})")

            r_, c_ = self._apply_action(action, unit)
            r.append(r_)
            if c_ is not None:
                c.append(c_)

        d = self._get_done()
        pdv_units = self.units

        self.turn *= -1
        if self.use_flip:
            self.flip()

        return r, d, c, pdv_units
