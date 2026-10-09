import numpy as np

class Unit:
    def __init__(self, pos, player, idx):
        self.pos = np.array(pos)
        self.player = player
        self.idx = idx

        self._possible_steps = np.array([
            [0,0],
            [1,0],
            [0,1],
            [-1,0],
            [0,-1]
        ])
        self.is_alive = True

    def get_possible_steps(self):
        return self._possible_steps

    def __repr__(self):
        return f"Unit_{self.idx}"
