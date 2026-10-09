from dataclasses import dataclass, field
from typing import ClassVar


@dataclass(frozen=True)
class EnvCfg:
    board_size: tuple[int, int] = (4, 4)
    num_lines: int = 1
    max_steps: int = 100

    win_reward: float = 0.0
    capture_reward: float = 10.0

    unit_image_size: tuple[int, int] = (32, 32)
    blue_unit_image: str = "images/blue.png"
    red_unit_image: str = "images/red.png"
    cell_image: str = "images/cell.png"

    unit_possible_actions: tuple[list] = (
        [0, 0],
        [1, 0],
        [0, 1],
        [-1, 0],
        [0, -1]
    )

    action_space: ClassVar[int] = len(unit_possible_actions)

    flatten_state_shape: int = field(init=False)
    obs_high: int = field(init=False)
    num_agents: int = field(init=False)

    row_range: tuple[int, ...] = field(init=False)
    col_range: tuple[int, ...] = field(init=False)

    def __post_init__(self):
        super().__setattr__("flatten_state_shape", self.board_size[0] * self.board_size[1])
        super().__setattr__("obs_high", self.board_size[1] * self.num_lines)
        super().__setattr__("num_agents", self.board_size[1] * self.num_lines)
        super().__setattr__("row_range", tuple(range(self.board_size[0])))
        super().__setattr__("col_range", tuple(range(self.board_size[1])))
