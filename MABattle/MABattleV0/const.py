import numpy as np

BOARD_SIZE = (8, 8)
BOARD_SHAPE = np.array(BOARD_SIZE)
FLATTEN_STATE_SHAPE = BOARD_SIZE[0] * BOARD_SIZE[1]

NUM_LINES = 2

ROW_RANGE = list(range(BOARD_SHAPE[0]))
COL_RANGE = list(range(BOARD_SHAPE[1]))

UNIT_POSSIBLE_ACTIONS = np.array([
    [0, 0],
    [1, 0],
    [0, 1],
    [-1, 0],
    [0, -1]
])

ACTION_SPACE = len(UNIT_POSSIBLE_ACTIONS)
OBS_HIGH = BOARD_SHAPE[1] * NUM_LINES

NUM_AGENTS = BOARD_SIZE[1] * NUM_LINES

WIN_REWARD = 0.0
CAPTURE_REWARD = 10.0

UNIT_IMAGE_SIZE = (32, 32)
BLUE_UNIT_IMAGE = r"images/blue.png"
RED_UNIT_IMAGE = r"images/red.png"
CELL_IMAGE = r"images/cell.png"
