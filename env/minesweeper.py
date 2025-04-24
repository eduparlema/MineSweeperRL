import pygame
import numpy as np
from enum import IntEnum

class Minesweeper:
    """
    RL-friendly Minesweeper environment (no flags).

    One action = pick an unopened tile to reveal.
    """
    class Tile(IntEnum):
        MINE = -1
        UNOPENED = 9
        # Integers 0 - 8 will represent the actual numbers (neighboring mines)
    
    def __init__(self, num_rows, num_cols, num_mines, first_click_safe=True, visualize=False, seed=None):

        if num_mines <= 0 or num_mines >= num_rows * num_cols:
            raise ValueError("`mine_count` must be in (0, rows*cols)")
        
        self.rows = num_rows
        self.cols = num_cols
        self.num_mines = num_mines
        self.first_click_safe = first_click_safe
        # Random numger generator
        self.np_random = np.random.RandomState(seed)
        self.minefield = np.zeros((num_rows, num_cols), dtype=np.int8) # Mine placement
        self.playerfield = np.full((num_rows, num_cols), self.Tile.UNOPENED, dtype=np.int8) # What the agent sees

        self.move_num         = 0
        self.exploded         = False
        self.done             = False
        self._mines_placed    = False

        # Reward constants
        self._REWARD_SAFE = 0.1
        self._REWARD_MINE = -1
        self._REWARD_WIN = 1

        self.gui = visualize
        if visualize:
            self._init_gui() # TODO: Implement this