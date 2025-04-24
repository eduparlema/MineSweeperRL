import pygame
import numpy as np
from enum import IntEnum
from collections import deque

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

        self.num_moves         = 0
        self.exploded         = False
        self.done             = False
        self.mines_placed    = False

        # Reward constants
        self.REWARD_SAFE = 0.1
        self.REWARD_MINE = -1
        self.REWARD_WIN = 1

        self.directions = [[-1,-1], [0,-1], [1,-1],
                           [-1,0], [1,0],
                           [-1,1], [0,1], [1,1]
                           ]

        self.gui = visualize
        if visualize:
            self.init_gui() # TODO: Implement this

    def reset(self):
        """
        Start a fresh episode.

        This returns a copy of the player field, where every tile is unopened.
        """
        self.exploded = False
        self.done = False
        self.num_moves = 0
        self.mines_placed = False # Mine will be added after first click

        self.playerfield.fill(self.Tile.UNOPENED)
        self.minefield.fill(0)

        return self.playerfield.copy()
    
    def step(self, action: int):
        if self.done:
            raise RuntimeError("Episode is done; call reset() to start a new one!")
        
        r, c = divmod(action, self.cols)
        if not self.is_valid(r, c):
            raise ValueError("Action index out of bounds")
        
        # Place mines on the first legal reveal
        if not self.mines_placed:
            self.place_mines((r, c))
            self.mines_placed = True

        info: dict[str, object] = {}

        # Illegal action (already opened) TODO: REMOVE THIS? ONLY LET AGENT CHOOSE VALID ACTIONS?
        if self.playerfield[r, c] != self.Tile.UNOPENED:
            reward = self.REWARD_MINE
            info["illegal"] = True
            return self.playerfield.copy(), reward, self.done, info
        
        # Hittin a mine
        if self.minefield[r, c] == self.Tile.MINE:
            self.playerfield[r, c] = self.Tile.MINE
            self.exploded = True
            self.done = True
            reward = self.REWARD_MINE
            info["exploded"] = True
            self.num_moves += 1
            return self.playerfield.copy(), reward, self.done, info

        # Safe reveal + flood-fill
        new_clues = self.reveal(r, c)
        reward = len(new_clues) * self.REWARD_SAFE
        self.num_moves += 1

        # Win condition - all non-mine tiles revealed
        unopened = np.count_nonzero(self.playerfield == self.Tile.UNOPENED)
        if unopened == self.num_mines:
            self.done = True
            reward += self.REWARD_WIN
            info["win"] = True
        
        info["newly_revealed"] = True
        return self.playerfield.copy(), reward, self.done, info



    def place_mines(self, first_click: tuple[int, int]):
        """
        Place mines uniformly at random, excluding first_click
        """
        fc_flat = first_click[0] * self.cols + first_click[1]

        # Get possible tiles (flat index) to place a mine
        candidates = np.arange(self.rows * self.cols)
        if self.first_click_safe:
            candidates = np.delete(candidates, fc_flat)
        
        mine_indices = self.np_random.choice(candidates, self.num_mines, replace=False)
        self.minefield.flat[mine_indices] = self.Tile.MINE

        # Write clue numbers around the mines:
        for idx in mine_indices:
            # formule to flat -> x[0] * cols + x[1]
            r, c = divmod(idx, self.cols)
            for dr, dc in self.directions:
                new_r = r + dr
                new_c = c + dc
                if self.is_valid(r, c) and self.minefield[new_r, new_c] != self.Tile.MINE:
                    self.minefield[new_r, new_c] += 1

    def is_valid(self, r, c):
        return r >= 0 and r < self.rows and c >= 0 and c < self.cols
            
    def reveal(self, r, c):
        """
        Flood-fill reveal starting from r, c. If the player clicks on a tile that
        contains a 0 (i.e., no mines around it), then it automatically reveals the
        tiles around it.

        Return list of coordinates that have been opened.
        """
        q = deque([r, c])
        opened = [] # list[(int, int)]
        while q:
            pr, pc = q.popleft()
            if self.playerfield[pr, pc] == self.Tile.UNOPENED:
                continue 

            self.playerfield[pr, pc] = self.minefield[pr, pc] # Show number to agent
            opened.append((pr, pc))

            # Continue flood only through zero-clue cells
            if self.minefield[pr, pc] == 0:
                for dr, dc in self.directions:
                    new_r, new_c = r + dr, c + dc
                    if self.playerfield[new_r, new_c] == self.Tile.UNOPENED:
                        q.append((new_r, new_c))
        return opened

    def init_gui(self):
        pass
        