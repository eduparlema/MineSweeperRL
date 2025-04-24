from minesweeper import Minesweeper
import pygame
import numpy as np
import time

env = Minesweeper(8, 8, 10, visualize=True)   # classic beginner board
obs = env.reset()

done = False
while not done:
    pygame.event.pump()                    # keep window responsive
    unopened = np.flatnonzero(obs == env.Tile.UNOPENED)
    action = int(env.np_random.choice(unopened))
    obs, reward, done, _ = env.step(action)
    env.plot_playerfield()
    time.sleep(3)
    print(f"Action: {action}")

env.plot_minefield()   # flash the full solution
pygame.time.wait(3000) # keep the window for 3 s
env.close()