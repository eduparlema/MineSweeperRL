from minesweeper import Minesweeper
import pygame
import numpy as np
import time
import pickle

def random_visualizer():

    env = Minesweeper(4, 4, 3, visualize=True)   # classic beginner board
    obs = env.reset()

    done = False
    while not done:
        pygame.event.pump()                    # keep window responsive
        unopened = np.flatnonzero(obs == env.Tile.UNOPENED)
        action = int(env.np_random.choice(unopened))
        obs, reward, done, _ = env.step(action)
        time.sleep(3)
        print(f"Action: {action}")

    env.plot_minefield()   # flash the full solution
    pygame.time.wait(3000) # keep the window for 3 s
    env.close()

def state_to_key(state_array):
    return tuple(state_array.tolist())

def q_learning_visualizer(Q, env_config):
    reward_safe = env_config['reward_safe']
    reward_mine = env_config['reward_mine']
    reward_win = env_config['reward_win']
    cols = env_config['cols']
    rows = env_config['rows']
    mines = env_config['mines']

    env = Minesweeper(cols, rows, mines, visualize=True, reward_safe=reward_safe, reward_mine=reward_mine, reward_win=reward_win)
    obs = env.reset()

    done = False
    while not done:
        pygame.event.pump()  # Keep the window responsive
        time.sleep(0.5)

        # Compute greedy action from Q-table
        unopened = np.flatnonzero(obs == env.Tile.UNOPENED)
        state_key = state_to_key(obs.flatten())
        q_vals = [Q[(state_key, a)] for a in unopened]
        action = unopened[int(np.argmax(q_vals))]

        obs, reward, done, info = env.step(action)
        print(f"Action: {action}, Reward: {reward:.2f}, Done: {done}, Info: {info}")

    # === Visualize the minefield solution ===
    env.plot_minefield()
    pygame.time.wait(3000)
    env.close()

if __name__ == '__main__':
    config = {
        'num_episodes': 50000,
        'eval_every': 100,
        'eval_episodes': 200,
        'alpha': 0.1,          # learning rate
        'gamma': 0.99,         # discount factor
        'epsilon_start': 1.0,  # initial exploration
        'epsilon_min': 0.1,    # minimum exploration
        'epsilon_decay': 0.995 # per-episode decay
    }

    env_config = {
        'reward_safe': 0.1,
        'reward_mine': -50,
        'reward_win': 10,
        'rows': 4,
        'cols': 4,
        'mines': 3
    }

    with open("../results/Q-learning/q_table.pkl", "rb") as f:
        Q = pickle.load(f)

    q_learning_visualizer(Q, env_config)
