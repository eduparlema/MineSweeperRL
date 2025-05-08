from minesweeper import Minesweeper
import pygame
import numpy as np
import time
import pickle
import json
import torch
from PPOagent import PPOAgent
import os

def random_visualizer():

    env = Minesweeper(4, 4, 3, visualize=True)
    obs = env.reset()

    done = False
    while not done:
        pygame.event.pump() 
        unopened = np.flatnonzero(obs == env.Tile.UNOPENED)
        action = int(env.np_random.choice(unopened))
        obs, reward, done, _ = env.step(action)
        time.sleep(3)
        print(f"Action: {action}")

    env.plot_minefield()   
    pygame.time.wait(3000) 
    env.close()

def state_to_key(state_array):
    return tuple(state_array.tolist())

def q_learning_visualizer(q_table_path: str, env_config_path: str, first_click_safe=True):
    """
    Load a trained Q-table and visualize greedy play on Minesweeper.

    Args:
      q_table_path: Path to the .pkl file containing the Q dict
      env_config_path: Path to the JSON file with environment settings
      first_click_safe: Whether first click is guaranteed safe
    """
    # 1) Load environment configuration
    with open(env_config_path, 'r') as f:
        env_config = json.load(f)

    rows = env_config['rows']
    cols = env_config['cols']
    mines = env_config['mines']
    rs = env_config['reward_safe']
    rm = env_config['reward_mine']
    rw = env_config['reward_win']

    # 2) Load Q-table
    with open(q_table_path, 'rb') as f:
        Q = pickle.load(f)
    print(f"[✓] Loaded Q-table with {len(Q)} entries from {q_table_path}")

    # 3) Initialize GUI environment
    env = Minesweeper(rows, cols, mines,
                      first_click_safe=first_click_safe,
                      visualize=True,
                      reward_safe=rs,
                      reward_mine=rm,
                      reward_win=rw)
    obs = env.reset()

    # 4) Greedy rollout
    done = False
    state = obs.flatten()
    while not done:
        pygame.event.pump()      
        time.sleep(1)   

        # Find valid actions
        unopened = np.flatnonzero(obs == env.Tile.UNOPENED)
        # Lookup Q-values and pick argmax
        key = state_to_key(state)
        q_vals = [Q.get((key, a), 0.0) for a in unopened]
        action = unopened[int(np.argmax(q_vals))]

        # Step environment
        obs, reward, done, info = env.step(action)
        state = obs.flatten()
        print(f"Action: {action}, Reward: {reward:.2f}, Done: {done}, Info: {info}")

    # 5) Show final minefield
    env.plot_minefield(action if info.get('exploded', False) else None)
    pygame.time.wait(3000)
    env.close()

def ppo_visualizer(policy_path: str, config_path: str, env_config_path: str, first_click_safe=True):
    """
    Load a trained PPO policy and visualize it playing Minesweeper.
    Arguments:
      policy_path: path to the .pth file containing the state_dict
      config_path: path to the JSON file with PPO hyperparameters
      env_config_path: path to the JSON file with environment settings
    """
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    with open(env_config_path, 'r') as f:
        env_config = json.load(f)

    rows = env_config['rows']
    cols = env_config['cols']
    mines = env_config['mines']
    rs = env_config['reward_safe']
    rm = env_config['reward_mine']
    rw = env_config['reward_win']

   
    env = Minesweeper(rows, cols, mines,
                      first_click_safe=first_click_safe,
                      visualize=True,
                      reward_safe=rs,
                      reward_mine=rm,
                      reward_win=rw)
    obs = env.reset()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    action_size = rows * cols
    agent = PPOAgent(rows, cols, action_size, device, config)

    state_dict = torch.load(policy_path, map_location=device)
    agent.policy_old.load_state_dict(state_dict)
    agent.policy.load_state_dict(state_dict)
    agent.policy_old.eval()

    done = False
    state = obs.flatten()
    time.sleep(5)
    while not done:
        pygame.event.pump() 
        time.sleep(1)   

        valid = np.flatnonzero(obs == env.Tile.UNOPENED)
        action = agent.act(state, valid)
        obs, reward, done, info = env.step(action)
        state = obs.flatten()
        print(f"Action: {action}, Reward: {reward:.2f}, Done: {done}")

    # 5) Reveal the full minefield
    env.plot_minefield(action if info.get('exploded', False) else None)
    pygame.time.wait(3000)
    env.close()

if __name__ == '__main__':
    # Example paths for mines3_sparse

    run_dir = os.path.join('../results', 'Q-learning', 'mines3_sparse')
    q_table_path = os.path.join(run_dir, 'q_table.pkl')
    env_config_path = os.path.join(run_dir, 'env_config.json')

    q_learning_visualizer(q_table_path, env_config_path)

    # run_dir = os.path.join('../results', 'PPO', 'mines4_shaped')
    # policy_path = os.path.join(run_dir, 'policy.pth')
    # config_path = os.path.join(run_dir, 'config.json')
    # env_config_path = os.path.join(run_dir, 'env_config.json')

    # ppo_visualizer(policy_path, config_path, env_config_path)