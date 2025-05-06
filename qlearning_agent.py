import os
import sys
import numpy as np
import random
from collections import defaultdict
import matplotlib.pyplot as plt
import csv
import pickle
import json  # NEW: import json for saving configs
from env.minesweeper import Minesweeper

RUN_DIR = './results/Q-learning'
# Ensure the root results directory exists
os.makedirs(RUN_DIR, exist_ok=True)  # FIXED: changed ios to os

def state_to_key(state_array):
    return tuple(state_array.tolist())

# Add run_mode parameter to distinguish between experiments
def run_qlearning(config: dict, env_config: dict, run_mode: str, first_click_safe=True) -> None:
    reward_safe = env_config['reward_safe']
    reward_mine = env_config['reward_mine']
    reward_win = env_config['reward_win']
    rows = env_config['rows']
    cols = env_config['cols']
    mines = env_config['mines']

    # Initialize environment with dynamic rewards
    env = Minesweeper(rows, cols, mines, first_click_safe=first_click_safe,
                      visualize=False,
                      reward_safe=reward_safe,
                      reward_mine=reward_mine,
                      reward_win=reward_win)
    Q = defaultdict(float)

    epsilon = config['epsilon_start']
    win_rates = []
    eval_log = []

    # Main training loop
    for ep in range(1, config['num_episodes'] + 1):
        state = env.reset().flatten()
        state_key = state_to_key(state)
        done = False

        while not done:
            valid = np.where(env.playerfield.flatten() == env.Tile.UNOPENED)[0]
            if random.random() < epsilon:
                action = random.choice(valid)
            else:
                q_vals = [Q[(state_key, a)] for a in valid]
                action = valid[int(np.argmax(q_vals))]

            next_field, reward, _, info = env.step(action)
            next_state = next_field.flatten()
            next_key = state_to_key(next_state)

            next_valid = np.where(env.playerfield.flatten() == env.Tile.UNOPENED)[0]
            if len(next_valid) > 0 and not info.get('exploded', False):
                next_max = max(Q[(next_key, a)] for a in next_valid)
            else:
                next_max = 0.0

            old_val = Q[(state_key, action)]
            Q[(state_key, action)] = old_val + config['alpha'] * (
                reward + config['gamma'] * next_max - old_val
            )

            state_key = next_key
            if info.get('exploded', False) or info.get('win', False):
                done = True

        # Decay exploration rate
        epsilon = max(config['epsilon_min'], epsilon * config['epsilon_decay'])

        # Periodic evaluation
        if ep % config['eval_every'] == 0:
            wins = 0
            for _ in range(config['eval_episodes']):
                s = env.reset().flatten()
                sk = state_to_key(s)
                d = False
                while not d:
                    valid = np.where(env.playerfield.flatten() == env.Tile.UNOPENED)[0]
                    q_vals = [Q[(sk, a)] for a in valid]
                    a = valid[int(np.argmax(q_vals))]
                    sf, _, _, inf = env.step(a)
                    sk = state_to_key(sf.flatten())
                    if inf.get('exploded', False):
                        d = True
                    elif inf.get('win', False):
                        d = True
                        wins += 1
            win_rate = wins / config['eval_episodes']
            win_rates.append((ep, win_rate))
            eval_log.append({'episode': ep, 'win_rate': win_rate, 'epsilon': epsilon})
            print(f"Episode {ep} | Eval Win rate: {win_rate:.2f} | Epsilon: {epsilon:.2f}")

    # Save results to a subdirectory named by run_mode
    run_path = os.path.join(RUN_DIR, run_mode)
    os.makedirs(run_path, exist_ok=True)

    # Write metrics CSV
    with open(os.path.join(run_path, "metrics.csv"), "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=['episode', 'win_rate', 'epsilon'])
        writer.writeheader()
        writer.writerows(eval_log)

    # Save Q-table
    with open(os.path.join(run_path, "q_table.pkl"), "wb") as f:
        pickle.dump(Q, f)

    # Save configs
    with open(os.path.join(run_path, "config.json"), "w") as f:  # NEW: save Q-learning config
        json.dump(config, f, indent=2)
    with open(os.path.join(run_path, "env_config.json"), "w") as f:  # NEW: save environment config
        json.dump(env_config, f, indent=2)

    # Plot win rates
    eps, rates = zip(*win_rates)
    plt.figure()
    plt.plot(eps, rates, alpha=0.4, label='Raw Win Rate')
    if len(rates) >= 5:
        sm = np.convolve(rates, np.ones(5)/5, mode='valid')
        eps_sm = eps[4:]
        plt.plot(eps_sm, sm, linewidth=2, label='5-Eval MA')
    plt.xlabel('Episode')
    plt.ylabel('Win Rate')
    plt.title(f'Q-Learning Win Rate ({run_mode})')
    plt.legend()
    plt.savefig(os.path.join(run_path, "win_rate.png"))
    plt.close()

    print(f"[✓] Q-learning run '{run_mode}' complete. Results saved to: {run_path}")

if __name__ == "__main__":
    # Q-learning configuration
    config = {
        'num_episodes': 100000,
        'eval_every': 100,
        'eval_episodes': 200,
        'alpha': 0.1,
        'gamma': 0.99,
        'epsilon_start': 1.0,
        'epsilon_min': 0.1,
        'epsilon_decay': 0.995
    }

    rows, cols = 4, 4
    mine_counts = [3, 4, 5]
    reward_schemes = {
        'sparse': {'reward_safe': 0.0, 'reward_mine': -1, 'reward_win': 1},
        'shaped': {'reward_safe': 0.1, 'reward_mine': -10, 'reward_win': 10}
    }

    # Run all combinations of mines and reward schemes
    for mines in mine_counts:
        for scheme_name, reward_config in reward_schemes.items():
            env_config = {
                'rows': rows,
                'cols': cols,
                'mines': mines,
                'reward_safe': reward_config['reward_safe'],
                'reward_mine': reward_config['reward_mine'],
                'reward_win': reward_config['reward_win']
            }
            run_mode = f"mines{mines}_{scheme_name}"
            run_qlearning(config, env_config, run_mode)
