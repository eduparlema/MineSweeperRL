import os
import sys
import numpy as np
import random
from collections import defaultdict, deque
import matplotlib.pyplot as plt

# Ensure the `env` folder is in Python path to import Minesweeper environment
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'env'))
from minesweeper import Minesweeper

# Q-learning configuration
config = {
    'num_episodes': 20000,
    'eval_every': 100,
    'eval_episodes': 200,
    'alpha': 0.1,          # learning rate
    'gamma': 0.99,         # discount factor
    'epsilon_start': 1.0,  # initial exploration
    'epsilon_min': 0.1,    # minimum exploration
    'epsilon_decay': 0.995 # per-episode decay
}

# Environment dimensions & settings
rows, cols, mines = 4, 4, 3
first_click_safe = True

def state_to_key(state_array):
    # convert flattened numpy array to immutable tuple for dict key
    return tuple(state_array.tolist())

if __name__ == "__main__":
    env = Minesweeper(rows, cols, mines, first_click_safe=first_click_safe, visualize=False)
    Q = defaultdict(float)  # Q[(state_key, action)] = value

    epsilon = config['epsilon_start']
    win_rates = []
    total_wins = 0

    for ep in range(1, config['num_episodes'] + 1):
        state = env.reset().flatten()
        state_key = state_to_key(state)
        done = False

        while not done:
            # epsilon-greedy over valid unopened positions
            valid = np.where(env.playerfield.flatten() == env.Tile.UNOPENED)[0]
            if random.random() < epsilon:
                action = random.choice(valid)
            else:
                q_vals = [Q[(state_key, a)] for a in valid]
                action = valid[int(np.argmax(q_vals))]

            next_field, reward, _, info = env.step(action)
            next_state = next_field.flatten()
            next_key = state_to_key(next_state)

            # determine max Q for next state
            next_valid = np.where(env.playerfield.flatten() == env.Tile.UNOPENED)[0]
            if len(next_valid) > 0 and not info.get('exploded', False):
                next_max = max(Q[(next_key, a)] for a in next_valid)
            else:
                next_max = 0.0

            # Q-learning update
            old_val = Q[(state_key, action)]
            Q[(state_key, action)] = old_val + config['alpha'] * (
                reward + config['gamma'] * next_max - old_val
            )

            state_key = next_key

            if info.get('exploded', False):
                done = True
            elif info.get('win', False):
                done = True
                total_wins += 1

        # decay exploration rate
        epsilon = max(config['epsilon_min'], epsilon * config['epsilon_decay'])

        # evaluate at intervals
        if ep % config['eval_every'] == 0:
            wins = 0
            for _ in range(config['eval_episodes']):
                s = env.reset().flatten()
                sk = state_to_key(s)
                d = False
                while not d:
                    valid = np.where(env.playerfield.flatten() == env.Tile.UNOPENED)[0]
                    # greedy action
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
            print(f"Episode {ep} | Eval Win rate: {win_rate:.2f} | Epsilon: {epsilon:.2f}")

    # plot evaluation win rates
    eps, rates = zip(*win_rates)
    plt.figure()
    plt.plot(eps, rates, alpha=0.4, label='Raw Win Rate')
    window = 5
    if len(rates) >= window:
        sm = np.convolve(rates, np.ones(window)/window, mode='valid')
        eps_sm = eps[window-1:]
        plt.plot(eps_sm, sm, linewidth=2, label=f'{window}-Eval MA')
    plt.xlabel('Episode')
    plt.ylabel('Win Rate')
    plt.title('Q-Learning Eval Win Rate')
    plt.legend()
    plt.savefig('qlearning_win_rate.png')
    print("Training complete. Win-rate plot saved to qlearning_win_rate.png.")
