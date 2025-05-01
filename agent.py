import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import matplotlib.pyplot as plt

# Ensure the `env` folder is in Python path to import Minesweeper environment
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'env'))
from minesweeper import Minesweeper

# PPO configuration
config = {
    'num_episodes': 10000,
    'eval_every': 50,
    'eval_episodes': 100,
    'max_resets_per_episode': 25,
    'gamma': 0.99,
    'lambda': 0.95,
    'lr': 1e-4,
    'eps_clip': 0.1,
    'value_clip': 0.2,
    'entropy_coef': 0.01,
    'update_timestep': 20000,
    'K_epochs': 4
}

# Environment dimensions & settings
rows, cols, mines = 4, 4, 3
first_click_safe = True

def compute_gae(rewards, values, next_values, dones, gamma, lam):
    gae = 0
    returns = []
    for step in reversed(range(len(rewards))):
        mask = 1.0 - dones[step]
        delta = rewards[step] + gamma * next_values[step] * mask - values[step]
        gae = delta + gamma * lam * mask * gae
        returns.insert(0, gae + values[step])
    return returns


class ActorCritic(nn.Module):
    def __init__(self, rows, cols, action_size):
        super().__init__()
        # conv: 1→32→64 on 4×4 → 3×3 → 2×2 map
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=2),
            nn.ReLU()
        )
        conv_out_size = 64 * (rows - 2) * (cols - 2)
        self.fc = nn.Linear(conv_out_size, 256)
        self.action_head = nn.Linear(256, action_size)
        self.value_head = nn.Linear(256, 1)

    def forward(self, x):
        batch = x.size(0)
        size = int(np.sqrt(x.size(1)))
        grid = x.view(batch, 1, size, size)
        h = self.conv(grid)
        h = h.view(batch, -1)
        h = F.relu(self.fc(h))
        logits = self.action_head(h)
        probs = F.softmax(logits, dim=-1)
        value = self.value_head(h).squeeze(1)
        return probs, value


class PPOAgent:
    def __init__(self, rows, cols, action_size, device, config):
        self.policy = ActorCritic(rows, cols, action_size).to(device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=config['lr'])
        self.scheduler = optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lr_lambda=lambda ep: 1 - (ep / config['num_episodes'])
        )
        self.policy_old = ActorCritic(rows, cols, action_size).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.memory = []
        self.eps_clip = config['eps_clip']
        self.value_clip = config['value_clip']
        self.gamma = config['gamma']
        self.lam = config['lambda']
        self.K_epochs = config['K_epochs']
        self.entropy_coef = config['entropy_coef']
        self.device = device

    def act(self, state, valid_actions):
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action_probs, _ = self.policy_old(state_tensor)
        mask = torch.zeros_like(action_probs)
        mask[0, valid_actions] = 1
        prot = action_probs * mask
        prot = prot / (prot.sum() + 1e-8)
        action = torch.multinomial(prot, 1).item()
        return action

    def update(self):
        states_list, actions_list, rewards_list, dones_list, next_states_list = zip(*self.memory)
        states = torch.FloatTensor(np.stack(states_list)).to(self.device)
        next_states = torch.FloatTensor(np.stack(next_states_list)).to(self.device)
        actions = torch.LongTensor(actions_list).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards_list).to(self.device)
        dones = torch.FloatTensor(dones_list).to(self.device)

        with torch.no_grad():
            old_probs_all, old_values = self.policy_old(states)
            old_probs = old_probs_all.gather(1, actions).squeeze(1)
            _, next_values = self.policy_old(next_states)

        returns = compute_gae(rewards, old_values, next_values, dones,
                              self.gamma, self.lam)
        returns = torch.FloatTensor(returns).to(self.device)
        advantages = returns - old_values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        for _ in range(self.K_epochs):
            probs_all, values = self.policy(states)
            probs = probs_all.gather(1, actions).squeeze(1)
            ratios = probs / (old_probs + 1e-8)
            s1 = ratios * advantages
            s2 = torch.clamp(ratios, 1 - self.eps_clip,
                             1 + self.eps_clip) * advantages
            policy_loss = -torch.min(s1, s2).mean()

            values_clipped = old_values + (values - old_values).clamp(
                -self.value_clip, self.value_clip)
            value_loss = F.mse_loss(values, returns) + \
                         F.mse_loss(values_clipped, returns)

            entropy = -(probs_all * torch.log(probs_all + 1e-8))
            entropy = entropy.sum(dim=1).mean()
            loss = policy_loss + 0.5 * value_loss - self.entropy_coef * entropy

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        self.scheduler.step()
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.memory.clear()


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    env = Minesweeper(rows, cols, mines, first_click_safe=first_click_safe, visualize=False)
    action_size = rows * cols
    agent = PPOAgent(rows, cols, action_size, device, config)

    training_rewards, win_rates = [], []
    timestep = 0

    for ep in range(1, config['num_episodes'] + 1):
        state = env.reset().flatten()
        total_reward, won, resets = 0.0, False, 0

        while not won and resets < config['max_resets_per_episode']:
            valid = np.where(env.playerfield.flatten() == env.Tile.UNOPENED)[0]
            action = agent.act(state, valid)
            next_field, reward, _, info = env.step(action)

            next_state = next_field.flatten()
            agent.memory.append((state, action, reward,
                                 info.get('exploded', False), next_state))

            total_reward += reward
            timestep += 1
            state = next_state

            if timestep % config['update_timestep'] == 0:
                agent.update()

            if info.get('exploded', False):
                resets += 1
                state = env.reset().flatten()
            if info.get('win', False):
                won = True

        training_rewards.append(total_reward)

        if ep % config['eval_every'] == 0:
            wins = 0
            for _ in range(config['eval_episodes']):
                s = env.reset().flatten(); w, r = False, 0
                while not w and r < config['max_resets_per_episode']:
                    valid = np.where(env.playerfield.flatten() == env.Tile.UNOPENED)[0]
                    a = agent.act(s, valid)
                    s_field, _, _, info = env.step(a)
                    s = s_field.flatten()
                    if info.get('exploded', False):
                        r += 1; s = env.reset().flatten()
                    if info.get('win', False):
                        wins += 1; w = True
            win_rate = wins / config['eval_episodes']
            win_rates.append((ep, win_rate))
            print(f"Episode {ep} | Reward: {total_reward:.2f} | Win rate: {win_rate:.2f}")

        # save performance plots for rewards
    plt.figure()
    plt.plot(training_rewards)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('PPO Training Rewards')
    plt.savefig('ppo_rewards.png')

        # plot raw win rate and smoothed moving average on same graph
    eps, rates = zip(*win_rates)
    plt.figure()
    plt.plot(eps, rates, alpha=0.4, label='Raw Win Rate')
    window = 5  # smoothing window in evaluation points
    if len(rates) >= window:
        smoothed = np.convolve(rates, np.ones(window)/window, mode='valid')
        eps_smoothed = eps[window-1:]
        plt.plot(eps_smoothed, smoothed, linewidth=2, label=f'{window}-Eval MA')
    plt.xlabel('Episode')
    plt.ylabel('Win Rate')
    plt.title('PPO Win Rate with Moving Average')
    plt.legend()
    plt.savefig('ppo_win_rate.png')

    print("Training and evaluation complete. Plots saved.")
