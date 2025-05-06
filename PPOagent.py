import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import matplotlib.pyplot as plt
import csv
import json

from env.minesweeper import Minesweeper

# PPO configuration
config = {
    'num_episodes': 2000,
    'eval_every': 10,
    'eval_episodes': 100,
    'max_resets_per_episode': 25,
    'gamma': 0.99,
    'lambda': 0.95,
    'lr': 1e-4,
    'eps_: 0.1,clip'
    'value_clip': 0.2,
    'entropy_coef': 0.01,
    'update_timestep': 20000,
    'K_epochs': 4
}

# Environment dimensions & settings
rows, cols, mines = 8, 8, 10
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

def run_ppo(config: dict, env_config: dict, run_mode: str, first_click_safe=True):
    rows = env_config['rows']
    cols = env_config['cols']
    mines = env_config['mines']
    rs = env_config['reward_safe']
    rm = env_config['reward_mine']
    rw = env_config['reward_win']

    # create result directory
    root = os.path.join('results', 'PPO', run_mode)
    os.makedirs(root, exist_ok=True)
    # save configs
    with open(os.path.join(root, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)
    with open(os.path.join(root, 'env_config.json'), 'w') as f:
        json.dump(env_config, f, indent=2)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    env = Minesweeper(rows, cols, mines, first_click_safe=first_click_safe,
                      visualize=False,
                      reward_safe=rs,
                      reward_mine=rm,
                      reward_win=rw)
    action_size = rows * cols
    agent = PPOAgent(rows, cols, action_size, device, config)

    win_rates = []
    episode_rewards = []
    timestep = 0

    # training loop
    for ep in range(1, config['num_episodes'] + 1):
        state = env.reset().flatten()
        total_reward = 0.0
        done = False
        resets = 0

        while not done and resets < config['max_resets_per_episode']:
            valid = np.where(env.playerfield.flatten() == env.Tile.UNOPENED)[0]
            action = agent.act(state, valid)
            next_field, reward, _, info = env.step(action)
            next_state = next_field.flatten()

            agent.memory.append((state, action, reward,
                                  info.get('exploded', False),
                                  next_state))
            state = next_state
            total_reward += reward
            timestep += 1

            if timestep % config['update_timestep'] == 0:
                agent.update()

            if info.get('exploded', False):
                resets += 1
                state = env.reset().flatten()
            if info.get('win', False):
                done = True

        episode_rewards.append(total_reward)

        # evaluation
        if ep % config['eval_every'] == 0:
            wins = 0
            for _ in range(config['eval_episodes']):
                s = env.reset().flatten()
                d_eval = False
                res_eval = 0
                while not d_eval and res_eval < config['max_resets_per_episode']:
                    valid = np.where(env.playerfield.flatten() == env.Tile.UNOPENED)[0]
                    act = agent.act(s, valid)
                    sf, _, _, info = env.step(act)
                    s = sf.flatten()
                    if info.get('exploded', False):
                        res_eval += 1
                        s = env.reset().flatten()
                    if info.get('win', False):
                        wins += 1
                        d_eval = True
            win_rate = wins / config['eval_episodes']
            win_rates.append((ep, win_rate))
            print(f"PPO {run_mode} Ep {ep} | Win rate: {win_rate:.2f}")

    # save reward plot
    plt.figure()
    plt.plot(episode_rewards)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title(f'PPO Rewards ({run_mode})')
    plt.savefig(os.path.join(root, 'ppo_rewards.png'))
    plt.close()

    # save win rates CSV and plot
    with open(os.path.join(root, 'win_rates.csv'), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['episode', 'win_rate'])
        writer.writerows(win_rates)
    if win_rates:
        eps, rates = zip(*win_rates)
    else:
        eps, rates = [], []
    plt.figure()
    plt.plot(eps, rates, alpha=0.4, label='Raw')
    if len(rates) >= 5:
        sm = np.convolve(rates, np.ones(5)/5, mode='valid')
        plt.plot(eps[4:], sm, linewidth=2, label='5-Eval MA')
    plt.xlabel('Episode')
    plt.ylabel('Win Rate')
    plt.title(f'PPO Win Rate ({run_mode})')
    plt.legend()
    plt.savefig(os.path.join(root, 'ppo_win_rate.png'))
    plt.close()

    # save policy
    torch.save(agent.policy_old.state_dict(), os.path.join(root, 'policy.pth'))

    print(f"[✓] PPO run '{run_mode}' complete. Results saved to: {root}")

if __name__ == "__main__":
    # Base PPO configuration
    config = {
        'num_episodes': 1000,
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
    # Experiment settings
    rows, cols = 4, 4
    mine_counts = [3, 4, 5]
    reward_schemes = {
        'sparse': {'reward_safe': 0.0, 'reward_mine': -1.0, 'reward_win': 1.0},
        'shaped': {'reward_safe': 0.1, 'reward_mine': -5.0, 'reward_win': 10.0}
    }
    # Run all combinations
    for mines in mine_counts:
        for scheme_name, reward_cfg in reward_schemes.items():
            env_config = {
                'rows': rows,
                'cols': cols,
                'mines': mines,
                'reward_safe': reward_cfg['reward_safe'],
                'reward_mine': reward_cfg['reward_mine'],
                'reward_win': reward_cfg['reward_win']
            }
            run_mode = f"mines{mines}_{scheme_name}"
            run_ppo(config, env_config, run_mode)