# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.4
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # V3-15: PPO for RLHF — Solution

# %%
import torch
import torch.nn as nn
from torch.utils.data import Dataset


class DummyDataGenerator:
    def __init__(self):
        torch.manual_seed(10)
        self.states = torch.randn(8, 16)
        self.actions = torch.randint(0, 5, (8,))
        self.rewards = torch.randn(8)
        self.values = torch.randn(8)
        self.logp_old = torch.randn(8)
    def rollout(self):
        return self.states, self.actions, self.rewards, self.values, self.logp_old


class RolloutDataset(Dataset):
    def __init__(self, s, a, r): self.s, self.a, self.r = s, a, r
    def __len__(self): return len(self.s)
    def __getitem__(self, i): return self.s[i], self.a[i], self.r[i]


class PolicyModel(nn.Module):
    def __init__(self, d=16, n_actions=5):
        super().__init__()
        self.net = nn.Linear(d, n_actions)
    def forward(self, s):
        return torch.log_softmax(self.net(s), dim=-1)


class ValueModel(nn.Module):
    def __init__(self, d=16):
        super().__init__()
        self.v = nn.Linear(d, 1)
    def forward(self, s):
        return self.v(s).squeeze(-1)


class RewardModel(nn.Module):
    def __init__(self, d=16):
        super().__init__()
        self.r = nn.Linear(d, 1)
    def forward(self, s):
        return self.r(s).squeeze(-1)


def compute_gae(rewards, values, gamma=0.99, lam=0.95):
    adv = torch.zeros_like(rewards)
    gae = 0
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * values[t+1] - values[t] if t < len(rewards)-1 else rewards[t] - values[t]
        gae = delta + gamma * lam * gae
        adv[t] = gae
    return adv


def ppo_step(policy, logp_old, states, actions, advantages, clip=0.2):
    logp = policy(states).gather(1, actions.unsqueeze(1)).squeeze(1)
    ratio = (logp - logp_old).exp()
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1-clip, 1+clip) * advantages
    return -torch.min(surr1, surr2).mean()


s, a, r, v, logp_old = DummyDataGenerator().rollout()
v_pad = torch.cat([v, torch.zeros(1)])
adv = compute_gae(r, v_pad)
policy = PolicyModel()
loss = ppo_step(policy, logp_old, s, a, adv)
print(f"PPO surrogate loss: {loss.item():.4f}")
print("✓ PPO step implemented")
