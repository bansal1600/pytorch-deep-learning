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
# # V3-27: GRPO — Solution

# %%
import torch
import torch.nn as nn
from torch.utils.data import Dataset


class DummyDataGenerator:
    def __init__(self):
        torch.manual_seed(11)
        self.prompts = torch.randint(0, 20, (2, 5))
        self.completions = torch.randint(0, 20, (2, 4, 6))
        self.rewards = torch.randn(2, 4)
    def batch(self):
        return self.prompts, self.completions, self.rewards


class GRPODataset(Dataset):
    def __init__(self, prompts): self.prompts = prompts
    def __len__(self): return len(self.prompts)
    def __getitem__(self, i): return self.prompts[i]


class PolicyModel(nn.Module):
    def __init__(self, vocab=20, d=32):
        super().__init__()
        self.embed = nn.Embedding(vocab, d)
        self.head = nn.Linear(d, vocab)
    def forward(self, ids):
        return self.head(self.embed(ids))


class RewardModel(nn.Module):
    def __init__(self, d=32):
        super().__init__()
        self.r = nn.Linear(d, 1)
    def forward(self, h):
        return self.r(h.mean(1)).squeeze(-1)


def compute_group_advantages(rewards):
    mean = rewards.mean(dim=1, keepdim=True)
    std = rewards.std(dim=1, keepdim=True) + 1e-8
    return (rewards - mean) / std


def grpo_loss(logp, logp_old, advantages, beta=0.1, clip=0.2):
    ratio = (logp - logp_old).exp()
    surr = torch.min(ratio * advantages, torch.clamp(ratio, 1-clip, 1+clip) * advantages)
    return -(surr.mean())


prompts, comps, rewards = DummyDataGenerator().batch()
adv = compute_group_advantages(rewards)
print(f"group advantages shape: {adv.shape}")
print("✓ GRPO group-relative advantages")
