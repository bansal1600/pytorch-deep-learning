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
# # V3-20: DDPM — Solution

# %%
import torch
import torch.nn as nn
from torch.utils.data import Dataset


class DummyDataGenerator:
    def __init__(self, n=64):
        angles = torch.rand(n) * 6.28
        self.data = torch.stack([torch.cos(angles), torch.sin(angles)], dim=1)
    def samples(self): return self.data


class PointDataset(Dataset):
    def __init__(self, data): self.data = data
    def __len__(self): return len(self.data)
    def __getitem__(self, i): return self.data[i]


def linear_beta_schedule(T, beta_start=1e-4, beta_end=0.02):
    return torch.linspace(beta_start, beta_end, T)


def q_sample(x0, t, betas):
    alpha = 1 - betas
    alpha_bar = torch.cumprod(alpha, dim=0)
    a = alpha_bar[t].unsqueeze(1)
    noise = torch.randn_like(x0)
    return torch.sqrt(a) * x0 + torch.sqrt(1 - a) * noise, noise


class SimpleDenoiser(nn.Module):
    def __init__(self, d=2, t_dim=16):
        super().__init__()
        self.t_emb = nn.Embedding(300, t_dim)
        self.net = nn.Sequential(nn.Linear(d + t_dim, 64), nn.ReLU(), nn.Linear(64, d))
    def forward(self, x, t):
        return self.net(torch.cat([x, self.t_emb(t)], dim=-1))


data = DummyDataGenerator().samples()
betas = linear_beta_schedule(100)
t = torch.randint(0, 100, (8,))
xt, noise = q_sample(data[:8], t, betas)
pred = SimpleDenoiser()(xt, t)
print(f"DDPM denoiser loss: {nn.MSELoss()(pred, noise).item():.4f}")
print("✓ DDPM forward noising + denoiser")
