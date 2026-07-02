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
# # V1-27: GAN — Solution

# %%
import torch
import torch.nn as nn
from torch.utils.data import Dataset


class DummyDataGenerator:
    def __init__(self, n=256):
        self.real = torch.randn(n, 8)
    def real_data(self): return self.real


class RealDataset(Dataset):
    def __init__(self, x): self.x = x
    def __len__(self): return len(self.x)
    def __getitem__(self, i): return self.x[i]


class Generator(nn.Module):
    def __init__(self, z=16, d=8):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(z, 32), nn.ReLU(), nn.Linear(32, d), nn.Tanh())
    def forward(self, z):
        return self.net(z)


class Discriminator(nn.Module):
    def __init__(self, d=8):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, 32), nn.LeakyReLU(0.2), nn.Linear(32, 1), nn.Sigmoid())
    def forward(self, x):
        return self.net(x)


real = DummyDataGenerator().real_data()
G, D = Generator(), Discriminator()
opt_g = torch.optim.Adam(G.parameters(), lr=2e-4)
opt_d = torch.optim.Adam(D.parameters(), lr=2e-4)
bce = nn.BCELoss()
for _ in range(50):
    z = torch.randn(32, 16)
    fake = G(z)
    loss_d = bce(D(real[:32]), torch.ones(32,1)) + bce(D(fake.detach()), torch.zeros(32,1))
    opt_d.zero_grad(); loss_d.backward(); opt_d.step()
    loss_g = bce(D(G(z)), torch.ones(32,1))
    opt_g.zero_grad(); loss_g.backward(); opt_g.step()
print(f"G loss={loss_g.item():.3f}, D loss={loss_d.item():.3f}")
print("✓ GAN training loop")
