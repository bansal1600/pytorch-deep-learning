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
# # V1-35: Variational Autoencoder — Solution

# %%
import torch
import torch.nn as nn
from torch.utils.data import Dataset


class DummyDataGenerator:
    def __init__(self, n=128):
        self.x = torch.randn(n, 8)
    def data(self): return self.x


class VAEDataset(Dataset):
    def __init__(self, x): self.x = x
    def __len__(self): return len(self.x)
    def __getitem__(self, i): return self.x[i]


class Encoder(nn.Module):
    def __init__(self, d=8, z=4):
        super().__init__()
        self.fc = nn.Linear(d, 32)
        self.mu = nn.Linear(32, z)
        self.logvar = nn.Linear(32, z)
    def forward(self, x):
        h = torch.relu(self.fc(x))
        return self.mu(h), self.logvar(h)


class Decoder(nn.Module):
    def __init__(self, z=4, d=8):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(z, 32), nn.ReLU(), nn.Linear(32, d))
    def forward(self, z):
        return self.net(z)


class VAEModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = Encoder()
        self.dec = Decoder()
    def reparameterize(self, mu, logvar):
        std = (0.5 * logvar).exp()
        return mu + std * torch.randn_like(std)
    def forward(self, x):
        mu, logvar = self.enc(x)
        z = self.reparameterize(mu, logvar)
        return self.dec(z), mu, logvar


def vae_loss(x, recon, mu, logvar):
    recon_l = nn.MSELoss()(recon, x)
    kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(1).mean()
    return recon_l + kl


x = DummyDataGenerator().data()
model = VAEModel()
recon, mu, lv = model(x)
print(f"VAE loss: {vae_loss(x, recon, mu, lv).item():.4f}")
print("✓ VAE with reparameterization")
