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
# # V3-21: DDIM + Classifier-Free Guidance — Solution

# %%
import torch
import torch.nn as nn
from torch.utils.data import Dataset


class DummyDataGenerator:
    def __init__(self, n=32):
        self.x = torch.randn(n, 2)
        self.labels = torch.randint(0, 3, (n,))
    def tensors(self): return self.x, self.labels


class ConditionalDataset(Dataset):
    def __init__(self, x, y): self.x, self.y = x, y
    def __len__(self): return len(self.x)
    def __getitem__(self, i): return self.x[i], self.y[i]


class ConditionalDenoiser(nn.Module):
    def __init__(self, n_classes=3):
        super().__init__()
        self.label_emb = nn.Embedding(n_classes + 1, 8)  # +1 for null/uncond
        self.net = nn.Sequential(nn.Linear(2 + 8, 32), nn.ReLU(), nn.Linear(32, 2))
    def forward(self, x, t, labels):
        return self.net(torch.cat([x, self.label_emb(labels)], dim=-1))


def guided_predict(model, x, t, label, scale=3.0):
    B = x.size(0)
    null = torch.full_like(label, 3)
    eps_cond = model(x, t, label)
    eps_uncond = model(x, t, null)
    return eps_uncond + scale * (eps_cond - eps_uncond)


def ddim_sample_step(x, eps, beta, eta=0.0):
    alpha = 1 - beta
    return x - eps * beta  # simplified single-step update


x, labels = DummyDataGenerator().tensors()
model = ConditionalDenoiser()
eps = guided_predict(model, x, torch.zeros(len(x), dtype=torch.long), labels)
print(f"CFG eps: {eps.shape}")
print("✓ Classifier-free guidance prediction")
