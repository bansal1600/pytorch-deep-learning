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
# # V1-29: Distributed Training (DDP pattern) — Solution
#
# CPU simulation of DDP gradient sync.

# %%
import torch
import torch.nn as nn
from torch.utils.data import Dataset


class DummyDataGenerator:
    def __init__(self, n=64):
        self.X = torch.randn(n, 8)
        self.y = torch.randint(0, 2, (n,))
    def tensors(self): return self.X, self.y


class ShardDataset(Dataset):
    def __init__(self, X, y, rank, world):
        self.X = X[rank::world]
        self.y = y[rank::world]
    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i], self.y[i]


class SimpleClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(8, 2)
    def forward(self, x):
        return self.fc(x)


def all_reduce_grads(model):
    for p in model.parameters():
        if p.grad is not None:
            p.grad = p.grad.clone()  # simulate sync on 1 process


X, y = DummyDataGenerator().tensors()
model = SimpleClassifier()
ds = ShardDataset(X, y, rank=0, world=2)
xb, yb = ds[0]
loss = nn.CrossEntropyLoss()(model(xb.unsqueeze(0)), yb.unsqueeze(0))
loss.backward()
all_reduce_grads(model)
print("✓ DDP-style gradient sync pattern demonstrated")
