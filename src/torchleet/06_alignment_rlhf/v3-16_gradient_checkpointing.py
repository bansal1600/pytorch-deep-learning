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
# # V3-16: Gradient Checkpointing — Solution

# %%
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.checkpoint import checkpoint as torch_checkpoint


class DummyDataGenerator:
    def __init__(self):
        self.x = torch.randn(4, 16)
    def input(self): return self.x


class TensorDataset(Dataset):
    def __init__(self, x): self.x = x
    def __len__(self): return len(self.x)
    def __getitem__(self, i): return self.x[i]


class DeepBlock(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.lin = nn.Linear(d, d)
    def forward(self, x):
        return torch.relu(self.lin(x))


class DeepNetwork(nn.Module):
    def __init__(self, d=16, layers=8, use_ckpt=True):
        super().__init__()
        self.blocks = nn.ModuleList([DeepBlock(d) for _ in range(layers)])
        self.use_ckpt = use_ckpt
    def forward(self, x):
        for block in self.blocks:
            if self.use_ckpt:
                x = torch_checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)
        return x


x = DummyDataGenerator().input()
model = DeepNetwork(use_ckpt=True)
loss = model(x).sum()
loss.backward()
print("✓ Gradient checkpointing backward OK")
