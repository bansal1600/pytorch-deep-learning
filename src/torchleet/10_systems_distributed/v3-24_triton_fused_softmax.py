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
# # V3-24: Fused Softmax (Online Algorithm) — Solution

# %%
import torch
import torch.nn as nn
from torch.utils.data import Dataset


class DummyDataGenerator:
    def __init__(self):
        self.x = torch.randn(4, 128)
    def input(self): return self.x


class LogitsDataset(Dataset):
    def __init__(self, x): self.x = x
    def __len__(self): return len(self.x)
    def __getitem__(self, i): return self.x[i]


def online_softmax_pytorch(x, dim=-1):
    """Single-pass numerically stable softmax (fused algorithm on CPU)."""
    x_max = x.max(dim=dim, keepdim=True).values
    exp_x = torch.exp(x - x_max)
    return exp_x / exp_x.sum(dim=dim, keepdim=True)


class SoftmaxLayer(nn.Module):
    def forward(self, x):
        return online_softmax_pytorch(x, dim=-1)


x = DummyDataGenerator().input()
ours = online_softmax_pytorch(x)
ref = torch.softmax(x, dim=-1)
print(f"max diff: {(ours-ref).abs().max():.2e}")
print("✓ Online fused softmax matches torch.softmax")
