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
# # V2-02: RMS Norm — Solution

# %%
import torch
import torch.nn as nn
from torch.utils.data import Dataset


class DummyDataGenerator:
    def __init__(self, batch=4, seq=8, dim=16):
        torch.manual_seed(5)
        self.x = torch.randn(batch, seq, dim)
    def input(self):
        return self.x


class TensorDataset(Dataset):
    def __init__(self, x):
        self.x = x
    def __len__(self): return len(self.x)
    def __getitem__(self, i): return self.x[i]


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-8):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))
    def forward(self, x):
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (x / rms) * self.scale


class TransformerBlockStub(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = RMSNorm(dim)
        self.ff = nn.Linear(dim, dim)
    def forward(self, x):
        return self.ff(self.norm(x))


x = DummyDataGenerator().input()
block = TransformerBlockStub(16)
out = block(x)
print(f"RMSNorm output shape: {out.shape}")
print("✓ RMSNorm integrated in block")
