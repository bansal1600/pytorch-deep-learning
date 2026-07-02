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
# # V3-22: Mamba (Selective SSM) — Solution

# %%
import torch
import torch.nn as nn
from torch.utils.data import Dataset


class DummyDataGenerator:
    def __init__(self, batch=2, seq=16, d=16):
        self.x = torch.randn(batch, seq, d)
    def input(self): return self.x


class SequenceDataset(Dataset):
    def __init__(self, x): self.x = x
    def __len__(self): return len(self.x)
    def __getitem__(self, i): return self.x[i]


def selective_scan(x, A, B, C):
    """Simplified selective scan along sequence."""
    Bsz, L, D = x.shape
    h = torch.zeros(Bsz, D, device=x.device)
    outputs = []
    for t in range(L):
        h = torch.exp(A) * h + B[:, t] * x[:, t]
        outputs.append(C[:, t] * h)
    return torch.stack(outputs, dim=1)


class MambaBlock(nn.Module):
    def __init__(self, d=16):
        super().__init__()
        self.in_proj = nn.Linear(d, d * 2)
        self.A = nn.Parameter(-torch.ones(d))
        self.B_proj = nn.Linear(d, d)
        self.C_proj = nn.Linear(d, d)
        self.out_proj = nn.Linear(d, d)
    def forward(self, x):
        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)
        B, C = self.B_proj(x), self.C_proj(x)
        y = selective_scan(x, self.A, B, C)
        return self.out_proj(y * torch.sigmoid(z))


class MambaModel(nn.Module):
    def __init__(self, d=16, layers=2):
        super().__init__()
        self.blocks = nn.ModuleList([MambaBlock(d) for _ in range(layers)])
    def forward(self, x):
        for b in self.blocks:
            x = x + b(x)
        return x


x = DummyDataGenerator().input()
print(f"Mamba output: {MambaModel()(x).shape}")
print("✓ Selective scan Mamba block")
