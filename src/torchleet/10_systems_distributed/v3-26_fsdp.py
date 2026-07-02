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
# # V3-26: FSDP from Scratch — Solution

# %%
import torch
import torch.nn as nn
from enum import Enum
from torch.utils.data import Dataset


class DummyDataGenerator:
    def __init__(self):
        self.x = torch.randn(8, 16)
    def input(self): return self.x


class TensorDataset(Dataset):
    def __init__(self, x): self.x = x
    def __len__(self): return len(self.x)
    def __getitem__(self, i): return self.x[i]


class ShardingStrategy(Enum):
    FULL_SHARD = "full_shard"
    SHARD_GRAD = "shard_grad"
    NO_SHARD = "no_shard"


class FakeDistributed:
    def __init__(self, world=2):
        self.world = world
    def all_gather(self, tensor, shard):
        return shard  # single-process stub
    def reduce_scatter(self, grad, world):
        return grad / world


class FSDPLinear(nn.Module):
    def __init__(self, in_f, out_f, strategy=ShardingStrategy.FULL_SHARD):
        super().__init__()
        self.strategy = strategy
        self.weight = nn.Parameter(torch.randn(out_f, in_f) * 0.02)
        self.bias = nn.Parameter(torch.zeros(out_f))
        self.dist = FakeDistributed()
    def forward(self, x):
        w = self.dist.all_gather(None, self.weight)
        return x @ w.T + self.bias


class FSDPModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = FSDPLinear(16, 32)
        self.fc2 = FSDPLinear(32, 8)
    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))


x = DummyDataGenerator().input()
model = FSDPModel()
print(f"FSDP output: {model(x).shape}")
print("✓ FSDP linear layers with sharding stub")
