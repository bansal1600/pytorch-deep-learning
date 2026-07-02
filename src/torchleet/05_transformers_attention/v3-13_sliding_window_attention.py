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
# # V3-13: Sliding Window Attention — Solution

# %%
import torch
import torch.nn as nn
from torch.utils.data import Dataset


class DummyDataGenerator:
    def __init__(self, s=12, d=16):
        self.q = torch.randn(1, 4, s, d)
        self.k = self.v = torch.randn(1, 4, s, d)
    def qkv(self): return self.q, self.k, self.v


class AttnDataset(Dataset):
    def __init__(self, q, k, v): self.q, self.k, self.v = q, k, v
    def __len__(self): return 1
    def __getitem__(self, i): return self.q, self.k, self.v


def create_sliding_window_mask(seq_len, window_size):
    idx = torch.arange(seq_len)
    dist = (idx.unsqueeze(0) - idx.unsqueeze(1)).abs()
    return (dist <= window_size).float()


def sliding_window_attention(q, k, v, window_size):
    S = q.size(-2)
    mask = create_sliding_window_mask(S, window_size).to(q.device)
    d = q.size(-1)
    scores = (q @ k.transpose(-2, -1)) / (d ** 0.5)
    scores = scores.masked_fill(mask == 0, float("-inf"))
    return torch.softmax(scores, dim=-1) @ v


class SWAModel(nn.Module):
    def __init__(self, window=2):
        super().__init__()
        self.window = window
    def forward(self, q, k, v):
        return sliding_window_attention(q, k, v, self.window)


q, k, v = DummyDataGenerator().qkv()
out = SWAModel(3)(q, k, v)
print(f"SWA output: {out.shape}")
print("✓ Sliding window attention")
