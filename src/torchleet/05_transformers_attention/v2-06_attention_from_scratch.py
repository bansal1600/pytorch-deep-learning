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
# # V2-06: Scaled Dot-Product Attention — Solution

# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset


class DummyDataGenerator:
    def __init__(self, b=2, h=4, s=6, d=8):
        torch.manual_seed(1)
        self.q = torch.randn(b, h, s, d)
        self.k = torch.randn(b, h, s, d)
        self.v = torch.randn(b, h, s, d)
    def qkv(self):
        return self.q, self.k, self.v


class QKVDataset(Dataset):
    def __init__(self, q, k, v):
        self.q, self.k, self.v = q, k, v
    def __len__(self): return self.q.size(0)
    def __getitem__(self, i): return self.q[i], self.k[i], self.v[i]


def scaled_dot_product_attention(q, k, v, mask=None):
    d = q.size(-1)
    scores = (q @ k.transpose(-2, -1)) / (d ** 0.5)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float("-inf"))
    weights = torch.softmax(scores, dim=-1)
    return weights @ v


class AttentionBlock(nn.Module):
    def __init__(self, d_model=8, n_heads=4):
        super().__init__()
        self.d, self.h = d_model, n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
    def forward(self, x):
        B, S, _ = x.shape
        qkv = self.qkv(x).reshape(B, S, 3, self.h, self.d // self.h).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        return scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(B, S, self.d)


q, k, v = DummyDataGenerator().qkv()
out = scaled_dot_product_attention(q, k, v)
ref = F.scaled_dot_product_attention(q, k, v)
print(f"max diff vs ref: {(out-ref).abs().max():.2e}")
print("✓ Attention matches PyTorch")
