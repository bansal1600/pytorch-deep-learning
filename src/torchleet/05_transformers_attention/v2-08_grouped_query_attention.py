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
# # V2-08: Grouped Query Attention — Solution

# %%
import torch
import torch.nn as nn
from torch.utils.data import Dataset


class DummyDataGenerator:
    def __init__(self):
        torch.manual_seed(3)
        self.q = torch.randn(1, 8, 4, 16)  # B, n_heads, S, d
        self.k = torch.randn(1, 2, 4, 16)  # B, n_kv_heads, S, d
        self.v = torch.randn(1, 2, 4, 16)
    def qkv(self): return self.q, self.k, self.v


class GQADataset(Dataset):
    def __init__(self, q, k, v):
        self.q, self.k, self.v = q, k, v
    def __len__(self): return 1
    def __getitem__(self, i): return self.q, self.k, self.v


def grouped_query_attention(q, k, v, num_query_groups):
    n_qh = q.size(1)
    n_kvh = k.size(1)
    repeat = n_qh // n_kvh
    k = k.repeat_interleave(repeat, dim=1)
    v = v.repeat_interleave(repeat, dim=1)
    d = q.size(-1)
    scores = (q @ k.transpose(-2, -1)) / (d ** 0.5)
    return torch.softmax(scores, dim=-1) @ v


class GQAModel(nn.Module):
    def __init__(self, d=16, n_q_heads=8, n_kv_heads=2):
        super().__init__()
        self.n_q_heads, self.n_kv_heads = n_q_heads, n_kv_heads
        self.wq = nn.Linear(d, d)
        self.wk = nn.Linear(d, d * n_kv_heads // n_q_heads)
        self.wv = nn.Linear(d, d * n_kv_heads // n_q_heads)
    def forward(self, x):
        B, S, D = x.shape
        q = self.wq(x).view(B, S, self.n_q_heads, D // self.n_q_heads).transpose(1, 2)
        k = self.wk(x).view(B, S, self.n_kv_heads, D // self.n_q_heads).transpose(1, 2)
        v = self.wv(x).view(B, S, self.n_kv_heads, D // self.n_q_heads).transpose(1, 2)
        return grouped_query_attention(q, k, v, self.n_kv_heads)


q, k, v = DummyDataGenerator().qkv()
out = grouped_query_attention(q, k, v, 2)
print(f"GQA output: {out.shape}")
print("✓ GQA implemented")
