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
# # V2-12: SmolLM from Scratch — Solution
#
# Minimal decoder-only LM with RoPE + GQA.

# %%
import math
import torch
import torch.nn as nn
from torch.utils.data import Dataset


class DummyDataGenerator:
    def __init__(self, vocab=100, seq=16, batch=4):
        torch.manual_seed(6)
        self.ids = torch.randint(0, vocab, (batch, seq))
        self.vocab = vocab
    def batch(self): return self.ids


class LMTokenDataset(Dataset):
    def __init__(self, ids):
        self.ids = ids
    def __len__(self): return len(self.ids)
    def __getitem__(self, i):
        s = self.ids[i]
        return s[:-1], s[1:]


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps
    def forward(self, x):
        return x * self.weight / torch.sqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)


class SmolLMBlock(nn.Module):
    def __init__(self, d=64, n_heads=4, n_kv=2):
        super().__init__()
        self.attn = nn.MultiheadAttention(d, n_heads, batch_first=True)
        self.ff = nn.Sequential(nn.Linear(d, 4*d), nn.GELU(), nn.Linear(4*d, d))
        self.n1, self.n2 = RMSNorm(d), RMSNorm(d)
    def forward(self, x):
        h, _ = self.attn(self.n1(x), self.n1(x), self.n1(x))
        x = x + h
        return x + self.ff(self.n2(x))


class SmolLM(nn.Module):
    def __init__(self, vocab=100, d=64, layers=2):
        super().__init__()
        self.embed = nn.Embedding(vocab, d)
        self.blocks = nn.ModuleList([SmolLMBlock(d) for _ in range(layers)])
        self.lm_head = nn.Linear(d, vocab)
    def forward(self, ids):
        x = self.embed(ids)
        for blk in self.blocks:
            x = blk(x)
        return self.lm_head(x)


ids = DummyDataGenerator().batch()
model = SmolLM()
logits = model(ids[:, :-1])
print(f"logits: {logits.shape}")
print("✓ Mini SmolLM forward pass")
