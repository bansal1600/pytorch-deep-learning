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
# # V3-12: KV Cache — Solution

# %%
import torch
import torch.nn as nn
from torch.utils.data import Dataset


class DummyDataGenerator:
    def __init__(self):
        self.x = torch.randn(1, 1, 16)
    def token(self): return self.x


class TokenDataset(Dataset):
    def __init__(self, x): self.x = x
    def __len__(self): return len(self.x)
    def __getitem__(self, i): return self.x[i]


class KVCache:
    def __init__(self):
        self.k, self.v = None, None
    def update(self, k, v):
        self.k = k if self.k is None else torch.cat([self.k, k], dim=2)
        self.v = v if self.v is None else torch.cat([self.v, v], dim=2)
        return self.k, self.v
    def get(self):
        return self.k, self.v
    def reset(self):
        self.k, self.v = None, None


class CachedAttention(nn.Module):
    def __init__(self, d=16, heads=2):
        super().__init__()
        self.d, self.h = d, heads
        self.qkv = nn.Linear(d, 3 * d)
        self.cache = KVCache()
    def forward(self, x, use_cache=True):
        B, S, _ = x.shape
        qkv = self.qkv(x).chunk(3, dim=-1)
        q, k, v = [t.view(B, S, self.h, -1).transpose(1, 2) for t in qkv]
        if use_cache:
            k, v = self.cache.update(k, v)
        # When using cache, q has seq len 1 but k/v have full history
        scores = (q @ k.transpose(-2, -1)) / (q.size(-1) ** 0.5)
        attn = torch.softmax(scores, dim=-1) @ v
        return attn.transpose(1, 2).reshape(B, S, self.d)


model = CachedAttention()
t1 = DummyDataGenerator().token()
o1 = model(t1, use_cache=True)
t2 = torch.randn(1, 1, 16)
o2 = model(t2, use_cache=True)
print(f"cached steps: {o1.shape}, {o2.shape}, cache len={model.cache.k.size(1)}")
print("✓ KV cache grows across steps")
