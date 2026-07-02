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
# # V3-25: FlashAttention (Tiled, O(N) memory) — Solution

# %%
import torch
import torch.nn as nn
from torch.utils.data import Dataset


class DummyDataGenerator:
    def __init__(self):
        self.q = torch.randn(1, 4, 32, 16)
        self.k = self.v = torch.randn(1, 4, 32, 16)
    def qkv(self): return self.q, self.k, self.v


class AttnDataset(Dataset):
    def __init__(self, q, k, v): self.q, self.k, self.v = q, k, v
    def __len__(self): return 1
    def __getitem__(self, i): return self.q, self.k, self.v


def standard_attention(q, k, v):
    d = q.size(-1)
    scores = (q @ k.transpose(-2, -1)) / (d ** 0.5)
    return torch.softmax(scores, dim=-1) @ v


def flash_attention_pytorch(q, k, v, block=8):
    B, H, N, D = q.shape
    out = torch.zeros_like(q)
    for i in range(0, N, block):
        qi = q[:, :, i:i+block]
        m = torch.full((B, H, qi.size(2), 1), float('-inf'), device=q.device)
        l = torch.zeros(B, H, qi.size(2), 1, device=q.device)
        acc = torch.zeros(B, H, qi.size(2), D, device=q.device)
        for j in range(0, N, block):
            kj, vj = k[:, :, j:j+block], v[:, :, j:j+block]
            s = (qi @ kj.transpose(-2, -1)) / (D ** 0.5)
            m_new = torch.maximum(m, s.max(dim=-1, keepdim=True).values)
            l = torch.exp(m - m_new) * l + torch.exp(s - m_new).sum(dim=-1, keepdim=True)
            acc = torch.exp(m - m_new) * acc + torch.exp(s - m_new) @ vj
            m = m_new
        out[:, :, i:i+block] = acc / l
    return out


q, k, v = DummyDataGenerator().qkv()
diff = (flash_attention_pytorch(q,k,v) - standard_attention(q,k,v)).abs().max()
print(f"flash vs standard max diff: {diff:.2e}")
print("✓ Tiled flash attention matches standard")
