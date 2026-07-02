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
# # V2-11: RoPE Embeddings — Solution

# %%
import torch
import torch.nn as nn
from torch.utils.data import Dataset


class DummyDataGenerator:
    def __init__(self):
        torch.manual_seed(5)
        self.q = torch.randn(1, 4, 8, 16)
        self.k = torch.randn(1, 4, 8, 16)
    def qk(self): return self.q, self.k


class RotaryDataset(Dataset):
    def __init__(self, q, k): self.q, self.k = q, k
    def __len__(self): return 1
    def __getitem__(self, i): return self.q, self.k


def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin):
    q_rot = q * cos + rotate_half(q) * sin
    k_rot = k * cos + rotate_half(k) * sin
    return q_rot, k_rot


class Rotary(nn.Module):
    def __init__(self, dim, max_seq=128):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        t = torch.arange(max_seq).float()
        freqs = torch.einsum("i,j->ij", t, inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer("cos", emb.cos()[None, None, :, :])
        self.register_buffer("sin", emb.sin()[None, None, :, :])
    def forward(self, q, k):
        S = q.size(2)
        return apply_rotary_pos_emb(q, k, self.cos[:, :, :S], self.sin[:, :, :S])


q, k = DummyDataGenerator().qk()
rope = Rotary(16)
qr, kr = rope(q, k)
print(f"rotated q: {qr.shape}")
print("✓ RoPE applied to Q/K")
