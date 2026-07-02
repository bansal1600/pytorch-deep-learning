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
# # V2-07: Multi-Head Attention — Solution

# %%
import torch
import torch.nn as nn
from torch.utils.data import Dataset


class DummyDataGenerator:
    def __init__(self, b=2, s=8, d=32):
        torch.manual_seed(2)
        self.x = torch.randn(b, s, d)
    def input(self): return self.x


class SequenceDataset(Dataset):
    def __init__(self, x): self.x = x
    def __len__(self): return len(self.x)
    def __getitem__(self, i): return self.x[i]


def multi_head_attention(x, d_model, num_heads, mask=None):
    B, S, _ = x.shape
    d_h = d_model // num_heads
    Wq = Wk = Wv = torch.randn(d_model, d_model) / d_model**0.5
    Wo = torch.randn(d_model, d_model) / d_model**0.5
    q = (x @ Wq).view(B, S, num_heads, d_h).transpose(1, 2)
    k = (x @ Wk).view(B, S, num_heads, d_h).transpose(1, 2)
    v = (x @ Wv).view(B, S, num_heads, d_h).transpose(1, 2)
    scores = (q @ k.transpose(-2, -1)) / (d_h ** 0.5)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float("-inf"))
    attn = torch.softmax(scores, dim=-1) @ v
    concat = attn.transpose(1, 2).reshape(B, S, d_model)
    return concat @ Wo


class MHAWrapper(nn.Module):
    def __init__(self, d=32, heads=4):
        super().__init__()
        self.mha = nn.MultiheadAttention(d, heads, batch_first=True)
    def forward(self, x):
        out, _ = self.mha(x, x, x)
        return out


x = DummyDataGenerator().input()
out = multi_head_attention(x, 32, 4)
print(f"MHA output shape: {out.shape}")
print("✓ Multi-head attention from scratch")
