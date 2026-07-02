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
# # V2-10: Sinusoidal Positional Embeddings — Solution

# %%
import math
import torch
import torch.nn as nn
from torch.utils.data import Dataset


class DummyDataGenerator:
    def __init__(self, seq_len=10, d_model=32):
        torch.manual_seed(4)
        self.tokens = torch.randn(2, seq_len, d_model)
        self.seq_len, self.d_model = seq_len, d_model
    def input(self): return self.tokens


class TokenDataset(Dataset):
    def __init__(self, x): self.x = x
    def __len__(self): return len(self.x)
    def __getitem__(self, i): return self.x[i]


class SinusoidalPositionalEmbedding(nn.Module):
    def __init__(self, max_seq_len, d_model):
        super().__init__()
        pe = torch.zeros(max_seq_len, d_model)
        pos = torch.arange(max_seq_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))
    def forward(self, x):
        return self.pe[:, : x.size(1)]


class EmbeddingModel(nn.Module):
    def __init__(self, d_model=32, max_len=128):
        super().__init__()
        self.pos = SinusoidalPositionalEmbedding(max_len, d_model)
    def forward(self, x):
        return x + self.pos(x)


x = DummyDataGenerator().input()
model = EmbeddingModel(32)
out = model(x)
print(f"embedded shape: {out.shape}")
print("✓ Sinusoidal PE applied")
