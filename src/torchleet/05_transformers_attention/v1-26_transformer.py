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
# # V1-26: Transformer from Scratch — Solution

# %%
import math
import torch
import torch.nn as nn
from torch.utils.data import Dataset


class DummyDataGenerator:
    def __init__(self, vocab=50, seq=12, batch=4):
        self.ids = torch.randint(1, vocab, (batch, seq))
    def batch(self): return self.ids


class TokenDataset(Dataset):
    def __init__(self, ids): self.ids = ids
    def __len__(self): return len(self.ids)
    def __getitem__(self, i): return self.ids[i]


class PositionalEncoding(nn.Module):
    def __init__(self, d, max_len=128):
        super().__init__()
        pe = torch.zeros(max_len, d)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d, 2).float() * (-math.log(10000)/d))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class TransformerModel(nn.Module):
    def __init__(self, vocab=50, d=32, heads=4, layers=2):
        super().__init__()
        self.embed = nn.Embedding(vocab, d)
        self.pos = PositionalEncoding(d)
        enc_layer = nn.TransformerEncoderLayer(d, heads, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, layers)
        self.head = nn.Linear(d, vocab)
    def forward(self, ids):
        x = self.pos(self.embed(ids))
        return self.head(self.encoder(x))


ids = DummyDataGenerator().batch()
print(f"transformer logits: {TransformerModel()(ids).shape}")
print("✓ Full transformer encoder")
