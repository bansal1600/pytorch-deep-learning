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
# # V1-28: Seq2Seq with Attention — Solution

# %%
import torch
import torch.nn as nn
from torch.utils.data import Dataset


class DummyDataGenerator:
    def __init__(self):
        self.src = torch.randint(1, 20, (4, 6))
        self.tgt = torch.randint(1, 20, (4, 5))
    def pairs(self): return self.src, self.tgt


class Seq2SeqDataset(Dataset):
    def __init__(self, src, tgt): self.src, self.tgt = src, tgt
    def __len__(self): return len(self.src)
    def __getitem__(self, i): return self.src[i], self.tgt[i]


class Encoder(nn.Module):
    def __init__(self, vocab=20, d=32):
        super().__init__()
        self.embed = nn.Embedding(vocab, d)
        self.lstm = nn.LSTM(d, d, batch_first=True)
    def forward(self, src):
        out, (h, c) = self.lstm(self.embed(src))
        return out, h, c


class Decoder(nn.Module):
    def __init__(self, vocab=20, d=32):
        super().__init__()
        self.embed = nn.Embedding(vocab, d)
        self.attn = nn.Linear(d*2, d)
        self.lstm = nn.LSTM(d, d, batch_first=True)
        self.fc = nn.Linear(d, vocab)
    def forward(self, tgt, enc_out, h, c):
        emb = self.embed(tgt)
        ctx = enc_out.mean(dim=1, keepdim=True).expand_as(emb)
        out, (h, c) = self.lstm(emb + self.attn(torch.cat([emb, ctx], -1)), (h, c))
        return self.fc(out), h, c


src, tgt = DummyDataGenerator().pairs()
enc = Encoder(); dec = Decoder()
enc_out, h, c = enc(src)
logits, _, _ = dec(tgt, enc_out, h, c)
print(f"seq2seq logits: {logits.shape}")
print("✓ Encoder-decoder with attention context")
