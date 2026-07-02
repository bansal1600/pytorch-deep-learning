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
# # V3-07: Top-p (Nucleus) Sampling — Solution

# %%
import torch
import torch.nn as nn
from torch.utils.data import Dataset


class DummyDataGenerator:
    def __init__(self, vocab=50):
        torch.manual_seed(7)
        self._logits = torch.randn(vocab)
    def get_logits(self):
        return self._logits


class LogitsDataset(Dataset):
    def __init__(self, logits): self.logits = logits
    def __len__(self): return 1
    def __getitem__(self, i): return self.logits


def top_p_sample(logits, p=0.9, temperature=1.0):
    logits = logits / temperature
    probs = torch.softmax(logits, dim=-1)
    sorted_probs, sorted_idx = torch.sort(probs, descending=True)
    cum = torch.cumsum(sorted_probs, dim=-1)
    mask = cum - sorted_probs > p
    sorted_probs = sorted_probs.masked_fill(mask, 0.0)
    sorted_probs /= sorted_probs.sum()
    i = torch.multinomial(sorted_probs, 1)
    return sorted_idx[i]


class LMHead(nn.Module):
    def __init__(self, vocab=50, d=32):
        super().__init__()
        self.proj = nn.Linear(d, vocab)
    def forward(self, h):
        return self.proj(h)


logits = DummyDataGenerator().get_logits()
tok = top_p_sample(logits, p=0.9)
print(f"sampled token id: {tok.item()}")
print("✓ Top-p sampling works")
