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
# # V3-08: Top-k Sampling — Solution

# %%
import torch
import torch.nn as nn
from torch.utils.data import Dataset


class DummyDataGenerator:
    def __init__(self):
        self._logits = torch.randn(100)
    def get_logits(self):
        return self._logits


class LogitsDataset(Dataset):
    def __init__(self, l): self.l = l
    def __len__(self): return 1
    def __getitem__(self, i): return self.l


def top_k_sample(logits, k=10, temperature=1.0):
    logits = logits / temperature
    topk_vals, _ = torch.topk(logits, k)
    threshold = topk_vals[-1]
    filtered = torch.where(logits < threshold, torch.full_like(logits, float("-inf")), logits)
    probs = torch.softmax(filtered, dim=-1)
    return torch.multinomial(probs, 1)


class DecoderModel(nn.Module):
    def __init__(self, vocab=100):
        super().__init__()
        self.head = nn.Linear(16, vocab)
    def forward(self, h):
        return self.head(h)


tok = top_k_sample(DummyDataGenerator().get_logits(), k=5)
print(f"top-k sample: {tok.item()}")
print("✓ Top-k sampling")
