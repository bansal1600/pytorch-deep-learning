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
# # V3-10: Temperature Sampling — Solution

# %%
import torch
import torch.nn as nn
from torch.utils.data import Dataset


class DummyDataGenerator:
    def __init__(self):
        self._logits = torch.tensor([1., 2., 3., 4.])
    def get_logits(self):
        return self._logits


class LogitsDataset(Dataset):
    def __init__(self, l): self.l = l
    def __len__(self): return 1
    def __getitem__(self, i): return self.l


def temperature_sample(logits, temperature=1.0):
    scaled = logits / max(temperature, 1e-8)
    probs = torch.softmax(scaled, dim=-1)
    return torch.multinomial(probs, 1)


def compute_entropy(probs):
    return -(probs * (probs + 1e-8).log()).sum()


class LMModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 4)
    def forward(self, x):
        return self.fc(x)


logits = DummyDataGenerator().get_logits()
for T in [0.1, 1.0, 5.0]:
    p = torch.softmax(logits / T, dim=-1)
    print(f"T={T} entropy={compute_entropy(p):.3f}")
print("✓ Temperature affects distribution sharpness")
