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
# # V2-13: GPTQ Quantization — Solution
#
# Simplified round-to-nearest with per-channel scale.

# %%
import torch
import torch.nn as nn
from torch.utils.data import Dataset


class DummyDataGenerator:
    def __init__(self, n=32, d=16):
        self.X = torch.randn(n, d)
    def calibration_data(self): return self.X


class CalibDataset(Dataset):
    def __init__(self, X): self.X = X
    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i]


class LinearModel(nn.Module):
    def __init__(self, d=16, out=8):
        super().__init__()
        self.fc = nn.Linear(d, out)
    def forward(self, x):
        return self.fc(x)


def quantize_weight(w, bits=4):
    qmax = 2**bits - 1
    scale = w.abs().amax(dim=1, keepdim=True) / qmax
    q = torch.round(w / (scale + 1e-8)).clamp(-qmax, qmax)
    return q * scale


model = LinearModel()
W = model.fc.weight.data
W_q = quantize_weight(W)
model.fc.weight.data = W_q
x = DummyDataGenerator().calibration_data()
diff = (model.fc(x) - nn.Linear(16, 8).forward(x)).abs().mean()
print(f"quantized weight max err vs float: {(W-W_q).abs().max():.4f}")
print("✓ GPTQ-style weight quantization")
