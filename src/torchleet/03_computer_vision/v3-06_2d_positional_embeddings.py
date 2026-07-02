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
# # V3-06: 2D Positional Embeddings — Solution

# %%
import math
import torch
import torch.nn as nn
from torch.utils.data import Dataset


class DummyDataGenerator:
    def __init__(self, h=4, w=4, d=32):
        self.patches = torch.randn(1, h*w, d)
        self.h, self.w, self.d = h, w, d
    def input(self): return self.patches, self.h, self.w


class PatchDataset(Dataset):
    def __init__(self, patches): self.patches = patches
    def __len__(self): return len(self.patches)
    def __getitem__(self, i): return self.patches[i]


def create_2d_sinusoidal_embeddings(height, width, d_model):
    assert d_model % 2 == 0
    half = d_model // 2
    pe = torch.zeros(height, width, d_model)
    y_pos = torch.arange(height).float()
    x_pos = torch.arange(width).float()
    div = torch.exp(torch.arange(0, half, 2).float() * (-math.log(10000) / half))
    for y in range(height):
        for x in range(width):
            pe[y, x, 0:half:2] = torch.sin(y_pos[y] * div)
            pe[y, x, 1:half:2] = torch.cos(y_pos[y] * div)
            pe[y, x, half::2] = torch.sin(x_pos[x] * div)
            pe[y, x, half+1::2] = torch.cos(x_pos[x] * div)
    return pe.view(height * width, d_model)


class ViTStub(nn.Module):
    def __init__(self, d=32):
        super().__init__()
        self.d = d
    def forward(self, patches, pe):
        return patches + pe.unsqueeze(0)


patches, h, w = DummyDataGenerator().input()
pe = create_2d_sinusoidal_embeddings(h, w, 32)
out = ViTStub()(patches, pe)
print(f"2D PE output: {out.shape}")
print("✓ 2D sinusoidal positional embeddings")
