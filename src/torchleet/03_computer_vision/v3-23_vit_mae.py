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
# # V3-23: Vision Transformer + MAE — Solution

# %%
import torch
import torch.nn as nn
from torch.utils.data import Dataset


class DummyDataGenerator:
    def __init__(self, n=8):
        self.imgs = torch.rand(n, 3, 32, 32)
    def get_images(self):
        return self.imgs


class ImageDataset(Dataset):
    def __init__(self, imgs): self.imgs = imgs
    def __len__(self): return len(self.imgs)
    def __getitem__(self, i): return self.imgs[i]


class PatchEmbedding(nn.Module):
    def __init__(self, patch=8, d=32):
        super().__init__()
        self.patch = patch
        self.proj = nn.Conv2d(3, d, patch, stride=patch)
    def forward(self, x):
        x = self.proj(x)
        return x.flatten(2).transpose(1, 2)


class ViT(nn.Module):
    def __init__(self, d=32, heads=4, layers=2):
        super().__init__()
        self.patch = PatchEmbedding(d=d)
        enc = nn.TransformerEncoderLayer(d, heads, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc, layers)
    def forward(self, x):
        return self.encoder(self.patch(x))


class MAE(nn.Module):
    def __init__(self, mask_ratio=0.75):
        super().__init__()
        self.encoder = ViT()
        self.decoder = nn.Linear(32, 3 * 8 * 8)
        self.mask_ratio = mask_ratio
    def forward(self, x):
        patches = PatchEmbedding()(x)
        N = patches.size(1)
        n_mask = int(N * self.mask_ratio)
        mask = torch.rand(patches.size(0), N) < self.mask_ratio
        visible = patches.clone()
        visible[mask] = 0
        h = self.encoder.encoder(self.encoder.patch(x))
        return self.decoder(h.mean(1))


imgs = DummyDataGenerator().get_images()
vit = ViT()
print(f"ViT patch tokens: {vit(imgs).shape}")
print("✓ ViT + MAE components")
