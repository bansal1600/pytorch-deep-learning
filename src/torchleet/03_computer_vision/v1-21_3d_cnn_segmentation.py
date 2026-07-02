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
# # V1-21: 3D CNN Segmentation — Solution

# %%
import torch
import torch.nn as nn
from torch.utils.data import Dataset


class DummyDataGenerator:
    def __init__(self):
        self.vol = torch.rand(2, 1, 16, 32, 32)
        self.mask = (torch.rand(2, 1, 16, 32, 32) > 0.5).float()
    def tensors(self): return self.vol, self.mask


class VolumeDataset(Dataset):
    def __init__(self, vol, mask): self.vol, self.mask = vol, mask
    def __len__(self): return len(self.vol)
    def __getitem__(self, i): return self.vol[i], self.mask[i]


class MedCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv3d(1, 8, 3, padding=1), nn.ReLU(),
            nn.Conv3d(8, 16, 3, stride=2, padding=1), nn.ReLU(),
        )
        self.dec = nn.Sequential(
            nn.ConvTranspose3d(16, 8, 4, stride=2, padding=1), nn.ReLU(),
            nn.Conv3d(8, 1, 1),
        )
    def forward(self, x):
        return torch.sigmoid(self.dec(self.enc(x)))


def dice_loss(pred, target, eps=1e-6):
    inter = (pred * target).sum()
    return 1 - (2 * inter + eps) / (pred.sum() + target.sum() + eps)


vol, mask = DummyDataGenerator().tensors()
pred = MedCNN()(vol)
print(f"pred: {pred.shape}, dice={dice_loss(pred, mask).item():.4f}")
print("✓ 3D segmentation model")
