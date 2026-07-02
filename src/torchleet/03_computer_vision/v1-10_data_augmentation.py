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
# # V1-10: Data Augmentation — Solution

# %%
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class DummyDataGenerator:
    def __init__(self, n=64):
        torch.manual_seed(5)
        self.images = torch.rand(n, 1, 28, 28)

    def tensors(self):
        return self.images


class MNISTDummyDataset(Dataset):
    def __init__(self, images, transform=None):
        self.images = images
        self.transform = transform
    def __len__(self): return len(self.images)
    def __getitem__(self, i):
        img = self.images[i]
        if self.transform:
            img = self.transform(img)
        return img, 0


class AugmentationModel(torch.nn.Module):
    """Tiny classifier on augmented images."""
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 4, 3, padding=1)
    def forward(self, x):
        return self.conv(x).mean(dim=(2, 3))


transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomCrop(28),
    transforms.Normalize((0.5,), (0.5,)),
])

imgs = DummyDataGenerator().tensors()
ds = MNISTDummyDataset(imgs, transform=transform)
aug = ds[0][0]
model = AugmentationModel()
print(f"augmented shape: {aug.shape}, model out: {model(aug.unsqueeze(0)).shape}")
print("✓ Augmentation pipeline applied")
