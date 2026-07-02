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
# # V1-08: CNN on CIFAR-10 — Solution
#
# Uses dummy 32×32 RGB tensors mimicking CIFAR-10.

# %%
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


class DummyDataGenerator:
    def __init__(self, n=200, num_classes=10):
        torch.manual_seed(11)
        self.images = torch.rand(n, 3, 32, 32)
        self.labels = torch.randint(0, num_classes, (n,))

    def tensors(self):
        return self.images, self.labels


class CIFAR10DummyDataset(Dataset):
    def __init__(self, images, labels):
        self.images, self.labels = images, labels
    def __len__(self): return len(self.images)
    def __getitem__(self, i): return self.images[i], self.labels[i]


class CNNModel(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 128), nn.ReLU(),
            nn.Linear(128, num_classes),
        )
    def forward(self, x):
        return self.classifier(self.features(x))


imgs, labels = DummyDataGenerator().tensors()
loader = DataLoader(CIFAR10DummyDataset(imgs, labels), batch_size=32, shuffle=True)
model = CNNModel()
opt = torch.optim.Adam(model.parameters(), lr=1e-3)

for _ in range(3):
    for xb, yb in loader:
        opt.zero_grad()
        nn.CrossEntropyLoss()(model(xb), yb).backward()
        opt.step()

print(f"output shape: {model(imgs[:2]).shape}")
print("✓ CNN forward + training step OK")
