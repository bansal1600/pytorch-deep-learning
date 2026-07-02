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
# # V1-31: GradCAM — Solution

# %%
import torch
import torch.nn as nn
from torch.utils.data import Dataset


class DummyDataGenerator:
    def __init__(self):
        self.img = torch.rand(1, 3, 32, 32)
        self.label = torch.tensor([3])
    def sample(self): return self.img, self.label


class ImageDataset(Dataset):
    def __init__(self, img, label): self.img, self.label = img, label
    def __len__(self): return 1
    def __getitem__(self, i): return self.img, self.label


class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 8, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(8, 10)
        self.activations = None
        self.gradients = None
    def forward(self, x):
        x = torch.relu(self.conv(x))
        self.activations = x
        x.register_hook(lambda g: setattr(self, 'gradients', g))
        return self.fc(self.pool(x).flatten(1))


def grad_cam(model, class_idx):
    model.zero_grad()
    logits = model(img)
    logits[0, class_idx].backward()
    weights = model.gradients.mean(dim=(2, 3), keepdim=True)
    cam = torch.relu((weights * model.activations).sum(dim=1, keepdim=True))
    return cam / cam.max()


img, label = DummyDataGenerator().sample()
model = SmallCNN()
heatmap = grad_cam(model, label.item())
print(f"GradCAM heatmap: {heatmap.shape}")
print("✓ GradCAM computed")
