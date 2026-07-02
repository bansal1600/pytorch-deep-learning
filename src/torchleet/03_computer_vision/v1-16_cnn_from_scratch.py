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
# # V1-16: CNN from Scratch — Solution

# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset


class DummyDataGenerator:
    def __init__(self, n=32):
        self.X = torch.rand(n, 3, 32, 32)
        self.y = torch.randint(0, 10, (n,))
    def tensors(self):
        return self.X, self.y


class ScratchDataset(Dataset):
    def __init__(self, X, y):
        self.X, self.y = X, y
    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i], self.y[i]


class Conv2dCustom(nn.Module):
    def __init__(self, in_ch, out_ch, kernel=3, stride=1, padding=1):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_ch, in_ch, kernel, kernel) * 0.1)
        self.bias = nn.Parameter(torch.zeros(out_ch))
        self.stride, self.padding = stride, padding
    def forward(self, x):
        return F.conv2d(x, self.weight, self.bias, stride=self.stride, padding=self.padding)


class MaxPool2dCustom(nn.Module):
    def __init__(self, kernel=2, stride=2):
        super().__init__()
        self.kernel, self.stride = kernel, stride
    def forward(self, x):
        return F.max_pool2d(x, self.kernel, self.stride)


class CNNModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = Conv2dCustom(3, 8)
        self.conv2 = Conv2dCustom(8, 16)
        self.pool = MaxPool2dCustom()
        self.fc = nn.Linear(16 * 8 * 8, 10)
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        return self.fc(x.flatten(1))


X, y = DummyDataGenerator().tensors()
model = CNNModel()
out = model(X[:2])
print(f"scratch CNN output: {out.shape}")
print("✓ Custom conv + pool layers work")
