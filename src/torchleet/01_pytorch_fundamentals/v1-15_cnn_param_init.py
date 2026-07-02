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
# # V1-15: CNN Parameter Initialization — Solution

# %%
import torch
import torch.nn as nn
import math
from torch.utils.data import Dataset


class DummyDataGenerator:
    def __init__(self, n=64):
        torch.manual_seed(12)
        self.X = torch.rand(n, 3, 32, 32)
        self.y = torch.randint(0, 10, (n,))
    def tensors(self):
        return self.X, self.y


class InitDataset(Dataset):
    def __init__(self, X, y):
        self.X, self.y = X, y
    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i], self.y[i]


class VanillaCNNModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 16, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(16, 10)
    def forward(self, x):
        return self.fc(self.pool(torch.relu(self.conv(x))).flatten(1))


def config_init(model, init_type="kaiming"):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            if init_type == "zero":
                nn.init.zeros_(m.weight); nn.init.zeros_(m.bias)
            elif init_type == "random":
                nn.init.uniform_(m.weight, -0.1, 0.1)
            elif init_type == "xavier":
                fan_in = m.weight.size(1) * m.weight[0][0].numel()
                std = math.sqrt(1.0 / fan_in)
                nn.init.normal_(m.weight, 0, std)
            elif init_type == "kaiming":
                fan_in = m.weight.size(1) * m.weight[0][0].numel()
                std = math.sqrt(2.0 / fan_in)
                nn.init.normal_(m.weight, 0, std)

X, y = DummyDataGenerator().tensors()
for init in ["zero", "random", "xavier", "kaiming"]:
    m = VanillaCNNModel()
    config_init(m, init)
    with torch.no_grad():
        logits = m(X[:8])
    print(f"{init:8s} logits std={logits.std().item():.4f}")
print("✓ Init strategies compared")
