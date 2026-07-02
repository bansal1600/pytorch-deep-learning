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
# # V1-07: Save & Load Model — Solution

# %%
import torch
import torch.nn as nn
from torch.utils.data import Dataset


class DummyDataGenerator:
    def __init__(self):
        self.x = torch.randn(10, 3)

    def input(self):
        return self.x


class InferenceDataset(Dataset):
    def __init__(self, X):
        self.X = X
    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i]


class MLPModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(3, 8), nn.ReLU(), nn.Linear(8, 2))
    def forward(self, x):
        return self.net(x)


model = MLPModel()
path = "/tmp/torchleet_v1_07.pth"
torch.save(model.state_dict(), path)

model2 = MLPModel()
model2.load_state_dict(torch.load(path, weights_only=True))
model2.eval()

x = DummyDataGenerator().input()
with torch.no_grad():
    diff = (model(x) - model2(x)).abs().max().item()
print(f"max diff after reload: {diff:.2e}")
assert diff < 1e-6
print("✓ Save/load verified")
