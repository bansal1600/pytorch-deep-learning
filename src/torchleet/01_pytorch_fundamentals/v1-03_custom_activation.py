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
# # V1-03: Custom Activation Function — Solution
#
# Activation: `tanh(x) + x`

# %%
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


class DummyDataGenerator:
    def __init__(self, n=256):
        torch.manual_seed(1)
        self.X = torch.randn(n, 3)
        self.y = (self.X.sum(dim=1, keepdim=True) > 0).float()

    def tensors(self):
        return self.X, self.y


class CustomActivation(nn.Module):
    def forward(self, x):
        return torch.tanh(x) + x


class ActivationModel(nn.Module):
    def __init__(self, d_in=3):
        super().__init__()
        self.fc1 = nn.Linear(d_in, 8)
        self.act = CustomActivation()
        self.fc2 = nn.Linear(8, 1)

    def forward(self, x):
        return torch.sigmoid(self.fc2(self.act(self.fc1(x))))


class BinaryDataset(Dataset):
    def __init__(self, X, y):
        self.X, self.y = X, y
    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i], self.y[i]


X, y = DummyDataGenerator().tensors()
loader = DataLoader(BinaryDataset(X, y), batch_size=32, shuffle=True)
model = ActivationModel()
opt = torch.optim.Adam(model.parameters(), lr=0.05)

for _ in range(100):
    for xb, yb in loader:
        opt.zero_grad()
        loss = nn.functional.binary_cross_entropy(model(xb), yb)
        loss.backward()
        opt.step()

x_test = torch.tensor([[1., 1., 1.]])
print(f"sample pred={model(x_test).item():.3f}")
print("✓ Custom activation integrated")
