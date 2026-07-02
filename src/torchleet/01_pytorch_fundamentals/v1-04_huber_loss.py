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
# # V1-04: Huber Loss — Solution

# %%
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


class HuberLoss(nn.Module):
    def __init__(self, delta: float = 1.0):
        super().__init__()
        self.delta = delta

    def forward(self, pred, target):
        err = pred - target
        abs_err = err.abs()
        quad = 0.5 * err.pow(2)
        lin = self.delta * (abs_err - 0.5 * self.delta)
        return torch.where(abs_err <= self.delta, quad, lin).mean()


class DummyDataGenerator:
    def __init__(self, n=300):
        torch.manual_seed(7)
        self.X = torch.randn(n, 2)
        self.y = self.X[:, 0:1] * 3 + torch.randn(n, 1) * 2

    def tensors(self):
        return self.X, self.y


class RegressionDataset(Dataset):
    def __init__(self, X, y):
        self.X, self.y = X, y
    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i], self.y[i]


class LinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 1)
    def forward(self, x):
        return self.linear(x)


X, y = DummyDataGenerator().tensors()
loader = DataLoader(RegressionDataset(X, y), batch_size=32, shuffle=True)
model = LinearRegressionModel()
criterion = HuberLoss(delta=1.0)
opt = torch.optim.Adam(model.parameters(), lr=0.03)

for _ in range(200):
    for xb, yb in loader:
        opt.zero_grad()
        criterion(model(xb), yb).backward()
        opt.step()

# Huber should be robust to outliers
outliers = torch.tensor([[10., 10.]])
print(f"Huber on outlier pred={model(outliers).item():.2f}")
print("✓ Huber loss training complete")
