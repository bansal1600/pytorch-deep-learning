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
# # V1-01: Linear Regression — Solution
#
# **TorchLeet Basics** | Difficulty: Basic
#
# Implement linear regression with `LinearRegressionModel`, custom `Dataset`, and synthetic data.

# %%
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


class DummyDataGenerator:
    """Synthetic linear data: y = 2x + 3 + noise."""
    def __init__(self, n: int = 200, seed: int = 42):
        torch.manual_seed(seed)
        self.X = torch.rand(n, 1) * 10
        self.y = 2 * self.X + 3 + torch.randn(n, 1) * 0.5

    def tensors(self):
        return self.X, self.y


class RegressionDataset(Dataset):
    def __init__(self, X: torch.Tensor, y: torch.Tensor):
        self.X, self.y = X, y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class LinearRegressionModel(nn.Module):
    def __init__(self, in_features: int = 1, out_features: int = 1):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x):
        return self.linear(x)


# --- Demo ---
gen = DummyDataGenerator()
dataset = RegressionDataset(*gen.tensors())
loader = DataLoader(dataset, batch_size=32, shuffle=True)
model = LinearRegressionModel()
opt = torch.optim.SGD(model.parameters(), lr=0.01)
loss_fn = nn.MSELoss()

for epoch in range(300):
    total = 0.0
    for xb, yb in loader:
        opt.zero_grad()
        loss = loss_fn(model(xb), yb)
        loss.backward()
        opt.step()
        total += loss.item()
    if (epoch + 1) % 100 == 0:
        w, b = model.linear.weight.item(), model.linear.bias.item()
        print(f"epoch {epoch+1} loss={total/len(loader):.4f} w≈{w:.3f} b≈{b:.3f}")

assert abs(model.linear.weight.item() - 2.0) < 0.3
print("✓ Linear regression converged near y=2x+3")
