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
# # V1-22: Custom Autograd (Learned-SiLU) — Solution

# %%
import torch
import torch.nn as nn
from torch.utils.data import Dataset


class DummyDataGenerator:
    def __init__(self, n=100):
        self.X = torch.randn(n, 1)
        self.y = 2 * self.X + 1 + torch.randn(n, 1) * 0.1
    def tensors(self): return self.X, self.y


class RegressionDataset(Dataset):
    def __init__(self, X, y): self.X, self.y = X, y
    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i], self.y[i]


class LearnedSiLUFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, slope):
        ctx.save_for_backward(x, slope)
        sig = torch.sigmoid(x)
        return slope * x * sig
    @staticmethod
    def backward(ctx, grad_out):
        x, slope = ctx.saved_tensors
        sig = torch.sigmoid(x)
        grad_x = grad_out * slope * (sig + x * sig * (1 - sig))
        grad_slope = (grad_out * x * sig).sum()
        return grad_x, grad_slope


class LearnedSiLU(nn.Module):
    def __init__(self):
        super().__init__()
        self.slope = nn.Parameter(torch.tensor(1.0))
    def forward(self, x):
        return LearnedSiLUFunction.apply(x, self.slope)


class LinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)
        self.act = LearnedSiLU()
    def forward(self, x):
        return self.act(self.linear(x))


X, y = DummyDataGenerator().tensors()
model = LinearRegressionModel()
opt = torch.optim.SGD(model.parameters(), lr=0.05)
for _ in range(200):
    opt.zero_grad()
    nn.MSELoss()(model(X), y).backward()
    opt.step()
print(f"slope={model.act.slope.item():.3f}")
print("✓ Custom autograd SiLU trained")
