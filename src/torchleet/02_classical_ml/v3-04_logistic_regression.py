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
# # V3-04: Logistic Regression (Manual GD) — Solution

# %%
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


class DummyDataGenerator:
    def __init__(self, n=400):
        torch.manual_seed(3)
        self.X = torch.randn(n, 2)
        self.y = (self.X[:, 0] + self.X[:, 1] > 0).float()
    def tensors(self):
        return self.X, self.y


class BinaryDataset(Dataset):
    def __init__(self, X, y):
        self.X, self.y = X, y
    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i], self.y[i]


def sigmoid(z):
    return 1 / (1 + torch.exp(-z))


def bce_loss(y, p):
    return -(y * torch.log(p + 1e-8) + (1 - y) * torch.log(1 - p + 1e-8)).mean()


def train_logistic_regression(X, y, lr=0.1, epochs=500):
    N, D = X.shape
    w = torch.zeros(D)
    b = torch.zeros(1)
    for _ in range(epochs):
        p = sigmoid(X @ w + b)
        dw = (X.T @ (p - y)) / N
        db = (p - y).mean()
        w -= lr * dw
        b -= lr * db
    return w, b


class LogisticModel(nn.Module):
    def __init__(self, w, b):
        super().__init__()
        self.register_buffer("w", w)
        self.register_buffer("b", b)
    def forward(self, x):
        return sigmoid(x @ self.w + self.b)


X, y = DummyDataGenerator().tensors()
w, b = train_logistic_regression(X, y)
model = LogisticModel(w, b)
acc = ((model(X) > 0.5).float() == y).float().mean()
print(f"accuracy={acc:.2%}")
assert acc > 0.85
print("✓ Manual logistic regression >85% acc")
