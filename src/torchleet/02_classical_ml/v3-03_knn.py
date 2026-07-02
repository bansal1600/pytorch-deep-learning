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
# # V3-03: KNN — Solution

# %%
import torch
import torch.nn as nn
from torch.utils.data import Dataset


class DummyDataGenerator:
    def __init__(self, n_train=100, n_test=20):
        torch.manual_seed(2)
        self.X_train = torch.randn(n_train, 4)
        self.y_train = (self.X_train[:, 0] > 0).long()
        self.X_test = torch.randn(n_test, 4)
    def splits(self):
        return self.X_train, self.y_train, self.X_test


class KNNDataset(Dataset):
    def __init__(self, X, y=None):
        self.X, self.y = X, y
    def __len__(self): return len(self.X)
    def __getitem__(self, i):
        return (self.X[i], self.y[i]) if self.y is not None else self.X[i]


def knn_predict(X_train, y_train, X_test, k=5):
    dists = torch.cdist(X_test, X_train)
    _, idx = dists.topk(k, largest=False, dim=1)
    neighbors = y_train[idx]
    return torch.mode(neighbors, dim=1).values


class KNNModel(nn.Module):
    def __init__(self, X_train, y_train, k=5):
        super().__init__()
        self.register_buffer("X_train", X_train)
        self.register_buffer("y_train", y_train)
        self.k = k
    def forward(self, x):
        return knn_predict(self.X_train, self.y_train, x, self.k)


Xt, yt, Xte = DummyDataGenerator().splits()
pred = knn_predict(Xt, yt, Xte, k=5)
model = KNNModel(Xt, yt)
assert torch.equal(pred, model(Xte))
print(f"KNN predictions: {pred[:10].tolist()}")
print("✓ KNN classification done")
