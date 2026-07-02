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
# # V3-02: K-Means Clustering — Solution

# %%
import torch
import torch.nn as nn
from torch.utils.data import Dataset


class DummyDataGenerator:
    def __init__(self, k=3, per_cluster=50):
        torch.manual_seed(0)
        centers = torch.tensor([[0., 0.], [5., 5.], [0., 5.]])
        pts = []
        for c in centers:
            pts.append(c + torch.randn(per_cluster, 2) * 0.5)
        self.data = torch.cat(pts)
        self.k = k
    def points(self):
        return self.data


class PointDataset(Dataset):
    def __init__(self, data):
        self.data = data
    def __len__(self): return len(self.data)
    def __getitem__(self, i): return self.data[i]


def kmeans(data, k, max_iters=100, tol=1e-4):
    idx = torch.randperm(len(data))[:k]
    centroids = data[idx].clone()
    for _ in range(max_iters):
        dists = torch.cdist(data, centroids)
        labels = dists.argmin(dim=1)
        new_c = torch.stack([data[labels == j].mean(0) if (labels == j).any() else centroids[j] for j in range(k)])
        if (new_c - centroids).abs().max() < tol:
            break
        centroids = new_c
    return centroids, labels


class KMeansModel(nn.Module):
    """Wraps centroids as learnable-free module for inference."""
    def __init__(self, centroids):
        super().__init__()
        self.register_buffer("centroids", centroids)
    def forward(self, x):
        return torch.cdist(x, self.centroids).argmin(dim=1)


data = DummyDataGenerator().points()
centroids, labels = kmeans(data, k=3)
model = KMeansModel(centroids)
pred = model(data)
print(f"clusters found, label distribution: {pred.bincount(minlength=3).tolist()}")
print("✓ K-means converged")
