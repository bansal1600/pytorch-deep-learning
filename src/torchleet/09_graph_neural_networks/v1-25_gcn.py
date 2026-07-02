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
# # V1-25: Graph Convolutional Network — Solution

# %%
import torch
import torch.nn as nn
from torch.utils.data import Dataset


class DummyDataGenerator:
    def __init__(self, n=6):
        self.x = torch.randn(n, 4)
        self.adj = torch.zeros(n, n)
        for i in range(n-1):
            self.adj[i, i+1] = self.adj[i+1, i] = 1
        self.adj += torch.eye(n)
        self.y = torch.randint(0, 3, (n,))
    def graph(self): return self.x, self.adj, self.y


class NodeDataset(Dataset):
    def __init__(self, x, y): self.x, self.y = x, y
    def __len__(self): return len(self.x)
    def __getitem__(self, i): return self.x[i], self.y[i]


class GCNLayer(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(d_in, d_out) * 0.1)
    def forward(self, x, adj):
        deg = adj.sum(dim=1, keepdim=True).clamp_min(1)
        norm_adj = adj / deg
        return torch.relu(norm_adj @ x @ self.weight)


class GCNModel(nn.Module):
    def __init__(self, d_in=4, d_h=8, c=3):
        super().__init__()
        self.gcn1 = GCNLayer(d_in, d_h)
        self.gcn2 = GCNLayer(d_h, c)
    def forward(self, x, adj):
        return self.gcn2(self.gcn1(x, adj), adj)


x, adj, y = DummyDataGenerator().graph()
out = GCNModel()(x, adj)
print(f"GCN output: {out.shape}")
print("✓ GCN with adjacency aggregation")
