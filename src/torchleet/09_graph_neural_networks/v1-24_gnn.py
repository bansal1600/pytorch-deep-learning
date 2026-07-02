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
# # V1-24: Graph Neural Network — Solution

# %%
import torch
import torch.nn as nn
from torch.utils.data import Dataset


class DummyDataGenerator:
    def __init__(self, n=5):
        self.x = torch.randn(n, 8)
        self.edge_index = torch.tensor([[0,1,1,2,3,4],[1,0,2,1,4,3]])
        self.y = torch.randint(0, 2, (n,))
    def graph(self): return self.x, self.edge_index, self.y


class GraphDataset(Dataset):
    def __init__(self, x, y): self.x, self.y = x, y
    def __len__(self): return len(self.x)
    def __getitem__(self, i): return self.x[i], self.y[i]


class GNNLayer(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.lin = nn.Linear(d_in, d_out)
    def forward(self, x, edge_index):
        src, dst = edge_index
        agg = torch.zeros_like(x)
        agg.index_add_(0, dst, x[src])
        deg = torch.bincount(dst, minlength=x.size(0)).float().unsqueeze(1).clamp_min(1)
        return torch.relu(self.lin(agg / deg))


class GNNModel(nn.Module):
    def __init__(self, d_in=8, d_h=16, n_classes=2):
        super().__init__()
        self.l1 = GNNLayer(d_in, d_h)
        self.l2 = GNNLayer(d_h, d_h)
        self.head = nn.Linear(d_h, n_classes)
    def forward(self, x, edge_index):
        x = self.l2(self.l1(x, edge_index), edge_index)
        return self.head(x)


x, ei, y = DummyDataGenerator().graph()
logits = GNNModel()(x, ei)
print(f"GNN logits: {logits.shape}")
print("✓ Message passing GNN")
