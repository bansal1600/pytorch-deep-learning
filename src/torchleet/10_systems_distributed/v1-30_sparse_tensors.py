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
# # V1-30: Sparse Tensors — Solution

# %%
import torch
import torch.nn as nn
from torch.utils.data import Dataset


class DummyDataGenerator:
    def __init__(self, n=100, d=50):
        dense = torch.zeros(n, d)
        idx = torch.randint(0, d, (n, 3))
        for i in range(n):
            dense[i, idx[i]] = torch.randn(3)
        self.dense = dense
    def dense_matrix(self): return self.dense


class SparseDataset(Dataset):
    def __init__(self, dense):
        self.coo = dense.to_sparse_coo()
    def __len__(self): return self.coo.size(0)
    def __getitem__(self, i):
        return self.coo[i].to_dense()


class SparseLinear(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(d_in, d_out) * 0.1)
    def forward(self, x_sparse):
        return torch.sparse.mm(x_sparse, self.weight)


dense = DummyDataGenerator().dense_matrix()
sparse = dense.to_sparse()
out = SparseLinear(50, 8)(sparse)
print(f"sparse mm output: {out.shape}, nnz={sparse._nnz()}")
print("✓ Sparse COO tensor operations")
