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
# # V1-02: Custom Dataset & DataLoader — Solution

# %%
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


class DummyDataGenerator:
    def __init__(self, n=150, seed=0):
        torch.manual_seed(seed)
        self.features = torch.randn(n, 4)
        self.targets = self.features @ torch.tensor([1., -2., 0.5, 3.]) + torch.randn(n) * 0.1

    def as_tensors(self):
        return self.features, self.targets


class TabularDataset(Dataset):
    def __init__(self, features, targets):
        self.features = features.float()
        self.targets = targets.float()

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]


class BatchPredictor(nn.Module):
    """Simple model consuming batched dataset samples."""
    def __init__(self, d_in=4):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, 16), nn.ReLU(), nn.Linear(16, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


gen = DummyDataGenerator()
ds = TabularDataset(*gen.as_tensors())
loader = DataLoader(ds, batch_size=16, shuffle=True)
model = BatchPredictor()

for xb, yb in loader:
    assert xb.shape == (16, 4) and yb.shape == (16,)
    preds = model(xb)
    assert preds.shape == (16,)
    break

print(f"Dataset size={len(ds)}, batches={len(loader)}")
print("✓ Custom Dataset + DataLoader working")
