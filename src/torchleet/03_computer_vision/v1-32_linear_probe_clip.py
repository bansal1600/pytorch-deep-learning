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
# # V1-32: Linear Probe on CLIP Features — Solution

# %%
import torch
import torch.nn as nn
from torch.utils.data import Dataset


class DummyDataGenerator:
    def __init__(self, n=100, d=32):
        torch.manual_seed(0)
        self.features = torch.randn(n, d)
        self.labels = (self.features[:, 0] > 0).long()
    def tensors(self): return self.features, self.labels


class FeatureDataset(Dataset):
    def __init__(self, feats, labels): self.feats, self.labels = feats, labels
    def __len__(self): return len(self.feats)
    def __getitem__(self, i): return self.feats[i], self.labels[i]


class FrozenCLIPEncoder(nn.Module):
    """Simulates frozen CLIP image embeddings."""
    def __init__(self, d=32):
        super().__init__()
        self.proj = nn.Linear(d, d, bias=False)
        for p in self.parameters(): p.requires_grad = False
    def forward(self, x):
        return torch.nn.functional.normalize(self.proj(x), dim=-1)


class LinearProbe(nn.Module):
    def __init__(self, d=32, n_classes=2):
        super().__init__()
        self.head = nn.Linear(d, n_classes)
    def forward(self, x):
        return self.head(x)


feats, labels = DummyDataGenerator().tensors()
encoder = FrozenCLIPEncoder()
with torch.no_grad():
    emb = encoder(feats)
probe = LinearProbe()
opt = torch.optim.Adam(probe.parameters(), lr=0.05)
for _ in range(100):
    opt.zero_grad()
    nn.CrossEntropyLoss()(probe(emb), labels).backward()
    opt.step()
acc = (probe(emb).argmax(1) == labels).float().mean()
print(f"linear probe acc: {acc:.2%}")
print("✓ Frozen CLIP features + linear probe")
