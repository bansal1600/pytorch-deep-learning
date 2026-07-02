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
# # V3-01: Softmax from Scratch — Solution

# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset


class DummyDataGenerator:
    def __init__(self):
        torch.manual_seed(42)
        self.logits_1d = torch.randn(5)
        self.logits_2d = torch.randn(3, 5)
        self.logits_large = torch.tensor([1000., 2000., 3000.])
    def samples(self):
        return self.logits_1d, self.logits_2d, self.logits_large


class LogitsDataset(Dataset):
    def __init__(self, logits):
        self.logits = logits
    def __len__(self): return len(self.logits)
    def __getitem__(self, i): return self.logits[i]


def stable_softmax(x, dim=-1):
    x_max = x.max(dim=dim, keepdim=True).values
    exp_x = torch.exp(x - x_max)
    return exp_x / exp_x.sum(dim=dim, keepdim=True)


class SoftmaxClassifier(nn.Module):
  def __init__(self, d_in=5, n_classes=5):
    super().__init__()
    self.fc = nn.Linear(d_in, n_classes)
  def forward(self, x):
    return stable_softmax(self.fc(x), dim=-1)


l1, l2, ll = DummyDataGenerator().samples()
for name, x in [("1d", l1), ("2d", l2), ("large", ll)]:
    ours = stable_softmax(x)
    ref = F.softmax(x, dim=-1)
    assert torch.allclose(ours, ref, atol=1e-5)
    print(f"{name}: sums to {ours.sum(dim=-1)}")

model = SoftmaxClassifier()
probs = model(l2)
print(f"classifier probs shape: {probs.shape}")
print("✓ Stable softmax matches PyTorch")
