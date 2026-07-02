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
# # V2-01: KL Divergence Loss — Solution

# %%
import torch
import torch.nn as nn
from torch.utils.data import Dataset


class DummyDataGenerator:
    def __init__(self):
        torch.manual_seed(4)
        self.p = torch.softmax(torch.randn(5), dim=0)
        self.q = torch.softmax(torch.randn(5), dim=0)
        self.logits_p = torch.randn(8, 5)
        self.logits_q = torch.randn(8, 5)
    def distributions(self):
        return self.p, self.q, self.logits_p, self.logits_q


class DistDataset(Dataset):
    def __init__(self, logits):
        self.logits = logits
    def __len__(self): return len(self.logits)
    def __getitem__(self, i): return self.logits[i]


def kl_divergence(p, q, eps=1e-8):
    p, q = p.clamp_min(eps), q.clamp_min(eps)
    return (p * (p / q).log()).sum(dim=-1)


class DistillationHead(nn.Module):
    def __init__(self, d=5):
        super().__init__()
        self.fc = nn.Linear(d, d)
    def forward(self, x):
        return torch.softmax(self.fc(x), dim=-1)


p, q, lp, lq = DummyDataGenerator().distributions()
kl = kl_divergence(p, q)
ref = torch.distributions.kl.kl_divergence(
    torch.distributions.Categorical(p), torch.distributions.Categorical(q)
)
print(f"KL(p||q)={kl.item():.4f}, ref={ref.item():.4f}")
print("✓ KL divergence from scratch")
