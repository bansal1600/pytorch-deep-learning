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
# # V3-17: Mixture of Experts — Solution

# %%
import torch
import torch.nn as nn
from dataclasses import dataclass
from torch.utils.data import Dataset


@dataclass
class MoEConfig:
    d_model: int = 32
    num_experts: int = 4
    top_k: int = 2


class DummyDataGenerator:
    def __init__(self, n=16, d=32):
        self.x = torch.randn(n, d)
    def input(self): return self.x


class MoEDataset(Dataset):
    def __init__(self, x): self.x = x
    def __len__(self): return len(self.x)
    def __getitem__(self, i): return self.x[i]


class Expert(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.ff = nn.Sequential(nn.Linear(d, 4*d), nn.ReLU(), nn.Linear(4*d, d))
    def forward(self, x):
        return self.ff(x)


class MoELayer(nn.Module):
    def __init__(self, cfg: MoEConfig):
        super().__init__()
        self.cfg = cfg
        self.gate = nn.Linear(cfg.d_model, cfg.num_experts)
        self.experts = nn.ModuleList([Expert(cfg.d_model) for _ in range(cfg.num_experts)])
    def forward(self, x):
        logits = self.gate(x)
        topk = torch.topk(logits, self.cfg.top_k, dim=-1)
        weights = torch.softmax(topk.values, dim=-1)
        out = torch.zeros_like(x)
        for i in range(self.cfg.top_k):
            idx = topk.indices[:, i]
            for e_id in range(self.cfg.num_experts):
                mask = idx == e_id
                if mask.any():
                    out[mask] += weights[mask, i:i+1] * self.experts[e_id](x[mask])
        load = torch.softmax(logits, dim=-1).mean(0)
        aux_loss = (load * load.log()).sum() * self.cfg.num_experts
        return out, aux_loss


x = DummyDataGenerator().input()
out, aux = MoELayer(MoEConfig())(x)
print(f"MoE out: {out.shape}, aux_loss={aux.item():.4f}")
print("✓ MoE routing + load balance loss")
