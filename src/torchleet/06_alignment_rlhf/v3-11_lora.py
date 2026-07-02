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
# # V3-11: LoRA — Solution

# %%
import torch
import torch.nn as nn
from torch.utils.data import Dataset


class DummyDataGenerator:
    def __init__(self, n=64, d=32):
        torch.manual_seed(8)
        self.X = torch.randn(n, d)
        self.y = torch.randint(0, 4, (n,))
    def tensors(self): return self.X, self.y


class LoRADataset(Dataset):
    def __init__(self, X, y): self.X, self.y = X, y
    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i], self.y[i]


class LoRALinear(nn.Module):
    def __init__(self, linear: nn.Linear, rank=4, alpha=8):
        super().__init__()
        self.linear = linear
        for p in self.linear.parameters():
            p.requires_grad = False
        d_in, d_out = linear.in_features, linear.out_features
        self.A = nn.Parameter(torch.randn(d_in, rank) * 0.01)
        self.B = nn.Parameter(torch.zeros(rank, d_out))
        self.scale = alpha / rank
    def forward(self, x):
        return self.linear(x) + (x @ self.A @ self.B) * self.scale


class LoRAModel(nn.Module):
    def __init__(self, d=32, c=4, rank=4):
        super().__init__()
        base = nn.Linear(d, c)
        self.lora = LoRALinear(base, rank=rank)
    def forward(self, x):
        return self.lora(x)


X, y = DummyDataGenerator().tensors()
model = LoRAModel()
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
print(f"trainable {trainable}/{total} params ({100*trainable/total:.1f}%)")
print("✓ LoRA layer with frozen base weights")
