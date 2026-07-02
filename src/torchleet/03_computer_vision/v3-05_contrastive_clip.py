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
# # V3-05: Contrastive Loss (InfoNCE) + CLIP — Solution

# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset


class DummyDataGenerator:
    def __init__(self, n=8, d=16):
        torch.manual_seed(2)
        self.images = torch.randn(n, d)
        self.texts = self.images + torch.randn(n, d) * 0.1
    def pairs(self): return self.images, self.texts


class PairDataset(Dataset):
    def __init__(self, img, txt): self.img, self.txt = img, txt
    def __len__(self): return len(self.img)
    def __getitem__(self, i): return self.img[i], self.txt[i]


def info_nce_loss(img_e, txt_e, temperature=0.07):
    img_e = F.normalize(img_e, dim=-1)
    txt_e = F.normalize(txt_e, dim=-1)
    logits = img_e @ txt_e.T / temperature
    labels = torch.arange(len(img_e))
    return (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2


class SimpleCLIP(nn.Module):
    def __init__(self, d=16, out=32):
        super().__init__()
        self.img_enc = nn.Linear(d, out)
        self.txt_enc = nn.Linear(d, out)
        self.log_temp = nn.Parameter(torch.log(torch.tensor(0.07)))
    def forward(self, img, txt):
        return self.img_enc(img), self.txt_enc(txt)


img, txt = DummyDataGenerator().pairs()
model = SimpleCLIP()
ie, te = model(img, txt)
loss = info_nce_loss(ie, te)
print(f"InfoNCE loss: {loss.item():.4f}")
print("✓ CLIP-style contrastive training")
