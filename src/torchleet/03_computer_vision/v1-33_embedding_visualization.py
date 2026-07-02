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
# # V1-33: Cross-Modal Embedding Visualization — Solution

# %%
import torch
import torch.nn as nn
from torch.utils.data import Dataset


class DummyDataGenerator:
    def __init__(self, n=50, d=16):
        torch.manual_seed(1)
        self.image_emb = torch.randn(n, d)
        self.text_emb = self.image_emb + torch.randn(n, d) * 0.3
        self.labels = torch.randint(0, 3, (n,))
    def embeddings(self): return self.image_emb, self.text_emb, self.labels


class EmbeddingDataset(Dataset):
    def __init__(self, img_e, txt_e, labels):
        self.img_e, self.txt_e, self.labels = img_e, txt_e, labels
    def __len__(self): return len(self.labels)
    def __getitem__(self, i): return self.img_e[i], self.txt_e[i], self.labels[i]


class EmbeddingProjector(nn.Module):
    def __init__(self, d=16, out=2):
        super().__init__()
        self.proj = nn.Linear(d, out, bias=False)
    def forward(self, x):
        return self.proj(x)


def tsne_simple(x, n_iter=50, lr=10.0):
    """Lightweight 2D projection via learned linear map (stand-in for t-SNE)."""
    proj = EmbeddingProjector(x.size(1), 2)
    opt = torch.optim.SGD(proj.parameters(), lr=lr)
    for _ in range(n_iter):
        y = proj(x)
        dist = torch.cdist(y, y)
        target = torch.cdist(x, x)
        loss = (dist - target).pow(2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    return proj(x).detach()


img_e, txt_e, labels = DummyDataGenerator().embeddings()
combined = torch.cat([img_e, txt_e])
coords = tsne_simple(combined)
print(f"2D coords shape: {coords.shape}")
print("✓ Multimodal embeddings projected to 2D")
