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
# # V1-23: Neural Style Transfer — Solution

# %%
import torch
import torch.nn as nn
from torch.utils.data import Dataset


class DummyDataGenerator:
    def __init__(self):
        self.content = torch.rand(1, 3, 64, 64)
        self.style = torch.rand(1, 3, 64, 64)
    def images(self): return self.content, self.style


class ImagePairDataset(Dataset):
    def __init__(self, content, style):
        self.content, self.style = content, style
    def __len__(self): return 1
    def __getitem__(self, i): return self.content, self.style


class FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
    def forward(self, x):
        h1 = torch.relu(self.conv1(x))
        return torch.relu(self.conv2(h1)), h1


def gram_matrix(f):
    B, C, H, W = f.shape
    f = f.view(B, C, -1)
    return f @ f.transpose(1, 2) / (C * H * W)


class StyleTransferModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = FeatureExtractor()
    def content_loss(self, gen, target):
        return nn.MSELoss()(gen, target)
    def style_loss(self, gen_g, style_g):
        return nn.MSELoss()(gram_matrix(gen_g), gram_matrix(style_g))


content, style = DummyDataGenerator().images()
model = StyleTransferModel()
gen_img = content.clone().requires_grad_(True)
opt = torch.optim.Adam([gen_img], lr=0.1)
with torch.no_grad():
    c_feat, _ = model.encoder(content)
    _, s_feat = model.encoder(style)
c_feat = c_feat.detach()
s_feat = s_feat.detach()
for _ in range(20):
    opt.zero_grad()
    g_deep, g_shallow = model.encoder(gen_img)
    loss = model.content_loss(g_deep, c_feat) + model.style_loss(g_shallow, s_feat)
    loss.backward()
    opt.step()
print(f"style transfer loss={loss.item():.4f}")
print("✓ NST optimization loop")
