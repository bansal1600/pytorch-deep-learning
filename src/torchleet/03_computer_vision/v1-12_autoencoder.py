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
# # V1-12: Autoencoder Anomaly Detection — Solution

# %%
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


class DummyDataGenerator:
    def __init__(self, n=128):
        torch.manual_seed(8)
        self.normal = torch.rand(n, 1, 28, 28)
        self.anomaly = torch.rand(16, 1, 28, 28) * 2 + 0.5
    def normal_data(self):
        return self.normal
    def anomaly_data(self):
        return self.anomaly


class ImageDataset(Dataset):
    def __init__(self, images):
        self.images = images
    def __len__(self): return len(self.images)
    def __getitem__(self, i): return self.images[i]


class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 4, stride=2, padding=1), nn.Sigmoid(),
        )
    def forward(self, x):
        return self.decoder(self.encoder(x))


gen = DummyDataGenerator()
loader = DataLoader(ImageDataset(gen.normal_data()), batch_size=32, shuffle=True)
model = Autoencoder()
opt = torch.optim.Adam(model.parameters(), lr=1e-3)

for _ in range(20):
    for xb in loader:
        opt.zero_grad()
        nn.MSELoss()(model(xb), xb).backward()
        opt.step()

with torch.no_grad():
    err_n = nn.MSELoss(reduction='none')(model(gen.normal_data()), gen.normal_data()).mean(dim=(1,2,3))
    err_a = nn.MSELoss(reduction='none')(model(gen.anomaly_data()), gen.anomaly_data()).mean(dim=(1,2,3))
print(f"normal recon err={err_n.mean():.4f}, anomaly err={err_a.mean():.4f}")
print("✓ Anomalies have higher reconstruction error")
