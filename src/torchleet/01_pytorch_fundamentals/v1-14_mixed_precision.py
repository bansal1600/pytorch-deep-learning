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
# # V1-14: Mixed Precision Training — Solution

# %%
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import Dataset, DataLoader


class DummyDataGenerator:
    def __init__(self, n=256, d=32):
        torch.manual_seed(10)
        self.X = torch.randn(n, d)
        self.y = torch.randint(0, 4, (n,))
    def tensors(self):
        return self.X, self.y


class TabularDataset(Dataset):
    def __init__(self, X, y):
        self.X, self.y = X, y
    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i], self.y[i]


class AMPModel(nn.Module):
    def __init__(self, d=32, c=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, c),
        )
    def forward(self, x):
        return self.net(x)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
X, y = DummyDataGenerator().tensors()
loader = DataLoader(TabularDataset(X, y), batch_size=32, shuffle=True)
model = AMPModel().to(device)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
scaler = GradScaler(enabled=device.type == "cuda")

for xb, yb in loader:
    xb, yb = xb.to(device), yb.to(device)
    opt.zero_grad()
    with autocast(enabled=device.type == "cuda"):
        loss = nn.CrossEntropyLoss()(model(xb), yb)
    scaler.scale(loss).backward()
    scaler.step(opt)
    scaler.update()
    break

print(f"AMP training on {device}")
print("✓ Mixed precision step complete")
