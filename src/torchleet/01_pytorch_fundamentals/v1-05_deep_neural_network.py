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
# # V1-05: Deep Neural Network — Solution

# %%
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


class DummyDataGenerator:
    def __init__(self, n=500):
        torch.manual_seed(0)
        self.X = torch.randn(n, 2)
        self.y = (self.X[:, 0]**2 + self.X[:, 1]**2).unsqueeze(1)

    def tensors(self):
        return self.X, self.y


class NonlinearDataset(Dataset):
    def __init__(self, X, y):
        self.X, self.y = X, y
    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i], self.y[i]


class DNNModel(nn.Module):
    def __init__(self, d_in=2, hidden=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )
    def forward(self, x):
        return self.net(x)


X, y = DummyDataGenerator().tensors()
loader = DataLoader(NonlinearDataset(X, y), batch_size=64, shuffle=True)
model = DNNModel()
opt = torch.optim.Adam(model.parameters(), lr=0.01)

for _ in range(150):
    for xb, yb in loader:
        opt.zero_grad()
        nn.MSELoss()(model(xb), yb).backward()
        opt.step()

test = torch.tensor([[1., 1.]])
print(f"pred x²+y²≈{model(test).item():.2f} (true=2)")
print("✓ DNN trained")
