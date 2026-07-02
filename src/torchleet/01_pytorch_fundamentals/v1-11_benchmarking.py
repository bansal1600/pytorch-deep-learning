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
# # V1-11: Benchmarking — Solution

# %%
import time
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


class DummyDataGenerator:
    def __init__(self, n=512, d=20, c=5):
        torch.manual_seed(6)
        self.X = torch.randn(n, d)
        self.y = torch.randint(0, c, (n,))
    def tensors(self):
        return self.X, self.y


class ClassificationDataset(Dataset):
    def __init__(self, X, y):
        self.X, self.y = X, y
    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i], self.y[i]


class SimpleNN(nn.Module):
    def __init__(self, d_in=20, n_classes=5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, n_classes),
        )
    def forward(self, x):
        return self.net(x)


X, y = DummyDataGenerator().tensors()
loader = DataLoader(ClassificationDataset(X, y), batch_size=64, shuffle=True)
model = SimpleNN()
opt = torch.optim.SGD(model.parameters(), lr=0.1)
criterion = nn.CrossEntropyLoss()

t0 = time.perf_counter()
for xb, yb in loader:
    opt.zero_grad()
    criterion(model(xb), yb).backward()
    opt.step()
train_time = time.perf_counter() - t0

model.eval()
t1 = time.perf_counter()
correct = 0
with torch.no_grad():
    for xb, yb in loader:
        correct += (model(xb).argmax(1) == yb).sum().item()
test_time = time.perf_counter() - t1
acc = correct / len(X)
print(f"train: {train_time*1000:.1f}ms | test: {test_time*1000:.1f}ms | acc={acc:.2%}")
print("✓ Benchmark complete")
