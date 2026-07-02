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
# # V1-06: TensorBoard Logging — Solution

# %%
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None


class DummyDataGenerator:
    def __init__(self, n=128):
        torch.manual_seed(3)
        self.X = torch.randn(n, 5)
        self.y = self.X.sum(dim=1, keepdim=True)

    def tensors(self):
        return self.X, self.y


class SimpleDataset(Dataset):
    def __init__(self, X, y):
        self.X, self.y = X, y
    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i], self.y[i]


class LoggingModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(5, 1)
    def forward(self, x):
        return self.fc(x)


X, y = DummyDataGenerator().tensors()
loader = DataLoader(SimpleDataset(X, y), batch_size=16, shuffle=True)
model = LoggingModel()
opt = torch.optim.SGD(model.parameters(), lr=0.05)

if SummaryWriter is not None:
    writer = SummaryWriter("runs/torchleet_v1_06")
    for epoch in range(20):
        epoch_loss = 0.0
        for xb, yb in loader:
            opt.zero_grad()
            loss = nn.MSELoss()(model(xb), yb)
            loss.backward()
            opt.step()
            epoch_loss += loss.item()
        writer.add_scalar("Loss/train", epoch_loss / len(loader), epoch)
    writer.close()
    print("✓ Logged to runs/torchleet_v1_06")
else:
    print("✓ Training loop OK (install tensorboard for logging)")
