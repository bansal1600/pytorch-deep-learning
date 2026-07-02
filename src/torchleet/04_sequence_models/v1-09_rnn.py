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
# # V1-09: RNN from Scratch — Solution

# %%
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


class DummyDataGenerator:
    def __init__(self, seq_len=10, n_seq=100):
        torch.manual_seed(4)
        self.data = torch.randn(n_seq, seq_len, 4)
        self.targets = self.data.sum(dim=(1, 2))

    def tensors(self):
        return self.data, self.targets


class SequenceDataset(Dataset):
    def __init__(self, seqs, targets):
        self.seqs, self.targets = seqs, targets
    def __len__(self): return len(self.seqs)
    def __getitem__(self, i): return self.seqs[i], self.targets[i]


class RNNModel(nn.Module):
    def __init__(self, input_size=4, hidden=16):
        super().__init__()
        self.rnn = nn.RNN(input_size, hidden, batch_first=True)
        self.fc = nn.Linear(hidden, 1)
    def forward(self, x):
        out, _ = self.rnn(x)
        return self.fc(out[:, -1, :]).squeeze(-1)


seqs, targets = DummyDataGenerator().tensors()
loader = DataLoader(SequenceDataset(seqs, targets), batch_size=16, shuffle=True)
model = RNNModel()
opt = torch.optim.Adam(model.parameters(), lr=0.01)

for _ in range(30):
    for xb, yb in loader:
        opt.zero_grad()
        nn.MSELoss()(model(xb), yb).backward()
        opt.step()

print(f"RNN output shape: {model(seqs[:2]).shape}")
print("✓ RNN trained on sequences")
