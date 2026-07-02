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
# # V1-17: LSTM from Scratch — Solution

# %%
import torch
import torch.nn as nn
from torch.utils.data import Dataset


class DummyDataGenerator:
    def __init__(self, batch=4, seq=6, d=8):
        torch.manual_seed(13)
        self.x = torch.randn(batch, seq, d)
    def input(self):
        return self.x


class SeqDataset(Dataset):
    def __init__(self, x):
        self.x = x
    def __len__(self): return len(self.x)
    def __getitem__(self, i): return self.x[i]


class CustomLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.W = nn.Linear(input_size + hidden_size, 4 * hidden_size)
    def forward(self, x, state):
        h, c = state
        gates = self.W(torch.cat([x, h], dim=-1))
        i, f, g, o = gates.chunk(4, dim=-1)
        i, f, o = torch.sigmoid(i), torch.sigmoid(f), torch.sigmoid(o)
        g = torch.tanh(g)
        c = f * c + i * g
        h = o * torch.tanh(c)
        return h, (h, c)


class CustomLSTMModel(nn.Module):
    def __init__(self, input_size=8, hidden=16):
        super().__init__()
        self.cell = CustomLSTMCell(input_size, hidden)
        self.fc = nn.Linear(hidden, 4)
    def forward(self, x):
        B, T, _ = x.shape
        h = torch.zeros(B, self.cell.hidden_size, device=x.device)
        c = torch.zeros_like(h)
        for t in range(T):
            h, (h, c) = self.cell(x[:, t], (h, c))
        return self.fc(h)


class LSTMModel(nn.Module):
    def __init__(self, input_size=8, hidden=16):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden, batch_first=True)
        self.fc = nn.Linear(hidden, 4)
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1])


x = DummyDataGenerator().input()
custom = CustomLSTMModel()
ref = LSTMModel()
print(f"custom: {custom(x).shape}, ref: {ref(x).shape}")
print("✓ Custom LSTM cell implemented")
