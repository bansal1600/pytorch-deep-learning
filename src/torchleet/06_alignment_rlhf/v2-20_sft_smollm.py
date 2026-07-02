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
# # V2-20: SFT on SmolLM — Solution

# %%
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


class DummyDataGenerator:
    def __init__(self):
        self.data = [
            {"prompt": "What is 2+2?", "response": "4"},
            {"prompt": "Capital of France?", "response": "Paris"},
        ]
    def instructions(self): return self.data


class SFTDataset(Dataset):
    def __init__(self, data, vocab, max_len=32):
        self.data = data
        self.vocab = vocab
        self.max_len = max_len
    def __len__(self): return len(self.data)
    def __getitem__(self, i):
        text = self.data[i]["prompt"] + " " + self.data[i]["response"]
        ids = [self.vocab.get(c, 0) for c in text[:self.max_len]]
        ids += [0] * (self.max_len - len(ids))
        x = torch.tensor(ids[:-1])
        y = torch.tensor(ids[1:])
        return x, y


class SmolLM(nn.Module):
    def __init__(self, vocab=128, d=64):
        super().__init__()
        self.embed = nn.Embedding(vocab, d)
        self.blocks = nn.TransformerEncoder(nn.TransformerEncoderLayer(d, 4, batch_first=True), 2)
        self.head = nn.Linear(d, vocab)
    def forward(self, ids):
        return self.head(self.blocks(self.embed(ids)))


vocab = {c: i+1 for i, c in enumerate("abcdefghijklmnopqrstuvwxyz0123456789 ?")}
data = DummyDataGenerator().instructions()
loader = DataLoader(SFTDataset(data, vocab), batch_size=2)
model = SmolLM(vocab=len(vocab)+1)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
for x, y in loader:
    opt.zero_grad()
    nn.CrossEntropyLoss()(model(x).view(-1, model.head.out_features), y.view(-1)).backward()
    opt.step()
print("✓ SFT training step on instruction data")
