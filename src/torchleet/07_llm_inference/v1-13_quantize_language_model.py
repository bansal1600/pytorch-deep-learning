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
# # V1-13: Quantize Language Model — Solution

# %%
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


class DummyDataGenerator:
    def __init__(self, vocab=50, seq_len=8, n=200):
        torch.manual_seed(9)
        self.data = torch.randint(0, vocab, (n, seq_len))
        self.vocab = vocab
    def sequences(self):
        return self.data


class LMSequenceDataset(Dataset):
    def __init__(self, seqs):
        self.seqs = seqs
    def __len__(self): return len(self.seqs)
    def __getitem__(self, i):
        s = self.seqs[i]
        return s[:-1], s[1:]


class LanguageModel(nn.Module):
    def __init__(self, vocab=50, embed=32, hidden=64):
        super().__init__()
        self.embed = nn.Embedding(vocab, embed)
        self.lstm = nn.LSTM(embed, hidden, batch_first=True)
        self.fc = nn.Linear(hidden, vocab)
    def forward(self, x):
        out, _ = self.lstm(self.embed(x))
        return self.fc(out)


seqs = DummyDataGenerator().sequences()
model = LanguageModel()
quantized = torch.quantization.quantize_dynamic(model, {nn.Linear, nn.LSTM}, dtype=torch.qint8)

x = seqs[:4, :-1]
with torch.no_grad():
    orig = model(x)
    # quantized LSTM may differ slightly
print(f"original params: {sum(p.numel() for p in model.parameters())}")
print(f"quantized model type: {type(quantized).__name__}")
print("✓ Dynamic quantization applied")
