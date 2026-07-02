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
# # V3-09: Beam Search — Solution

# %%
import torch
import torch.nn as nn
from torch.utils.data import Dataset


class DummyDataGenerator:
    def __init__(self, vocab=10):
        self.start = 0
        self.vocab = vocab
    def start_token(self): return self.start


class StartDataset(Dataset):
    def __init__(self, start): self.start = start
    def __len__(self): return 1
    def __getitem__(self, i): return self.start


class DummyLM(nn.Module):
    def __init__(self, vocab=10, d=16):
        super().__init__()
        self.embed = nn.Embedding(vocab, d)
        self.head = nn.Linear(d, vocab)
    def forward(self, ids):
        return self.head(self.embed(ids))


def beam_search(model, start_token, beam_width=3, max_len=5):
    beams = [([start_token], 0.0)]
    for _ in range(max_len):
        candidates = []
        for seq, score in beams:
            logits = model(torch.tensor([seq]))[0, -1]
            logp = torch.log_softmax(logits, dim=-1)
            topk = torch.topk(logp, beam_width)
            for lp, tok in zip(topk.values, topk.indices):
                candidates.append((seq + [tok.item()], score + lp.item()))
        beams = sorted(candidates, key=lambda x: x[1], reverse=True)[:beam_width]
    return beams[0][0]


model = DummyLM()
result = beam_search(model, DummyDataGenerator().start_token())
print(f"beam result: {result}")
print("✓ Beam search decoding")
