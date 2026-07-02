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
# # V3-28: LLM Inference Engine — Solution

# %%
import torch
import torch.nn as nn
from torch.utils.data import Dataset


class DummyDataGenerator:
    def prompts(self):
        return ["abc", "xy"]


class PromptDataset(Dataset):
    def __init__(self, texts): self.texts = texts
    def __len__(self): return len(self.texts)
    def __getitem__(self, i): return self.texts[i]


class SimpleTokenizer:
  def __init__(self):
    self.chars = list("abcdefghijklmnopqrstuvwxyz ")
    self.stoi = {c: i+2 for i, c in enumerate(self.chars)}
    self.stoi["<pad>"] = 0; self.stoi["<eos>"] = 1
  def encode(self, s): return [self.stoi.get(c, 0) for c in s.lower()] + [1]
  def decode(self, ids): return "".join(self.chars[i-2] for i in ids if i >= 2)


class KVCache:
    def __init__(self): self.k = self.v = None
    def update(self, k, v):
        self.k = k if self.k is None else torch.cat([self.k, k], dim=2)
        self.v = v if self.v is None else torch.cat([self.v, v], dim=2)
        return self.k, self.v


class MiniTransformer(nn.Module):
    def __init__(self, vocab=30, d=32):
        super().__init__()
        self.embed = nn.Embedding(vocab, d)
        self.attn = nn.MultiheadAttention(d, 2, batch_first=True)
        self.ff = nn.Sequential(nn.Linear(d, 4*d), nn.GELU(), nn.Linear(4*d, d))
        self.head = nn.Linear(d, vocab)
    def forward(self, ids):
        x = self.embed(ids)
        x, _ = self.attn(x, x, x)
        return self.head(self.ff(x))


class InferenceEngine:
    def __init__(self, model, tokenizer):
        self.model, self.tok = model, tokenizer
    def generate(self, prompts, max_new_tokens=5, temperature=1.0):
        results = []
        self.model.eval()
        with torch.no_grad():
            for p in prompts:
                ids = torch.tensor([self.tok.encode(p)])
                for _ in range(max_new_tokens):
                    logits = self.model(ids)[0, -1] / temperature
                    nxt = torch.multinomial(torch.softmax(logits, -1), 1)
                    ids = torch.cat([ids, nxt.unsqueeze(0)], dim=1)
                    if nxt.item() == 1: break
                results.append(self.tok.decode(ids[0].tolist()))
        return results


tok = SimpleTokenizer()
engine = InferenceEngine(MiniTransformer(), tok)
print(engine.generate(DummyDataGenerator().prompts(), max_new_tokens=3))
print("✓ Inference engine generates text")
