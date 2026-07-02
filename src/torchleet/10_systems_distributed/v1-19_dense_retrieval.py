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
# # V1-19: Dense Retrieval — Solution

# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset


class DummyDataGenerator:
    def __init__(self):
        self.docs = ["pytorch tensors", "neural networks", "gradient descent"]
        self.query = "deep learning optimization"
    def corpus(self): return self.docs, self.query


class RetrievalDataset(Dataset):
    def __init__(self, texts):
        self.texts = texts
    def __len__(self): return len(self.texts)
    def __getitem__(self, i): return self.texts[i]


class Encoder(nn.Module):
    def __init__(self, vocab=64, d=32):
        super().__init__()
        self.embed = nn.Embedding(vocab, d)
    def forward(self, token_ids):
        return self.embed(token_ids).mean(dim=1)


class DenseRetriever(nn.Module):
    def __init__(self, d=32):
        super().__init__()
        self.query_enc = Encoder(d=d)
        self.doc_enc = Encoder(d=d)
    def encode_query(self, ids): return F.normalize(self.query_enc(ids), dim=-1)
    def encode_doc(self, ids): return F.normalize(self.doc_enc(ids), dim=-1)
    def search(self, query_emb, doc_embs, k=2):
        scores = query_emb @ doc_embs.T
        return scores.topk(k)


def char_ids(s, vocab=64):
    return torch.tensor([[min(ord(c), vocab-1) for c in s]])


docs, query = DummyDataGenerator().corpus()
model = DenseRetriever()
doc_embs = torch.cat([model.encode_doc(char_ids(d)) for d in docs])
q_emb = model.encode_query(char_ids(query))
scores, idx = model.search(q_emb, doc_embs)
print(f"top docs: {[docs[i] for i in idx[0].tolist()]}")
print("✓ Dense retrieval search")
