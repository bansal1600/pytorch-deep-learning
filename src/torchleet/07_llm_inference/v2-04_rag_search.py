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
# # V2-04: RAG Search of Embeddings — Solution

# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset


class DummyDataGenerator:
    def __init__(self):
        self.corpus = [
            "PyTorch is a deep learning framework",
            "Transformers use attention mechanisms",
            "Gradient descent optimizes loss functions",
        ]
        self.query = "How does attention work in neural networks?"
    def data(self): return self.corpus, self.query


class CorpusDataset(Dataset):
    def __init__(self, texts): self.texts = texts
    def __len__(self): return len(self.texts)
    def __getitem__(self, i): return self.texts[i]


class EmbeddingModel(nn.Module):
    def __init__(self, vocab=128, d=32):
        super().__init__()
        self.embed = nn.Embedding(vocab, d)
    def encode_text(self, text):
        ids = torch.tensor([[min(ord(c), 127) for c in text[:32]]])
        return F.normalize(self.embed(ids).mean(1), dim=-1)


class RAGGenerator(nn.Module):
    def __init__(self, d=32, vocab=128):
        super().__init__()
        self.fc = nn.Linear(d, vocab)
    def forward(self, context_emb):
        return self.fc(context_emb)


corpus, query = DummyDataGenerator().data()
emb_model = EmbeddingModel()
doc_embs = torch.cat([emb_model.encode_text(d) for d in corpus])
q_emb = emb_model.encode_text(query)
scores = (q_emb @ doc_embs.T).squeeze()
top_idx = scores.argmax().item()
context = doc_embs[top_idx]
gen = RAGGenerator()
print(f"retrieved: {corpus[top_idx][:50]}...")
print(f"generation logits: {gen(context).shape}")
print("✓ RAG retrieve + condition generation")
