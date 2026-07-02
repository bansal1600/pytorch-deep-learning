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
# # V3-18: Speculative Decoding — Solution

# %%
import torch
import torch.nn as nn
from torch.utils.data import Dataset


class DummyDataGenerator:
    def __init__(self, vocab=20):
        self.vocab = vocab
        self.prompt = torch.tensor([1, 2, 3])
    def prompt_ids(self): return self.prompt


class PromptDataset(Dataset):
    def __init__(self, ids): self.ids = ids
    def __len__(self): return 1
    def __getitem__(self, i): return self.ids


class DraftModel(nn.Module):
    def __init__(self, vocab=20, d=16):
        super().__init__()
        self.embed = nn.Embedding(vocab, d)
        self.head = nn.Linear(d, vocab)
    def forward(self, ids):
        return self.head(self.embed(ids))


class TargetModel(nn.Module):
    def __init__(self, vocab=20, d=32):
        super().__init__()
        self.embed = nn.Embedding(vocab, d)
        self.head = nn.Linear(d, vocab)
    def forward(self, ids):
        return self.head(self.embed(ids))


def speculative_decode(draft, target, prompt, K=3, max_new=6):
    seq = prompt.tolist()
    draft.eval(); target.eval()
    with torch.no_grad():
        while len(seq) < len(prompt) + max_new:
            draft_tokens = []
            ctx = torch.tensor([seq])
            for _ in range(K):
                logits = draft(ctx)[0, -1]
                t = logits.argmax().item()
                draft_tokens.append(t)
                ctx = torch.tensor([seq + draft_tokens])
            verify_logits = target(torch.tensor([seq + draft_tokens]))[0]
            accepted = 0
            for i, t in enumerate(draft_tokens):
                p_t = torch.softmax(verify_logits[len(seq)+i-1], dim=-1)
                p_d = torch.softmax(draft(ctx)[0, -1], dim=-1)
                if torch.rand(1).item() < min(1.0, (p_t[t] / (p_d[t] + 1e-8)).item()):
                    seq.append(t); accepted += 1
                else:
                    break
            if accepted == 0:
                seq.append(verify_logits[len(seq)-1].argmax().item())
    return torch.tensor(seq)


prompt = DummyDataGenerator().prompt_ids()
out = speculative_decode(DraftModel(), TargetModel(), prompt, K=2, max_new=4)
print(f"speculative output: {out.tolist()}")
print("✓ Speculative decoding loop")
