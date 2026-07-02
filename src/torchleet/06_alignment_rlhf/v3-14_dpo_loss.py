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
# # V3-14: DPO Loss — Solution

# %%
import torch
import torch.nn as nn
from torch.utils.data import Dataset


class DummyDataGenerator:
    def __init__(self, vocab=20, seq=6, batch=4):
        torch.manual_seed(9)
        self.chosen = torch.randint(0, vocab, (batch, seq))
        self.rejected = torch.randint(0, vocab, (batch, seq))
    def pairs(self): return self.chosen, self.rejected


class PreferenceDataset(Dataset):
    def __init__(self, chosen, rejected):
        self.chosen, self.rejected = chosen, rejected
    def __len__(self): return len(self.chosen)
    def __getitem__(self, i): return self.chosen[i], self.rejected[i]


class SimpleLM(nn.Module):
    def __init__(self, vocab=20, d=32):
        super().__init__()
        self.embed = nn.Embedding(vocab, d)
        self.head = nn.Linear(d, vocab)
    def forward(self, ids):
        return self.head(self.embed(ids))


def get_batch_logps(model, ids):
    logits = model(ids)
    logp = torch.log_softmax(logits, dim=-1)
    return logp.gather(-1, ids.unsqueeze(-1)).squeeze(-1).sum(-1)


def dpo_loss(pi_c, pi_r, ref_c, ref_r, beta=0.1):
    return -torch.log(torch.sigmoid(beta * ((pi_c - pi_r) - (ref_c - ref_r)))).mean()


chosen, rejected = DummyDataGenerator().pairs()
policy, ref = SimpleLM(), SimpleLM()
ref.eval()
for p in ref.parameters(): p.requires_grad = False
loss = dpo_loss(
    get_batch_logps(policy, chosen), get_batch_logps(policy, rejected),
    get_batch_logps(ref, chosen), get_batch_logps(ref, rejected),
)
print(f"DPO loss: {loss.item():.4f}")
print("✓ DPO loss computed")
