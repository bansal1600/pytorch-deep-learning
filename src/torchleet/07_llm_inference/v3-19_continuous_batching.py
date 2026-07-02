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
# # V3-19: Continuous Batching — Solution

# %%
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from dataclasses import dataclass, field


class DummyDataGenerator:
    @staticmethod
    def requests():
        return [
            {"id": 0, "input_ids": [1, 2], "max_gen": 3},
            {"id": 1, "input_ids": [3], "max_gen": 4},
            {"id": 2, "input_ids": [4, 5, 6], "max_gen": 2},
        ]


@dataclass
class Request:
    id: int
    input_ids: list
    generated_ids: list = field(default_factory=list)
    max_gen_len: int = 5
    @property
    def is_done(self):
        return len(self.generated_ids) >= self.max_gen_len
    @property
    def all_ids(self):
        return self.input_ids + self.generated_ids


class RequestDataset(Dataset):
    def __init__(self, reqs): self.reqs = reqs
    def __len__(self): return len(self.reqs)
    def __getitem__(self, i): return self.reqs[i]


class DummyLLM(nn.Module):
    def __init__(self, vocab=10):
        super().__init__()
        self.embed = nn.Embedding(vocab, 8)
        self.head = nn.Linear(8, vocab)
    def forward(self, ids):
        return self.head(self.embed(ids))


class ContinuousBatchScheduler:
    def __init__(self, model, max_batch=4):
        self.model = model
        self.max_batch = max_batch
        self.queue = []
        self.active = []
    def add_request(self, req: Request):
        self.queue.append(req)
    def step(self):
        while self.queue and len(self.active) < self.max_batch:
            self.active.append(self.queue.pop(0))
        if not self.active:
            return []
        still = []
        for req in self.active:
            if not req.is_done:
                logits = self.model(torch.tensor([req.all_ids]))
                req.generated_ids.append(logits[0, -1].argmax().item())
            if not req.is_done:
                still.append(req)
        done = [r for r in self.active if r.is_done]
        self.active = still
        return done


sched = ContinuousBatchScheduler(DummyLLM())
for r in DummyDataGenerator.requests():
    sched.add_request(Request(id=r["id"], input_ids=r["input_ids"], max_gen_len=r["max_gen"]))
finished = []
while sched.queue or sched.active:
    finished.extend(sched.step())
print(f"completed {len(finished)} requests")
print("✓ Continuous batching scheduler")
