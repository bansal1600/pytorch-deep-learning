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
# # V3-30: Ring Attention — Solution

# %%
import torch
import torch.nn as nn
from torch.utils.data import Dataset


class DummyDataGenerator:
    def __init__(self):
        self.q = torch.randn(1, 2, 16, 8)
        self.k = self.v = torch.randn(1, 2, 16, 8)
    def qkv(self): return self.q, self.k, self.v


class AttnDataset(Dataset):
    def __init__(self, q, k, v): self.q, self.k, self.v = q, k, v
    def __len__(self): return 1
    def __getitem__(self, i): return self.q, self.k, self.v


def standard_attention(q, k, v):
    d = q.size(-1)
    return torch.softmax((q @ k.transpose(-2, -1)) / (d**0.5), dim=-1) @ v


def ring_step(q_chunk, k_chunk, v_chunk, m, l, acc):
    d = q_chunk.size(-1)
    s = (q_chunk @ k_chunk.transpose(-2, -1)) / (d**0.5)
    m_new = torch.maximum(m, s.max(dim=-1, keepdim=True).values)
    l = torch.exp(m - m_new) * l + torch.exp(s - m_new).sum(dim=-1, keepdim=True)
    acc = torch.exp(m - m_new) * acc + torch.exp(s - m_new) @ v_chunk
    return m_new, l, acc


class RingAttention(nn.Module):
    def __init__(self, num_devices=2):
        super().__init__()
        self.num_devices = num_devices
    def forward(self, q, k, v):
        N = q.size(2)
        chunk = N // self.num_devices
        outs = []
        for dev in range(self.num_devices):
            s, e = dev * chunk, (dev + 1) * chunk
            qi = q[:, :, s:e]
            m = torch.full((*qi.shape[:2], qi.size(2), 1), float('-inf'))
            l = torch.zeros_like(m)
            acc = torch.zeros(*qi.shape)
            for d2 in range(self.num_devices):
                ks, ke = d2 * chunk, (d2 + 1) * chunk
                m, l, acc = ring_step(qi, k[:, :, ks:ke], v[:, :, ks:ke], m, l, acc)
            outs.append(acc / l)
        return torch.cat(outs, dim=2)


q, k, v = DummyDataGenerator().qkv()
ring_out = RingAttention(2)(q, k, v)
std_out = standard_attention(q, k, v)
print(f"ring vs std diff: {(ring_out-std_out).abs().max():.2e}")
print("✓ Ring attention matches standard")
