#!/usr/bin/env python3
"""Extra tutorials: sequences, probability, data augmentation."""

from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[1] / "tutorials"
SETUP = """import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (10, 5)
torch.manual_seed(42); np.random.seed(42)
"""

def nb(cells):
    return {"nbformat": 4, "nbformat_minor": 5,
            "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                         "colab": {"provenance": []}}, "cells": cells}

def md(t): return {"cell_type": "markdown", "metadata": {}, "source": t.splitlines(keepends=True)}
def code(t): return {"cell_type": "code", "metadata": {}, "outputs": [], "execution_count": None,
                     "source": t.splitlines(keepends=True)}

def save(folder, name, cells):
    p = ROOT / folder / name
    if p.exists():
        print(f"  SKIP {folder}/{name}")
        return
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(nb(cells), indent=1))
    print(f"  wrote {folder}/{name}")

EXTRA = [
("10_sequence_models", "01_sequences_and_rnn_unfold.ipynb", [
    md("# Sequences & RNN — Unfolding Time\n\nAn RNN processes one time step at a time, carrying **hidden state** forward."),
    code(SETUP),
    md("## Synthetic sequence data"),
    code("""class DummyDataGenerator:
    def __init__(self, seq_len=20, n=1):
        t = torch.linspace(0, 4*np.pi, seq_len)
        self.data = torch.sin(t) + 0.1*torch.randn(seq_len)
        self.seq_len = seq_len
    def get_series(self): return self.data

class SequenceDataset(Dataset):
    def __init__(self, data, window=5):
        self.data = data; self.window = window
    def __len__(self): return len(self.data) - self.window
    def __getitem__(self, i):
        x = self.data[i:i+self.window]
        y = self.data[i+self.window]
        return x.unsqueeze(-1), y

gen = DummyDataGenerator()
data = gen.get_series()
fig, ax = plt.subplots(figsize=(10, 3))
ax.plot(data.numpy(), 'b-', lw=2); ax.set_title('Time series — each point is one time step')
ax.set_xlabel('time t'); ax.set_ylabel('value')
plt.show()"""),
    md("## RNN cell — hidden state at each step"),
    code("""class TinyRNN(nn.Module):
    def __init__(self, d_in=1, hidden=8):
        super().__init__()
        self.rnn = nn.RNN(d_in, hidden, batch_first=True)
    def forward(self, x):
        out, h = self.rnn(x)
        return out, h

x = data[:10].unsqueeze(0).unsqueeze(-1)  # (1, 10, 1)
model = TinyRNN()
out, h = model(x)
print(f"output per step shape: {out.shape}, final hidden: {h.shape}")

fig, axes = plt.subplots(2, 1, figsize=(10, 5))
axes[0].plot(out.squeeze().detach().numpy(), 'g-o', label='RNN output each step')
axes[0].plot(x.squeeze().numpy(), 'b--', alpha=0.5, label='input')
axes[0].legend(); axes[0].set_title('RNN outputs at each time step')
axes[1].bar(range(out.shape[-1]), h.squeeze().detach().numpy(), color='coral')
axes[1].set_title('Final hidden state vector (memory)')
plt.tight_layout(); plt.show()"""),
]),
("10_sequence_models", "02_lstm_gates_intuition.ipynb", [
    md("# LSTM Gates — Forget, Input, Output\n\nLSTMs use gates to control what information flows through time."),
    code(SETUP),
    code("""class DummyDataGenerator:
    def __init__(self, seq_len=30):
        self.x = torch.randn(1, seq_len, 4)
    def input(self): return self.x

class SequenceDataset(Dataset):
    def __init__(self, x): self.x = x
    def __len__(self): return 1
    def __getitem__(self, i): return self.x[i]

class LSTMModel(nn.Module):
    def __init__(self, d_in=4, hidden=16):
        super().__init__()
        self.lstm = nn.LSTM(d_in, hidden, batch_first=True)
        self.fc = nn.Linear(hidden, 1)
    def forward(self, x):
        out, (h, c) = self.lstm(x)
        return self.fc(out[:, -1]), out

x = DummyDataGenerator().input()
model = LSTMModel()
pred, all_h = model(x)

fig, ax = plt.subplots(figsize=(10, 4))
for i in range(min(4, all_h.shape[-1])):
    ax.plot(all_h[0, :, i].detach().numpy(), label=f'hidden dim {i}', alpha=0.8)
ax.set_title('LSTM hidden state evolution over time'); ax.legend(); ax.set_xlabel('time step')
plt.show()

# Gate concept diagram (schematic)
fig, ax = plt.subplots(figsize=(8, 3))
gates = ['Forget gate', 'Input gate', 'Output gate', 'Cell state']
vals = [0.2, 0.7, 0.5, 1.0]
ax.barh(gates, vals, color=['#e74c3c','#2ecc71','#3498db','#9b59b6'])
ax.set_xlim(0, 1); ax.set_title('LSTM gates (example values at one step — σ(output) ∈ [0,1])')
plt.tight_layout(); plt.show()"""),
]),
("10_sequence_models", "03_attention_weights_heatmap.ipynb", [
    md("# Attention Weights as a Heatmap\n\nAttention learns **which positions to focus on** when producing each output."),
    code(SETUP),
    code("""class DummyDataGenerator:
    def __init__(self, seq_len=8, d=16):
        self.q = torch.randn(1, seq_len, d)
        self.k = torch.randn(1, seq_len, d)
        self.v = torch.randn(1, seq_len, d)
    def qkv(self): return self.q, self.k, self.v

class AttentionModel(nn.Module):
    def __init__(self, d=16):
        super().__init__()
        self.d = d
    def forward(self, q, k, v):
        scores = (q @ k.transpose(-2, -1)) / (self.d ** 0.5)
        weights = torch.softmax(scores, dim=-1)
        out = weights @ v
        return out, weights

q, k, v = DummyDataGenerator().qkv()
model = AttentionModel(d=16)
out, weights = model(q, k, v)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
im = axes[0].imshow(weights[0].detach().numpy(), cmap='hot', aspect='auto')
axes[0].set_xlabel('Key position'); axes[0].set_ylabel('Query position')
axes[0].set_title('Attention weights (softmax of QK^T/√d)')
plt.colorbar(im, ax=axes[0])
axes[1].bar(range(weights.shape[-1]), weights[0, -1].detach().numpy(), color='steelblue')
axes[1].set_title('Last query attends to these keys'); axes[1].set_xlabel('key index')
plt.tight_layout(); plt.show()"""),
]),
("11_probability", "01_softmax_temperature.ipynb", [
    md("# Softmax & Temperature\n\nSoftmax converts logits to probabilities. Temperature controls sharpness."),
    code(SETUP),
    code("""logits = torch.tensor([2.0, 1.0, 0.5, 0.1])

def softmax_temp(logits, T=1.0):
    x = logits / T
    return torch.softmax(x, dim=0)

temps = [0.1, 0.5, 1.0, 2.0, 5.0]
fig, axes = plt.subplots(1, len(temps), figsize=(14, 3), sharey=True)
for ax, T in zip(axes, temps):
    p = softmax_temp(logits, T)
    ax.bar(range(4), p.numpy(), color=plt.cm.viridis(T/5))
    ax.set_title(f'T={T}'); ax.set_ylim(0, 1)
plt.suptitle('Low T → peaked (confident), High T → uniform (exploratory)'); plt.tight_layout(); plt.show()"""),
]),
("11_probability", "02_sampling_strategies.ipynb", [
    md("# Sampling: Argmax vs Multinomial vs Top-k"),
    code(SETUP),
    code("""logits = torch.tensor([1., 3., 2., 0.5, 0.1])
probs = torch.softmax(logits, dim=0)

strategies = {
    'argmax': logits.argmax().item(),
    'multinomial': torch.multinomial(probs, 1).item(),
}
# top-k
k = 3
topk = torch.topk(logits, k)
mask = torch.full_like(logits, float('-inf'))
mask[topk.indices] = logits[topk.indices]
topk_probs = torch.softmax(mask, dim=0)

fig, axes = plt.subplots(1, 3, figsize=(12, 3))
axes[0].bar(range(5), probs.numpy(), color='gray', alpha=0.5)
axes[0].bar(strategies['argmax'], probs[strategies['argmax']].item(), color='red')
axes[0].set_title(f"Argmax → token {strategies['argmax']}")
axes[1].bar(range(5), probs.numpy(), color='steelblue')
axes[1].set_title('Full distribution (multinomial samples from this)')
axes[2].bar(range(5), topk_probs.numpy(), color='coral')
axes[2].set_title('Top-k=3 masked distribution')
plt.tight_layout(); plt.show()"""),
]),
("12_data_pipeline", "01_normalization_standardization.ipynb", [
    md("# Normalization & Standardization\n\nScale features so models train faster and more stably."),
    code(SETUP),
    code("""class DummyDataGenerator:
    def __init__(self, n=500):
        self.x = torch.cat([torch.randn(n//2)*1 + 2, torch.randn(n//2)*5 + 10])
    def data(self): return self.x

raw = DummyDataGenerator().data()
standardized = (raw - raw.mean()) / raw.std()
normalized = (raw - raw.min()) / (raw.max() - raw.min())

fig, axes = plt.subplots(1, 3, figsize=(14, 4))
for ax, d, title in zip(axes, [raw, standardized, normalized],
                        ['Raw', 'Standardized (z-score)', 'Min-Max [0,1]']):
    ax.hist(d.numpy(), bins=40, color='teal', edgecolor='black', alpha=0.8)
    ax.set_title(f'{title}\\nμ={d.mean():.2f}, σ={d.std():.2f}')
plt.tight_layout(); plt.show()"""),
]),
("12_data_pipeline", "02_train_augmentation_demo.ipynb", [
    md("# Data Augmentation — Why It Helps\n\nAugment training data to reduce overfitting (demo on synthetic images)."),
    code(SETUP + "\nfrom torchvision import transforms"),
    code("""class DummyDataGenerator:
    def __init__(self):
        self.img = torch.rand(1, 28, 28)
    def image(self): return self.img

class ImageDataset(Dataset):
    def __init__(self, img, tfm=None):
        self.img = img; self.tfm = tfm
    def __len__(self): return 1
    def __getitem__(self, i):
        x = self.img
        return (self.tfm(x) if self.tfm else x), 0

tfm = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomAffine(degrees=15, translate=(0.1, 0.1)),
])
base = DummyDataGenerator().image()
fig, axes = plt.subplots(2, 5, figsize=(12, 5))
for ax in axes.flat:
    aug, _ = ImageDataset(base, tfm)[0]
    ax.imshow(aug.squeeze().numpy(), cmap='gray'); ax.axis('off')
plt.suptitle('Random augmentations of the same synthetic image'); plt.tight_layout(); plt.show()"""),
]),
]

if __name__ == "__main__":
    print("Generating extra tutorials...")
    for folder, name, cells in EXTRA:
        save(folder, name, cells)
