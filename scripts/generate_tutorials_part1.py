#!/usr/bin/env python3
"""Generate visual PyTorch tutorial notebooks with graphs and component breakdowns."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "tutorials"

SETUP = """import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = (10, 5)
plt.rcParams['font.size'] = 11
torch.manual_seed(42)
np.random.seed(42)
"""


def nb(cells):
    return {
        "nbformat": 4, "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "colab": {"provenance": []},
        },
        "cells": cells,
    }


def md(t):
    return {"cell_type": "markdown", "metadata": {}, "source": t.splitlines(keepends=True)}


def code(t):
    return {"cell_type": "code", "metadata": {}, "outputs": [], "execution_count": None,
            "source": t.splitlines(keepends=True)}


def save(folder, name, cells):
    p = ROOT / folder / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(nb(cells), indent=1))
    print(f"  {folder}/{name}")


# =============================================================================
# 01 TENSORS
# =============================================================================

TENSOR_TUTORIALS = [
    ("01_tensors_and_basics", "01_scalar_vector_matrix_tensor.ipynb", [
        md("# Tutorial 01: Scalar → Vector → Matrix → Tensor\n\nUnderstand how data rank grows from a single number to multi-dimensional tensors."),
        code(SETUP),
        md("## 1. Scalar (0-D)\nA single number. In PyTorch: a 0-dimensional tensor."),
        code("""scalar = torch.tensor(3.14)
print(f"value={scalar.item()}, shape={scalar.shape}, ndim={scalar.ndim}")

fig, axes = plt.subplots(1, 4, figsize=(14, 3))
axes[0].scatter([0], [scalar.item()], s=200, c='steelblue')
axes[0].set_title('Scalar (0-D)'); axes[0].set_ylim(0, 5); axes[0].axhline(0, color='gray', lw=0.5)
plt.tight_layout(); plt.show()"""),
        md("## 2. Vector (1-D)\nAn ordered list of numbers — one **axis**."),
        code("""vector = torch.tensor([1., 2., 3., 4., 5.])
print(f"shape={vector.shape}")

fig, ax = plt.subplots(figsize=(8, 3))
ax.bar(range(len(vector)), vector.numpy(), color='coral', edgecolor='black')
ax.set_title('Vector (1-D)'); ax.set_xlabel('index'); ax.set_ylabel('value')
plt.show()"""),
        md("## 3. Matrix (2-D)\nRows × columns — two axes."),
        code("""matrix = torch.arange(12).reshape(3, 4).float()
print(matrix)

fig, ax = plt.subplots(figsize=(6, 4))
im = ax.imshow(matrix.numpy(), cmap='viridis', aspect='auto')
plt.colorbar(im); ax.set_title('Matrix (2-D) — heatmap view')
ax.set_xlabel('column'); ax.set_ylabel('row')
plt.show()"""),
        md("## 4. Tensor (3-D+)\nStack matrices → images `(C,H,W)`, sequences `(B,S,D)`, etc."),
        code("""tensor_3d = torch.randn(3, 4, 5)  # e.g. 3 channels × 4×5 image
print(f"3-D tensor shape: {tensor_3d.shape}")

fig, axes = plt.subplots(1, 3, figsize=(12, 3))
for i, ax in enumerate(axes):
    im = ax.imshow(tensor_3d[i].numpy(), cmap='coolwarm', aspect='auto')
    ax.set_title(f'Slice [{i}, :, :]'); plt.colorbar(im, ax=ax, fraction=0.046)
plt.suptitle('3-D Tensor — each slice is a 2-D matrix'); plt.tight_layout(); plt.show()"""),
        md("## Summary\n| Rank | Name | Shape example |\n|------|------|---------------|\n| 0 | Scalar | `()` |\n| 1 | Vector | `(5,)` |\n| 2 | Matrix | `(3, 4)` |\n| 3+ | Tensor | `(3, 4, 5)` |"),
    ]),
    ("01_tensors_and_basics", "02_creating_tensors.ipynb", [
        md("# Tutorial 02: Every Way to Create a Tensor\n\n`torch.tensor`, `zeros`, `ones`, `rand`, `arange`, `linspace`, `eye`, from NumPy."),
        code(SETUP),
        md("## Factory functions"),
        code("""methods = {
    'torch.tensor([1,2,3])': torch.tensor([1, 2, 3]),
    'torch.zeros(2,3)': torch.zeros(2, 3),
    'torch.ones(2,3)': torch.ones(2, 3),
    'torch.full((2,3), 7)': torch.full((2, 3), 7),
    'torch.eye(4)': torch.eye(4),
    'torch.arange(0,10,2)': torch.arange(0, 10, 2),
    'torch.linspace(0,1,5)': torch.linspace(0, 1, 5),
    'torch.rand(2,3)': torch.rand(2, 3),
    'torch.randn(2,3)': torch.randn(2, 3),
}
for name, t in methods.items():
    print(f"{name:30s} shape={tuple(t.shape)}")"""),
        md("## Visual: `rand` vs `randn` distribution"),
        code("""uniform = torch.rand(10000)
normal = torch.randn(10000)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].hist(uniform.numpy(), bins=50, color='skyblue', edgecolor='black')
axes[0].set_title('torch.rand — Uniform [0, 1)'); axes[0].set_xlabel('value')
axes[1].hist(normal.numpy(), bins=50, color='salmon', edgecolor='black')
axes[1].set_title('torch.randn — Standard Normal'); axes[1].set_xlabel('value')
plt.tight_layout(); plt.show()"""),
        md("## From NumPy (shares memory!)"),
        code("""import numpy as np
arr = np.array([1., 2., 3.])
t = torch.from_numpy(arr)
arr[0] = 99
print(f"NumPy changed → tensor also changed: {t}")"""),
    ]),
    ("01_tensors_and_basics", "03_tensor_dtypes_and_devices.ipynb", [
        md("# Tutorial 03: dtypes & Devices (CPU / GPU)\n\n`float32` vs `float64`, `int64`, moving tensors with `.to(device)`."),
        code(SETUP + "\ndevice = torch.device('cuda' if torch.cuda.is_available() else 'cpu')\nprint(f'Using: {device}')"),
        md("## Data types matter for memory & precision"),
        code("""dtypes = [torch.float16, torch.float32, torch.float64, torch.int32, torch.int64, torch.bool]
fig, ax = plt.subplots(figsize=(8, 4))
names, sizes = [], []
for dt in dtypes:
    t = torch.zeros(1, dtype=dt)
    names.append(str(dt).split('.')[-1])
    sizes.append(t.element_size() * 8)
ax.barh(names, sizes, color='teal', edgecolor='black')
ax.set_xlabel('bits per element'); ax.set_title('Tensor dtype → memory per element')
plt.tight_layout(); plt.show()"""),
        md("## CPU vs GPU"),
        code("""x_cpu = torch.randn(3, 3)
x_gpu = x_cpu.to(device)
print(f"CPU: {x_cpu.device}  |  GPU: {x_gpu.device}")

# Timing demo (matrix multiply)
size = 2000
a = torch.randn(size, size)
b = torch.randn(size, size)
import time
t0 = time.perf_counter(); _ = a @ b; t_cpu = time.perf_counter() - t0
if device.type == 'cuda':
    ag, bg = a.to(device), b.to(device)
    torch.cuda.synchronize()
    t0 = time.perf_counter(); _ = ag @ bg; torch.cuda.synchronize(); t_gpu = time.perf_counter() - t0
    print(f"CPU matmul: {t_cpu:.3f}s | GPU matmul: {t_gpu:.3f}s")
else:
    print(f"CPU matmul: {t_cpu:.3f}s (no GPU in this session)")"""),
    ]),
    ("01_tensors_and_basics", "04_indexing_slicing_broadcasting.ipynb", [
        md("# Tutorial 04: Indexing, Slicing & Broadcasting\n\nHow to access parts of tensors and combine different shapes."),
        code(SETUP),
        md("## Indexing & slicing"),
        code("""t = torch.arange(24).reshape(4, 6)
print("Full tensor:\\n", t)
print("\\nRow 1:", t[1])
print("Column 2:", t[:, 2])
print("Block [1:3, 2:5]:\\n", t[1:3, 2:5])

fig, axes = plt.subplots(1, 3, figsize=(14, 3))
axes[0].imshow(t.numpy(), cmap='Blues'); axes[0].set_title('Full tensor')
mask = torch.zeros_like(t, dtype=bool); mask[1:3, 2:5] = True
highlighted = t.numpy().astype(float).copy(); highlighted[~mask.numpy()] = np.nan
axes[1].imshow(highlighted, cmap='Reds'); axes[1].set_title('Sliced block')
axes[2].imshow(t[:, 2].numpy().reshape(-1, 1), cmap='Greens', aspect=0.3)
axes[2].set_title('Column 2')
plt.tight_layout(); plt.show()"""),
        md("## Broadcasting — align shapes without copying"),
        code("""a = torch.tensor([[1.],[2.],[3.]])  # (3,1)
b = torch.tensor([10., 20., 30.])       # (3,)
result = a + b
print(f"a shape {a.shape} + b shape {b.shape} → {result.shape}\\n{result}")

fig, axes = plt.subplots(1, 3, figsize=(12, 3))
axes[0].imshow(a.numpy(), cmap='coolwarm', aspect='auto'); axes[0].set_title('a (3,1)')
axes[1].imshow(b.numpy().reshape(1,-1), cmap='coolwarm', aspect='auto'); axes[1].set_title('b (1,3) broadcast')
axes[2].imshow(result.numpy(), cmap='coolwarm', aspect='auto'); axes[2].set_title('result (3,3)')
plt.tight_layout(); plt.show()"""),
    ]),
    ("01_tensors_and_basics", "05_elementwise_vs_matrix_ops.ipynb", [
        md("# Tutorial 05: Element-wise vs Matrix Operations\n\n`*` is element-wise; `@` or `torch.mm` is matrix multiply."),
        code(SETUP),
        code("""A = torch.tensor([[1., 2.], [3., 4.]])
B = torch.tensor([[5., 6.], [7., 8.]])
elem = A * B
mat = A @ B
print("Element-wise A*B:\\n", elem)
print("\\nMatrix A@B:\\n", mat)

fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
for ax, m, title in zip(axes, [A, B, elem], ['A', 'B', 'A * B (element-wise)']):
    im = ax.imshow(m.numpy(), cmap='plasma', vmin=0, vmax=10)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f'{m[i,j]:.0f}', ha='center', va='center', color='white', fontsize=14)
    ax.set_title(title); plt.colorbar(im, ax=ax, fraction=0.046)
plt.tight_layout(); plt.show()

fig, ax = plt.subplots(figsize=(4, 3.5))
im = ax.imshow(mat.numpy(), cmap='plasma')
for i in range(2):
    for j in range(2):
        ax.text(j, i, f'{mat[i,j]:.0f}', ha='center', va='center', color='white', fontsize=14)
ax.set_title('A @ B (matrix multiply)'); plt.colorbar(im); plt.show()"""),
    ]),
    ("01_tensors_and_basics", "06_reshape_view_contiguous.ipynb", [
        md("# Tutorial 06: Reshape, View & Contiguous Memory\n\nHow tensor layout in memory affects `view` vs `reshape`."),
        code(SETUP),
        code("""t = torch.arange(12)
print("Original:", t)

views = {
    '(4,3)': t.reshape(4, 3),
    '(2,2,3)': t.reshape(2, 2, 3),
    '(3,4)': t.view(3, 4),
}
fig, axes = plt.subplots(1, 3, figsize=(12, 3))
for ax, (name, v) in zip(axes, views.items()):
    im = ax.imshow(v.numpy(), cmap='tab20', aspect='auto')
    ax.set_title(f'reshape{name}'); plt.colorbar(im, ax=ax, fraction=0.046)
plt.suptitle('Same 12 elements, different shapes'); plt.tight_layout(); plt.show()

# Transpose changes memory layout
m = torch.arange(6).reshape(2, 3)
print(f"\\nBefore T: is_contiguous={m.is_contiguous()}")
mt = m.T
print(f"After  T: is_contiguous={mt.is_contiguous()} → need .contiguous() before .view()")"""),
    ]),
    ("01_tensors_and_basics", "07_tensor_aggregation.ipynb", [
        md("# Tutorial 07: Sum, Mean, Max, Argmax along Dimensions\n\nReducing tensors along `dim` — the foundation of loss functions."),
        code(SETUP),
        code("""t = torch.arange(12).reshape(3, 4).float()
print(t)
print(f"\\nGlobal sum: {t.sum()}")
print(f"Sum along rows (dim=0): {t.sum(dim=0)}")
print(f"Sum along cols (dim=1): {t.sum(dim=1)}")

fig, axes = plt.subplots(1, 3, figsize=(13, 3.5))
axes[0].imshow(t.numpy(), cmap='YlOrRd')
for i in range(3):
    for j in range(4):
        axes[0].text(j, i, f'{int(t[i,j])}', ha='center', va='center')
axes[0].set_title('Original (3×4)')

axes[1].bar(range(4), t.sum(dim=0).numpy(), color='steelblue')
axes[1].set_title('sum(dim=0) — collapse rows')

axes[2].barh(range(3), t.sum(dim=1).numpy(), color='coral')
axes[2].set_title('sum(dim=1) — collapse cols')
plt.tight_layout(); plt.show()"""),
    ]),
]

# I'll continue with more tutorial groups in part 2 of the file
# For now write part 1 and a runner

if __name__ == "__main__":
    print("Generating tensor tutorials...")
    for folder, name, cells in TENSOR_TUTORIALS:
        save(folder, name, cells)
