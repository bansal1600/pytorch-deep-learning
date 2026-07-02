#!/usr/bin/env python3
"""Generate visual PyTorch tutorial notebooks (parts 2–9). Skips existing files."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "tutorials"

SETUP = """import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = (10, 5)
plt.rcParams['font.size'] = 11
torch.manual_seed(42)
np.random.seed(42)
"""


def nb(cells):
    return {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "colab": {"provenance": []},
        },
        "cells": cells,
    }


def md(t):
    return {"cell_type": "markdown", "metadata": {}, "source": t.splitlines(keepends=True)}


def code(t):
    return {
        "cell_type": "code",
        "metadata": {},
        "outputs": [],
        "execution_count": None,
        "source": t.splitlines(keepends=True),
    }


def save(folder, name, cells):
    p = ROOT / folder / name
    if p.exists():
        print(f"  SKIP (exists) {folder}/{name}")
        return False
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(nb(cells), indent=1))
    print(f"  CREATED {folder}/{name}")
    return True


# Shared snippets
DUMMY_GEN = """class DummyDataGenerator:
    \"\"\"Synthetic classification data for CPU-only tutorials.\"\"\"
    def __init__(self, n_samples=256, n_features=8, n_classes=3, seed=42):
        g = torch.Generator().manual_seed(seed)
        self.X = torch.randn(n_samples, n_features, generator=g)
        self.y = torch.randint(0, n_classes, (n_samples,), generator=g)

    def tensors(self):
        return self.X, self.y
"""

DATASET_CLASS = """class TabularDataset(Dataset):
    def __init__(self, X, y):
        self.X, self.y = X, y
    def __len__(self):
        return len(self.y)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
"""

SIMPLE_MLP = """class SimpleMLP(nn.Module):
    def __init__(self, in_dim=8, hidden=16, n_classes=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, n_classes),
        )
    def forward(self, x):
        return self.net(x)
"""

CREATED = 0


# =============================================================================
# 02 AUTOGRAD (5 notebooks)
# =============================================================================

AUTOGRAD_TUTORIALS = [
    ("02_autograd", "01_computation_graph.ipynb", [
        md("# Computation Graph for y = x²\n\nPyTorch builds a **dynamic computation graph** during the forward pass. Each operation becomes a node; edges carry tensors and gradients flow backward along them."),
        code(SETUP),
        md("## 1. Forward pass: build the graph\n`x` is a leaf tensor with `requires_grad=True`. Squaring creates node `PowBackward`."),
        code("""x = torch.tensor(3.0, requires_grad=True)
y = x ** 2
print(f"x={x.item():.2f}, y={y.item():.2f}")
print(f"y.grad_fn: {y.grad_fn}")"""),
        md("## 2. Visualize nodes and values"),
        code("""fig, ax = plt.subplots(figsize=(8, 4))
nodes = ['x\\n(3.0)', 'x²\\n(9.0)']
xs = [0.2, 0.8]
for i, (node, xp) in enumerate(zip(nodes, xs)):
    ax.add_patch(plt.Circle((xp, 0.5), 0.12, fc='lightblue', ec='navy', lw=2))
    ax.text(xp, 0.5, node, ha='center', va='center', fontsize=11)
ax.annotate('', xy=(0.68, 0.5), xytext=(0.32, 0.5),
            arrowprops=dict(arrowstyle='->', lw=2, color='darkgreen'))
ax.text(0.5, 0.62, 'forward: y = x²', ha='center', fontsize=12)
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off')
ax.set_title('Computation graph: one multiply path')
plt.tight_layout(); plt.show()"""),
        md("## 3. Backward pass: gradient dy/dx = 2x"),
        code("""y.backward()
print(f"dy/dx = {x.grad.item():.2f}  (expected 2×3 = 6)")

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].bar(['x', 'y'], [x.item(), y.item()], color=['steelblue', 'coral'])
axes[0].set_title('Forward values'); axes[0].set_ylabel('value')
axes[1].bar(['∂y/∂x'], [x.grad.item()], color='seagreen')
axes[1].set_title('Backward gradient'); axes[1].set_ylabel('gradient')
plt.tight_layout(); plt.show()"""),
        md("## 4. Chained operations extend the graph\nEach intermediate tensor keeps a link to its parent via `.grad_fn`."),
        code("""x2 = torch.tensor(2.0, requires_grad=True)
z = (x2 ** 2) + 3 * x2
z.backward()
print(f"z = {z.item():.2f}, dz/dx = {x2.grad.item():.2f}")

fig, ax = plt.subplots(figsize=(9, 3))
labels = ['x²', '+3x', 'z']
vals = [x2.item()**2, 3*x2.item(), z.item()]
ax.plot(labels, vals, 'o-', color='purple', lw=2, markersize=10)
ax.set_title('Intermediate forward values along the chain')
ax.set_ylabel('value'); ax.grid(True, alpha=0.3)
plt.tight_layout(); plt.show()"""),
    ]),
    ("02_autograd", "02_forward_backward_pass.ipynb", [
        md("# Forward & Backward Pass on a Small Network\n\nTrace **activations** in the forward pass and **gradients** stored on parameters after `loss.backward()`."),
        code(SETUP + "\n" + DUMMY_GEN + "\n" + SIMPLE_MLP),
        md("## 1. Forward pass — record layer outputs"),
        code("""gen = DummyDataGenerator(n_samples=4, n_features=8, n_classes=3)
X, y = gen.tensors()
model = SimpleMLP(in_dim=8, hidden=16, n_classes=3)

h = model.net[0](X)
logits = model.net[2](F.relu(h))
loss = F.cross_entropy(logits, y)
print(f"logits shape: {logits.shape}, loss: {loss.item():.4f}")"""),
        md("## 2. Visualize forward activations"),
        code("""fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].imshow(h.detach().numpy(), aspect='auto', cmap='viridis')
axes[0].set_title('Hidden activations (batch × hidden)'); axes[0].set_xlabel('neuron')
axes[1].imshow(logits.detach().numpy(), aspect='auto', cmap='coolwarm')
axes[1].set_title('Output logits'); axes[1].set_xlabel('class')
plt.tight_layout(); plt.show()"""),
        md("## 3. Backward pass — gradients on weights"),
        code("""model.zero_grad()
loss.backward()
w1_grad = model.net[0].weight.grad
w2_grad = model.net[2].weight.grad
print(f"W1 grad norm: {w1_grad.norm():.4f}, W2 grad norm: {w2_grad.norm():.4f}")"""),
        md("## 4. Plot gradient magnitudes per layer"),
        code("""fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].imshow(w1_grad.numpy(), aspect='auto', cmap='RdBu_r')
axes[0].set_title('∂L/∂W1 (input→hidden)')
axes[1].imshow(w2_grad.numpy(), aspect='auto', cmap='RdBu_r')
axes[1].set_title('∂L/∂W2 (hidden→output)')
plt.tight_layout(); plt.show()

fig, ax = plt.subplots(figsize=(6, 4))
layers = ['W1', 'b1', 'W2', 'b2']
norms = [model.net[i].weight.grad.norm().item() if hasattr(model.net[i], 'weight')
         else model.net[i].bias.grad.norm().item() for i in [0, 0, 2, 2]]
# fix layer names
params = [('W1', model.net[0].weight), ('b1', model.net[0].bias),
          ('W2', model.net[2].weight), ('b2', model.net[2].bias)]
names, gnorms = zip(*[(n, p.grad.norm().item()) for n, p in params])
ax.bar(names, gnorms, color='teal', edgecolor='black')
ax.set_title('Gradient L2 norm per parameter tensor')
ax.set_ylabel('||grad||')
plt.tight_layout(); plt.show()"""),
    ]),
    ("02_autograd", "03_gradients_multivariable.ipynb", [
        md("# Multivariable Gradients: f(x, y) = x² + y²\n\nThe gradient points **uphill**; negative gradient is the direction of steepest descent."),
        code(SETUP),
        md("## 1. Define f and compute gradients at sample points"),
        code("""def f(x, y):
    return x**2 + y**2

pts = torch.tensor([[1., 2.], [2., 1.], [-1., 1.]], requires_grad=True)
vals = (pts[:, 0]**2 + pts[:, 1]**2)
for i in range(len(pts)):
    if pts.grad is not None:
        pts.grad.zero_()
    vals[i].backward(retain_graph=True)
print("Points and gradients:")
for i, p in enumerate(pts):
    print(f"  ({p[0]:.1f}, {p[1]:.1f}) → ∇f = ({2*p[0]:.1f}, {2*p[1]:.1f})")"""),
        md("## 2. Contour plot of f(x, y)"),
        code("""xx, yy = np.meshgrid(np.linspace(-3, 3, 100), np.linspace(-3, 3, 100))
zz = xx**2 + yy**2

fig, ax = plt.subplots(figsize=(7, 6))
cs = ax.contour(xx, yy, zz, levels=15, cmap='viridis')
ax.clabel(cs, inline=True, fontsize=8)
ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_title('Contour: f(x,y) = x² + y²')
ax.set_aspect('equal')
plt.tight_layout(); plt.show()"""),
        md("## 3. Gradient arrows on the contour map"),
        code("""grid_x = np.linspace(-2.5, 2.5, 8)
grid_y = np.linspace(-2.5, 2.5, 8)
GX, GY = np.meshgrid(grid_x, grid_y)
U = 2 * GX
V = 2 * GY

fig, ax = plt.subplots(figsize=(7, 6))
ax.contour(xx, yy, zz, levels=15, cmap='viridis', alpha=0.7)
ax.quiver(GX, GY, U, V, color='red', alpha=0.8, scale=40, width=0.004)
ax.set_xlabel('x'); ax.set_ylabel('y')
ax.set_title('Gradient field (red arrows) on contours')
ax.set_aspect('equal')
plt.tight_layout(); plt.show()"""),
        md("## 4. Gradient descent steps toward the minimum at (0, 0)"),
        code("""pos = torch.tensor([2.5, 2.0])
lr = 0.1
path = [pos.clone()]
for _ in range(15):
    pos = pos - lr * 2 * pos
    path.append(pos.clone())
path = torch.stack(path).numpy()

fig, ax = plt.subplots(figsize=(7, 6))
ax.contour(xx, yy, zz, levels=15, cmap='viridis', alpha=0.6)
ax.plot(path[:, 0], path[:, 1], 'ro-', lw=2, markersize=5, label='GD path')
ax.scatter([0], [0], s=120, c='gold', edgecolors='black', zorder=5, label='minimum')
ax.legend(); ax.set_aspect('equal')
ax.set_title('Gradient descent with lr=0.1')
plt.tight_layout(); plt.show()"""),
    ]),
    ("02_autograd", "04_detach_and_no_grad.ipynb", [
        md("# When Gradients Stop: `detach()` and `torch.no_grad()`\n\nUse these to **freeze** parts of the graph for inference, metrics, or target networks."),
        code(SETUP),
        md("## 1. Normal backward vs detached tensor"),
        code("""x = torch.tensor(2.0, requires_grad=True)
y = x ** 2
z = y.detach()  # breaks graph here
w = z * 3
try:
    w.backward()
except RuntimeError as e:
    print(f"Cannot backprop through detach: {type(e).__name__}")

y.backward()
print(f"Gradient on x from y branch: {x.grad.item()}")"""),
        md("## 2. Compare gradient flow with and without detach"),
        code("""fig, axes = plt.subplots(1, 2, figsize=(10, 4))
for ax, title, grad_on in zip(axes, ['with grad', 'detached'], [True, False]):
    x2 = torch.tensor(2.0, requires_grad=True)
    y2 = x2 ** 2
    z2 = y2 if grad_on else y2.detach()
    loss = z2 * 2
    if grad_on:
        loss.backward()
        g = x2.grad.item()
    else:
        g = 0.0
    ax.bar(['∂loss/∂x'], [g], color='steelblue' if grad_on else 'gray')
    ax.set_ylim(0, 10); ax.set_title(title)
plt.suptitle('detach() blocks gradients to x'); plt.tight_layout(); plt.show()"""),
        md("## 3. `torch.no_grad()` for inference"),
        code("""model = nn.Linear(4, 2)
x = torch.randn(8, 4)
with torch.no_grad():
    out = model(x)
    print(f"inference output shape: {out.shape}, requires_grad: {out.requires_grad}")"""),
        md("## 4. Training vs eval mode — loss with and without grad"),
        code("""gen_vals = []
with torch.enable_grad():
  x = torch.randn(16, 4, requires_grad=True)
  gen_vals.append(F.mse_loss(x, torch.zeros_like(x)).item())
with torch.no_grad():
  gen_vals.append(F.mse_loss(x, torch.zeros_like(x)).item())

fig, ax = plt.subplots(figsize=(6, 4))
ax.bar(['enable_grad', 'no_grad'], gen_vals, color=['coral', 'lightgray'], edgecolor='black')
ax.set_title('Same forward op — grad context differs'); ax.set_ylabel('MSE loss')
plt.tight_layout(); plt.show()

# requires_grad flags visualization
flags = {'x': True, 'y=x²': True, 'detach(y)': False, 'no_grad block': False}
fig, ax = plt.subplots(figsize=(7, 3))
ax.barh(list(flags.keys()), [1 if v else 0 for v in flags.values()],
        color=['green' if v else 'red' for v in flags.values()])
ax.set_xlim(0, 1.2); ax.set_title('requires_grad flags along the pipeline')
plt.tight_layout(); plt.show()"""),
    ]),
    ("02_autograd", "05_gradient_accumulation.ipynb", [
        md("# Gradient Accumulation\n\nCalling `backward()` **multiple times** without `zero_grad()` **adds** gradients — useful for large effective batch sizes."),
        code(SETUP + "\n" + SIMPLE_MLP),
        md("## 1. Three backward passes without zeroing"),
        code("""model = SimpleMLP(in_dim=4, hidden=8, n_classes=2)
opt = torch.optim.SGD(model.parameters(), lr=0.1)
grad_norms_per_step = []

for step in range(3):
    x = torch.randn(4, 4)
    y = torch.randint(0, 2, (4,))
    loss = F.cross_entropy(model(x), y)
    loss.backward()
    total_norm = sum(p.grad.norm().item() for p in model.parameters() if p.grad is not None)
    grad_norms_per_step.append(total_norm)
    print(f"Step {step+1}: loss={loss.item():.3f}, total grad norm={total_norm:.3f}")"""),
        md("## 2. Bar chart — accumulated gradient norm grows"),
        code("""fig, ax = plt.subplots(figsize=(7, 4))
ax.bar(range(1, 4), grad_norms_per_step, color='steelblue', edgecolor='black')
ax.set_xlabel('backward() call #'); ax.set_ylabel('total ||grad||')
ax.set_title('Gradients accumulate across backward() calls')
ax.set_xticks([1, 2, 3])
plt.tight_layout(); plt.show()"""),
        md("## 3. Compare: with vs without `zero_grad()`"),
        code("""def run_accumulate(zero_between=False, n=3):
    m = SimpleMLP(in_dim=4, hidden=8, n_classes=2)
    norms = []
    for _ in range(n):
        if zero_between:
            m.zero_grad()
        loss = F.cross_entropy(m(torch.randn(4, 4)), torch.randint(0, 2, (4,)))
        loss.backward()
        norms.append(sum(p.grad.norm().item() for p in m.parameters() if p.grad is not None))
    return norms

acc = run_accumulate(zero_between=False)
reset = run_accumulate(zero_between=True)

fig, ax = plt.subplots(figsize=(8, 4))
x = np.arange(1, 4)
w = 0.35
ax.bar(x - w/2, acc, w, label='accumulate', color='coral')
ax.bar(x + w/2, reset, w, label='zero_grad each step', color='seagreen')
ax.set_xlabel('backward() step'); ax.set_ylabel('total ||grad||')
ax.legend(); ax.set_title('zero_grad() resets accumulation')
plt.tight_layout(); plt.show()"""),
        md("## 4. Effective batch size via accumulation\nSimulate batch_size=32 with micro-batches of 8."),
        code("""model = SimpleMLP(in_dim=4, hidden=8, n_classes=2)
opt = torch.optim.SGD(model.parameters(), lr=0.05)
micro = 8
steps = 4
model.zero_grad()
losses = []
for s in range(steps):
    loss = F.cross_entropy(model(torch.randn(micro, 4)), torch.randint(0, 2, (micro,)))
    (loss / steps).backward()
    losses.append(loss.item())
opt.step()

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].plot(range(1, steps+1), losses, 'o-', color='purple')
axes[0].set_title('Micro-batch losses'); axes[0].set_xlabel('micro-step')
axes[1].bar(['single optimizer.step'], [sum(p.grad.norm().item() for p in model.parameters() if p.grad is not None)],
            color='teal')
axes[1].set_title('One weight update after 4 accumulated grads')
plt.tight_layout(); plt.show()"""),
    ]),
]


# =============================================================================
# 03 NN MODULES (6 notebooks)
# =============================================================================

NN_MODULES_TUTORIALS = [
    ("03_nn_modules", "01_parameters_and_buffers.ipynb", [
        md("# Parameters vs Buffers\n\n**Parameters** are learned (`requires_grad=True`). **Buffers** are state (e.g. BatchNorm running stats) saved in `state_dict` but not updated by optimizers."),
        code(SETUP),
        md("## 1. Inspect `named_parameters()`"),
        code("""class DemoNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 3)
        self.register_buffer('running_mean', torch.zeros(3))

model = DemoNet()
for name, p in model.named_parameters():
    print(f"{name:20s} shape={tuple(p.shape)} requires_grad={p.requires_grad}")"""),
        md("## 2. `state_dict` keys — parameters + buffers"),
        code("""sd = model.state_dict()
print("state_dict keys:", list(sd.keys()))"""),
        md("## 3. Visualize parameter tensor shapes"),
        code("""names, sizes = [], []
for name, p in model.named_parameters():
    names.append(name.replace('.', '\\n'))
    sizes.append(p.numel())

fig, ax = plt.subplots(figsize=(7, 4))
ax.bar(names, sizes, color='steelblue', edgecolor='black')
ax.set_title('Parameter element counts'); ax.set_ylabel('# elements')
plt.tight_layout(); plt.show()"""),
        md("## 4. Heatmap of weight matrix"),
        code("""W = model.fc.weight.detach()
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].imshow(W.numpy(), cmap='RdBu_r', aspect='auto')
axes[0].set_title('fc.weight'); axes[0].set_xlabel('in'); axes[0].set_ylabel('out')
axes[1].bar(range(3), model.fc.bias.detach().numpy(), color='coral')
axes[1].set_title('fc.bias')
plt.tight_layout(); plt.show()"""),
        md("## 5. Buffers vs parameters in optimizer"),
        code("""opt = torch.optim.SGD(model.parameters(), lr=0.01)
opt_params = [n for n, _ in model.named_parameters()]
buf_names = [n for n, _ in model.named_buffers()]
fig, ax = plt.subplots(figsize=(6, 3))
ax.bar(['optimizer tracks', 'buffers only'], [len(opt_params), len(buf_names)],
       color=['seagreen', 'lightgray'], edgecolor='black')
ax.set_title('Parameters vs buffers')
plt.tight_layout(); plt.show()"""),
    ]),
    ("03_nn_modules", "02_linear_layer_math.ipynb", [
        md("# Linear Layer Math: y = Wx + b\n\nEach output neuron is a **weighted sum** of inputs plus bias."),
        code(SETUP),
        md("## 1. Manual vs `nn.Linear`"),
        code("""torch.manual_seed(0)
layer = nn.Linear(3, 2, bias=True)
x = torch.tensor([1., 2., 3.])
y = layer(x)
W, b = layer.weight, layer.bias
y_manual = W @ x + b
print(f"nn.Linear: {y}")
print(f"manual:    {y_manual}")"""),
        md("## 2. Weight heatmap"),
        code("""fig, ax = plt.subplots(figsize=(5, 4))
im = ax.imshow(W.detach().numpy(), cmap='coolwarm', aspect='auto')
ax.set_xlabel('input dim'); ax.set_ylabel('output dim')
ax.set_title('Weight matrix W (out × in)')
plt.colorbar(im); plt.tight_layout(); plt.show()"""),
        md("## 3. Batch of inputs — matrix multiply view"),
        code("""X = torch.randn(6, 3)
Y = layer(X)
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].imshow(X.detach().numpy(), aspect='auto', cmap='viridis')
axes[0].set_title('Input batch X (6×3)')
axes[1].imshow(Y.detach().numpy(), aspect='auto', cmap='plasma')
axes[1].set_title('Output Y = XWᵀ + b (6×2)')
plt.tight_layout(); plt.show()"""),
        md("## 4. Geometric view: mapping 2D → 1D"),
        code("""lin2d = nn.Linear(2, 1, bias=False)
lin2d.weight.data = torch.tensor([[1., -0.5]])
pts = torch.randn(50, 2)
out = lin2d(pts).squeeze()

fig, ax = plt.subplots(figsize=(6, 5))
sc = ax.scatter(pts[:, 0], pts[:, 1], c=out.detach(), cmap='coolwarm', s=40)
plt.colorbar(sc, label='y = w·x')
ax.set_xlabel('x1'); ax.set_ylabel('x2'); ax.set_title('Linear projection colors points')
plt.tight_layout(); plt.show()"""),
    ]),
    ("03_nn_modules", "03_activation_functions.ipynb", [
        md("# Activation Functions Compared\n\nNonlinearities let networks approximate complex functions. Compare **ReLU**, **Sigmoid**, **Tanh**, and **GELU**."),
        code(SETUP),
        md("## 1. Define activations"),
        code("""x = torch.linspace(-4, 4, 400)
acts = {
    'ReLU': F.relu(x),
    'Sigmoid': torch.sigmoid(x),
    'Tanh': torch.tanh(x),
    'GELU': F.gelu(x),
}"""),
        md("## 2. Overlay all curves"),
        code("""fig, ax = plt.subplots(figsize=(9, 5))
colors = ['#e74c3c', '#3498db', '#2ecc71', '#9b59b6']
for (name, y), c in zip(acts.items(), colors):
    ax.plot(x.numpy(), y.numpy(), label=name, lw=2, color=c)
ax.axhline(0, color='gray', lw=0.5); ax.axvline(0, color='gray', lw=0.5)
ax.legend(); ax.set_title('Activation functions'); ax.set_xlabel('x')
plt.tight_layout(); plt.show()"""),
        md("## 3. Derivatives (numerical) — slope matters for training"),
        code("""fig, axes = plt.subplots(2, 2, figsize=(10, 8))
for ax, (name, y) in zip(axes.ravel(), acts.items()):
    grad = torch.gradient(y, spacing=(x[1]-x[0]).item())[0]
    ax.plot(x.numpy(), grad.numpy(), color='navy')
    ax.set_title(f"d({name})/dx"); ax.grid(True, alpha=0.3)
plt.suptitle('Activation derivatives (numerical)'); plt.tight_layout(); plt.show()"""),
        md("## 4. Output distribution after activation on random inputs"),
        code("""z = torch.randn(5000)
fig, axes = plt.subplots(2, 2, figsize=(10, 7))
for ax, (name, fn) in zip(axes.ravel(), [
    ('ReLU', F.relu), ('Sigmoid', torch.sigmoid), ('Tanh', torch.tanh), ('GELU', F.gelu)]):
    out = fn(z)
    ax.hist(out.numpy(), bins=50, color='steelblue', edgecolor='black', alpha=0.8)
    ax.set_title(f'{name}(z), z~N(0,1)')
plt.tight_layout(); plt.show()"""),
    ]),
    ("03_nn_modules", "04_loss_functions.ipynb", [
        md("# Loss Functions: MSE, BCE, CrossEntropy\n\nEach loss measures a different prediction task — regression vs binary vs multi-class classification."),
        code(SETUP),
        md("## 1. MSE for regression"),
        code("""pred = torch.linspace(-2, 2, 100)
target = torch.zeros_like(pred)
mse = (pred - target) ** 2"""),
        md("## 2. BCE for binary probabilities"),
        code("""p = torch.linspace(0.01, 0.99, 100)
y1 = torch.ones_like(p)
bce_pos = -torch.log(p)
bce_neg = -torch.log(1 - p)"""),
        md("## 3. Plot MSE and BCE curves"),
        code("""fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].plot(pred.numpy(), mse.numpy(), 'b-', lw=2)
axes[0].set_title('MSE = (pred - 0)²'); axes[0].set_xlabel('prediction')
axes[1].plot(p.numpy(), bce_pos.numpy(), label='y=1', lw=2)
axes[1].plot(p.numpy(), bce_neg.numpy(), label='y=0', lw=2)
axes[1].legend(); axes[1].set_title('Binary Cross-Entropy'); axes[1].set_xlabel('p(pred=1)')
plt.tight_layout(); plt.show()"""),
        md("## 4. Cross-entropy over class logits"),
        code("""logits = torch.linspace(-3, 3, 100)
# CE when true class is 0 vs 1 (2-class softmax)
ce0 = -F.log_softmax(torch.stack([logits, torch.zeros_like(logits)]), dim=0)[0]
ce1 = -F.log_softmax(torch.stack([torch.zeros_like(logits), logits]), dim=0)[1]

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(logits.numpy(), ce0.numpy(), label='true class 0', lw=2)
ax.plot(logits.numpy(), ce1.numpy(), label='true class 1', lw=2)
ax.set_xlabel('logit for favored class'); ax.set_ylabel('CE loss')
ax.legend(); ax.set_title('Cross-entropy (2-class)')
plt.tight_layout(); plt.show()"""),
        md("## 5. Compare loss values on synthetic batch"),
        code("""gen = torch.randn(32, 4)
reg_t = torch.randn(32, 1)
bin_p = torch.sigmoid(torch.randn(32))
bin_y = torch.randint(0, 2, (32,)).float()
cls_logits = torch.randn(32, 3)
cls_y = torch.randint(0, 3, (32,))

losses = {
    'MSE': F.mse_loss(gen[:, :1], reg_t).item(),
    'BCE': F.binary_cross_entropy(bin_p, bin_y).item(),
    'CE': F.cross_entropy(cls_logits, cls_y).item(),
}
fig, ax = plt.subplots(figsize=(6, 4))
ax.bar(losses.keys(), losses.values(), color=['steelblue', 'coral', 'seagreen'], edgecolor='black')
ax.set_title('Sample batch loss values'); ax.set_ylabel('loss')
plt.tight_layout(); plt.show()"""),
    ]),
    ("03_nn_modules", "05_optimizers_compared.ipynb", [
        md("# SGD vs Adam on a 2D Loss Landscape\n\nDifferent optimizers take different paths toward a minimum."),
        code(SETUP),
        md("## 1. Loss surface f(w1, w2) = w1² + 10·w2²"),
        code("""def loss_fn(w):
    return w[0]**2 + 10 * w[1]**2

def grad_fn(w):
    return torch.tensor([2*w[0], 20*w[1]], dtype=torch.float32)

w1, w2 = np.meshgrid(np.linspace(-2, 2, 80), np.linspace(-2, 2, 80))
Z = w1**2 + 10 * w2**2"""),
        md("## 2. Contour plot of the bowl"),
        code("""fig, ax = plt.subplots(figsize=(7, 6))
ax.contour(w1, w2, Z, levels=20, cmap='viridis')
ax.set_xlabel('w1'); ax.set_ylabel('w2'); ax.set_title('Loss landscape (elongated bowl)')
ax.set_aspect('equal')
plt.tight_layout(); plt.show()"""),
        md("## 3. Run SGD and Adam from same start"),
        code("""def optimize(optimizer_cls, steps=40, lr=0.1, **kwargs):
    w = torch.tensor([1.8, 1.5], requires_grad=True)
    opt = optimizer_cls([w], lr=lr, **kwargs)
    path = [w.detach().clone().numpy()]
    for _ in range(steps):
        l = loss_fn(w)
        opt.zero_grad()
        l.backward()
        opt.step()
        path.append(w.detach().clone().numpy())
    return np.array(path)

path_sgd = optimize(torch.optim.SGD, lr=0.08)
path_adam = optimize(torch.optim.Adam, lr=0.3)"""),
        md("## 4. Overlay optimizer paths"),
        code("""fig, ax = plt.subplots(figsize=(8, 6))
ax.contour(w1, w2, Z, levels=20, cmap='viridis', alpha=0.7)
ax.plot(path_sgd[:, 0], path_sgd[:, 1], 'r.-', label='SGD', lw=2)
ax.plot(path_adam[:, 0], path_adam[:, 1], 'b.-', label='Adam', lw=2)
ax.scatter([0], [0], c='gold', s=100, zorder=5, edgecolors='k', label='min')
ax.legend(); ax.set_aspect('equal')
ax.set_title('Optimizer trajectories on f(w)=w1²+10w2²')
plt.tight_layout(); plt.show()"""),
        md("## 5. Loss vs step"),
        code("""def loss_path(path):
    return [loss_fn(torch.tensor(p)).item() for p in path]

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(loss_path(path_sgd), 'r-', label='SGD', lw=2)
ax.plot(loss_path(path_adam), 'b-', label='Adam', lw=2)
ax.set_xlabel('step'); ax.set_ylabel('loss'); ax.legend()
ax.set_title('Loss convergence comparison')
plt.tight_layout(); plt.show()"""),
    ]),
    ("03_nn_modules", "06_building_blocks_sequential.ipynb", [
        md("# Building Blocks with `nn.Sequential`\n\nStack layers and track **output shapes** through the pipeline."),
        code(SETUP),
        md("## 1. Define a Sequential model"),
        code("""class ShapeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Linear(8, 16), nn.ReLU(),
            nn.Linear(16, 32), nn.ReLU(),
            nn.Linear(32, 10),
        )
    def forward(self, x):
        return self.features(x)

model = ShapeNet()
x = torch.randn(4, 8)
print(model)"""),
        md("## 2. Hook intermediate shapes"),
        code("""shapes = [('input', tuple(x.shape))]
h = x
for i, layer in enumerate(model.features):
    h = layer(h)
    shapes.append((f'{i}: {layer.__class__.__name__}', tuple(h.shape)))
for name, sh in shapes:
    print(f"{name:25s} → {sh}")"""),
        md("## 3. Diagram: layer blocks with shapes"),
        code("""fig, ax = plt.subplots(figsize=(12, 3))
labels = [s[0] for s in shapes]
ypos = np.arange(len(labels))
ax.barh(ypos, [int(np.prod(s[1])) if len(s[1])>1 else s[1][0] for s in shapes],
        color=plt.cm.viridis(np.linspace(0.2, 0.9, len(labels))))
ax.set_yticks(ypos); ax.set_yticklabels(labels)
ax.set_xlabel('tensor elements (batch flattened)')
ax.set_title('nn.Sequential — output size per stage')
plt.tight_layout(); plt.show()"""),
        md("## 4. Activation map after first hidden layer"),
        code("""with torch.no_grad():
    h1 = model.features[1](model.features[0](x))
fig, ax = plt.subplots(figsize=(8, 4))
ax.imshow(h1.numpy(), aspect='auto', cmap='magma')
ax.set_title('Hidden activations after Linear+ReLU'); ax.set_xlabel('neuron')
plt.tight_layout(); plt.show()"""),
        md("## 5. Parameter count per block"),
        code("""blocks = ['Linear(8→16)', 'ReLU', 'Linear(16→32)', 'ReLU', 'Linear(32→10)']
params = [8*16+16, 0, 16*32+32, 0, 32*10+10]
fig, ax = plt.subplots(figsize=(9, 4))
ax.bar(blocks, params, color='teal', edgecolor='black')
ax.set_title('Learnable parameters per layer'); ax.set_ylabel('# params')
plt.xticks(rotation=15); plt.tight_layout(); plt.show()"""),
    ]),
]


# =============================================================================
# 04 TRAINING (5 notebooks)
# =============================================================================

TRAINING_TUTORIALS = [
    ("04_training", "01_dataset_and_dataloader.ipynb", [
        md("# Dataset & DataLoader\n\nWrap synthetic data in a **`Dataset`** and batch it with **`DataLoader`** for efficient training loops."),
        code(SETUP + "\n" + DUMMY_GEN + "\n" + DATASET_CLASS),
        md("## 1. Generate synthetic data"),
        code("""gen = DummyDataGenerator(n_samples=128, n_features=8, n_classes=3)
X, y = gen.tensors()
ds = TabularDataset(X, y)
print(f"Dataset length: {len(ds)}, sample X shape: {ds[0][0].shape}")"""),
        md("## 2. DataLoader — batching & shuffling"),
        code("""loader = DataLoader(ds, batch_size=16, shuffle=True)
batch_x, batch_y = next(iter(loader))
print(f"Batch X: {batch_x.shape}, Batch y: {batch_y.shape}")"""),
        md("## 3. Visualize one batch — feature heatmap"),
        code("""fig, ax = plt.subplots(figsize=(9, 4))
ax.imshow(batch_x.numpy(), aspect='auto', cmap='viridis')
ax.set_title('One batch of features (16 × 8)'); ax.set_xlabel('feature')
plt.tight_layout(); plt.show()"""),
        md("## 4. Class distribution in batch vs full dataset"),
        code("""full_counts = torch.bincount(y, minlength=3).numpy()
batch_counts = torch.bincount(batch_y, minlength=3).numpy()
fig, ax = plt.subplots(figsize=(7, 4))
x = np.arange(3); w = 0.35
ax.bar(x - w/2, full_counts, w, label='full dataset', color='steelblue')
ax.bar(x + w/2, batch_counts, w, label='one batch', color='coral')
ax.set_xticks(x); ax.set_xlabel('class'); ax.legend()
ax.set_title('Label distribution')
plt.tight_layout(); plt.show()"""),
        md("## 5. Iterate multiple batches"),
        code("""batch_sizes = [len(by) for bx, by in loader]
fig, ax = plt.subplots(figsize=(8, 3))
ax.bar(range(len(batch_sizes)), batch_sizes, color='seagreen', edgecolor='black')
ax.set_xlabel('batch index'); ax.set_ylabel('batch size')
ax.set_title(f'DataLoader yields {len(batch_sizes)} batches (batch_size=16)')
plt.tight_layout(); plt.show()"""),
    ]),
    ("04_training", "02_train_val_test_split.ipynb", [
        md("# Train / Validation / Test Split\n\nHold out data for **unbiased evaluation** — typically 70/15/15 or 80/10/10."),
        code(SETUP + "\n" + DUMMY_GEN + "\n" + DATASET_CLASS),
        md("## 1. Split indices"),
        code("""gen = DummyDataGenerator(n_samples=500, n_features=8, n_classes=3)
X, y = gen.tensors()
n = len(y)
n_train = int(0.7 * n)
n_val = int(0.15 * n)
perm = torch.randperm(n)
train_idx = perm[:n_train]
val_idx = perm[n_train:n_train + n_val]
test_idx = perm[n_train + n_val:]
print(f"train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")"""),
        md("## 2. Pie chart of splits"),
        code("""sizes = [len(train_idx), len(val_idx), len(test_idx)]
labels = ['Train 70%', 'Val 15%', 'Test 15%']
colors = ['#3498db', '#f39c12', '#e74c3c']
fig, ax = plt.subplots(figsize=(7, 6))
ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
ax.set_title('Dataset split proportions')
plt.tight_layout(); plt.show()"""),
        md("## 3. Class balance per split"),
        code("""def class_hist(idx, title):
    counts = torch.bincount(y[idx], minlength=3).numpy()
    return counts

fig, axes = plt.subplots(1, 3, figsize=(12, 4))
for ax, idx, name in zip(axes, [train_idx, val_idx, test_idx], ['Train', 'Val', 'Test']):
    c = class_hist(idx, name)
    ax.bar(range(3), c, color='teal', edgecolor='black')
    ax.set_title(f'{name} class counts'); ax.set_xlabel('class')
plt.tight_layout(); plt.show()"""),
        md("## 4. Create DataLoaders per split"),
        code("""train_ds = TabularDataset(X[train_idx], y[train_idx])
val_ds = TabularDataset(X[val_idx], y[val_idx])
test_ds = TabularDataset(X[test_idx], y[test_idx])
loaders = {
    'train': DataLoader(train_ds, batch_size=32, shuffle=True),
    'val': DataLoader(val_ds, batch_size=32),
    'test': DataLoader(test_ds, batch_size=32),
}
for k, ld in loaders.items():
    print(f"{k}: {len(ld)} batches")"""),
        md("## 5. Sample counts bar chart"),
        code("""fig, ax = plt.subplots(figsize=(7, 4))
ax.bar(['Train', 'Val', 'Test'], sizes, color=colors, edgecolor='black')
ax.set_ylabel('# samples'); ax.set_title('Absolute split sizes')
plt.tight_layout(); plt.show()"""),
    ]),
    ("04_training", "03_training_loop_anatomy.ipynb", [
        md("# Training Loop Anatomy\n\nThe core cycle: **forward → loss → backward → optimizer.step()**."),
        code(SETUP + "\n" + DUMMY_GEN + "\n" + DATASET_CLASS + "\n" + SIMPLE_MLP),
        md("## 1. Setup model, loss, optimizer, data"),
        code("""gen = DummyDataGenerator(n_samples=200, n_features=8, n_classes=3)
ds = TabularDataset(*gen.tensors())
loader = DataLoader(ds, batch_size=32, shuffle=True)
model = SimpleMLP(in_dim=8, hidden=16, n_classes=3)
opt = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()
loss_history = []"""),
        md("## 2. One epoch — step by step"),
        code("""for epoch in range(1):
    for batch_x, batch_y in loader:
        # 1. Forward
        logits = model(batch_x)
        # 2. Loss
        loss = criterion(logits, batch_y)
        # 3. Backward
        opt.zero_grad()
        loss.backward()
        # 4. Update
        opt.step()
        loss_history.append(loss.item())
print(f"Steps: {len(loss_history)}, final loss: {loss_history[-1]:.4f}")"""),
        md("## 3. Loss per training step"),
        code("""fig, ax = plt.subplots(figsize=(9, 4))
ax.plot(loss_history, color='steelblue', lw=1.5)
ax.set_xlabel('step'); ax.set_ylabel('loss')
ax.set_title('Loss during one epoch')
plt.tight_layout(); plt.show()"""),
        md("## 4. Annotated pipeline diagram (conceptual)"),
        code("""steps = ['forward', 'loss', 'backward', 'step']
colors = ['#3498db', '#e74c3c', '#9b59b6', '#2ecc71']
fig, ax = plt.subplots(figsize=(10, 2.5))
for i, (s, c) in enumerate(zip(steps, colors)):
    ax.add_patch(plt.Rectangle((i*2.2, 0.2), 1.8, 0.6, fc=c, ec='black', alpha=0.8))
    ax.text(i*2.2 + 0.9, 0.5, s, ha='center', va='center', color='white', fontsize=12, fontweight='bold')
    if i < 3:
        ax.annotate('', xy=((i+1)*2.2, 0.5), xytext=(i*2.2+1.8, 0.5),
                    arrowprops=dict(arrowstyle='->', lw=2))
ax.set_xlim(-0.2, 9); ax.set_ylim(0, 1); ax.axis('off')
ax.set_title('Training step pipeline')
plt.tight_layout(); plt.show()"""),
        md("## 5. Multi-epoch training curve"),
        code("""epoch_losses = []
for epoch in range(5):
    ep_loss = []
    for bx, by in loader:
        opt.zero_grad()
        loss = criterion(model(bx), by)
        loss.backward(); opt.step()
        ep_loss.append(loss.item())
    epoch_losses.append(np.mean(ep_loss))

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(range(1, 6), epoch_losses, 'o-', color='coral', lw=2)
ax.set_xlabel('epoch'); ax.set_ylabel('mean loss')
ax.set_title('Loss decreases over epochs')
plt.tight_layout(); plt.show()"""),
    ]),
    ("04_training", "04_learning_curves.ipynb", [
        md("# Learning Curves — Spotting Overfitting\n\n**Train loss ↓** but **val loss ↑** → model memorizes training data."),
        code(SETUP + "\n" + DUMMY_GEN + "\n" + DATASET_CLASS + "\n" + SIMPLE_MLP),
        md("## 1. Simulate train vs val metrics"),
        code("""epochs = np.arange(1, 21)
# Synthetic curves mimicking overfitting after epoch 12
train_loss = 1.0 * np.exp(-epochs / 8) + 0.05 + np.random.randn(20) * 0.02
val_loss = 1.0 * np.exp(-epochs / 10) + 0.1 + np.maximum(0, (epochs - 12) * 0.03) + np.random.randn(20) * 0.02
train_acc = 1 - train_loss / 1.2
val_acc = 1 - val_loss / 1.3"""),
        md("## 2. Train vs val loss"),
        code("""fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(epochs, train_loss, 'b-o', label='train loss', lw=2)
ax.plot(epochs, val_loss, 'r-o', label='val loss', lw=2)
ax.axvline(12, color='gray', ls='--', label='overfitting starts')
ax.set_xlabel('epoch'); ax.set_ylabel('loss'); ax.legend()
ax.set_title('Train vs validation loss')
plt.tight_layout(); plt.show()"""),
        md("## 3. Train vs val accuracy"),
        code("""fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(epochs, train_acc, 'b-o', label='train acc', lw=2)
ax.plot(epochs, val_acc, 'r-o', label='val acc', lw=2)
ax.axvline(12, color='gray', ls='--')
ax.set_xlabel('epoch'); ax.set_ylabel('accuracy'); ax.legend()
ax.set_title('Accuracy curves — gap widens when overfitting')
plt.tight_layout(); plt.show()"""),
        md("## 4. Real mini-training on synthetic data"),
        code("""gen = DummyDataGenerator(n_samples=300, n_features=8, n_classes=3)
X, y = gen.tensors()
n = len(y); tr, va = X[:200], X[200:]; ytr, yva = y[:200], y[200:]
model = SimpleMLP()
opt = torch.optim.Adam(model.parameters(), lr=0.02)
tl, vl = [], []
for ep in range(30):
    model.train()
    opt.zero_grad()
    loss = F.cross_entropy(model(tr), ytr)
    loss.backward(); opt.step()
    tl.append(loss.item())
    model.eval()
    with torch.no_grad():
        vl.append(F.cross_entropy(model(va), yva).item())

fig, ax = plt.subplots(figsize=(9, 4))
ax.plot(tl, label='train'); ax.plot(vl, label='val')
ax.legend(); ax.set_title('Live training on DummyDataGenerator')
plt.tight_layout(); plt.show()"""),
        md("## 5. Generalization gap"),
        code("""gap = np.array(vl) - np.array(tl)
fig, ax = plt.subplots(figsize=(8, 4))
ax.fill_between(range(len(gap)), 0, gap, alpha=0.4, color='orange')
ax.plot(gap, color='darkorange', lw=2)
ax.set_xlabel('epoch'); ax.set_ylabel('val_loss - train_loss')
ax.set_title('Generalization gap (wider = more overfitting)')
plt.tight_layout(); plt.show()"""),
    ]),
    ("04_training", "05_batch_size_effect.ipynb", [
        md("# Batch Size Effects\n\nLarger batches: **faster per epoch** but sometimes **less stable** loss. Smaller batches: noisier gradients."),
        code(SETUP + "\n" + DUMMY_GEN + "\n" + DATASET_CLASS + "\n" + SIMPLE_MLP),
        md("## 1. Measure steps per epoch vs batch size"),
        code("""gen = DummyDataGenerator(n_samples=512, n_features=8, n_classes=3)
ds = TabularDataset(*gen.tensors())
batch_sizes = [8, 16, 32, 64, 128]
steps_per_epoch = [len(DataLoader(ds, batch_size=bs)) for bs in batch_sizes]
print(dict(zip(batch_sizes, steps_per_epoch)))"""),
        md("## 2. Steps per epoch vs batch size"),
        code("""fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(batch_sizes, steps_per_epoch, 'o-', color='steelblue', lw=2, markersize=8)
ax.set_xlabel('batch size'); ax.set_ylabel('steps per epoch')
ax.set_title('Fewer steps with larger batches (fixed dataset)')
plt.tight_layout(); plt.show()"""),
        md("## 3. Loss stability across batch sizes"),
        code("""def train_trace(bs, steps=50):
    m = SimpleMLP()
    opt = torch.optim.SGD(m.parameters(), lr=0.05)
    ld = DataLoader(ds, batch_size=bs, shuffle=True)
    losses = []
    it = iter(ld)
    for _ in range(steps):
        try:
            bx, by = next(it)
        except StopIteration:
            it = iter(ld); bx, by = next(it)
        opt.zero_grad()
        l = F.cross_entropy(m(bx), by)
        l.backward(); opt.step()
        losses.append(l.item())
    return losses

traces = {bs: train_trace(bs) for bs in [8, 64]}"""),
        md("## 4. Compare loss traces"),
        code("""fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(traces[8], alpha=0.8, label='batch=8 (noisy)', lw=1.5)
ax.plot(traces[64], alpha=0.8, label='batch=64 (smooth)', lw=1.5)
ax.set_xlabel('step'); ax.set_ylabel('loss'); ax.legend()
ax.set_title('Loss stability: small vs large batch')
plt.tight_layout(); plt.show()"""),
        md("## 5. Simulated time per epoch vs batch size"),
        code("""# time ≈ steps * (base + batch_factor * bs) — illustrative
time_per_epoch = [s * (0.01 + 0.0001 * bs) for s, bs in zip(steps_per_epoch, batch_sizes)]
fig, ax = plt.subplots(figsize=(8, 4))
ax.bar([str(b) for b in batch_sizes], time_per_epoch, color='coral', edgecolor='black')
ax.set_xlabel('batch size'); ax.set_ylabel('relative time / epoch')
ax.set_title('Wall-clock tradeoff (synthetic timing model)')
plt.tight_layout(); plt.show()"""),
    ]),
]


# =============================================================================
# 06 CNN BASICS (5 notebooks)
# =============================================================================

CNN_TUTORIALS = [
    ("06_cnn_basics", "01_what_is_convolution.ipynb", [
        md("# What is Convolution? (1D)\n\nA **kernel** slides along a signal, computing dot products at each position."),
        code(SETUP),
        md("## 1. 1D signal and kernel"),
        code("""signal = torch.tensor([0., 1., 2., 3., 2., 1., 0., -1., -2., -1., 0.])
kernel = torch.tensor([1., 0., -1.])  # simple edge-like
print(f"signal length: {len(signal)}, kernel length: {len(kernel)}")"""),
        md("## 2. Manual convolution steps"),
        code("""def conv1d_manual(x, k):
    out_len = len(x) - len(k) + 1
    out = []
    positions = []
    for i in range(out_len):
        patch = x[i:i+len(k)]
        out.append(float((patch * k).sum()))
        positions.append(i)
    return torch.tensor(out), positions

out, positions = conv1d_manual(signal, kernel)
print(f"output: {out}")"""),
        md("## 3. Plot signal and kernel"),
        code("""fig, axes = plt.subplots(2, 1, figsize=(10, 5))
axes[0].stem(range(len(signal)), signal.numpy(), linefmt='C0-', markerfmt='C0o', basefmt=' ')
axes[0].set_title('Input 1D signal'); axes[0].set_ylabel('value')
axes[1].stem(range(len(kernel)), kernel.numpy(), linefmt='C1-', markerfmt='C1o', basefmt=' ')
axes[1].set_title('Kernel [1, 0, -1]'); axes[1].set_xlabel('index')
plt.tight_layout(); plt.show()"""),
        md("## 4. Kernel sliding — multi-panel animation-style"),
        code("""fig, axes = plt.subplots(2, 4, figsize=(14, 5))
for i, pos in enumerate([0, 2, 4, 6]):
    patch = signal[pos:pos+3]
    axes[0, i].bar(range(3), patch.numpy(), color='steelblue')
    axes[0, i].set_title(f'patch @ pos {pos}'); axes[0, i].set_ylim(-3, 4)
    axes[1, i].bar([0], [out[pos].item()], color='coral', width=0.5)
    axes[1, i].set_title(f'output[{pos}]={out[pos]:.1f}'); axes[1, i].set_ylim(-3, 4)
plt.suptitle('Kernel sliding left → right'); plt.tight_layout(); plt.show()"""),
        md("## 5. Full output vs PyTorch conv1d"),
        code("""pt_out = F.conv1d(signal.view(1, 1, -1), kernel.view(1, 1, -1)).squeeze()
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(out.numpy(), 'o-', label='manual', lw=2)
ax.plot(pt_out.numpy(), 'x--', label='F.conv1d', lw=2)
ax.set_xlabel('output index'); ax.legend()
ax.set_title('Convolution output')
plt.tight_layout(); plt.show()"""),
    ]),
    ("06_cnn_basics", "02_filters_edge_detection.ipynb", [
        md("# Filters & Edge Detection\n\n**Sobel** kernels highlight horizontal and vertical intensity changes."),
        code(SETUP),
        md("## 1. Synthetic grayscale image"),
        code("""img = torch.zeros(1, 1, 32, 32)
img[:, :, 8:24, 8:24] = 1.0  # bright square on dark background
img[:, :, :, 16:] += 0.3 * torch.randn(1, 1, 32, 16)
print(f"image shape: {img.shape}")"""),
        md("## 2. Sobel kernels"),
        code("""sobel_x = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]).view(1, 1, 3, 3)
sobel_y = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]).view(1, 1, 3, 3)
gx = F.conv2d(img, sobel_x, padding=1)
gy = F.conv2d(img, sobel_y, padding=1)
mag = torch.sqrt(gx**2 + gy**2)"""),
        md("## 3. Original image"),
        code("""fig, ax = plt.subplots(figsize=(5, 5))
ax.imshow(img.squeeze().numpy(), cmap='gray')
ax.set_title('Synthetic image'); ax.axis('off')
plt.tight_layout(); plt.show()"""),
        md("## 4. Feature maps: Sobel X, Y, magnitude"),
        code("""fig, axes = plt.subplots(1, 3, figsize=(12, 4))
for ax, m, t in zip(axes, [gx, gy, mag], ['Sobel X', 'Sobel Y', 'Magnitude']):
    ax.imshow(m.squeeze().numpy(), cmap='RdBu_r')
    ax.set_title(t); ax.axis('off')
plt.tight_layout(); plt.show()"""),
        md("## 5. Kernel heatmaps"),
        code("""fig, axes = plt.subplots(1, 2, figsize=(8, 3))
axes[0].imshow(sobel_x.squeeze().numpy(), cmap='coolwarm')
axes[0].set_title('sobel_x kernel')
axes[1].imshow(sobel_y.squeeze().numpy(), cmap='coolwarm')
axes[1].set_title('sobel_y kernel')
plt.tight_layout(); plt.show()"""),
    ]),
    ("06_cnn_basics", "03_pooling_max_vs_avg.ipynb", [
        md("# Max Pooling vs Average Pooling\n\nPooling **downsamples** feature maps — max takes the strongest activation; avg smooths."),
        code(SETUP),
        md("## 1. Feature map with localized peaks"),
        code("""feat = torch.zeros(1, 1, 8, 8)
feat[0, 0, 2, 3] = 9; feat[0, 0, 5, 6] = 7; feat[0, 0, 1, 1] = 4
feat += torch.rand_like(feat) * 0.5
print(f"feature map shape: {feat.shape}")"""),
        md("## 2. Apply 2×2 pooling stride 2"),
        code("""max_pool = F.max_pool2d(feat, kernel_size=2, stride=2)
avg_pool = F.avg_pool2d(feat, kernel_size=2, stride=2)
print(f"after pool: {max_pool.shape}")"""),
        md("## 3. Input feature map"),
        code("""fig, ax = plt.subplots(figsize=(5, 5))
ax.imshow(feat.squeeze().numpy(), cmap='viridis')
ax.set_title('8×8 feature map (with peaks)'); ax.grid(True, color='white', alpha=0.3)
plt.tight_layout(); plt.show()"""),
        md("## 4. Max vs avg pooled output"),
        code("""fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].imshow(max_pool.squeeze().numpy(), cmap='viridis')
axes[0].set_title('MaxPool 2×2'); axes[0].grid(True, color='white', alpha=0.3)
axes[1].imshow(avg_pool.squeeze().numpy(), cmap='viridis')
axes[1].set_title('AvgPool 2×2'); axes[1].grid(True, color='white', alpha=0.3)
plt.tight_layout(); plt.show()"""),
        md("## 5. On synthetic 'image' — compare effects"),
        code("""img = torch.randn(1, 1, 16, 16)
mp = F.max_pool2d(img, 2, 2)
ap = F.avg_pool2d(img, 2, 2)
fig, axes = plt.subplots(1, 3, figsize=(12, 4))
axes[0].imshow(img.squeeze().numpy(), cmap='coolwarm')
axes[0].set_title('Input 16×16')
axes[1].imshow(mp.squeeze().numpy(), cmap='coolwarm'); axes[1].set_title('MaxPool → 8×8')
axes[2].imshow(ap.squeeze().numpy(), cmap='coolwarm'); axes[2].set_title('AvgPool → 8×8')
for ax in axes: ax.axis('off')
plt.tight_layout(); plt.show()"""),
    ]),
    ("06_cnn_basics", "04_padding_and_stride.ipynb", [
        md("# Padding & Stride — Output Size Formula\n\n`out = floor((in + 2*pad - kernel) / stride) + 1`"),
        code(SETUP),
        md("## 1. Output size helper"),
        code("""def conv_out_size(in_size, kernel, stride=1, padding=0):
    return (in_size + 2 * padding - kernel) // stride + 1

for k, s, p in [(3, 1, 0), (3, 1, 1), (3, 2, 1)]:
    print(f"in=7, k={k}, s={s}, p={p} → out={conv_out_size(7, k, s, p)}")"""),
        md("## 2. Visualize padding expanding input"),
        code("""in_sz, k, pad = 5, 3, 1
padded = in_sz + 2 * pad
fig, ax = plt.subplots(figsize=(8, 2))
ax.add_patch(plt.Rectangle((0, 0.3), in_sz, 0.4, fc='steelblue', ec='black'))
ax.add_patch(plt.Rectangle((-pad, 0.2), pad, 0.6, fc='lightgray', ec='black'))
ax.add_patch(plt.Rectangle((in_sz, 0.2), pad, 0.6, fc='lightgray', ec='black'))
ax.text(in_sz/2, 0.5, f'input {in_sz}', ha='center', va='center', color='white')
ax.text(-pad/2, 0.5, 'pad', ha='center', fontsize=9)
ax.set_xlim(-1.5, in_sz+1.5); ax.set_ylim(0, 1); ax.axis('off')
ax.set_title('Padding=1 adds border pixels')
plt.tight_layout(); plt.show()"""),
        md("## 3. Stride effect on output length"),
        code("""x = torch.arange(16).float().view(1, 1, 1, 16)
kernel = torch.ones(1, 1, 1, 3) / 3
outs = {}
for stride in [1, 2, 3]:
    outs[stride] = F.conv2d(x, kernel, stride=stride)

fig, axes = plt.subplots(1, 3, figsize=(12, 3))
for ax, (s, o) in zip(axes, outs.items()):
    ax.plot(o.squeeze().numpy(), 'o-')
    ax.set_title(f'stride={s}, out_len={o.shape[-1]}')
plt.suptitle('Larger stride → smaller output'); plt.tight_layout(); plt.show()"""),
        md("## 4. Output size vs stride (chart)"),
        code("""strides = [1, 2, 3, 4]
out_sizes = [conv_out_size(32, 3, s, 1) for s in strides]
fig, ax = plt.subplots(figsize=(8, 4))
ax.bar([str(s) for s in strides], out_sizes, color='teal', edgecolor='black')
ax.set_xlabel('stride'); ax.set_ylabel('output spatial size')
ax.set_title('Output size for 32×32 input, k=3, pad=1')
plt.tight_layout(); plt.show()"""),
        md("## 5. 2D conv shape table"),
        code("""configs = [(32, 3, 1, 1), (32, 5, 2, 2), (16, 3, 1, 0)]
labels = [f'k={k},s={s},p={p}' for _, k, s, p in configs]
sizes = [conv_out_size(inp, k, s, p) for inp, k, s, p in configs]
fig, ax = plt.subplots(figsize=(8, 4))
ax.barh(labels, sizes, color='coral', edgecolor='black')
ax.set_xlabel('output H=W'); ax.set_title('Output size for different conv configs')
plt.tight_layout(); plt.show()"""),
    ]),
    ("06_cnn_basics", "05_mini_cnn_mnist.ipynb", [
        md("# Mini CNN on Synthetic Digit-like Data\n\nA small CNN on **`DummyDataGenerator`** images — no downloads required."),
        code(SETUP + """
class DummyDataGenerator:
    \"\"\"Synthetic 28×28 grayscale images and 10 class labels.\"\"\"
    def __init__(self, n_samples=256, n_classes=10, seed=42):
        g = torch.Generator().manual_seed(seed)
        self.X = torch.randn(n_samples, 1, 28, 28, generator=g)
        # add blob-like structure
        for i in range(n_samples):
            cx, cy = torch.randint(6, 22, (2,), generator=g)
            self.X[i, 0, cx-3:cx+3, cy-3:cy+3] += 2.0
        self.y = torch.randint(0, n_classes, (n_samples,), generator=g)

class ImageDataset(Dataset):
    def __init__(self, X, y):
        self.X, self.y = X, y
    def __len__(self):
        return len(self.y)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class MiniCNN(nn.Module):
    def __init__(self, n_classes=10):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.fc = nn.Linear(16 * 7 * 7, n_classes)
    def forward(self, x):
        x = self.conv(x)
        return self.fc(x.flatten(1))
"""),
        md("## 1. Load synthetic data"),
        code("""gen = DummyDataGenerator(n_samples=128)
ds = ImageDataset(gen.X, gen.y)
loader = DataLoader(ds, batch_size=16, shuffle=True)
bx, by = next(iter(loader))
print(f"batch: {bx.shape}, labels: {by.shape}")"""),
        md("## 2. Sample images grid"),
        code("""fig, axes = plt.subplots(2, 4, figsize=(10, 5))
for ax, i in zip(axes.ravel(), range(8)):
    ax.imshow(gen.X[i, 0].numpy(), cmap='gray')
    ax.set_title(f'label {gen.y[i].item()}'); ax.axis('off')
plt.suptitle('Synthetic 28×28 images'); plt.tight_layout(); plt.show()"""),
        md("## 3. Train mini CNN"),
        code("""model = MiniCNN()
opt = torch.optim.Adam(model.parameters(), lr=0.001)
losses = []
for epoch in range(5):
    for bx, by in loader:
        opt.zero_grad()
        loss = F.cross_entropy(model(bx), by)
        loss.backward(); opt.step()
        losses.append(loss.item())
print(f"final loss: {losses[-1]:.3f}")"""),
        md("## 4. Training loss curve"),
        code("""fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(losses, color='steelblue')
ax.set_xlabel('step'); ax.set_ylabel('loss')
ax.set_title('MiniCNN training loss')
plt.tight_layout(); plt.show()"""),
        md("## 5. Visualize first-layer filters"),
        code("""filters = model.conv[0].weight.detach()  # 8 filters
fig, axes = plt.subplots(2, 4, figsize=(10, 5))
for ax, i in zip(axes.ravel(), range(8)):
    ax.imshow(filters[i, 0].numpy(), cmap='RdBu_r')
    ax.set_title(f'filter {i}'); ax.axis('off')
plt.suptitle('Learned conv1 filters'); plt.tight_layout(); plt.show()"""),
    ]),
]


# =============================================================================
# 07 VISUALIZATION (5 notebooks)
# =============================================================================

VISUALIZATION_TUTORIALS = [
    ("07_visualization", "01_plotting_training_metrics.ipynb", [
        md("# Plotting Training Metrics\n\nTrack **loss** and **accuracy** over epochs with matplotlib."),
        code(SETUP + "\n" + DUMMY_GEN + "\n" + SIMPLE_MLP),
        md("## 1. Simulate training history"),
        code("""epochs = list(range(1, 16))
train_loss = [1.2 * np.exp(-0.25 * e) + 0.05 + np.random.rand() * 0.03 for e in epochs]
val_loss = [1.1 * np.exp(-0.2 * e) + 0.12 + np.random.rand() * 0.05 for e in epochs]
train_acc = [1 - l / 1.3 for l in train_loss]
val_acc = [1 - l / 1.4 for l in val_loss]"""),
        md("## 2. Loss curves"),
        code("""fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(epochs, train_loss, 'b-o', label='train loss', lw=2)
ax.plot(epochs, val_loss, 'r-o', label='val loss', lw=2)
ax.set_xlabel('epoch'); ax.set_ylabel('loss'); ax.legend()
ax.set_title('Training & validation loss')
ax.grid(True, alpha=0.3)
plt.tight_layout(); plt.show()"""),
        md("## 3. Accuracy curves"),
        code("""fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(epochs, train_acc, 'b-o', label='train acc', lw=2)
ax.plot(epochs, val_acc, 'r-o', label='val acc', lw=2)
ax.set_xlabel('epoch'); ax.set_ylabel('accuracy'); ax.legend()
ax.set_title('Training & validation accuracy')
plt.tight_layout(); plt.show()"""),
        md("## 4. Dual-axis plot"),
        code("""fig, ax1 = plt.subplots(figsize=(9, 5))
ax2 = ax1.twinx()
ax1.plot(epochs, train_loss, 'b-', label='loss')
ax2.plot(epochs, train_acc, 'g--', label='acc')
ax1.set_xlabel('epoch'); ax1.set_ylabel('loss', color='b')
ax2.set_ylabel('accuracy', color='g')
ax1.set_title('Loss and accuracy on shared epoch axis')
plt.tight_layout(); plt.show()"""),
        md("## 5. Live metrics from short training run"),
        code("""gen = DummyDataGenerator(n_samples=200, n_features=8, n_classes=3)
X, y = gen.tensors()
model = SimpleMLP()
opt = torch.optim.Adam(model.parameters(), lr=0.02)
live_loss, live_acc = [], []
for ep in range(20):
    opt.zero_grad()
    logits = model(X)
    loss = F.cross_entropy(logits, y)
    loss.backward(); opt.step()
    live_loss.append(loss.item())
    live_acc.append((logits.argmax(1) == y).float().mean().item())

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].plot(live_loss, color='steelblue'); axes[0].set_title('Live loss')
axes[1].plot(live_acc, color='seagreen'); axes[1].set_title('Live accuracy')
plt.tight_layout(); plt.show()"""),
    ]),
    ("07_visualization", "02_weight_histograms.ipynb", [
        md("# Weight Histograms — Before & After Training\n\nWeight distributions shift as the network learns."),
        code(SETUP + "\n" + SIMPLE_MLP),
        md("## 1. Capture weights before training"),
        code("""torch.manual_seed(42)
model = SimpleMLP(in_dim=8, hidden=32, n_classes=3)
before = {n: p.detach().clone() for n, p in model.named_parameters()}"""),
        md("## 2. Train briefly on synthetic data"),
        code("""X = torch.randn(256, 8)
y = torch.randint(0, 3, (256,))
opt = torch.optim.Adam(model.parameters(), lr=0.01)
for _ in range(100):
    opt.zero_grad()
    F.cross_entropy(model(X), y).backward()
    opt.step()
after = {n: p.detach().clone() for n, p in model.named_parameters()}"""),
        md("## 3. Histogram: first layer weights before/after"),
        code("""fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].hist(before['net.0.weight'].flatten().numpy(), bins=40, color='steelblue', edgecolor='black')
axes[0].set_title('W1 before training')
axes[1].hist(after['net.0.weight'].flatten().numpy(), bins=40, color='coral', edgecolor='black')
axes[1].set_title('W1 after training')
plt.tight_layout(); plt.show()"""),
        md("## 4. Overlay all weight layers"),
        code("""fig, axes = plt.subplots(1, 2, figsize=(11, 4))
for name in ['net.0.weight', 'net.2.weight']:
    axes[0].hist(before[name].flatten().numpy(), bins=30, alpha=0.6, label=name)
    axes[1].hist(after[name].flatten().numpy(), bins=30, alpha=0.6, label=name)
axes[0].legend(); axes[0].set_title('Before')
axes[1].legend(); axes[1].set_title('After')
plt.tight_layout(); plt.show()"""),
        md("## 5. Mean absolute weight per layer"),
        code("""layers = list(before.keys())
means_before = [before[k].abs().mean().item() for k in layers]
means_after = [after[k].abs().mean().item() for k in layers]
fig, ax = plt.subplots(figsize=(9, 4))
x = np.arange(len(layers)); w = 0.35
ax.bar(x - w/2, means_before, w, label='before', color='lightgray')
ax.bar(x + w/2, means_after, w, label='after', color='teal')
ax.set_xticks(x); ax.set_xticklabels([l.replace('.', '\\n') for l in layers], fontsize=8)
ax.legend(); ax.set_title('Mean |weight| per parameter tensor')
plt.tight_layout(); plt.show()"""),
    ]),
    ("07_visualization", "03_activation_distributions.ipynb", [
        md("# Activation Distributions per Layer\n\nInspect whether activations are **saturated**, **dead** (ReLU), or well-scaled."),
        code(SETUP),
        md("## 1. Model with hooks"),
        code("""class MLPAct(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(16, 64)
        self.l2 = nn.Linear(64, 32)
        self.l3 = nn.Linear(32, 3)
    def forward(self, x):
        self.a1 = F.relu(self.l1(x))
        self.a2 = F.relu(self.l2(self.a1))
        return self.l3(self.a2)

model = MLPAct()
X = torch.randn(500, 16)
with torch.no_grad():
    model(X)"""),
        md("## 2. Histogram per layer"),
        code("""acts = [('layer1', model.a1), ('layer2', model.a2)]
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
for ax, (name, a) in zip(axes, acts):
    ax.hist(a.flatten().numpy(), bins=50, color='steelblue', edgecolor='black', alpha=0.85)
    ax.set_title(f'{name} activations'); ax.set_xlabel('value')
plt.tight_layout(); plt.show()"""),
        md("## 3. Compare before/after BatchNorm-style scaling"),
        code("""raw = model.a1.flatten()
scaled = (raw - raw.mean()) / (raw.std() + 1e-5)
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].hist(raw.numpy(), bins=50, color='coral'); axes[0].set_title('Raw ReLU activations')
axes[1].hist(scaled.numpy(), bins=50, color='seagreen'); axes[1].set_title('Standardized (illustrative)')
plt.tight_layout(); plt.show()"""),
        md("## 4. Dead ReLU fraction per neuron"),
        code("""dead_frac = (model.a1 == 0).float().mean(dim=0).numpy()
fig, ax = plt.subplots(figsize=(10, 4))
ax.bar(range(len(dead_frac)), dead_frac, color='gray', edgecolor='black')
ax.set_xlabel('neuron index'); ax.set_ylabel('fraction zero')
ax.set_title('Dead ReLU rate per neuron (one batch)')
plt.tight_layout(); plt.show()"""),
        md("## 5. Activation mean/std per layer"),
        code("""stats = {'L1': model.a1, 'L2': model.a2}
means = [t.mean().item() for t in stats.values()]
stds = [t.std().item() for t in stats.values()]
fig, ax = plt.subplots(figsize=(7, 4))
x = np.arange(2); w = 0.35
ax.bar(x - w/2, means, w, label='mean', color='steelblue')
ax.bar(x + w/2, stds, w, label='std', color='orange')
ax.set_xticks(x); ax.set_xticklabels(list(stats.keys()))
ax.legend(); ax.set_title('Activation statistics')
plt.tight_layout(); plt.show()"""),
    ]),
    ("07_visualization", "04_decision_boundary_2d.ipynb", [
        md("# 2D Decision Boundary\n\nTrain a classifier on **2D blob** data and visualize the decision regions."),
        code(SETUP + """
class BlobDataGenerator:
    def __init__(self, n_per_class=80, seed=42):
        g = torch.Generator().manual_seed(seed)
        centers = torch.tensor([[0., 0.], [2., 2.], [-2., 1.]])
        X, y = [], []
        for c, center in enumerate(centers):
            pts = center + torch.randn(n_per_class, 2, generator=g) * 0.6
            X.append(pts)
            y.append(torch.full((n_per_class,), c))
        self.X = torch.cat(X)
        self.y = torch.cat(y)

class BlobDataset(Dataset):
    def __init__(self, X, y):
        self.X, self.y = X, y
    def __len__(self):
        return len(self.y)
    def __getitem__(self, i):
        return self.X[i], self.y[i]

class BlobClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2, 16), nn.ReLU(), nn.Linear(16, 3))
    def forward(self, x):
        return self.net(x)
"""),
        md("## 1. Generate 2D blobs"),
        code("""gen = BlobDataGenerator()
X, y = gen.X, gen.y
print(X.shape, y.shape)"""),
        md("## 2. Scatter plot of classes"),
        code("""fig, ax = plt.subplots(figsize=(7, 6))
colors = ['#e74c3c', '#3498db', '#2ecc71']
for c in range(3):
    mask = y == c
    ax.scatter(X[mask, 0], X[mask, 1], c=colors[c], label=f'class {c}', s=30, alpha=0.8)
ax.legend(); ax.set_title('2D blob dataset'); ax.set_aspect('equal')
plt.tight_layout(); plt.show()"""),
        md("## 3. Train classifier"),
        code("""model = BlobClassifier()
opt = torch.optim.Adam(model.parameters(), lr=0.05)
for _ in range(200):
    opt.zero_grad()
    F.cross_entropy(model(X), y).backward()
    opt.step()"""),
        md("## 4. Decision boundary mesh"),
        code("""xx, yy = np.meshgrid(np.linspace(-4, 4, 120), np.linspace(-3, 4, 120))
grid = torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32)
with torch.no_grad():
    preds = model(grid).argmax(1).numpy().reshape(xx.shape)

fig, ax = plt.subplots(figsize=(8, 6))
ax.contourf(xx, yy, preds, alpha=0.35, cmap='Pastel1')
for c in range(3):
    mask = y == c
    ax.scatter(X[mask, 0], X[mask, 1], c=colors[c], s=25, edgecolors='k', linewidths=0.3)
ax.set_title('Decision regions + data points'); ax.set_aspect('equal')
plt.tight_layout(); plt.show()"""),
        md("## 5. Confidence heatmap"),
        code("""with torch.no_grad():
    probs = F.softmax(model(grid), dim=1)[:, 0].numpy().reshape(xx.shape)
fig, ax = plt.subplots(figsize=(7, 6))
im = ax.imshow(probs, extent=[-4, 4, -3, 4], origin='lower', cmap='viridis', alpha=0.9)
plt.colorbar(im, label='P(class 0)')
ax.scatter(X[y==0, 0], X[y==0, 1], c='white', s=15, edgecolors='k')
ax.set_title('Class-0 probability surface')
plt.tight_layout(); plt.show()"""),
    ]),
    ("07_visualization", "05_confusion_matrix.ipynb", [
        md("# Confusion Matrix Heatmap\n\nSee **which classes get confused** after classification."),
        code(SETUP + "\n" + DUMMY_GEN + "\n" + SIMPLE_MLP),
        md("## 1. Train and predict"),
        code("""gen = DummyDataGenerator(n_samples=300, n_features=8, n_classes=3)
X, y = gen.tensors()
model = SimpleMLP()
opt = torch.optim.Adam(model.parameters(), lr=0.02)
for _ in range(80):
    opt.zero_grad()
    F.cross_entropy(model(X), y).backward()
    opt.step()
with torch.no_grad():
    preds = model(X).argmax(1)"""),
        md("## 2. Build confusion matrix"),
        code("""n_classes = 3
cm = torch.zeros(n_classes, n_classes, dtype=torch.int32)
for t, p in zip(y, preds):
    cm[t, p] += 1
print(cm)"""),
        md("## 3. Heatmap (counts)"),
        code("""fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(cm.numpy(), cmap='Blues')
ax.set_xticks(range(3)); ax.set_yticks(range(3))
ax.set_xlabel('predicted'); ax.set_ylabel('true')
for i in range(3):
    for j in range(3):
        ax.text(j, i, int(cm[i,j]), ha='center', va='center', color='white' if cm[i,j]>cm.max()/2 else 'black')
plt.colorbar(im); ax.set_title('Confusion matrix (counts)')
plt.tight_layout(); plt.show()"""),
        md("## 4. Normalized confusion matrix"),
        code("""cm_norm = cm.float() / cm.sum(dim=1, keepdim=True).clamp(min=1)
fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(cm_norm.numpy(), cmap='Oranges', vmin=0, vmax=1)
ax.set_xticks(range(3)); ax.set_yticks(range(3))
ax.set_xlabel('predicted'); ax.set_ylabel('true')
plt.colorbar(im, label='recall per true class')
ax.set_title('Row-normalized confusion matrix')
plt.tight_layout(); plt.show()"""),
        md("## 5. Per-class accuracy bar chart"),
        code("""diag = cm.diag().float()
row_sum = cm.sum(dim=1).float().clamp(min=1)
class_acc = (diag / row_sum).numpy()
fig, ax = plt.subplots(figsize=(6, 4))
ax.bar(range(3), class_acc, color=['#e74c3c', '#3498db', '#2ecc71'], edgecolor='black')
ax.set_ylim(0, 1.05); ax.set_xlabel('class'); ax.set_ylabel('recall')
ax.set_title('Per-class recall from confusion matrix')
plt.tight_layout(); plt.show()"""),
    ]),
]


# =============================================================================
# 08 OPTIMIZATION LANDSCAPES (4 notebooks)
# =============================================================================

OPT_LANDSCAPE_TUTORIALS = [
    ("08_optimization_landscapes", "01_gradient_descent_1d.ipynb", [
        md("# Gradient Descent on a 1D Curve\n\nWatch a **ball** roll down `f(x) = x²` via repeated gradient steps."),
        code(SETUP),
        md("## 1. Define function and derivative"),
        code("""def f(x):
    return x ** 2
def df(x):
    return 2 * x

x_start = 3.0
lr = 0.1
xs = [x_start]
for _ in range(12):
    xs.append(xs[-1] - lr * df(xs[-1]))
xs = np.array(xs)"""),
        md("## 2. Plot the curve"),
        code("""x_line = np.linspace(-3.5, 3.5, 200)
fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(x_line, f(x_line), 'b-', lw=2, label='f(x)=x²')
ax.set_xlabel('x'); ax.set_ylabel('f(x)'); ax.legend()
ax.set_title('1D loss landscape')
plt.tight_layout(); plt.show()"""),
        md("## 3. Multi-step descent path"),
        code("""fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(x_line, f(x_line), 'b-', lw=2, alpha=0.5)
ax.plot(xs, f(xs), 'ro-', lw=2, markersize=8, label='GD steps')
for i, x in enumerate(xs):
    ax.annotate(str(i), (x, f(x)), textcoords='offset points', xytext=(0, 8), ha='center', fontsize=8)
ax.legend(); ax.set_title('Ball rolling down the curve')
plt.tight_layout(); plt.show()"""),
        md("## 4. Loss vs iteration"),
        code("""losses = [f(x) for x in xs]
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(losses, 'o-', color='coral', lw=2)
ax.set_xlabel('step'); ax.set_ylabel('f(x)')
ax.set_title('Loss decreases each gradient step')
plt.tight_layout(); plt.show()"""),
        md("## 5. Position vs step"),
        code("""fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(xs, 's-', color='seagreen', lw=2)
ax.axhline(0, color='gray', ls='--', label='minimum at x=0')
ax.set_xlabel('step'); ax.set_ylabel('x')
ax.legend(); ax.set_title('Parameter value converges to 0')
plt.tight_layout(); plt.show()"""),
    ]),
    ("08_optimization_landscapes", "02_learning_rate_effect.ipynb", [
        md("# Learning Rate Effect\n\nSame problem, **different learning rates** — too large diverges, too small is slow."),
        code(SETUP),
        md("## 1. GD with multiple learning rates"),
        code("""def gd_path(lr, steps=15, x0=2.5):
    x = x0
    path = [x]
    for _ in range(steps):
        x = x - lr * 2 * x
        path.append(x)
    return np.array(path)

lrs = [0.05, 0.3, 0.55, 0.9]
paths = {lr: gd_path(lr) for lr in lrs}"""),
        md("## 2. Loss curve f(x)=x²"),
        code("""x_line = np.linspace(-3, 3, 200)
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(x_line, x_line**2, 'k-', lw=2)
ax.set_title('f(x) = x²'); ax.set_xlabel('x')
plt.tight_layout(); plt.show()"""),
        md("## 3. All LR paths on one plot"),
        code("""fig, ax = plt.subplots(figsize=(9, 5))
colors = ['green', 'blue', 'orange', 'red']
for lr, c in zip(lrs, colors):
    p = paths[lr]
    ax.plot(p, p**2, 'o-', color=c, label=f'lr={lr}', lw=1.5, markersize=5)
ax.plot(x_line, x_line**2, 'k-', alpha=0.3)
ax.legend(); ax.set_title('Optimizer paths for different learning rates')
plt.tight_layout(); plt.show()"""),
        md("## 4. Loss vs step for each LR"),
        code("""fig, ax = plt.subplots(figsize=(9, 5))
for lr, c in zip(lrs, colors):
    p = paths[lr]
    ax.plot(p**2, 'o-', color=c, label=f'lr={lr}', lw=1.5)
ax.set_xlabel('step'); ax.set_ylabel('f(x)'); ax.legend()
ax.set_title('Convergence speed vs instability')
plt.tight_layout(); plt.show()"""),
        md("## 5. Final |x| after 15 steps"),
        code("""finals = [abs(paths[lr][-1]) for lr in lrs]
fig, ax = plt.subplots(figsize=(7, 4))
ax.bar([str(l) for l in lrs], finals, color=colors, edgecolor='black')
ax.set_xlabel('learning rate'); ax.set_ylabel('|x| after 15 steps')
ax.set_title('Too-large LR may diverge (|x| grows)')
plt.tight_layout(); plt.show()"""),
    ]),
    ("08_optimization_landscapes", "03_momentum_and_adam.ipynb", [
        md("# Momentum & Adam on 2D Contours\n\n**Momentum** adds velocity; **Adam** adapts per-parameter step sizes."),
        code(SETUP),
        md("## 1. Rosenbrock-like elongated bowl"),
        code("""def loss2d(w):
    return (1 - w[0])**2 + 10 * (w[1] - w[0]**2)**2

w1, w2 = np.meshgrid(np.linspace(-2, 2, 100), np.linspace(-1, 3, 100))
Z = (1 - w1)**2 + 10 * (w2 - w1**2)**2"""),
        md("## 2. Contour plot"),
        code("""fig, ax = plt.subplots(figsize=(8, 6))
ax.contour(w1, w2, Z, levels=30, cmap='viridis')
ax.set_xlabel('w1'); ax.set_ylabel('w2')
ax.set_title('Non-convex optimization landscape')
plt.tight_layout(); plt.show()"""),
        md("## 3. Run SGD, SGD+momentum, Adam"),
        code("""def track(optimizer_cls, steps=60, lr=0.01, **kw):
    w = torch.tensor([-1.5, 2.0], requires_grad=True)
    opt = optimizer_cls([w], lr=lr, **kw)
    path = [w.detach().clone().numpy()]
    for _ in range(steps):
        l = loss2d(w)
        opt.zero_grad(); l.backward(); opt.step()
        path.append(w.detach().clone().numpy())
    return np.array(path)

p_sgd = track(torch.optim.SGD, lr=0.002)
p_mom = track(torch.optim.SGD, lr=0.002, momentum=0.9)
p_adam = track(torch.optim.Adam, lr=0.05)"""),
        md("## 4. Overlay paths on contours"),
        code("""fig, ax = plt.subplots(figsize=(9, 7))
ax.contour(w1, w2, Z, levels=30, cmap='viridis', alpha=0.7)
ax.plot(p_sgd[:,0], p_sgd[:,1], 'r.-', label='SGD', lw=1.5)
ax.plot(p_mom[:,0], p_mom[:,1], 'b.-', label='SGD+momentum', lw=1.5)
ax.plot(p_adam[:,0], p_adam[:,1], 'g.-', label='Adam', lw=1.5)
ax.legend(); ax.set_title('Optimizer paths compared')
plt.tight_layout(); plt.show()"""),
        md("## 5. Loss along paths"),
        code("""def path_loss(p):
    return [loss2d(torch.tensor(pt)).item() for pt in p]

fig, ax = plt.subplots(figsize=(9, 4))
ax.plot(path_loss(p_sgd), 'r-', label='SGD')
ax.plot(path_loss(p_mom), 'b-', label='momentum')
ax.plot(path_loss(p_adam), 'g-', label='Adam')
ax.set_xlabel('step'); ax.set_ylabel('loss'); ax.legend()
plt.tight_layout(); plt.show()"""),
    ]),
    ("08_optimization_landscapes", "04_local_minima_saddle.ipynb", [
        md("# Saddle Points & Local Structure\n\nSaddle points have **negative curvature** in some directions — optimizers can escape."),
        code(SETUP),
        md("## 1. Saddle function f(x,y) = x² - y²"),
        code("""xx, yy = np.meshgrid(np.linspace(-2, 2, 100), np.linspace(-2, 2, 100))
zz = xx**2 - yy**2"""),
        md("## 2. 3D surface view (contour + elevation color)"),
        code("""fig, ax = plt.subplots(figsize=(7, 6))
cf = ax.contourf(xx, yy, zz, levels=25, cmap='coolwarm')
plt.colorbar(cf); ax.set_xlabel('x'); ax.set_ylabel('y')
ax.set_title('Saddle: f(x,y) = x² − y²'); ax.set_aspect('equal')
plt.tight_layout(); plt.show()"""),
        md("## 3. Gradient field — escapes along -y direction"),
        code("""GX, GY = np.meshgrid(np.linspace(-1.5, 1.5, 10), np.linspace(-1.5, 1.5, 10))
U = 2 * GX
V = -2 * GY
fig, ax = plt.subplots(figsize=(7, 6))
ax.contour(xx, yy, zz, levels=15, cmap='gray', alpha=0.5)
ax.quiver(GX, GY, -U, -V, color='blue', alpha=0.8)
ax.scatter([0], [0], c='red', s=80, zorder=5)
ax.set_title('Negative gradient field (descent directions)')
ax.set_aspect('equal'); plt.tight_layout(); plt.show()"""),
        md("## 4. GD path starting near saddle"),
        code("""pos = torch.tensor([0.01, 1.2])
lr = 0.1
path = [pos.clone()]
for _ in range(25):
    pos = pos - lr * torch.tensor([2*pos[0], -2*pos[1]])
    path.append(pos.clone())
path = torch.stack(path).numpy()

fig, ax = plt.subplots(figsize=(7, 6))
ax.contour(xx, yy, zz, levels=20, cmap='coolwarm', alpha=0.7)
ax.plot(path[:,0], path[:,1], 'ko-', lw=2, markersize=4)
ax.scatter([0], [0], c='gold', s=100, zorder=5, edgecolors='k')
ax.set_title('Gradient descent escapes saddle along y-axis')
ax.set_aspect('equal'); plt.tight_layout(); plt.show()"""),
        md("## 5. Compare local minima basin x²+y² nearby"),
        code("""zz2 = xx**2 + yy**2
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].contourf(xx, yy, zz, levels=20, cmap='coolwarm')
axes[0].set_title('Saddle (can escape)')
axes[1].contourf(xx, yy, zz2, levels=20, cmap='viridis')
axes[1].set_title('Bowl (unique minimum)')
for ax in axes: ax.set_aspect('equal')
plt.tight_layout(); plt.show()"""),
    ]),
]


# =============================================================================
# 09 REGULARIZATION (4 notebooks)
# =============================================================================

REGULARIZATION_TUTORIALS = [
    ("09_regularization", "01_overfitting_polynomial.ipynb", [
        md("# Overfitting with High-Degree Polynomials\n\nA **high-degree** poly fits noise on training data but fails on validation."),
        code(SETUP),
        md("## 1. Synthetic noisy data"),
        code("""n = 40
x = torch.linspace(-1, 1, n)
y = torch.sin(3 * x) + 0.2 * torch.randn(n)
x_train, x_val = x[:30], x[30:]
y_train, y_val = y[:30], y[30:]"""),
        md("## 2. Fit degree 1 vs degree 9"),
        code("""def poly_fit(degree, x_tr, y_tr):
    X = torch.stack([x_tr**i for i in range(degree+1)], dim=1)
    w = torch.linalg.lstsq(X, y_tr.unsqueeze(1)).solution.squeeze()
    return w

def predict(w, x):
    deg = len(w) - 1
    return sum(w[i] * x**i for i in range(deg+1))

w1 = poly_fit(1, x_train, y_train)
w9 = poly_fit(9, x_train, y_train)
x_plot = torch.linspace(-1, 1, 200)"""),
        md("## 3. Plot fits"),
        code("""fig, ax = plt.subplots(figsize=(9, 5))
ax.scatter(x_train, y_train, label='train', color='steelblue', s=40)
ax.scatter(x_val, y_val, label='val', color='coral', s=40)
ax.plot(x_plot, predict(w1, x_plot), 'g-', lw=2, label='degree 1')
ax.plot(x_plot, predict(w9, x_plot), 'r-', lw=2, label='degree 9')
ax.legend(); ax.set_title('Low vs high degree polynomial')
plt.tight_layout(); plt.show()"""),
        md("## 4. Train vs val MSE across degrees"),
        code("""degrees = range(1, 12)
train_mse, val_mse = [], []
for d in degrees:
    w = poly_fit(d, x_train, y_train)
    train_mse.append(float(((predict(w, x_train) - y_train)**2).mean()))
    val_mse.append(float(((predict(w, x_val) - y_val)**2).mean()))

fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(list(degrees), train_mse, 'b-o', label='train MSE', lw=2)
ax.plot(list(degrees), val_mse, 'r-o', label='val MSE', lw=2)
ax.set_xlabel('polynomial degree'); ax.legend()
ax.set_title('Validation error rises when overfitting')
plt.tight_layout(); plt.show()"""),
        md("## 5. Gap between train and val error"),
        code("""gap = np.array(val_mse) - np.array(train_mse)
fig, ax = plt.subplots(figsize=(8, 4))
ax.bar(list(degrees), gap, color='orange', edgecolor='black')
ax.set_xlabel('degree'); ax.set_ylabel('val_mse - train_mse')
ax.set_title('Generalization gap grows with model complexity')
plt.tight_layout(); plt.show()"""),
    ]),
    ("09_regularization", "02_dropout_visual.ipynb", [
        md("# Dropout Visualization\n\nEach forward pass **randomly zeros** a fraction of neurons — different masks every time."),
        code(SETUP),
        md("## 1. Model with dropout"),
        code("""class DropNet(nn.Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.fc = nn.Linear(8, 16)
        self.drop = nn.Dropout(p=p)
    def forward(self, x):
        h = F.relu(self.fc(x))
        return self.drop(h)

model = DropNet(p=0.5)
x = torch.ones(1, 8)"""),
        md("## 2. Multiple forward passes — different masks"),
        code("""masks = []
for _ in range(6):
    model.train()
    out = model(x)
    mask = (out == 0).float().squeeze()
    masks.append(mask.numpy())
masks = np.array(masks)
print(f"mask shape: {masks.shape}")"""),
        md("## 3. Heatmap of dropped neurons per pass"),
        code("""fig, ax = plt.subplots(figsize=(10, 4))
ax.imshow(masks, aspect='auto', cmap='Greys', vmin=0, vmax=1)
ax.set_xlabel('neuron index'); ax.set_ylabel('forward pass #')
ax.set_title('White = dropped neuron (Dropout p=0.5)')
plt.tight_layout(); plt.show()"""),
        md("## 4. Drop fraction per neuron across passes"),
        code("""drop_rate = masks.mean(axis=0)
fig, ax = plt.subplots(figsize=(9, 4))
ax.bar(range(16), drop_rate, color='steelblue', edgecolor='black')
ax.axhline(0.5, color='red', ls='--', label='p=0.5 target')
ax.set_xlabel('neuron'); ax.legend()
ax.set_title('Empirical drop rate per neuron (6 passes)')
plt.tight_layout(); plt.show()"""),
        md("## 5. Train vs eval mode output scale"),
        code("""model.train()
train_outs = torch.stack([model(x) for _ in range(50)]).mean(0)
model.eval()
eval_out = model(x)
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].bar(range(16), train_outs.squeeze().numpy(), color='coral')
axes[0].set_title('Mean output in train mode (noisy)')
axes[1].bar(range(16), eval_out.squeeze().numpy(), color='seagreen')
axes[1].set_title('Output in eval mode (deterministic)')
plt.tight_layout(); plt.show()"""),
    ]),
    ("09_regularization", "03_batch_norm_effect.ipynb", [
        md("# BatchNorm Effect on Distributions\n\n**BatchNorm** re-centers and rescales activations batch-wise."),
        code(SETUP),
        md("## 1. Layer without BatchNorm"),
        code("""class PlainNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(32, 64)
    def forward(self, x):
        return self.fc(x)

class BNNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(32, 64)
        self.bn = nn.BatchNorm1d(64)
    def forward(self, x):
        return self.bn(self.fc(x))

X = torch.randn(128, 32) * 3 + 2"""),
        md("## 2. Activation distributions before BN"),
        code("""plain = PlainNet()
with torch.no_grad():
    h_plain = plain(X)
fig, ax = plt.subplots(figsize=(8, 4))
ax.hist(h_plain.flatten().numpy(), bins=50, color='coral', edgecolor='black')
ax.set_title('Linear output distribution (shifted, wide)')
plt.tight_layout(); plt.show()"""),
        md("## 3. After BatchNorm"),
        code("""bn = BNNet()
with torch.no_grad():
    h_bn = bn(X)
fig, ax = plt.subplots(figsize=(8, 4))
ax.hist(h_bn.flatten().numpy(), bins=50, color='seagreen', edgecolor='black')
ax.set_title('After BatchNorm (~0 mean, ~1 std per feature)')
plt.tight_layout(); plt.show()"""),
        md("## 4. Side-by-side histograms"),
        code("""fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].hist(h_plain.flatten().numpy(), bins=50, color='coral', alpha=0.8)
axes[0].set_title(f'before: μ={h_plain.mean():.2f}, σ={h_plain.std():.2f}')
axes[1].hist(h_bn.flatten().numpy(), bins=50, color='seagreen', alpha=0.8)
axes[1].set_title(f'after: μ={h_bn.mean():.2f}, σ={h_bn.std():.2f}')
plt.tight_layout(); plt.show()"""),
        md("## 5. Per-feature mean before/after"),
        code("""feat_mean_before = h_plain.mean(dim=0).numpy()
feat_mean_after = h_bn.mean(dim=0).numpy()
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(feat_mean_before, 'o-', label='before BN', color='coral')
ax.plot(feat_mean_after, 's-', label='after BN', color='seagreen')
ax.set_xlabel('feature index'); ax.legend()
ax.set_title('Per-feature batch mean')
plt.tight_layout(); plt.show()"""),
    ]),
    ("09_regularization", "04_weight_decay.ipynb", [
        md("# Weight Decay (L2 Regularization)\n\nL2 penalty **shrinks weights** toward zero, reducing overfitting."),
        code(SETUP + "\n" + DUMMY_GEN + "\n" + SIMPLE_MLP),
        md("## 1. Train with and without weight decay"),
        code("""def train_model(weight_decay=0.0, steps=150):
    gen = DummyDataGenerator(n_samples=200, n_features=8, n_classes=3)
    X, y = gen.tensors()
    model = SimpleMLP(hidden=64)
    opt = torch.optim.Adam(model.parameters(), lr=0.02, weight_decay=weight_decay)
    for _ in range(steps):
        opt.zero_grad()
        F.cross_entropy(model(X), y).backward()
        opt.step()
    return model

m_none = train_model(0.0)
m_l2 = train_model(0.1)"""),
        md("## 2. Weight histograms comparison"),
        code("""w_none = m_none.net[0].weight.detach().flatten().numpy()
w_l2 = m_l2.net[0].weight.detach().flatten().numpy()
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].hist(w_none, bins=40, color='coral', edgecolor='black')
axes[0].set_title('No weight decay')
axes[1].hist(w_l2, bins=40, color='seagreen', edgecolor='black')
axes[1].set_title('weight_decay=0.1')
plt.tight_layout(); plt.show()"""),
        md("## 3. |weight| distribution overlay"),
        code("""fig, ax = plt.subplots(figsize=(8, 4))
ax.hist(np.abs(w_none), bins=40, alpha=0.6, label='no decay', color='coral')
ax.hist(np.abs(w_l2), bins=40, alpha=0.6, label='L2 decay', color='seagreen')
ax.legend(); ax.set_title('Absolute weight magnitudes')
plt.tight_layout(); plt.show()"""),
        md("## 4. L2 norm per layer"),
        code("""def layer_norms(model):
    return [p.norm().item() for n, p in model.named_parameters() if 'weight' in n]

names = ['W1', 'W2']
n0 = layer_norms(m_none)
n1 = layer_norms(m_l2)
fig, ax = plt.subplots(figsize=(7, 4))
x = np.arange(2); w = 0.35
ax.bar(x - w/2, n0, w, label='no decay', color='coral')
ax.bar(x + w/2, n1, w, label='weight decay', color='seagreen')
ax.set_xticks(x); ax.set_xticklabels(names); ax.legend()
ax.set_title('L2 norm of weight tensors')
plt.tight_layout(); plt.show()"""),
        md("## 5. Weight decay strengths sweep"),
        code("""decs = [0.0, 0.01, 0.05, 0.1, 0.5]
norms = []
for d in decs:
    m = train_model(d, steps=80)
    norms.append(m.net[0].weight.norm().item())
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(decs, norms, 'o-', color='purple', lw=2)
ax.set_xlabel('weight_decay'); ax.set_ylabel('||W1||')
ax.set_title('Stronger L2 → smaller weights')
plt.tight_layout(); plt.show()"""),
    ]),
]


ALL_TUTORIALS = (
    AUTOGRAD_TUTORIALS
    + NN_MODULES_TUTORIALS
    + TRAINING_TUTORIALS
    + CNN_TUTORIALS
    + VISUALIZATION_TUTORIALS
    + OPT_LANDSCAPE_TUTORIALS
    + REGULARIZATION_TUTORIALS
)


if __name__ == "__main__":
    print(f"Generating tutorials under {ROOT} ...")
    created = 0
    skipped = 0
    for folder, name, cells in ALL_TUTORIALS:
        if save(folder, name, cells):
            created += 1
        else:
            skipped += 1
    print(f"\nDone: {created} notebooks created, {skipped} skipped (already exist).")
    print(f"Total defined: {len(ALL_TUTORIALS)}")
