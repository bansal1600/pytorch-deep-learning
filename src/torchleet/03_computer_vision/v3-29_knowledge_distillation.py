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
# # V3-29: Knowledge Distillation — Solution

# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset


class DummyDataGenerator:
    def __init__(self, n=128, d=32):
        self.X = torch.randn(n, d)
        self.y = torch.randint(0, 5, (n,))
    def tensors(self): return self.X, self.y


class DistillDataset(Dataset):
    def __init__(self, X, y): self.X, self.y = X, y
    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i], self.y[i]


class TeacherModel(nn.Module):
    def __init__(self, d=32, c=5):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, 128), nn.ReLU(), nn.Linear(128, c))
    def forward(self, x):
        return self.net(x)


class StudentModel(nn.Module):
    def __init__(self, d=32, c=5):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, 32), nn.ReLU(), nn.Linear(32, c))
    def forward(self, x):
        return self.net(x)


def distillation_loss(student_logits, teacher_logits, labels, T=4.0, alpha=0.7):
    soft = F.kl_div(
        F.log_softmax(student_logits / T, dim=-1),
        F.softmax(teacher_logits / T, dim=-1),
        reduction='batchmean',
    ) * (T * T)
    hard = F.cross_entropy(student_logits, labels)
    return alpha * soft + (1 - alpha) * hard


X, y = DummyDataGenerator().tensors()
teacher = TeacherModel(); student = StudentModel()
teacher.eval()
with torch.no_grad():
    t_logits = teacher(X)
loss = distillation_loss(student(X), t_logits, y)
print(f"distillation loss: {loss.item():.4f}")
print("✓ Knowledge distillation loss")
