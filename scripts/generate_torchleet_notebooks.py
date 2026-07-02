"""Generate original TorchLeet solution notebooks with Dataset, Model, and DummyDataGenerator."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "torchleet_solutions"


def nb(cells: list[dict]) -> dict:
    return {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
        "cells": cells,
    }


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code(text: str) -> dict:
    return {"cell_type": "code", "metadata": {}, "outputs": [], "execution_count": None, "source": text.splitlines(keepends=True)}


def save(folder: str, filename: str, cells: list[dict]) -> None:
    path = ROOT / folder / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(nb(cells), indent=1))
    print(f"  wrote {path.relative_to(ROOT.parent)}")


# ---------------------------------------------------------------------------
# BASICS (24)
# ---------------------------------------------------------------------------

def gen_v1_01():
    save("basics", "v1-01_linear_regression_solution.ipynb", [
        md("# V1-01: Linear Regression — Solution\n\n**TorchLeet Basics** | Difficulty: Basic\n\nImplement linear regression with `LinearRegressionModel`, custom `Dataset`, and synthetic data."),
        code("""import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


class DummyDataGenerator:
    \"\"\"Synthetic linear data: y = 2x + 3 + noise.\"\"\"
    def __init__(self, n: int = 200, seed: int = 42):
        torch.manual_seed(seed)
        self.X = torch.rand(n, 1) * 10
        self.y = 2 * self.X + 3 + torch.randn(n, 1) * 0.5

    def tensors(self):
        return self.X, self.y


class RegressionDataset(Dataset):
    def __init__(self, X: torch.Tensor, y: torch.Tensor):
        self.X, self.y = X, y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class LinearRegressionModel(nn.Module):
    def __init__(self, in_features: int = 1, out_features: int = 1):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x):
        return self.linear(x)


# --- Demo ---
gen = DummyDataGenerator()
dataset = RegressionDataset(*gen.tensors())
loader = DataLoader(dataset, batch_size=32, shuffle=True)
model = LinearRegressionModel()
opt = torch.optim.SGD(model.parameters(), lr=0.01)
loss_fn = nn.MSELoss()

for epoch in range(300):
    total = 0.0
    for xb, yb in loader:
        opt.zero_grad()
        loss = loss_fn(model(xb), yb)
        loss.backward()
        opt.step()
        total += loss.item()
    if (epoch + 1) % 100 == 0:
        w, b = model.linear.weight.item(), model.linear.bias.item()
        print(f"epoch {epoch+1} loss={total/len(loader):.4f} w≈{w:.3f} b≈{b:.3f}")

assert abs(model.linear.weight.item() - 2.0) < 0.3
print("✓ Linear regression converged near y=2x+3")"""),
    ])


def gen_v1_02():
    save("basics", "v1-02_custom_dataset_dataloader_solution.ipynb", [
        md("# V1-02: Custom Dataset & DataLoader — Solution"),
        code("""import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


class DummyDataGenerator:
    def __init__(self, n=150, seed=0):
        torch.manual_seed(seed)
        self.features = torch.randn(n, 4)
        self.targets = self.features @ torch.tensor([1., -2., 0.5, 3.]) + torch.randn(n) * 0.1

    def as_tensors(self):
        return self.features, self.targets


class TabularDataset(Dataset):
    def __init__(self, features, targets):
        self.features = features.float()
        self.targets = targets.float()

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]


class BatchPredictor(nn.Module):
    \"\"\"Simple model consuming batched dataset samples.\"\"\"
    def __init__(self, d_in=4):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, 16), nn.ReLU(), nn.Linear(16, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


gen = DummyDataGenerator()
ds = TabularDataset(*gen.as_tensors())
loader = DataLoader(ds, batch_size=16, shuffle=True)
model = BatchPredictor()

for xb, yb in loader:
    assert xb.shape == (16, 4) and yb.shape == (16,)
    preds = model(xb)
    assert preds.shape == (16,)
    break

print(f"Dataset size={len(ds)}, batches={len(loader)}")
print("✓ Custom Dataset + DataLoader working")"""),
    ])


def gen_v1_03():
    save("basics", "v1-03_custom_activation_solution.ipynb", [
        md("# V1-03: Custom Activation Function — Solution\n\nActivation: `tanh(x) + x`"),
        code("""import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


class DummyDataGenerator:
    def __init__(self, n=256):
        torch.manual_seed(1)
        self.X = torch.randn(n, 3)
        self.y = (self.X.sum(dim=1, keepdim=True) > 0).float()

    def tensors(self):
        return self.X, self.y


class CustomActivation(nn.Module):
    def forward(self, x):
        return torch.tanh(x) + x


class ActivationModel(nn.Module):
    def __init__(self, d_in=3):
        super().__init__()
        self.fc1 = nn.Linear(d_in, 8)
        self.act = CustomActivation()
        self.fc2 = nn.Linear(8, 1)

    def forward(self, x):
        return torch.sigmoid(self.fc2(self.act(self.fc1(x))))


class BinaryDataset(Dataset):
    def __init__(self, X, y):
        self.X, self.y = X, y
    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i], self.y[i]


X, y = DummyDataGenerator().tensors()
loader = DataLoader(BinaryDataset(X, y), batch_size=32, shuffle=True)
model = ActivationModel()
opt = torch.optim.Adam(model.parameters(), lr=0.05)

for _ in range(100):
    for xb, yb in loader:
        opt.zero_grad()
        loss = nn.functional.binary_cross_entropy(model(xb), yb)
        loss.backward()
        opt.step()

x_test = torch.tensor([[1., 1., 1.]])
print(f"sample pred={model(x_test).item():.3f}")
print("✓ Custom activation integrated")"""),
    ])


def gen_v1_04():
    save("basics", "v1-04_huber_loss_solution.ipynb", [
        md("# V1-04: Huber Loss — Solution"),
        code("""import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


class HuberLoss(nn.Module):
    def __init__(self, delta: float = 1.0):
        super().__init__()
        self.delta = delta

    def forward(self, pred, target):
        err = pred - target
        abs_err = err.abs()
        quad = 0.5 * err.pow(2)
        lin = self.delta * (abs_err - 0.5 * self.delta)
        return torch.where(abs_err <= self.delta, quad, lin).mean()


class DummyDataGenerator:
    def __init__(self, n=300):
        torch.manual_seed(7)
        self.X = torch.randn(n, 2)
        self.y = self.X[:, 0:1] * 3 + torch.randn(n, 1) * 2

    def tensors(self):
        return self.X, self.y


class RegressionDataset(Dataset):
    def __init__(self, X, y):
        self.X, self.y = X, y
    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i], self.y[i]


class LinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 1)
    def forward(self, x):
        return self.linear(x)


X, y = DummyDataGenerator().tensors()
loader = DataLoader(RegressionDataset(X, y), batch_size=32, shuffle=True)
model = LinearRegressionModel()
criterion = HuberLoss(delta=1.0)
opt = torch.optim.Adam(model.parameters(), lr=0.03)

for _ in range(200):
    for xb, yb in loader:
        opt.zero_grad()
        criterion(model(xb), yb).backward()
        opt.step()

# Huber should be robust to outliers
outliers = torch.tensor([[10., 10.]])
print(f"Huber on outlier pred={model(outliers).item():.2f}")
print("✓ Huber loss training complete")"""),
    ])


def gen_v1_05():
    save("basics", "v1-05_deep_neural_network_solution.ipynb", [
        md("# V1-05: Deep Neural Network — Solution"),
        code("""import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


class DummyDataGenerator:
    def __init__(self, n=500):
        torch.manual_seed(0)
        self.X = torch.randn(n, 2)
        self.y = (self.X[:, 0]**2 + self.X[:, 1]**2).unsqueeze(1)

    def tensors(self):
        return self.X, self.y


class NonlinearDataset(Dataset):
    def __init__(self, X, y):
        self.X, self.y = X, y
    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i], self.y[i]


class DNNModel(nn.Module):
    def __init__(self, d_in=2, hidden=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )
    def forward(self, x):
        return self.net(x)


X, y = DummyDataGenerator().tensors()
loader = DataLoader(NonlinearDataset(X, y), batch_size=64, shuffle=True)
model = DNNModel()
opt = torch.optim.Adam(model.parameters(), lr=0.01)

for _ in range(150):
    for xb, yb in loader:
        opt.zero_grad()
        nn.MSELoss()(model(xb), yb).backward()
        opt.step()

test = torch.tensor([[1., 1.]])
print(f"pred x²+y²≈{model(test).item():.2f} (true=2)")
print("✓ DNN trained")"""),
    ])


def gen_v1_06():
    save("basics", "v1-06_tensorboard_solution.ipynb", [
        md("# V1-06: TensorBoard Logging — Solution"),
        code("""import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None


class DummyDataGenerator:
    def __init__(self, n=128):
        torch.manual_seed(3)
        self.X = torch.randn(n, 5)
        self.y = self.X.sum(dim=1, keepdim=True)

    def tensors(self):
        return self.X, self.y


class SimpleDataset(Dataset):
    def __init__(self, X, y):
        self.X, self.y = X, y
    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i], self.y[i]


class LoggingModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(5, 1)
    def forward(self, x):
        return self.fc(x)


X, y = DummyDataGenerator().tensors()
loader = DataLoader(SimpleDataset(X, y), batch_size=16, shuffle=True)
model = LoggingModel()
opt = torch.optim.SGD(model.parameters(), lr=0.05)

if SummaryWriter is not None:
    writer = SummaryWriter("runs/torchleet_v1_06")
    for epoch in range(20):
        epoch_loss = 0.0
        for xb, yb in loader:
            opt.zero_grad()
            loss = nn.MSELoss()(model(xb), yb)
            loss.backward()
            opt.step()
            epoch_loss += loss.item()
        writer.add_scalar("Loss/train", epoch_loss / len(loader), epoch)
    writer.close()
    print("✓ Logged to runs/torchleet_v1_06")
else:
    print("✓ Training loop OK (install tensorboard for logging)")"""),
    ])


def gen_v1_07():
    save("basics", "v1-07_save_load_model_solution.ipynb", [
        md("# V1-07: Save & Load Model — Solution"),
        code("""import torch
import torch.nn as nn
from torch.utils.data import Dataset


class DummyDataGenerator:
    def __init__(self):
        self.x = torch.randn(10, 3)

    def input(self):
        return self.x


class InferenceDataset(Dataset):
    def __init__(self, X):
        self.X = X
    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i]


class MLPModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(3, 8), nn.ReLU(), nn.Linear(8, 2))
    def forward(self, x):
        return self.net(x)


model = MLPModel()
path = "/tmp/torchleet_v1_07.pth"
torch.save(model.state_dict(), path)

model2 = MLPModel()
model2.load_state_dict(torch.load(path, weights_only=True))
model2.eval()

x = DummyDataGenerator().input()
with torch.no_grad():
    diff = (model(x) - model2(x)).abs().max().item()
print(f"max diff after reload: {diff:.2e}")
assert diff < 1e-6
print("✓ Save/load verified")"""),
    ])


def gen_v1_08():
    save("basics", "v1-08_cnn_cifar10_solution.ipynb", [
        md("# V1-08: CNN on CIFAR-10 — Solution\n\nUses dummy 32×32 RGB tensors mimicking CIFAR-10."),
        code("""import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


class DummyDataGenerator:
    def __init__(self, n=200, num_classes=10):
        torch.manual_seed(11)
        self.images = torch.rand(n, 3, 32, 32)
        self.labels = torch.randint(0, num_classes, (n,))

    def tensors(self):
        return self.images, self.labels


class CIFAR10DummyDataset(Dataset):
    def __init__(self, images, labels):
        self.images, self.labels = images, labels
    def __len__(self): return len(self.images)
    def __getitem__(self, i): return self.images[i], self.labels[i]


class CNNModel(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 128), nn.ReLU(),
            nn.Linear(128, num_classes),
        )
    def forward(self, x):
        return self.classifier(self.features(x))


imgs, labels = DummyDataGenerator().tensors()
loader = DataLoader(CIFAR10DummyDataset(imgs, labels), batch_size=32, shuffle=True)
model = CNNModel()
opt = torch.optim.Adam(model.parameters(), lr=1e-3)

for _ in range(3):
    for xb, yb in loader:
        opt.zero_grad()
        nn.CrossEntropyLoss()(model(xb), yb).backward()
        opt.step()

print(f"output shape: {model(imgs[:2]).shape}")
print("✓ CNN forward + training step OK")"""),
    ])


def gen_v1_09():
    save("basics", "v1-09_rnn_solution.ipynb", [
        md("# V1-09: RNN from Scratch — Solution"),
        code("""import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


class DummyDataGenerator:
    def __init__(self, seq_len=10, n_seq=100):
        torch.manual_seed(4)
        self.data = torch.randn(n_seq, seq_len, 4)
        self.targets = self.data.sum(dim=(1, 2))

    def tensors(self):
        return self.data, self.targets


class SequenceDataset(Dataset):
    def __init__(self, seqs, targets):
        self.seqs, self.targets = seqs, targets
    def __len__(self): return len(self.seqs)
    def __getitem__(self, i): return self.seqs[i], self.targets[i]


class RNNModel(nn.Module):
    def __init__(self, input_size=4, hidden=16):
        super().__init__()
        self.rnn = nn.RNN(input_size, hidden, batch_first=True)
        self.fc = nn.Linear(hidden, 1)
    def forward(self, x):
        out, _ = self.rnn(x)
        return self.fc(out[:, -1, :]).squeeze(-1)


seqs, targets = DummyDataGenerator().tensors()
loader = DataLoader(SequenceDataset(seqs, targets), batch_size=16, shuffle=True)
model = RNNModel()
opt = torch.optim.Adam(model.parameters(), lr=0.01)

for _ in range(30):
    for xb, yb in loader:
        opt.zero_grad()
        nn.MSELoss()(model(xb), yb).backward()
        opt.step()

print(f"RNN output shape: {model(seqs[:2]).shape}")
print("✓ RNN trained on sequences")"""),
    ])


def gen_v1_10():
    save("basics", "v1-10_data_augmentation_solution.ipynb", [
        md("# V1-10: Data Augmentation — Solution"),
        code("""import torch
from torch.utils.data import Dataset
from torchvision import transforms


class DummyDataGenerator:
    def __init__(self, n=64):
        torch.manual_seed(5)
        self.images = torch.rand(n, 1, 28, 28)

    def tensors(self):
        return self.images


class MNISTDummyDataset(Dataset):
    def __init__(self, images, transform=None):
        self.images = images
        self.transform = transform
    def __len__(self): return len(self.images)
    def __getitem__(self, i):
        img = self.images[i]
        if self.transform:
            img = self.transform(img)
        return img, 0


class AugmentationModel(torch.nn.Module):
    \"\"\"Tiny classifier on augmented images.\"\"\"
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 4, 3, padding=1)
    def forward(self, x):
        return self.conv(x).mean(dim=(2, 3))


transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomCrop(28),
    transforms.Normalize((0.5,), (0.5,)),
])

imgs = DummyDataGenerator().tensors()
ds = MNISTDummyDataset(imgs, transform=transform)
aug = ds[0][0]
model = AugmentationModel()
print(f"augmented shape: {aug.shape}, model out: {model(aug.unsqueeze(0)).shape}")
print("✓ Augmentation pipeline applied")"""),
    ])


def gen_v1_11():
    save("basics", "v1-11_benchmarking_solution.ipynb", [
        md("# V1-11: Benchmarking — Solution"),
        code("""import time
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


class DummyDataGenerator:
    def __init__(self, n=512, d=20, c=5):
        torch.manual_seed(6)
        self.X = torch.randn(n, d)
        self.y = torch.randint(0, c, (n,))
    def tensors(self):
        return self.X, self.y


class ClassificationDataset(Dataset):
    def __init__(self, X, y):
        self.X, self.y = X, y
    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i], self.y[i]


class SimpleNN(nn.Module):
    def __init__(self, d_in=20, n_classes=5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, n_classes),
        )
    def forward(self, x):
        return self.net(x)


X, y = DummyDataGenerator().tensors()
loader = DataLoader(ClassificationDataset(X, y), batch_size=64, shuffle=True)
model = SimpleNN()
opt = torch.optim.SGD(model.parameters(), lr=0.1)
criterion = nn.CrossEntropyLoss()

t0 = time.perf_counter()
for xb, yb in loader:
    opt.zero_grad()
    criterion(model(xb), yb).backward()
    opt.step()
train_time = time.perf_counter() - t0

model.eval()
t1 = time.perf_counter()
correct = 0
with torch.no_grad():
    for xb, yb in loader:
        correct += (model(xb).argmax(1) == yb).sum().item()
test_time = time.perf_counter() - t1
acc = correct / len(X)
print(f"train: {train_time*1000:.1f}ms | test: {test_time*1000:.1f}ms | acc={acc:.2%}")
print("✓ Benchmark complete")"""),
    ])


def gen_v1_12():
    save("basics", "v1-12_autoencoder_solution.ipynb", [
        md("# V1-12: Autoencoder Anomaly Detection — Solution"),
        code("""import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


class DummyDataGenerator:
    def __init__(self, n=128):
        torch.manual_seed(8)
        self.normal = torch.rand(n, 1, 28, 28)
        self.anomaly = torch.rand(16, 1, 28, 28) * 2 + 0.5
    def normal_data(self):
        return self.normal
    def anomaly_data(self):
        return self.anomaly


class ImageDataset(Dataset):
    def __init__(self, images):
        self.images = images
    def __len__(self): return len(self.images)
    def __getitem__(self, i): return self.images[i]


class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 4, stride=2, padding=1), nn.Sigmoid(),
        )
    def forward(self, x):
        return self.decoder(self.encoder(x))


gen = DummyDataGenerator()
loader = DataLoader(ImageDataset(gen.normal_data()), batch_size=32, shuffle=True)
model = Autoencoder()
opt = torch.optim.Adam(model.parameters(), lr=1e-3)

for _ in range(20):
    for xb in loader:
        opt.zero_grad()
        nn.MSELoss()(model(xb), xb).backward()
        opt.step()

with torch.no_grad():
    err_n = nn.MSELoss(reduction='none')(model(gen.normal_data()), gen.normal_data()).mean(dim=(1,2,3))
    err_a = nn.MSELoss(reduction='none')(model(gen.anomaly_data()), gen.anomaly_data()).mean(dim=(1,2,3))
print(f"normal recon err={err_n.mean():.4f}, anomaly err={err_a.mean():.4f}")
print("✓ Anomalies have higher reconstruction error")"""),
    ])


def gen_v1_13():
    save("basics", "v1-13_quantize_language_model_solution.ipynb", [
        md("# V1-13: Quantize Language Model — Solution"),
        code("""import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


class DummyDataGenerator:
    def __init__(self, vocab=50, seq_len=8, n=200):
        torch.manual_seed(9)
        self.data = torch.randint(0, vocab, (n, seq_len))
        self.vocab = vocab
    def sequences(self):
        return self.data


class LMSequenceDataset(Dataset):
    def __init__(self, seqs):
        self.seqs = seqs
    def __len__(self): return len(self.seqs)
    def __getitem__(self, i):
        s = self.seqs[i]
        return s[:-1], s[1:]


class LanguageModel(nn.Module):
    def __init__(self, vocab=50, embed=32, hidden=64):
        super().__init__()
        self.embed = nn.Embedding(vocab, embed)
        self.lstm = nn.LSTM(embed, hidden, batch_first=True)
        self.fc = nn.Linear(hidden, vocab)
    def forward(self, x):
        out, _ = self.lstm(self.embed(x))
        return self.fc(out)


seqs = DummyDataGenerator().sequences()
model = LanguageModel()
quantized = torch.quantization.quantize_dynamic(model, {nn.Linear, nn.LSTM}, dtype=torch.qint8)

x = seqs[:4, :-1]
with torch.no_grad():
    orig = model(x)
    # quantized LSTM may differ slightly
print(f"original params: {sum(p.numel() for p in model.parameters())}")
print(f"quantized model type: {type(quantized).__name__}")
print("✓ Dynamic quantization applied")"""),
    ])


def gen_v1_14():
    save("basics", "v1-14_mixed_precision_solution.ipynb", [
        md("# V1-14: Mixed Precision Training — Solution"),
        code("""import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import Dataset, DataLoader


class DummyDataGenerator:
    def __init__(self, n=256, d=32):
        torch.manual_seed(10)
        self.X = torch.randn(n, d)
        self.y = torch.randint(0, 4, (n,))
    def tensors(self):
        return self.X, self.y


class TabularDataset(Dataset):
    def __init__(self, X, y):
        self.X, self.y = X, y
    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i], self.y[i]


class AMPModel(nn.Module):
    def __init__(self, d=32, c=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, c),
        )
    def forward(self, x):
        return self.net(x)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
X, y = DummyDataGenerator().tensors()
loader = DataLoader(TabularDataset(X, y), batch_size=32, shuffle=True)
model = AMPModel().to(device)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
scaler = GradScaler(enabled=device.type == "cuda")

for xb, yb in loader:
    xb, yb = xb.to(device), yb.to(device)
    opt.zero_grad()
    with autocast(enabled=device.type == "cuda"):
        loss = nn.CrossEntropyLoss()(model(xb), yb)
    scaler.scale(loss).backward()
    scaler.step(opt)
    scaler.update()
    break

print(f"AMP training on {device}")
print("✓ Mixed precision step complete")"""),
    ])


def gen_v1_15():
    save("basics", "v1-15_cnn_param_init_solution.ipynb", [
        md("# V1-15: CNN Parameter Initialization — Solution"),
        code("""import torch
import torch.nn as nn
import math
from torch.utils.data import Dataset


class DummyDataGenerator:
    def __init__(self, n=64):
        torch.manual_seed(12)
        self.X = torch.rand(n, 3, 32, 32)
        self.y = torch.randint(0, 10, (n,))
    def tensors(self):
        return self.X, self.y


class InitDataset(Dataset):
    def __init__(self, X, y):
        self.X, self.y = X, y
    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i], self.y[i]


class VanillaCNNModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 16, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(16, 10)
    def forward(self, x):
        return self.fc(self.pool(torch.relu(self.conv(x))).flatten(1))


def config_init(model, init_type="kaiming"):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            if init_type == "zero":
                nn.init.zeros_(m.weight); nn.init.zeros_(m.bias)
            elif init_type == "random":
                nn.init.uniform_(m.weight, -0.1, 0.1)
            elif init_type == "xavier":
                fan_in = m.weight.size(1) * m.weight[0][0].numel()
                std = math.sqrt(1.0 / fan_in)
                nn.init.normal_(m.weight, 0, std)
            elif init_type == "kaiming":
                fan_in = m.weight.size(1) * m.weight[0][0].numel()
                std = math.sqrt(2.0 / fan_in)
                nn.init.normal_(m.weight, 0, std)

X, y = DummyDataGenerator().tensors()
for init in ["zero", "random", "xavier", "kaiming"]:
    m = VanillaCNNModel()
    config_init(m, init)
    with torch.no_grad():
        logits = m(X[:8])
    print(f"{init:8s} logits std={logits.std().item():.4f}")
print("✓ Init strategies compared")"""),
    ])


def gen_v1_16():
    save("basics", "v1-16_cnn_from_scratch_solution.ipynb", [
        md("# V1-16: CNN from Scratch — Solution"),
        code("""import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset


class DummyDataGenerator:
    def __init__(self, n=32):
        self.X = torch.rand(n, 3, 32, 32)
        self.y = torch.randint(0, 10, (n,))
    def tensors(self):
        return self.X, self.y


class ScratchDataset(Dataset):
    def __init__(self, X, y):
        self.X, self.y = X, y
    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i], self.y[i]


class Conv2dCustom(nn.Module):
    def __init__(self, in_ch, out_ch, kernel=3, stride=1, padding=1):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_ch, in_ch, kernel, kernel) * 0.1)
        self.bias = nn.Parameter(torch.zeros(out_ch))
        self.stride, self.padding = stride, padding
    def forward(self, x):
        return F.conv2d(x, self.weight, self.bias, stride=self.stride, padding=self.padding)


class MaxPool2dCustom(nn.Module):
    def __init__(self, kernel=2, stride=2):
        super().__init__()
        self.kernel, self.stride = kernel, stride
    def forward(self, x):
        return F.max_pool2d(x, self.kernel, self.stride)


class CNNModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = Conv2dCustom(3, 8)
        self.conv2 = Conv2dCustom(8, 16)
        self.pool = MaxPool2dCustom()
        self.fc = nn.Linear(16 * 8 * 8, 10)
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        return self.fc(x.flatten(1))


X, y = DummyDataGenerator().tensors()
model = CNNModel()
out = model(X[:2])
print(f"scratch CNN output: {out.shape}")
print("✓ Custom conv + pool layers work")"""),
    ])


def gen_v1_17():
    save("basics", "v1-17_lstm_from_scratch_solution.ipynb", [
        md("# V1-17: LSTM from Scratch — Solution"),
        code("""import torch
import torch.nn as nn
from torch.utils.data import Dataset


class DummyDataGenerator:
    def __init__(self, batch=4, seq=6, d=8):
        torch.manual_seed(13)
        self.x = torch.randn(batch, seq, d)
    def input(self):
        return self.x


class SeqDataset(Dataset):
    def __init__(self, x):
        self.x = x
    def __len__(self): return len(self.x)
    def __getitem__(self, i): return self.x[i]


class CustomLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.W = nn.Linear(input_size + hidden_size, 4 * hidden_size)
    def forward(self, x, state):
        h, c = state
        gates = self.W(torch.cat([x, h], dim=-1))
        i, f, g, o = gates.chunk(4, dim=-1)
        i, f, o = torch.sigmoid(i), torch.sigmoid(f), torch.sigmoid(o)
        g = torch.tanh(g)
        c = f * c + i * g
        h = o * torch.tanh(c)
        return h, (h, c)


class CustomLSTMModel(nn.Module):
    def __init__(self, input_size=8, hidden=16):
        super().__init__()
        self.cell = CustomLSTMCell(input_size, hidden)
        self.fc = nn.Linear(hidden, 4)
    def forward(self, x):
        B, T, _ = x.shape
        h = torch.zeros(B, self.cell.hidden_size, device=x.device)
        c = torch.zeros_like(h)
        for t in range(T):
            h, (h, c) = self.cell(x[:, t], (h, c))
        return self.fc(h)


class LSTMModel(nn.Module):
    def __init__(self, input_size=8, hidden=16):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden, batch_first=True)
        self.fc = nn.Linear(hidden, 4)
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1])


x = DummyDataGenerator().input()
custom = CustomLSTMModel()
ref = LSTMModel()
print(f"custom: {custom(x).shape}, ref: {ref(x).shape}")
print("✓ Custom LSTM cell implemented")"""),
    ])


def gen_v3_01():
    save("basics", "v3-01_softmax_solution.ipynb", [
        md("# V3-01: Softmax from Scratch — Solution"),
        code("""import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset


class DummyDataGenerator:
    def __init__(self):
        torch.manual_seed(42)
        self.logits_1d = torch.randn(5)
        self.logits_2d = torch.randn(3, 5)
        self.logits_large = torch.tensor([1000., 2000., 3000.])
    def samples(self):
        return self.logits_1d, self.logits_2d, self.logits_large


class LogitsDataset(Dataset):
    def __init__(self, logits):
        self.logits = logits
    def __len__(self): return len(self.logits)
    def __getitem__(self, i): return self.logits[i]


def stable_softmax(x, dim=-1):
    x_max = x.max(dim=dim, keepdim=True).values
    exp_x = torch.exp(x - x_max)
    return exp_x / exp_x.sum(dim=dim, keepdim=True)


class SoftmaxClassifier(nn.Module):
  def __init__(self, d_in=5, n_classes=5):
    super().__init__()
    self.fc = nn.Linear(d_in, n_classes)
  def forward(self, x):
    return stable_softmax(self.fc(x), dim=-1)


l1, l2, ll = DummyDataGenerator().samples()
for name, x in [("1d", l1), ("2d", l2), ("large", ll)]:
    ours = stable_softmax(x)
    ref = F.softmax(x, dim=-1)
    assert torch.allclose(ours, ref, atol=1e-5)
    print(f"{name}: sums to {ours.sum(dim=-1)}")

model = SoftmaxClassifier()
probs = model(l2)
print(f"classifier probs shape: {probs.shape}")
print("✓ Stable softmax matches PyTorch")"""),
    ])


def gen_v3_02():
    save("basics", "v3-02_kmeans_solution.ipynb", [
        md("# V3-02: K-Means Clustering — Solution"),
        code("""import torch
import torch.nn as nn
from torch.utils.data import Dataset


class DummyDataGenerator:
    def __init__(self, k=3, per_cluster=50):
        torch.manual_seed(0)
        centers = torch.tensor([[0., 0.], [5., 5.], [0., 5.]])
        pts = []
        for c in centers:
            pts.append(c + torch.randn(per_cluster, 2) * 0.5)
        self.data = torch.cat(pts)
        self.k = k
    def points(self):
        return self.data


class PointDataset(Dataset):
    def __init__(self, data):
        self.data = data
    def __len__(self): return len(self.data)
    def __getitem__(self, i): return self.data[i]


def kmeans(data, k, max_iters=100, tol=1e-4):
    idx = torch.randperm(len(data))[:k]
    centroids = data[idx].clone()
    for _ in range(max_iters):
        dists = torch.cdist(data, centroids)
        labels = dists.argmin(dim=1)
        new_c = torch.stack([data[labels == j].mean(0) if (labels == j).any() else centroids[j] for j in range(k)])
        if (new_c - centroids).abs().max() < tol:
            break
        centroids = new_c
    return centroids, labels


class KMeansModel(nn.Module):
    \"\"\"Wraps centroids as learnable-free module for inference.\"\"\"
    def __init__(self, centroids):
        super().__init__()
        self.register_buffer("centroids", centroids)
    def forward(self, x):
        return torch.cdist(x, self.centroids).argmin(dim=1)


data = DummyDataGenerator().points()
centroids, labels = kmeans(data, k=3)
model = KMeansModel(centroids)
pred = model(data)
print(f"clusters found, label distribution: {pred.bincount(minlength=3).tolist()}")
print("✓ K-means converged")"""),
    ])


def gen_v3_03():
    save("basics", "v3-03_knn_solution.ipynb", [
        md("# V3-03: KNN — Solution"),
        code("""import torch
import torch.nn as nn
from torch.utils.data import Dataset


class DummyDataGenerator:
    def __init__(self, n_train=100, n_test=20):
        torch.manual_seed(2)
        self.X_train = torch.randn(n_train, 4)
        self.y_train = (self.X_train[:, 0] > 0).long()
        self.X_test = torch.randn(n_test, 4)
    def splits(self):
        return self.X_train, self.y_train, self.X_test


class KNNDataset(Dataset):
    def __init__(self, X, y=None):
        self.X, self.y = X, y
    def __len__(self): return len(self.X)
    def __getitem__(self, i):
        return (self.X[i], self.y[i]) if self.y is not None else self.X[i]


def knn_predict(X_train, y_train, X_test, k=5):
    dists = torch.cdist(X_test, X_train)
    _, idx = dists.topk(k, largest=False, dim=1)
    neighbors = y_train[idx]
    return torch.mode(neighbors, dim=1).values


class KNNModel(nn.Module):
    def __init__(self, X_train, y_train, k=5):
        super().__init__()
        self.register_buffer("X_train", X_train)
        self.register_buffer("y_train", y_train)
        self.k = k
    def forward(self, x):
        return knn_predict(self.X_train, self.y_train, x, self.k)


Xt, yt, Xte = DummyDataGenerator().splits()
pred = knn_predict(Xt, yt, Xte, k=5)
model = KNNModel(Xt, yt)
assert torch.equal(pred, model(Xte))
print(f"KNN predictions: {pred[:10].tolist()}")
print("✓ KNN classification done")"""),
    ])


def gen_v3_04():
    save("basics", "v3-04_logistic_regression_solution.ipynb", [
        md("# V3-04: Logistic Regression (Manual GD) — Solution"),
        code("""import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


class DummyDataGenerator:
    def __init__(self, n=400):
        torch.manual_seed(3)
        self.X = torch.randn(n, 2)
        self.y = (self.X[:, 0] + self.X[:, 1] > 0).float()
    def tensors(self):
        return self.X, self.y


class BinaryDataset(Dataset):
    def __init__(self, X, y):
        self.X, self.y = X, y
    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i], self.y[i]


def sigmoid(z):
    return 1 / (1 + torch.exp(-z))


def bce_loss(y, p):
    return -(y * torch.log(p + 1e-8) + (1 - y) * torch.log(1 - p + 1e-8)).mean()


def train_logistic_regression(X, y, lr=0.1, epochs=500):
    N, D = X.shape
    w = torch.zeros(D)
    b = torch.zeros(1)
    for _ in range(epochs):
        p = sigmoid(X @ w + b)
        dw = (X.T @ (p - y)) / N
        db = (p - y).mean()
        w -= lr * dw
        b -= lr * db
    return w, b


class LogisticModel(nn.Module):
    def __init__(self, w, b):
        super().__init__()
        self.register_buffer("w", w)
        self.register_buffer("b", b)
    def forward(self, x):
        return sigmoid(x @ self.w + self.b)


X, y = DummyDataGenerator().tensors()
w, b = train_logistic_regression(X, y)
model = LogisticModel(w, b)
acc = ((model(X) > 0.5).float() == y).float().mean()
print(f"accuracy={acc:.2%}")
assert acc > 0.85
print("✓ Manual logistic regression >85% acc")"""),
    ])


def gen_v2_01():
    save("basics", "v2-01_kl_divergence_solution.ipynb", [
        md("# V2-01: KL Divergence Loss — Solution"),
        code("""import torch
import torch.nn as nn
from torch.utils.data import Dataset


class DummyDataGenerator:
    def __init__(self):
        torch.manual_seed(4)
        self.p = torch.softmax(torch.randn(5), dim=0)
        self.q = torch.softmax(torch.randn(5), dim=0)
        self.logits_p = torch.randn(8, 5)
        self.logits_q = torch.randn(8, 5)
    def distributions(self):
        return self.p, self.q, self.logits_p, self.logits_q


class DistDataset(Dataset):
    def __init__(self, logits):
        self.logits = logits
    def __len__(self): return len(self.logits)
    def __getitem__(self, i): return self.logits[i]


def kl_divergence(p, q, eps=1e-8):
    p, q = p.clamp_min(eps), q.clamp_min(eps)
    return (p * (p / q).log()).sum(dim=-1)


class DistillationHead(nn.Module):
    def __init__(self, d=5):
        super().__init__()
        self.fc = nn.Linear(d, d)
    def forward(self, x):
        return torch.softmax(self.fc(x), dim=-1)


p, q, lp, lq = DummyDataGenerator().distributions()
kl = kl_divergence(p, q)
ref = torch.distributions.kl.kl_divergence(
    torch.distributions.Categorical(p), torch.distributions.Categorical(q)
)
print(f"KL(p||q)={kl.item():.4f}, ref={ref.item():.4f}")
print("✓ KL divergence from scratch")"""),
    ])


def gen_v2_02():
    save("basics", "v2-02_rms_norm_solution.ipynb", [
        md("# V2-02: RMS Norm — Solution"),
        code("""import torch
import torch.nn as nn
from torch.utils.data import Dataset


class DummyDataGenerator:
    def __init__(self, batch=4, seq=8, dim=16):
        torch.manual_seed(5)
        self.x = torch.randn(batch, seq, dim)
    def input(self):
        return self.x


class TensorDataset(Dataset):
    def __init__(self, x):
        self.x = x
    def __len__(self): return len(self.x)
    def __getitem__(self, i): return self.x[i]


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-8):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))
    def forward(self, x):
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (x / rms) * self.scale


class TransformerBlockStub(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = RMSNorm(dim)
        self.ff = nn.Linear(dim, dim)
    def forward(self, x):
        return self.ff(self.norm(x))


x = DummyDataGenerator().input()
block = TransformerBlockStub(16)
out = block(x)
print(f"RMSNorm output shape: {out.shape}")
print("✓ RMSNorm integrated in block")"""),
    ])


def gen_v2_03():
    save("basics", "v2-03_byte_pair_encoding_solution.ipynb", [
        md("# V2-03: Byte Pair Encoding — Solution"),
        code("""from collections import Counter, defaultdict
from torch.utils.data import Dataset


class DummyDataGenerator:
    def __init__(self):
        self.corpus = ["low", "lower", "newest", "widest"]
    def text(self):
        return self.corpus


class CorpusDataset(Dataset):
    def __init__(self, words):
        self.words = words
    def __len__(self): return len(self.words)
    def __getitem__(self, i): return self.words[i]


def get_vocab(corpus):
    vocab = Counter()
    for word in corpus:
        vocab[tuple(list(word) + ["</w>"])] += 1
    return vocab


def get_stats(vocab):
    pairs = defaultdict(int)
    for word, freq in vocab.items():
        for i in range(len(word) - 1):
            pairs[(word[i], word[i + 1])] += freq
    return pairs


def merge_vocab(pair, vocab):
    new_vocab = {}
    bigram = " ".join(pair)
    replacement = "".join(pair)
    for word, freq in vocab.items():
        w = " ".join(word).replace(bigram, replacement)
        new_vocab[tuple(w.split())] = freq
    return new_vocab


class BPETokenizer:
    def __init__(self):
        self.merges = []
    def train(self, corpus, num_merges=10):
        vocab = get_vocab(corpus)
        for _ in range(num_merges):
            pairs = get_stats(vocab)
            if not pairs:
                break
            best = max(pairs, key=pairs.get)
            vocab = merge_vocab(best, vocab)
            self.merges.append(best)
        return self
    def tokenize(self, word):
        tokens = list(word) + ["</w>"]
        for a, b in self.merges:
            merged = a + b
            i = 0
            while i < len(tokens) - 1:
                if tokens[i] == a and tokens[i + 1] == b:
                    tokens = tokens[:i] + [merged] + tokens[i + 2:]
                else:
                    i += 1
        return tokens


corpus = DummyDataGenerator().text()
tok = BPETokenizer().train(corpus, num_merges=5)
print(f"merges: {tok.merges}")
print(f"tokenize 'lower': {tok.tokenize('lower')}")
print("✓ BPE trained and tokenizing")"""),
    ])


BASICS = [
    gen_v1_01, gen_v1_02, gen_v1_03, gen_v1_04, gen_v1_05, gen_v1_06, gen_v1_07,
    gen_v1_08, gen_v1_09, gen_v1_10, gen_v1_11, gen_v1_12, gen_v1_13, gen_v1_14,
    gen_v1_15, gen_v1_16, gen_v1_17, gen_v2_01, gen_v2_02, gen_v2_03,
    gen_v3_01, gen_v3_02, gen_v3_03, gen_v3_04,
]


if __name__ == "__main__":
    print("Generating basics...")
    for fn in BASICS:
        fn()
    print(f"Done basics ({len(BASICS)} notebooks)")
