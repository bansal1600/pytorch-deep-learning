# PyTorch Deep Learning

A curated PyTorch learning repo: beginner tutorials, 75 [TorchLeet](https://torch-leet.vercel.app/) interview solutions, and Python source files you can open in **Google Colab**.

## Repository layout

```
pytorch-deep-learning/
├── tutorials/              # Step-by-step learning path (start here)
├── notebooks/torchleet/    # Colab-ready .ipynb (75 solutions, topic-wise)
├── src/torchleet/          # Same solutions as .py (edit here, sync to notebooks)
├── src/utils/              # Shared helpers
├── scripts/                # Convert & regenerate tooling
└── docs/COLAB.md           # How to use in Google Colab
```

## Quick start

### 1. Learning path (beginners)

| Order | Folder | Topic |
|-------|--------|-------|
| 1 | `tutorials/01_tensors_and_basics/` | Tensors |
| 2 | `tutorials/02_autograd/` | Autograd |
| 3 | `tutorials/03_nn_modules/` | `nn.Module` |
| 4 | `tutorials/04_training/` | Training pipeline |
| 5 | `tutorials/05_projects/` | MNIST & Fashion-MNIST projects |

### 2. TorchLeet interview prep (75 problems)

Solutions are grouped by topic under `notebooks/torchleet/` and `src/torchleet/`:

| Topic folder | Count | Examples |
|--------------|-------|----------|
| `01_pytorch_fundamentals` | 11 | Linear regression, custom Dataset, autograd |
| `02_classical_ml` | 5 | Softmax, k-means, KNN, logistic regression |
| `03_computer_vision` | 14 | CNN, ViT, GradCAM, CLIP |
| `04_sequence_models` | 4 | RNN, LSTM, seq2seq, Mamba |
| `05_transformers_attention` | 10 | Attention, RoPE, SmolLM, BPE |
| `06_alignment_rlhf` | 6 | LoRA, DPO, PPO, GRPO |
| `07_llm_inference` | 12 | KV cache, top-p, speculative decoding |
| `08_generative_models` | 4 | GAN, VAE, DDPM, DDIM |
| `09_graph_neural_networks` | 2 | GNN, GCN |
| `10_systems_distributed` | 7 | FSDP, flash attention, ring attention |

Each solution includes:
- `DummyDataGenerator` — synthetic data (no downloads)
- `Dataset` — PyTorch dataset class
- `Model` — main `nn.Module`

## Python vs notebooks

| Format | Location | Best for |
|--------|----------|----------|
| `.py` | `src/torchleet/` | Editing in IDE, version control, `%` cell format |
| `.ipynb` | `notebooks/torchleet/` | Google Colab, Jupyter |

Convert between them:

```bash
pip install jupytext
# notebooks -> Python
python scripts/convert_notebooks.py --to-py
# Python -> notebooks (Colab)
python scripts/convert_notebooks.py --to-ipynb
```

See **[docs/COLAB.md](docs/COLAB.md)** for opening in Google Colab.

## Requirements

```bash
pip install -r requirements.txt
```

## Regenerate TorchLeet notebooks from scratch

```bash
python scripts/run_all_generators.py   # create initial notebooks
python scripts/reorganize_repo.py      # topic folders + .py export
```

## Open in Colab (one click)

Replace the path with any notebook under `notebooks/torchleet/`:

```
https://colab.research.google.com/github/bansal1600/pytorch-deep-learning/blob/main/notebooks/torchleet/01_pytorch_fundamentals/v1-01_linear_regression_solution.ipynb
```

Full index: [notebooks/torchleet/README.md](notebooks/torchleet/README.md)
