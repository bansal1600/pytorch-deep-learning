# Using this repo in Google Colab

Three ways to run the code in Colab — pick what fits your workflow.

---

## Method 1: Open a notebook directly (easiest)

1. Go to [Google Colab](https://colab.research.google.com/)
2. **File → Open notebook → GitHub**
3. Paste repo URL: `https://github.com/bansal1600/pytorch-deep-learning`
4. Navigate to e.g. `notebooks/torchleet/01_pytorch_fundamentals/v1-01_linear_regression_solution.ipynb`

**Or use a direct link** (swap the path for any notebook):

```
https://colab.research.google.com/github/bansal1600/pytorch-deep-learning/blob/main/notebooks/torchleet/01_pytorch_fundamentals/v1-01_linear_regression_solution.ipynb
```

Colab installs PyTorch automatically. Run the first cell if you need extras:

```python
!pip install -q torchvision
```

---

## Method 2: Convert `.py` → notebook in Colab

The `src/torchleet/` files use **Jupytext percent format** (`# %%` cells). Colab can run them as scripts or you can convert:

### Option A — Run as a script

```python
# In a Colab cell:
!git clone https://github.com/bansal1600/pytorch-deep-learning.git
%cd pytorch-deep-learning
!python src/torchleet/01_pytorch_fundamentals/v1-01_linear_regression.py
```

### Option B — Convert to notebook inside Colab

```python
!pip install -q jupytext
!git clone https://github.com/bansal1600/pytorch-deep-learning.git
%cd pytorch-deep-learning
!python scripts/convert_notebooks.py --to-ipynb src/torchleet/01_pytorch_fundamentals/v1-01_linear_regression.py
```

Then open the generated file under `notebooks/torchleet/...`.

### Option C — Upload a `.py` file

1. In Colab: **File → Upload notebook** won't work for `.py`
2. Instead: **File → Upload to session storage**, then:

```python
!pip install -q jupytext
!jupytext --to ipynb your_file.py
# Open the generated .ipynb from the file browser (left panel)
```

---

## Method 3: Convert locally, then upload to Colab

On your machine:

```bash
git clone https://github.com/bansal1600/pytorch-deep-learning.git
cd pytorch-deep-learning
pip install jupytext

# One file
python scripts/convert_notebooks.py --to-ipynb src/torchleet/05_transformers_attention/v2-06_attention_from_scratch.py

# All Python files -> notebooks
python scripts/convert_notebooks.py --to-ipynb
```

Upload the `.ipynb` from `notebooks/torchleet/` to Colab via **File → Upload notebook**.

---

## Sync workflow (recommended for contributors)

1. **Edit** `src/torchleet/<topic>/<problem>.py` in your IDE
2. **Convert** to notebook: `python scripts/convert_notebooks.py --to-ipynb path/to/file.py`
3. **Commit** both `.py` and `.ipynb`
4. **Open in Colab** via GitHub link (Method 1)

Reverse (edited notebook in Colab → Python):

```bash
python scripts/convert_notebooks.py --to-py notebooks/torchleet/.../file.ipynb
```

---

## Colab tips

| Tip | Detail |
|-----|--------|
| GPU | **Runtime → Change runtime type → T4 GPU** for CNN/LLM notebooks |
| Clone fresh | `!git clone` or `!git pull` at the start of a session |
| Paths | After clone, `%cd pytorch-deep-learning` before running scripts |
| Tutorials | Beginner path is under `tutorials/` (same open-in-Colab flow) |

---

## Topic folders (where to find problems)

| Folder | What's inside |
|--------|----------------|
| `01_pytorch_fundamentals` | Datasets, losses, training basics |
| `02_classical_ml` | Softmax, k-means, KNN |
| `03_computer_vision` | CNN, ViT, XAI |
| `04_sequence_models` | RNN, LSTM, Mamba |
| `05_transformers_attention` | Attention, RoPE, BPE, SmolLM |
| `06_alignment_rlhf` | LoRA, DPO, PPO, GRPO |
| `07_llm_inference` | KV cache, sampling, MoE |
| `08_generative_models` | GAN, VAE, diffusion |
| `09_graph_neural_networks` | GNN, GCN |
| `10_systems_distributed` | FSDP, flash attention |

See the full list with Colab links in [notebooks/torchleet/README.md](../notebooks/torchleet/README.md).
