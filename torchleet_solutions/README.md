# TorchLeet Solutions

Original PyTorch solutions for all **75** problems from [TorchLeet](https://torch-leet.vercel.app/), written from scratch (not copied from the official repo).

Each notebook includes:
- **DummyDataGenerator** — synthetic data for quick local runs (no downloads required)
- **Dataset** — `torch.utils.data.Dataset` wrapper
- **Model** — main `nn.Module` (or algorithm class for non-training tasks)

## Structure

| Folder | Count | Topics |
|--------|-------|--------|
| `basics/` | 24 | Linear regression, CNN, RNN, classical ML, BPE, softmax, k-means, etc. |
| `llm_path/` | 20 | Attention, RoPE, LoRA, DPO, PPO, KV cache, MoE, inference engine |
| `advanced/` | 31 | GAN, VAE, DDPM, Mamba, ViT/MAE, FSDP, flash attention, GNN, RAG |

## Naming

`{id}_{slug}_solution.ipynb` — e.g. `v3-12_kv_cache_solution.ipynb`

## Requirements

```bash
pip install torch torchvision
```

Some notebooks optionally use CUDA; all run on CPU.

## Regenerate

```bash
python scripts/run_all_generators.py
```
