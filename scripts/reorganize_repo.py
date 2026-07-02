#!/usr/bin/env python3
"""Reorganize repo into topic-wise folders and convert notebooks <-> .py."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# TorchLeet notebook filename -> topic folder
TOPIC_MAP: dict[str, str] = {
    # 01 — PyTorch fundamentals
    "v1-01_linear_regression_solution.ipynb": "01_pytorch_fundamentals",
    "v1-02_custom_dataset_dataloader_solution.ipynb": "01_pytorch_fundamentals",
    "v1-03_custom_activation_solution.ipynb": "01_pytorch_fundamentals",
    "v1-04_huber_loss_solution.ipynb": "01_pytorch_fundamentals",
    "v1-05_deep_neural_network_solution.ipynb": "01_pytorch_fundamentals",
    "v1-06_tensorboard_solution.ipynb": "01_pytorch_fundamentals",
    "v1-07_save_load_model_solution.ipynb": "01_pytorch_fundamentals",
    "v1-11_benchmarking_solution.ipynb": "01_pytorch_fundamentals",
    "v1-14_mixed_precision_solution.ipynb": "01_pytorch_fundamentals",
    "v1-15_cnn_param_init_solution.ipynb": "01_pytorch_fundamentals",
    "v1-22_custom_autograd_silu_solution.ipynb": "01_pytorch_fundamentals",
    # 02 — Classical ML
    "v3-01_softmax_solution.ipynb": "02_classical_ml",
    "v3-02_kmeans_solution.ipynb": "02_classical_ml",
    "v3-03_knn_solution.ipynb": "02_classical_ml",
    "v3-04_logistic_regression_solution.ipynb": "02_classical_ml",
    "v2-01_kl_divergence_solution.ipynb": "02_classical_ml",
    # 03 — Computer vision
    "v1-08_cnn_cifar10_solution.ipynb": "03_computer_vision",
    "v1-10_data_augmentation_solution.ipynb": "03_computer_vision",
    "v1-12_autoencoder_solution.ipynb": "03_computer_vision",
    "v1-16_cnn_from_scratch_solution.ipynb": "03_computer_vision",
    "v1-18_alexnet_solution.ipynb": "03_computer_vision",
    "v1-21_3d_cnn_segmentation_solution.ipynb": "03_computer_vision",
    "v1-23_neural_style_transfer_solution.ipynb": "03_computer_vision",
    "v1-31_gradcam_xai_solution.ipynb": "03_computer_vision",
    "v1-32_linear_probe_clip_solution.ipynb": "03_computer_vision",
    "v1-33_embedding_visualization_solution.ipynb": "03_computer_vision",
    "v3-05_contrastive_clip_solution.ipynb": "03_computer_vision",
    "v3-06_2d_positional_embeddings_solution.ipynb": "03_computer_vision",
    "v3-23_vit_mae_solution.ipynb": "03_computer_vision",
    "v3-29_knowledge_distillation_solution.ipynb": "03_computer_vision",
    # 04 — Sequence models
    "v1-09_rnn_solution.ipynb": "04_sequence_models",
    "v1-17_lstm_from_scratch_solution.ipynb": "04_sequence_models",
    "v1-28_seq2seq_attention_solution.ipynb": "04_sequence_models",
    "v3-22_mamba_solution.ipynb": "04_sequence_models",
    # 05 — Transformers & attention
    "v2-02_rms_norm_solution.ipynb": "05_transformers_attention",
    "v2-03_byte_pair_encoding_solution.ipynb": "05_transformers_attention",
    "v2-06_attention_from_scratch_solution.ipynb": "05_transformers_attention",
    "v2-07_multi_head_attention_solution.ipynb": "05_transformers_attention",
    "v2-08_grouped_query_attention_solution.ipynb": "05_transformers_attention",
    "v2-10_sinusoidal_embeddings_solution.ipynb": "05_transformers_attention",
    "v2-11_rope_embeddings_solution.ipynb": "05_transformers_attention",
    "v2-12_smollm_from_scratch_solution.ipynb": "05_transformers_attention",
    "v1-26_transformer_solution.ipynb": "05_transformers_attention",
    "v3-13_sliding_window_attention_solution.ipynb": "05_transformers_attention",
    # 06 — Alignment & RLHF
    "v2-20_sft_smollm_solution.ipynb": "06_alignment_rlhf",
    "v3-11_lora_solution.ipynb": "06_alignment_rlhf",
    "v3-14_dpo_loss_solution.ipynb": "06_alignment_rlhf",
    "v3-15_ppo_rlhf_solution.ipynb": "06_alignment_rlhf",
    "v3-27_grpo_solution.ipynb": "06_alignment_rlhf",
    "v3-16_gradient_checkpointing_solution.ipynb": "06_alignment_rlhf",
    # 07 — LLM inference & decoding
    "v3-07_top_p_sampling_solution.ipynb": "07_llm_inference",
    "v3-08_top_k_sampling_solution.ipynb": "07_llm_inference",
    "v3-09_beam_search_solution.ipynb": "07_llm_inference",
    "v3-10_temperature_sampling_solution.ipynb": "07_llm_inference",
    "v3-12_kv_cache_solution.ipynb": "07_llm_inference",
    "v3-17_mixture_of_experts_solution.ipynb": "07_llm_inference",
    "v3-18_speculative_decoding_solution.ipynb": "07_llm_inference",
    "v3-19_continuous_batching_solution.ipynb": "07_llm_inference",
    "v3-28_inference_engine_solution.ipynb": "07_llm_inference",
    "v1-13_quantize_language_model_solution.ipynb": "07_llm_inference",
    "v2-04_rag_search_solution.ipynb": "07_llm_inference",
    "v2-13_gptq_quantization_solution.ipynb": "07_llm_inference",
    # 08 — Generative models
    "v1-27_gan_solution.ipynb": "08_generative_models",
    "v1-35_vae_solution.ipynb": "08_generative_models",
    "v3-20_ddpm_solution.ipynb": "08_generative_models",
    "v3-21_ddim_cfg_solution.ipynb": "08_generative_models",
    # 09 — Graph neural networks
    "v1-24_gnn_solution.ipynb": "09_graph_neural_networks",
    "v1-25_gcn_solution.ipynb": "09_graph_neural_networks",
    # 10 — Systems & distributed
    "v1-19_dense_retrieval_solution.ipynb": "10_systems_distributed",
    "v1-29_distributed_training_solution.ipynb": "10_systems_distributed",
    "v1-30_sparse_tensors_solution.ipynb": "10_systems_distributed",
    "v3-24_triton_fused_softmax_solution.ipynb": "10_systems_distributed",
    "v3-25_flash_attention_solution.ipynb": "10_systems_distributed",
    "v3-26_fsdp_solution.ipynb": "10_systems_distributed",
    "v3-30_ring_attention_solution.ipynb": "10_systems_distributed",
}

TUTORIAL_MOVES = {
    "tensors_in_pytorch.ipynb": "tutorials/01_tensors_and_basics/tensors_in_pytorch.ipynb",
    "pytorch_autograd.ipynb": "tutorials/02_autograd/pytorch_autograd.ipynb",
    "pytorch_nn_module.ipynb": "tutorials/03_nn_modules/pytorch_nn_module.ipynb",
    "pytorch_training_pipeline.ipynb": "tutorials/04_training/pytorch_training_pipeline.ipynb",
    "ANN_PyTorch_MNIST.ipynb": "tutorials/05_projects/mnist_ann.ipynb",
    "cnn_fashion_mnist_pytorch_gpu.ipynb": "tutorials/05_projects/fashion_mnist_cnn.ipynb",
    "ann_fashion_mnist_pytorch_gpu_optimized_optuna.ipynb": "tutorials/05_projects/fashion_mnist_ann_optuna.ipynb",
}

DELETE_FILES = [
    "dataset_and_dataloader_demo.ipynb",  # covered by torchleet v1-02
    "torchleet_solutions/README.md",
]


def notebook_to_py(nb_path: Path, py_path: Path) -> None:
    py_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [sys.executable, "-m", "jupytext", "--to", "py:percent", str(nb_path), "-o", str(py_path)],
        check=True,
        capture_output=True,
    )


def py_to_notebook(py_path: Path, nb_path: Path) -> None:
    nb_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [sys.executable, "-m", "jupytext", "--to", "ipynb", str(py_path), "-o", str(nb_path)],
        check=True,
        capture_output=True,
    )


def collect_torchleet_notebooks() -> list[Path]:
    old = ROOT / "torchleet_solutions"
    found = []
    for sub in ("basics", "llm_path", "advanced"):
        d = old / sub
        if d.exists():
            found.extend(sorted(d.glob("*.ipynb")))
    return found


def reorganize() -> None:
    src_nb = ROOT / "src" / "torchleet"
    colab_nb = ROOT / "notebooks" / "torchleet"

    # Clean target dirs
    for base in (src_nb, colab_nb):
        if base.exists():
            shutil.rmtree(base)

    notebooks = collect_torchleet_notebooks()
    missing = set(TOPIC_MAP) - {n.name for n in notebooks}
    if missing:
        raise RuntimeError(f"Missing notebooks in map: {missing}")

    for nb_path in notebooks:
        topic = TOPIC_MAP[nb_path.name]
        stem = nb_path.stem.replace("_solution", "")
        py_dest = src_nb / topic / f"{stem}.py"
        nb_dest = colab_nb / topic / nb_path.name
        nb_dest.parent.mkdir(parents=True, exist_ok=True)

        shutil.copy2(nb_path, nb_dest)
        notebook_to_py(nb_dest, py_dest)
        print(f"  {topic}/{stem}")

    # Move tutorials
    for src_name, dest_rel in TUTORIAL_MOVES.items():
        src = ROOT / src_name
        dest = ROOT / dest_rel
        if src.exists():
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src), str(dest))
            print(f"  tutorial -> {dest_rel}")

    # Move modeling_utilities
    util_src = ROOT / "modeling_utilities.py"
    if util_src.exists():
        util_dest = ROOT / "src" / "utils" / "modeling_utilities.py"
        util_dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(util_src), str(util_dest))

    # Remove old structure
    old = ROOT / "torchleet_solutions"
    if old.exists():
        shutil.rmtree(old)

    for f in DELETE_FILES:
        p = ROOT / f
        if p.exists():
            p.unlink()

    print(f"\nReorganized {len(notebooks)} TorchLeet solutions into {len(set(TOPIC_MAP.values()))} topics.")


if __name__ == "__main__":
    print("Reorganizing repository...\n")
    reorganize()
