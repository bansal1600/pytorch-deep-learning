#!/usr/bin/env python3
"""Generate notebooks/torchleet/README.md with Colab links."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NB_ROOT = ROOT / "notebooks" / "torchleet"
REPO = "bansal1600/pytorch-deep-learning"
BRANCH = "main"

TOPIC_TITLES = {
    "01_pytorch_fundamentals": "01 — PyTorch Fundamentals",
    "02_classical_ml": "02 — Classical ML",
    "03_computer_vision": "03 — Computer Vision",
    "04_sequence_models": "04 — Sequence Models",
    "05_transformers_attention": "05 — Transformers & Attention",
    "06_alignment_rlhf": "06 — Alignment & RLHF",
    "07_llm_inference": "07 — LLM Inference & Decoding",
    "08_generative_models": "08 — Generative Models",
    "09_graph_neural_networks": "09 — Graph Neural Networks",
    "10_systems_distributed": "10 — Systems & Distributed",
}


def colab_url(rel_path: str) -> str:
    return f"https://colab.research.google.com/github/{REPO}/blob/{BRANCH}/{rel_path}"


def main() -> None:
    lines = [
        "# TorchLeet Solutions — Topic Index\n",
        "75 original PyTorch interview solutions from [TorchLeet](https://torch-leet.vercel.app/).\n",
        "**Python source:** `src/torchleet/<topic>/`  \n",
        "**Colab notebooks:** this folder  \n",
        "**How to use Colab:** [docs/COLAB.md](../../docs/COLAB.md)\n",
    ]
    total = 0
    for topic_dir in sorted(NB_ROOT.iterdir()):
        if not topic_dir.is_dir():
            continue
        title = TOPIC_TITLES.get(topic_dir.name, topic_dir.name)
        notebooks = sorted(topic_dir.glob("*.ipynb"))
        lines.append(f"\n## {title} ({len(notebooks)})\n")
        lines.append("| # | Notebook | Colab | Python |\n")
        lines.append("|---|----------|-------|--------|\n")
        for i, nb in enumerate(notebooks, 1):
            rel_nb = nb.relative_to(ROOT).as_posix()
            rel_py = rel_nb.replace("notebooks/torchleet/", "src/torchleet/").replace("_solution.ipynb", ".py")
            name = nb.stem.replace("_solution", "").replace("_", " ")
            lines.append(
                f"| {i} | `{nb.name}` | [Open]({colab_url(rel_nb)}) | [`{Path(rel_py).name}`](../../{rel_py}) |\n"
            )
            total += 1
    lines.append(f"\n---\n\n**Total: {total} notebooks**\n")
    out = NB_ROOT / "README.md"
    out.write_text("".join(lines))
    print(f"Wrote {out} ({total} entries)")


if __name__ == "__main__":
    main()
