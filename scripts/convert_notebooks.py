#!/usr/bin/env python3
"""
Convert between .ipynb and .py (Jupytext percent format).

Usage:
  # All notebooks -> Python source
  python scripts/convert_notebooks.py --to-py

  # All Python source -> Colab-ready notebooks
  python scripts/convert_notebooks.py --to-ipynb

  # Single file
  python scripts/convert_notebooks.py --to-py notebooks/torchleet/01_pytorch_fundamentals/v1-01_linear_regression_solution.ipynb
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def to_py(path: Path) -> None:
    path = path.resolve()
    out = ROOT / "src" / "torchleet" / path.relative_to((ROOT / "notebooks" / "torchleet").resolve())
    out = out.with_suffix(".py")
    out.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [sys.executable, "-m", "jupytext", "--to", "py:percent", str(path), "-o", str(out)],
        check=True,
    )
    print(f"  {path.name} -> {out.relative_to(ROOT)}")


def to_ipynb(path: Path) -> None:
    path = path.resolve()
    rel = path.relative_to((ROOT / "src" / "torchleet").resolve())
    out = ROOT / "notebooks" / "torchleet" / rel.with_suffix(".ipynb")
    out.parent.mkdir(parents=True, exist_ok=True)
    # Preserve _solution suffix in notebook name
    if not out.name.endswith("_solution.ipynb"):
        out = out.with_name(out.stem + "_solution.ipynb")
    subprocess.run(
        [sys.executable, "-m", "jupytext", "--to", "ipynb", str(path), "-o", str(out)],
        check=True,
    )
    print(f"  {path.name} -> {out.relative_to(ROOT)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert notebooks <-> Python")
    parser.add_argument("--to-py", action="store_true", help="Convert .ipynb to src/*.py")
    parser.add_argument("--to-ipynb", action="store_true", help="Convert src/*.py to notebooks/")
    parser.add_argument("files", nargs="*", help="Optional specific files")
    args = parser.parse_args()

    if args.to_py:
        files = [Path(f) for f in args.files] if args.files else list((ROOT / "notebooks" / "torchleet").rglob("*.ipynb"))
        for f in files:
            to_py(f)
    elif args.to_ipynb:
        files = [Path(f) for f in args.files] if args.files else list((ROOT / "src" / "torchleet").rglob("*.py"))
        for f in files:
            to_ipynb(f)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
