#!/usr/bin/env python3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from generate_torchleet_notebooks import BASICS
import generate_torchleet_part2 as p2
import generate_torchleet_part3 as p3

if __name__ == "__main__":
    print("=== TorchLeet Solution Notebook Generator ===\n")
    for fn in BASICS:
        fn()
    for fn in p2.LLM_PATH:
        fn()
    for fn in p3.ADVANCED:
        fn()
    total = len(BASICS) + len(p2.LLM_PATH) + len(p3.ADVANCED)
    print(f"\n=== Generated {total} notebooks ===")
