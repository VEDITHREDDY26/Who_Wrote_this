"""Environment sanity check. Run from the project root: python scripts/check_env.py"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas
import sklearn
import torch
import transformers

from src.config import get_device


def main() -> None:
    """Print interpreter and library versions and run a small matmul on the chosen device."""
    print(f"Python        {sys.version.split()[0]}  ({sys.executable})")
    print(f"torch         {torch.__version__}")
    device = get_device()
    print(f"device        {device}")
    x = torch.randn(256, 256, device=device)
    print(f"device matmul ok, sum={float((x @ x).sum()):.2f}")
    print(f"transformers  {transformers.__version__}")
    print(f"sklearn       {sklearn.__version__} (import ok)")
    print(f"pandas        {pandas.__version__} (import ok)")


if __name__ == "__main__":
    main()
