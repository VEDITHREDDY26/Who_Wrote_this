"""Seeding and RNG-state capture for reproducible, resumable training.

`get_rng_state()` returns only plain Python types and tensors, so it can be stored in a
checkpoint and read back with `torch.load(..., weights_only=True)`.
"""

from __future__ import annotations

import logging
import os
import random
from typing import Any

import numpy as np
import torch

logger = logging.getLogger(__name__)

RNG_KEYS = ("python", "numpy", "torch", "cuda", "mps")


def _mps_available() -> bool:
    return bool(getattr(torch.backends, "mps", None)) and torch.backends.mps.is_available()


def seed_everything(seed: int, deterministic: bool = False) -> None:
    """Seed Python, NumPy and torch (CPU, every CUDA device, and MPS).

    Args:
        seed: The seed, normally `cfg["seed"]`.
        deterministic: Also force deterministic cuDNN / torch algorithms (slower; ops with
            no deterministic version warn instead of failing).
    """
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise TypeError(f"seed must be an int, got {type(seed).__name__}: {seed!r}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if _mps_available():
        torch.mps.manual_seed(seed)
    # Only affects subprocesses started later (e.g. DataLoader workers), not this process.
    os.environ["PYTHONHASHSEED"] = str(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)


def get_rng_state() -> dict[str, Any]:
    """Capture every RNG state so a resumed run continues the exact same random stream."""
    name, keys, pos, has_gauss, cached_gaussian = np.random.get_state(legacy=True)
    return {
        "python": random.getstate(),
        "numpy": (name, keys.tolist(), int(pos), int(has_gauss), float(cached_gaussian)),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "mps": torch.mps.get_rng_state() if _mps_available() else None,
    }


def set_rng_state(state: dict[str, Any]) -> None:
    """Restore RNG states captured by `get_rng_state()`.

    A CUDA or MPS state is skipped with a warning when that device is not present, so a
    checkpoint from a GPU machine can still be resumed elsewhere.
    """
    missing = [k for k in RNG_KEYS if k not in state]
    if missing:
        raise KeyError(f"RNG state is missing keys {missing}; expected {list(RNG_KEYS)}")

    version, internal, gauss = state["python"]
    random.setstate((version, tuple(internal), gauss))
    name, keys, pos, has_gauss, cached_gaussian = state["numpy"]
    np.random.set_state((name, np.asarray(keys, dtype=np.uint32), pos, has_gauss,
                         cached_gaussian))
    torch.set_rng_state(state["torch"])

    if state["cuda"] is not None:
        if torch.cuda.is_available():
            torch.cuda.set_rng_state_all(state["cuda"])
        else:
            logger.warning("Checkpoint has a CUDA RNG state but CUDA is unavailable; skipped")
    if state["mps"] is not None:
        if _mps_available():
            torch.mps.set_rng_state(state["mps"])
        else:
            logger.warning("Checkpoint has an MPS RNG state but MPS is unavailable; skipped")
