"""Class weights, computed once from the full train split and shared by every model.

The weights are saved to <results_dir>/class_weights.json together with the train fingerprint.
Later calls load that file; if the train split has changed, they refuse instead of silently
mixing weights from different data.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.utils.class_weight import compute_class_weight

from src.data.splits import fingerprint_ids
from src.utils.io import atomic_write_json, safe_load_json

logger = logging.getLogger(__name__)

SUPPORTED_METHODS = ("balanced",)


def compute_class_weights(labels: np.ndarray, num_classes: int, method: str) -> np.ndarray:
    """Compute per-class weights with sklearn ("balanced": n_samples / (n_classes * count))."""
    if method not in SUPPORTED_METHODS:
        raise ValueError(f"class weight method {method!r} not supported; use {SUPPORTED_METHODS}")
    labels = np.asarray(labels)
    missing = sorted(set(range(num_classes)) - set(labels.tolist()))
    if missing:
        raise ValueError(f"train labels have no examples of class ids {missing}")
    return compute_class_weight(method, classes=np.arange(num_classes), y=labels)


def get_class_weights(train_df: pd.DataFrame, class_names: Sequence[str], method: str,
                      path: Path) -> torch.Tensor:
    """Load the saved class weights, or compute them from `train_df` and save them.

    Args:
        train_df: The FULL train split (columns `id`, `label_id`), never a subset.
        class_names: Class names in label-id order.
        method: Weighting method (see SUPPORTED_METHODS).
        path: Where the JSON lives, normally <results_dir>/class_weights.json.
    """
    fingerprint = fingerprint_ids(train_df["id"])
    saved = safe_load_json(path, default=None)
    if saved is not None:
        if saved.get("train_fingerprint") != fingerprint:
            raise ValueError(
                f"{path} was computed from a different train split (fingerprint "
                f"{saved.get('train_fingerprint')}, current {fingerprint}). Delete it only if "
                "the splits were rebuilt on purpose.")
        if saved.get("classes") != list(class_names) or saved.get("method") != method:
            raise ValueError(f"{path} has classes {saved.get('classes')} / method "
                             f"{saved.get('method')!r}; expected {list(class_names)} / {method!r}")
        return torch.tensor(saved["weights"], dtype=torch.float32)

    labels = train_df["label_id"].to_numpy()
    weights = compute_class_weights(labels, len(class_names), method)
    counts = np.bincount(labels, minlength=len(class_names))
    atomic_write_json(path, {
        "method": method,
        "classes": list(class_names),
        "weights": [float(w) for w in weights],
        "by_class": {c: float(w) for c, w in zip(class_names, weights, strict=True)},
        "class_counts": {c: int(n) for c, n in zip(class_names, counts, strict=True)},
        "train_fingerprint": fingerprint,
    })
    logger.info("Computed class weights from %d train rows and saved %s", len(labels), path)
    return torch.tensor(weights, dtype=torch.float32)
