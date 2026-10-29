"""The single evaluation function shared by all four models (and the baselines).

`evaluate()` computes the metrics; `save_results()` writes them, a confusion-matrix figure and
the raw probabilities (for later calibration) to results_dir. The test split is evaluated once,
at the very end, by src/final_test.py.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import confusion_matrix, log_loss, precision_recall_fscore_support

from src.utils.io import atomic_open, atomic_write_json

logger = logging.getLogger(__name__)

REQUIRED_EXTRA = ("train_time_seconds", "num_parameters")
PROB_SUM_TOLERANCE = 1e-3  # rows of y_prob must sum to 1 within this (float32 rounding)
NAME_PATTERN = re.compile(r"^[A-Za-z0-9_.-]+$")
METRIC_KEYS = ("n_samples", "accuracy", "macro_f1", "human_fpr", "log_loss", "per_class",
               "confusion_matrix", "class_names")


def _check_inputs(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray,
                  n_classes: int) -> None:
    if y_true.ndim != 1 or y_pred.ndim != 1:
        raise ValueError(f"y_true and y_pred must be 1-D, got shapes {y_true.shape} and "
                         f"{y_pred.shape}")
    if len(y_true) == 0:
        raise ValueError("cannot evaluate zero samples")
    if len(y_true) != len(y_pred):
        raise ValueError(f"y_true has {len(y_true)} rows but y_pred has {len(y_pred)}")
    if y_prob.shape != (len(y_true), n_classes):
        raise ValueError(f"y_prob must have shape ({len(y_true)}, {n_classes}), "
                         f"got {y_prob.shape}")
    for name, arr in (("y_true", y_true), ("y_pred", y_pred)):
        if arr.min() < 0 or arr.max() >= n_classes:
            raise ValueError(f"{name} has label ids outside 0..{n_classes - 1}")
    if (y_prob < 0).any() or not np.allclose(y_prob.sum(axis=1), 1.0, atol=PROB_SUM_TOLERANCE):
        raise ValueError("y_prob rows must be non-negative and sum to 1 (pass softmax outputs)")


def evaluate(y_true: Sequence[int], y_pred: Sequence[int], y_prob: np.ndarray,
             class_names: Sequence[str], human_label: str = "Human") -> dict[str, Any]:
    """Compute every reported metric.

    Returns a dict with: n_samples, accuracy, macro_f1, human_fpr (share of true `human_label`
    rows predicted as any other class; None if there are none), log_loss, per_class
    {name: precision, recall, f1, support}, confusion_matrix (rows = true, cols = predicted)
    and class_names.
    """
    y_true_arr = np.asarray(y_true, dtype=np.int64)
    y_pred_arr = np.asarray(y_pred, dtype=np.int64)
    y_prob_arr = np.asarray(y_prob, dtype=np.float64)
    names = list(class_names)
    _check_inputs(y_true_arr, y_pred_arr, y_prob_arr, len(names))
    y_prob_arr = y_prob_arr / y_prob_arr.sum(axis=1, keepdims=True)  # remove rounding drift
    if human_label not in names:
        raise ValueError(f"human_label {human_label!r} is not one of the classes {names}")

    labels = np.arange(len(names))
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true_arr, y_pred_arr, labels=labels, average=None, zero_division=0)

    human = names.index(human_label)
    true_human = y_true_arr == human
    if true_human.any():
        human_fpr: float | None = float((y_pred_arr[true_human] != human).mean())
    else:
        logger.warning("No true %s rows; human_fpr is undefined (None)", human_label)
        human_fpr = None

    return {
        "n_samples": int(len(y_true_arr)),
        "accuracy": float((y_true_arr == y_pred_arr).mean()),
        "macro_f1": float(f1.mean()),
        "human_fpr": human_fpr,
        "log_loss": float(log_loss(y_true_arr, y_prob_arr, labels=labels)),
        "per_class": {
            name: {"precision": float(p), "recall": float(r), "f1": float(f), "support": int(s)}
            for name, p, r, f, s in zip(names, precision, recall, f1, support, strict=True)},
        "confusion_matrix": confusion_matrix(y_true_arr, y_pred_arr, labels=labels).tolist(),
        "class_names": names,
    }


def plot_confusion_matrix(matrix: np.ndarray, class_names: Sequence[str], title: str,
                          path: Path) -> None:
    """Save counts and row-normalized confusion matrices side by side (rows = true class)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    counts = np.asarray(matrix, dtype=np.int64)
    row_sums = counts.sum(axis=1, keepdims=True)
    normalized = np.divide(counts, row_sums, out=np.zeros(counts.shape), where=row_sums > 0)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), dpi=150)
    panels = ((counts, "Counts", "{:,}"), (normalized, "Row-normalized (recall)", "{:.0%}"))
    for ax, (values, subtitle, fmt) in zip(axes, panels, strict=True):
        ax.imshow(values, cmap="Blues", vmin=0)
        threshold = values.max() / 2 if values.max() > 0 else 1
        for i in range(values.shape[0]):
            for j in range(values.shape[1]):
                ax.text(j, i, fmt.format(values[i, j]), ha="center", va="center", fontsize=8,
                        color="white" if values[i, j] > threshold else "#0b0b0b")
        ax.set_xticks(range(len(class_names)), class_names, rotation=30, ha="right")
        ax.set_yticks(range(len(class_names)), class_names)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(subtitle, loc="left", fontsize=11)
    fig.suptitle(title, x=0.01, ha="left", fontsize=12)
    fig.tight_layout()
    with atomic_open(path) as f:
        fig.savefig(f, format="png")
    plt.close(fig)


def save_results(metrics: Mapping[str, Any], model_name: str, split: str,
                 extra: Mapping[str, Any], y_true: Sequence[int], y_prob: np.ndarray,
                 results_dir: Path) -> dict[str, Path]:
    """Write <model>_<split>_metrics.json, _confusion.png and _probs.npz into `results_dir`.

    Args:
        metrics: Output of `evaluate()`.
        model_name: e.g. "mlp" (letters, digits, "_", "-", "." only).
        split: e.g. "val" or "test".
        extra: Must hold train_time_seconds and num_parameters; may hold more.
        y_true: True label ids, saved with the probabilities for calibration.
        y_prob: (n, n_classes) predicted probabilities.
        results_dir: Output folder.
    """
    for name, value in (("model_name", model_name), ("split", split)):
        if not NAME_PATTERN.match(value):
            raise ValueError(f"{name} {value!r} may only contain letters, digits, '_', '-', '.'")
    missing = [k for k in METRIC_KEYS if k not in metrics]
    if missing:
        raise KeyError(f"metrics is missing {missing}; pass the output of evaluate()")
    missing = [k for k in REQUIRED_EXTRA if k not in extra]
    if missing:
        raise KeyError(f"extra must include {list(REQUIRED_EXTRA)}; missing {missing}")
    y_true_arr = np.asarray(y_true, dtype=np.int64)
    y_prob_arr = np.asarray(y_prob, dtype=np.float32)
    if len(y_true_arr) != metrics["n_samples"] or y_prob_arr.shape[0] != len(y_true_arr):
        raise ValueError(f"y_true ({len(y_true_arr)}) / y_prob ({y_prob_arr.shape[0]}) rows do "
                         f"not match metrics n_samples ({metrics['n_samples']})")

    stem = f"{model_name}_{split}"
    paths = {"metrics": results_dir / f"{stem}_metrics.json",
             "confusion": results_dir / f"{stem}_confusion.png",
             "probs": results_dir / f"{stem}_probs.npz"}
    atomic_write_json(paths["metrics"], {
        "model": model_name, "split": split,
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        **metrics, "extra": dict(extra)})
    plot_confusion_matrix(np.asarray(metrics["confusion_matrix"]), metrics["class_names"],
                          f"{model_name} — {split} (macro-F1 {metrics['macro_f1']:.3f})",
                          paths["confusion"])
    with atomic_open(paths["probs"]) as f:
        np.savez_compressed(f, y_true=y_true_arr, y_prob=y_prob_arr,
                            class_names=np.asarray(metrics["class_names"]))
    logger.info("Saved %s results to %s", stem, results_dir)
    return paths
