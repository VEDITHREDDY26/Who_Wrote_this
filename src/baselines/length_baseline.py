"""Length-only baseline: how far can text length alone go?

Logistic regression on (word count, character count, mean word length), with the shared class
weights. Trained on train, scored on val (never test); writes
results/tuning/length_baseline.json.

Usage: python -m src.baselines.length_baseline [--config PATH]
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

from src.config import get_path, load_config, use_config
from src.data.class_weights import get_class_weights
from src.data.splits import load_split
from src.evaluate import evaluate
from src.utils.io import atomic_write_json
from src.utils.logging import setup_logging

logger = logging.getLogger(__name__)

FEATURE_NAMES = ("word_count", "char_count", "mean_word_length")
OUTPUT_SUBDIR = "tuning"
OUTPUT_FILE = "length_baseline.json"


def length_features(texts: pd.Series) -> np.ndarray:
    """(n, 3) array: word count, character count, mean word length (0 for empty texts)."""
    words = texts.str.split().str.len().fillna(0).to_numpy(dtype=np.float64)
    chars = texts.str.len().to_numpy(dtype=np.float64)
    letters = texts.str.replace(r"\s+", "", regex=True).str.len().to_numpy(dtype=np.float64)
    mean_len = np.divide(letters, words, out=np.zeros_like(letters), where=words > 0)
    return np.column_stack([words, chars, mean_len])


def fit_length_baseline(train: pd.DataFrame, class_weights: Sequence[float],
                        params: Mapping[str, Any], seed: int) -> Pipeline:
    """Fit StandardScaler + LogisticRegression on the length features of `train`."""
    model = make_pipeline(StandardScaler(), LogisticRegression(
        C=params["C"], max_iter=params["max_iter"], random_state=seed,
        class_weight={i: float(w) for i, w in enumerate(class_weights)}))
    model.fit(length_features(train["text"]), train["label_id"].to_numpy())
    return model


def run(config_path: str | None = None) -> dict[str, Any]:
    """Train on train, evaluate on val, save and return the result."""
    use_config(config_path)
    cfg = load_config()
    columns = ["id", "text", "label_id"]
    train, val = load_split("train", columns=columns), load_split("val", columns=columns)
    weights = get_class_weights(train, cfg["classes"], cfg["class_weights"]["method"],
                                get_path("results_dir") / cfg["class_weights"]["file"])
    start = time.monotonic()
    model = fit_length_baseline(train, weights.tolist(), cfg["baselines"]["length"],
                                cfg["seed"])
    train_seconds = time.monotonic() - start
    y_prob = model.predict_proba(length_features(val["text"]))
    metrics = evaluate(val["label_id"].to_numpy(), y_prob.argmax(axis=1), y_prob,
                       cfg["classes"], cfg["training"]["human_label"])
    lr = model[-1]
    result = {
        "model": "length_baseline", "split": "val", "features": list(FEATURE_NAMES),
        "params": dict(cfg["baselines"]["length"]), "val_macro_f1": metrics["macro_f1"],
        "metrics": metrics, "train_time_seconds": train_seconds,
        "num_parameters": int(lr.coef_.size + lr.intercept_.size),
    }
    out = get_path("results_dir") / OUTPUT_SUBDIR / OUTPUT_FILE
    atomic_write_json(out, result)
    logger.info("Saved %s", out)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default=None)
    args = parser.parse_args(argv)
    setup_logging()
    result = run(args.config)
    m = result["metrics"]
    print(f"\nLength-only baseline (val, {m['n_samples']:,} rows)")
    print(f"  macro-F1   {m['macro_f1']:.4f}")
    print(f"  accuracy   {m['accuracy']:.4f}")
    print(f"  human FPR  {m['human_fpr']:.4f}")
    for name, pc in m["per_class"].items():
        print(f"  {name:8s} P {pc['precision']:.3f}  R {pc['recall']:.3f}  F1 {pc['f1']:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
