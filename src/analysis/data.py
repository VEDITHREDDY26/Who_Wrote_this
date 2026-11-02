"""Load the saved results the figures are drawn from, and the derived numbers behind them.

Everything comes from files written by src.data.build_splits, src.train, src.tune and
src.final_test. The test split itself is never loaded: test-set analysis uses the per-text
predictions that src.final_test saved. Text lengths come from the val split only.
"""

from __future__ import annotations

import ast
import json
import logging
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from src.data.splits import load_split
from src.tune import TUNING_SUBDIR
from src.utils.io import safe_load_json

logger = logging.getLogger(__name__)

SUMMARY_FILE = "final_test_summary.json"
SPLIT_STATS_FILE = "split_stats.json"
BASELINE = "length_baseline"
DEFAULT_RUN = "default"
PROB_PREFIX = "prob_"
# A logged step of the Hugging Face Trainer, as written by src.models.transformer.LogCallback.
LOG_STEP = re.compile(r"step (\d+)/\d+\s+epoch [\d.]+\s+(\{.*\})\s*$")


class ResultsError(FileNotFoundError):
    """A results file the figures need is missing or does not match the others."""


@dataclass(frozen=True)
class ModelResults:
    """Everything saved for one model scored by src.final_test."""

    name: str
    run: str | None
    summary: dict[str, Any]
    test_metrics: dict[str, Any]
    y_true: np.ndarray
    y_pred: np.ndarray
    y_prob: np.ndarray
    domains: np.ndarray
    val_true: np.ndarray | None
    val_prob: np.ndarray | None
    history: list[dict[str, Any]]
    train_log: Path | None


@dataclass(frozen=True)
class AnalysisData:
    """All inputs of the figures: the tested models (best first) and dataset statistics."""

    class_names: list[str]
    human_label: str
    models: list[ModelResults]
    baseline: ModelResults | None
    tuning: pd.DataFrame
    split_stats: dict[str, Any] | None
    train_meta: pd.DataFrame | None
    val_words: np.ndarray | None

    @property
    def human_id(self) -> int:
        """Label id of the human class."""
        return self.class_names.index(self.human_label)


def result_label(model: str, run: str | None) -> str:
    """File prefix of a run's results: the model name, plus the run name unless "default"."""
    return model if run in (None, DEFAULT_RUN) else f"{model}-{run}"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Records of a JSON-lines file ([] if it does not exist)."""
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def load_summary(results_dir: Path) -> list[dict[str, Any]]:
    """Rows of final_test_summary.json, best test macro-F1 first."""
    summary = safe_load_json(results_dir / SUMMARY_FILE, default=None)
    if not summary or not summary.get("models"):
        raise ResultsError(f"{results_dir / SUMMARY_FILE} not found or empty; run "
                           "`python -m src.final_test` first")
    return sorted(summary["models"], key=lambda r: -r["test_macro_f1"])


def load_test_predictions(results_dir: Path, model: str,
                          class_names: Sequence[str]) -> pd.DataFrame:
    """The per-text test predictions saved by src.final_test, checked against the classes."""
    path = results_dir / f"{model}_test_predictions.parquet"
    if not path.exists():
        raise ResultsError(f"{path} not found; it is written by src.final_test")
    df = pd.read_parquet(path)
    expected = {"label_id", "pred_id", "domain"} | {PROB_PREFIX + c for c in class_names}
    missing = sorted(expected - set(df.columns))
    if missing:
        raise ResultsError(f"{path} lacks columns {missing}")
    return df


def load_val_probs(results_dir: Path, model: str,
                   run: str | None) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Saved val labels and probabilities of the tested run (None, None if not saved)."""
    path = results_dir / f"{result_label(model, run)}_val_probs.npz"
    if not path.exists():
        # The length baseline never saves val probabilities; any other model should.
        log = logger.info if model == BASELINE else logger.warning
        log("%s not found; %s is left out of the val-based figures", path, model)
        return None, None
    saved = np.load(path)
    return saved["y_true"], saved["y_prob"]


def load_model_results(results_dir: Path, row: Mapping[str, Any],
                       class_names: Sequence[str]) -> ModelResults:
    """Load one tested model's saved test predictions, val probabilities and history."""
    model, run = row["model"], row.get("run_name")
    metrics = safe_load_json(results_dir / f"{model}_test_metrics.json", default=None)
    if metrics is None:
        raise ResultsError(f"{results_dir / f'{model}_test_metrics.json'} not found")
    preds = load_test_predictions(results_dir, model, class_names)
    val_true, val_prob = load_val_probs(results_dir, model, run)
    run_dir = results_dir / model / (run or DEFAULT_RUN)
    train_log = run_dir / "train.log"
    return ModelResults(
        name=model, run=run, summary=dict(row), test_metrics=metrics,
        y_true=preds["label_id"].to_numpy(), y_pred=preds["pred_id"].to_numpy(),
        y_prob=preds[[PROB_PREFIX + c for c in class_names]].to_numpy(),
        domains=preds["domain"].to_numpy(), val_true=val_true, val_prob=val_prob,
        history=read_jsonl(run_dir / "history.jsonl"),
        train_log=train_log if train_log.exists() else None)


def load_tuning_runs(results_dir: Path, models: Sequence[str]) -> pd.DataFrame:
    """Every tuning run's val macro-F1, from all the places runs are recorded.

    Sources, first one wins per (model, run): the val metrics files of src.train
    (<model>[-<run>]_val_metrics.json, full-data runs only), the sweep summaries
    (tuning/<model>_tuning.json) and hand-collected tables (tuning/*.csv with the columns
    model, run, val_macro_f1).
    """
    rows: list[dict[str, Any]] = []
    for model in models:
        for path in sorted(results_dir.glob(f"{model}*_val_metrics.json")):
            stem = path.name.removesuffix("_val_metrics.json")
            if stem != model and not stem.startswith(f"{model}-"):
                continue
            metrics = safe_load_json(path)
            if metrics.get("extra", {}).get("subset") is not None:
                continue
            rows.append({"model": model, "run": stem.removeprefix(model).lstrip("-")
                         or DEFAULT_RUN, "val_macro_f1": metrics["macro_f1"],
                         "source": path.name})
    tuning_dir = results_dir / TUNING_SUBDIR
    for model in models:
        sweep = safe_load_json(tuning_dir / f"{model}_tuning.json", default=None) or {}
        rows += [{"model": model, "run": r["run_name"], "val_macro_f1": r["val_macro_f1"],
                  "source": f"{model}_tuning.json"} for r in sweep.get("runs", [])]
    for path in sorted(tuning_dir.glob("*.csv")):
        table = pd.read_csv(path)
        if not {"model", "run", "val_macro_f1"} <= set(table.columns):
            logger.warning("%s lacks the columns model, run, val_macro_f1; skipped", path)
            continue
        rows += [{"model": r.model, "run": r.run, "val_macro_f1": float(r.val_macro_f1),
                  "source": path.name} for r in table.itertuples() if r.model in models]
    columns = ["model", "run", "val_macro_f1", "source"]
    runs = pd.DataFrame(rows, columns=columns).drop_duplicates(["model", "run"])
    return runs.reset_index(drop=True)


def parse_train_log(path: Path) -> pd.DataFrame:
    """Training loss per logged step and val results per epoch from a Hugging Face train.log."""
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        match = LOG_STEP.search(line)
        if not match:
            continue
        values = ast.literal_eval(match.group(2))
        if "loss" in values:
            rows.append({"step": int(match.group(1)), "kind": "train", "loss": values["loss"],
                         "epoch": values.get("epoch")})
        elif "eval_loss" in values:
            rows.append({"step": int(match.group(1)), "kind": "eval",
                         "loss": values["eval_loss"], "epoch": values.get("epoch"),
                         "macro_f1": values.get("eval_macro_f1")})
    return pd.DataFrame(rows)


def load_val_words(splits_dir: Path | None, val_models: Sequence[ModelResults]) -> np.ndarray:
    """Words per val text, in the val split's order (checked against the saved val labels)."""
    val = load_split("val", splits_dir, columns=["text", "label_id"])
    labels = val["label_id"].to_numpy()
    for m in val_models:
        if m.val_true is not None and not np.array_equal(m.val_true, labels):
            raise ResultsError(f"{m.name}: saved val labels are not in the val split's order")
    return val["text"].str.split().str.len().to_numpy()


def load_analysis_data(results_dir: Path, class_names: Sequence[str], human_label: str,
                       splits_dir: Path | None = None) -> AnalysisData:
    """Load everything the figures need (train and val splits only; never the test split)."""
    rows = load_summary(results_dir)
    loaded = [load_model_results(results_dir, r, class_names) for r in rows]
    models = [m for m in loaded if m.name != BASELINE]
    baseline = next((m for m in loaded if m.name == BASELINE), None)
    train_meta = load_split("train", splits_dir, columns=["label_id", "domain"])
    return AnalysisData(
        class_names=list(class_names), human_label=human_label, models=models,
        baseline=baseline, tuning=load_tuning_runs(results_dir, [m.name for m in models]),
        split_stats=safe_load_json(results_dir / SPLIT_STATS_FILE, default=None),
        train_meta=train_meta, val_words=load_val_words(splits_dir, models))


def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int) -> float:
    """Expected calibration error of the top-class confidence (equal-width bins)."""
    curve = reliability_curve(y_true, y_prob, n_bins)
    weights = curve["count"] / curve["count"].sum()
    return float((weights * (curve["accuracy"] - curve["confidence"]).abs()).sum())


def reliability_curve(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int) -> pd.DataFrame:
    """Mean confidence, accuracy and count per confidence bin (empty bins dropped)."""
    confidence = y_prob.max(axis=1)
    correct = y_prob.argmax(axis=1) == y_true
    bins = np.minimum((confidence * n_bins).astype(int), n_bins - 1)
    frame = pd.DataFrame({"bin": bins, "confidence": confidence, "correct": correct})
    grouped = frame.groupby("bin").agg(confidence=("confidence", "mean"),
                                       accuracy=("correct", "mean"),
                                       count=("correct", "size"))
    return grouped.reset_index()


def macro_f1_by_group(y_true: np.ndarray, y_pred: np.ndarray, groups: np.ndarray,
                      n_classes: int) -> pd.Series:
    """Macro-F1 within each group (e.g. domain), over all classes."""
    labels = list(range(n_classes))
    return pd.Series({g: f1_score(y_true[groups == g], y_pred[groups == g], labels=labels,
                                  average="macro", zero_division=0)
                      for g in np.unique(groups)})


def per_class_f1_by_group(y_true: np.ndarray, y_pred: np.ndarray, groups: np.ndarray,
                          class_names: Sequence[str]) -> pd.DataFrame:
    """F1 of each class (rows) within each group (columns)."""
    labels = list(range(len(class_names)))
    return pd.DataFrame({g: f1_score(y_true[groups == g], y_pred[groups == g], labels=labels,
                                     average=None, zero_division=0)
                         for g in np.unique(groups)}, index=list(class_names))


def length_buckets(words: np.ndarray, edges: Sequence[int]) -> np.ndarray:
    """Bucket label per text ("100-149", ..., "400+"); texts below the first edge get ""."""
    labels = bucket_labels(edges)
    index = np.searchsorted(np.asarray(edges), words, side="right") - 1
    return np.array([labels[i] if i >= 0 else "" for i in index], dtype=object)


def bucket_labels(edges: Sequence[int]) -> list[str]:
    """Labels of the length buckets defined by `edges` (the last bucket is open-ended)."""
    if list(edges) != sorted(set(edges)) or len(edges) < 2:
        raise ValueError(f"length_bins must be at least two increasing numbers, got {edges}")
    return [f"{lo}-{hi - 1}" for lo, hi in zip(edges[:-1], edges[1:], strict=True)] + \
        [f"{edges[-1]}+"]
