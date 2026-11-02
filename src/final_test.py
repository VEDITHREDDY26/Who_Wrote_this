"""One-time evaluation on the test split, at the very end of the project.

    python -m src.final_test --i-understand-this-uses-the-test-set [--config PATH]
        [--models mlp cnn lstm transformer length_baseline] [--run MODEL=RUN ...]

For each model this takes the run chosen on validation (results/tuning/<model>_tuning.json, or
--run MODEL=RUN) and checks that the run finished on the full current train split, that its
features still match, and that the reloaded model reproduces its saved val macro-F1. Only then
does it score the test split with the shared evaluate(). Outputs, per model:
- results/<model>_test_metrics.json (+ _confusion.png, _probs.npz), via save_results();
- results/<model>_test_predictions.parquet: id, domain, true and predicted label and the
  probabilities, so per-domain and error analysis never needs another pass over test;
and results/final_test_summary.json comparing every model scored so far.

The test set is used once: a model whose test metrics already exist is skipped, which also makes
the command safe to rerun after an interruption. To re-score a model, delete its
<model>_test_metrics.json deliberately.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from src.baselines.length_baseline import count_parameters, fit_length_baseline, length_features
from src.config import get_device, get_path, load_config, use_config
from src.data.class_weights import get_class_weights
from src.data.splits import fingerprint_ids, load_split
from src.evaluate import evaluate, save_results
from src.models import REGISTRY, build_model, get_model_spec
from src.models.transformer import (
    BEST_MODEL_DIR,
    RUN_CONFIG_FILE,
    TransformerParams,
    load_best_model,
    predict_proba_texts,
)
from src.train import SPLIT_COLUMNS, build_datasets, training_config
from src.trainer import BEST_FILE, predict_proba
from src.tune import TUNING_SUBDIR, metrics_file
from src.utils.io import atomic_open, atomic_write_json, safe_load_json
from src.utils.logging import setup_logging
from src.utils.seed import seed_everything

logger = logging.getLogger(__name__)

TEST_FLAG = "--i-understand-this-uses-the-test-set"
LENGTH_BASELINE = "length_baseline"
LENGTH_BASELINE_FILE = "length_baseline.json"  # written by src.baselines.length_baseline
DEFAULT_MODELS = ("mlp", "cnn", "lstm", "transformer", LENGTH_BASELINE)
TEST_COLUMNS = (*SPLIT_COLUMNS, "domain")
TEST = "test"
SUMMARY_FILE = "final_test_summary.json"
LOG_FILE = "final_test.log"
CARRIED_EXTRA = ("train_time_seconds", "num_parameters", "best_epoch", "epochs_completed")
USAGE_ERROR_EXIT_CODE = 2


class FinalTestError(RuntimeError):
    """A run is not fit to be scored on the test split (unfinished, subset, stale, ...)."""


def parse_runs(items: Sequence[str]) -> dict[str, str]:
    """Parse --run MODEL=RUN items into {model: run name}."""
    runs = {}
    for item in items:
        model, sep, run = item.partition("=")
        if not sep or not model or not run:
            raise ValueError(f"--run expects MODEL=RUN (e.g. mlp=h512x128-d50), got {item!r}")
        runs[model] = run
    return runs


def final_metrics_path(results_dir: Path, model: str) -> Path:
    """Where save_results() writes a model's test metrics (also the 'already done' marker)."""
    return results_dir / f"{model}_{TEST}_metrics.json"


def final_predictions_path(results_dir: Path, model: str) -> Path:
    """Per-row test predictions of one model."""
    return results_dir / f"{model}_{TEST}_predictions.parquet"


def choose_run(model: str, overrides: Mapping[str, str], results_dir: Path) -> str:
    """The run to test: --run if given, else the run chosen by the val tuning sweep."""
    if model in overrides:
        return overrides[model]
    path = results_dir / TUNING_SUBDIR / f"{model}_tuning.json"
    chosen = safe_load_json(path, default={}).get("chosen")
    if chosen is None:
        raise FinalTestError(f"{model}: no chosen run in {path}; run `python -m src.tune "
                             f"--model {model}` or pass --run {model}=<run name>")
    return chosen


def load_val_metrics(model: str, run: str, results_dir: Path) -> dict[str, Any]:
    """The run's saved val metrics; they only exist once training has finished."""
    path = metrics_file(results_dir, model, run)
    metrics = safe_load_json(path, default=None)
    if metrics is None:
        raise FinalTestError(f"{model}: {path} not found, so run {run!r} has not finished "
                             "training; finish it before testing")
    if metrics["extra"].get("subset") is not None:
        raise FinalTestError(f"{model}: run {run!r} was trained on a "
                             f"{metrics['extra']['subset']}-row subset; only full-data runs "
                             "may be tested")
    return metrics


def check_run_config(model: str, run_cfg: Mapping[str, Any], train_fingerprint: str,
                     source: Path) -> None:
    """The run must have trained on the full train split that exists now."""
    if run_cfg.get("subset") is not None:
        raise FinalTestError(f"{model}: {source} was trained on a {run_cfg['subset']}-row "
                             "subset; only full-data runs may be tested")
    if run_cfg.get("train_fingerprint") != train_fingerprint:
        raise FinalTestError(f"{model}: {source} was trained on train fingerprint "
                             f"{run_cfg.get('train_fingerprint')}, but the current train split "
                             f"is {train_fingerprint}; the splits changed since training")


def score_trainer_model(model: str, run: str, splits: Mapping[str, pd.DataFrame],
                        cfg: Mapping[str, Any]) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Val and test probabilities from a src.trainer run's best.pt (MLP, CNN, LSTM)."""
    path = get_path("checkpoint_dir") / model / run / BEST_FILE
    if not path.exists():
        raise FinalTestError(f"{model}: {path} not found")
    best = torch.load(path, map_location="cpu", weights_only=True)
    run_cfg = best["config"]["run"]
    check_run_config(model, run_cfg, fingerprint_ids(splits["train"]["id"]), path)
    data, input_info, dims = build_datasets(get_model_spec(model), splits, cfg)
    if input_info["cache_key"] != run_cfg["input"]["cache_key"]:
        raise FinalTestError(f"{model}: the feature settings changed since run {run!r} was "
                             f"trained (cache key {run_cfg['input']['cache_key']} -> "
                             f"{input_info['cache_key']})")
    net = build_model(model, run_cfg["model_params"], **dims)
    net.load_state_dict(best["model"])
    device = get_device()
    settings = training_config(cfg, model, None)
    net.to(device)
    probs = {name: predict_proba(net, data[name], device=device,
                                 batch_size=settings.eval_batch_size,
                                 use_amp=settings.amp and device.type == "cuda")
             for name in ("val", TEST)}
    return probs, {"checkpoint": str(path)}


def score_transformer(run: str, splits: Mapping[str, pd.DataFrame],
                      cfg: Mapping[str, Any]) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Val and test probabilities from a Transformer run's best_model/."""
    checkpoint_dir = get_path("checkpoint_dir") / "transformer" / run
    saved = safe_load_json(checkpoint_dir / RUN_CONFIG_FILE, default=None)
    if saved is None:
        raise FinalTestError(f"transformer: {checkpoint_dir / RUN_CONFIG_FILE} not found")
    run_cfg = saved["config"]["run"]
    check_run_config("transformer", run_cfg, fingerprint_ids(splits["train"]["id"]),
                     checkpoint_dir)
    hp = TransformerParams.from_config(run_cfg["model_params"])
    net, tokenizer = load_best_model(checkpoint_dir)
    settings = training_config(cfg, "transformer", None)
    device = get_device()
    probs = {name: predict_proba_texts(net, tokenizer, splits[name]["text"].tolist(),
                                       max_len=hp.max_len, batch_size=settings.eval_batch_size,
                                       device=device, use_amp=settings.amp)
             for name in ("val", TEST)}
    return probs, {"checkpoint": str(checkpoint_dir / BEST_MODEL_DIR),
                   "pretrained_name": hp.pretrained_name}


def score_length_baseline(splits: Mapping[str, pd.DataFrame], cfg: Mapping[str, Any],
                          class_weights: Sequence[float]
                          ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Refit the length-only baseline on train (deterministic) and score val and test."""
    start = time.monotonic()
    model = fit_length_baseline(splits["train"], class_weights, cfg["baselines"]["length"],
                                cfg["seed"])
    extra = {"train_time_seconds": time.monotonic() - start,
             "num_parameters": count_parameters(model)}
    probs = {name: model.predict_proba(length_features(splits[name]["text"]))
             for name in ("val", TEST)}
    return probs, extra


def check_val_reproduced(model: str, y_val: np.ndarray, val_prob: np.ndarray,
                         saved_macro_f1: float, cfg: Mapping[str, Any]) -> float:
    """Refuse to score test unless the reloaded model matches its saved val macro-F1."""
    metrics = evaluate(y_val, val_prob.argmax(axis=1), val_prob, cfg["classes"],
                       cfg["training"]["human_label"])
    tolerance = cfg["final_test"]["val_check_tolerance"]
    if abs(metrics["macro_f1"] - saved_macro_f1) > tolerance:
        raise FinalTestError(f"{model}: the reloaded model scores val macro-F1 "
                             f"{metrics['macro_f1']:.4f}, but {saved_macro_f1:.4f} was saved "
                             f"(tolerance {tolerance}); the checkpoint does not match its "
                             "results, so test was not scored")
    logger.info("%s: val macro-F1 reproduced (%.4f, saved %.4f)", model, metrics["macro_f1"],
                saved_macro_f1)
    return metrics["macro_f1"]


def save_predictions(test: pd.DataFrame, y_prob: np.ndarray, class_names: Sequence[str],
                     path: Path) -> None:
    """Write id, domain, true and predicted label ids and per-class probabilities."""
    out = pd.DataFrame({"id": test["id"].to_numpy(), "domain": test["domain"].to_numpy(),
                        "label_id": test["label_id"].to_numpy(),
                        "pred_id": y_prob.argmax(axis=1)})
    for i, name in enumerate(class_names):
        out[f"prob_{name}"] = y_prob[:, i].astype(np.float32)
    with atomic_open(path) as f:
        out.to_parquet(f, index=False)


def score_model(model: str, run: str | None, splits: Mapping[str, pd.DataFrame],
             cfg: Mapping[str, Any], class_weights: Sequence[float],
             results_dir: Path) -> dict[str, Any]:
    """Validate one model's run, then score it on test and save everything."""
    start = time.monotonic()
    if model == LENGTH_BASELINE:
        saved = safe_load_json(results_dir / TUNING_SUBDIR / LENGTH_BASELINE_FILE, default=None)
        if saved is None:
            raise FinalTestError(f"{LENGTH_BASELINE}: no val result yet; run "
                                 "`python -m src.baselines.length_baseline` first")
        saved_val_f1, extra = saved["val_macro_f1"], {}
        probs, scored_extra = score_length_baseline(splits, cfg, class_weights)
    else:
        val_metrics = load_val_metrics(model, run, results_dir)
        saved_val_f1 = val_metrics["macro_f1"]
        extra = {k: val_metrics["extra"].get(k) for k in CARRIED_EXTRA}
        if get_model_spec(model).input_kind == "hf":
            probs, scored_extra = score_transformer(run, splits, cfg)
        else:
            probs, scored_extra = score_trainer_model(model, run, splits, cfg)

    val_f1 = check_val_reproduced(model, splits["val"]["label_id"].to_numpy(), probs["val"],
                                  saved_val_f1, cfg)
    test, y_prob = splits[TEST], probs[TEST]
    y_true = test["label_id"].to_numpy()
    metrics = evaluate(y_true, y_prob.argmax(axis=1), y_prob, cfg["classes"],
                       cfg["training"]["human_label"])
    # Predictions first: the metrics file is written last and marks the model as done.
    save_predictions(test, y_prob, cfg["classes"], final_predictions_path(results_dir, model))
    save_results(metrics, model, TEST, extra | scored_extra | {
        "run_name": run, "val_macro_f1": saved_val_f1, "val_macro_f1_recomputed": val_f1,
        "scoring_seconds": time.monotonic() - start}, y_true, y_prob, results_dir)
    return metrics


def summarize(results_dir: Path, models: Sequence[str]) -> list[dict[str, Any]]:
    """One row per model with test results, best test macro-F1 first."""
    rows = []
    for model in models:
        m = safe_load_json(final_metrics_path(results_dir, model), default=None)
        if m is None:
            continue
        rows.append({"model": model, "run_name": m["extra"]["run_name"],
                     "test_macro_f1": m["macro_f1"], "test_accuracy": m["accuracy"],
                     "test_human_fpr": m["human_fpr"], "val_macro_f1": m["extra"]["val_macro_f1"],
                     "num_parameters": m["extra"]["num_parameters"],
                     "train_time_seconds": m["extra"]["train_time_seconds"]})
    return sorted(rows, key=lambda r: -r["test_macro_f1"])


def print_summary(rows: Sequence[Mapping[str, Any]]) -> None:
    """CLI table of the test results."""
    print(f"\n{'model':16s} {'run':16s} {'test F1':>8s} {'acc':>7s} {'hFPR':>6s} "
          f"{'val F1':>7s} {'params':>12s} {'train min':>9s}")
    for r in rows:
        print(f"{r['model']:16s} {r['run_name'] or '-':16s} {r['test_macro_f1']:8.4f} "
              f"{r['test_accuracy']:7.4f} {r['test_human_fpr'] or 0:6.3f} "
              f"{r['val_macro_f1']:7.4f} {r['num_parameters'] or 0:12,d} "
              f"{(r['train_time_seconds'] or 0) / 60:9.1f}")


def run(args: argparse.Namespace, overrides: Mapping[str, str]) -> int:
    """Score every requested model on test once; return the process exit code."""
    use_config(args.config)
    cfg = load_config()
    results_dir = get_path("results_dir")
    setup_logging(results_dir / LOG_FILE)

    failed: dict[str, str] = {}
    runs: dict[str, str | None] = {}
    for model in args.models:
        if final_metrics_path(results_dir, model).exists():
            logger.info("%s: already scored on test; skipping (the test set is used once)",
                        model)
            continue
        try:
            runs[model] = (None if model == LENGTH_BASELINE
                           else choose_run(model, overrides, results_dir))
            if runs[model] is not None:
                load_val_metrics(model, runs[model], results_dir)
        except FinalTestError as err:
            runs.pop(model, None)
            failed[model] = str(err)
            logger.error("%s", err)

    if runs:
        logger.info("Scoring on test: %s", ", ".join(f"{m} ({r or 'refit'})"
                                                      for m, r in runs.items()))
        seed_everything(cfg["seed"])
        splits = {"train": load_split("train", columns=SPLIT_COLUMNS),
                  "val": load_split("val", columns=SPLIT_COLUMNS),
                  TEST: load_split(TEST, columns=TEST_COLUMNS, allow_test=True)}
        class_weights = get_class_weights(
            splits["train"], cfg["classes"], cfg["class_weights"]["method"],
            results_dir / cfg["class_weights"]["file"]).tolist()
        for model, chosen in runs.items():
            try:
                metrics = score_model(model, chosen, splits, cfg, class_weights, results_dir)
            except FinalTestError as err:
                failed[model] = str(err)
                logger.error("%s", err)
                continue
            logger.info("%s: test macro-F1 %.4f", model, metrics["macro_f1"])

    rows = summarize(results_dir, [*REGISTRY, LENGTH_BASELINE])
    atomic_write_json(results_dir / SUMMARY_FILE, {"models": rows})
    print_summary(rows)
    if failed:
        print("\nNot scored:\n  " + "\n  ".join(f"{m}: {e}" for m, e in failed.items()))
        return 1
    return 0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Score the chosen runs on the test split, "
                                                 "once.")
    parser.add_argument(TEST_FLAG, dest="confirmed", action="store_true",
                        help="required: confirms that tuning is over")
    parser.add_argument("--config", default=None, help="config YAML (default: configs/config.yaml)")
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS),
                        choices=[*sorted(REGISTRY), LENGTH_BASELINE])
    parser.add_argument("--run", nargs="+", default=[], metavar="MODEL=RUN",
                        help="test this run instead of the tuning sweep's choice")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point."""
    args = parse_args(argv)
    setup_logging()
    if not args.confirmed:
        logger.error("The test split is used once, after all tuning is finished. Rerun with "
                     "%s to confirm.", TEST_FLAG)
        return USAGE_ERROR_EXIT_CODE
    try:
        overrides = parse_runs(args.run)
    except ValueError as err:
        logger.error("%s", err)
        return USAGE_ERROR_EXIT_CODE
    unknown = sorted(set(overrides) - set(args.models))
    if unknown:
        logger.error("--run names models that are not being tested: %s", unknown)
        return USAGE_ERROR_EXIT_CODE
    return run(args, overrides)


if __name__ == "__main__":
    sys.exit(main())
