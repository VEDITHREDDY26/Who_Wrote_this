"""Training entry point.

    python -m src.train --model {dummy,mlp,cnn,lstm,transformer} [--config PATH]
        [--run-name NAME] [--resume auto|none] [--max-epochs N] [--subset N] [--force-new]

Trains with the shared seed, splits and class weights, keeps the best checkpoint by
validation macro-F1, then writes results/<model>_val_metrics.json (+ confusion matrix and
probabilities). Rerunning the same command after any interruption resumes the run.

This module only ever reads the train and val splits. The test split is evaluated once, at
the very end, by src/final_test.py.
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Mapping, Sequence
from dataclasses import replace
from datetime import datetime
from typing import Any

import pandas as pd
import torch

from src.config import get_path, load_config, use_config
from src.data.class_weights import get_class_weights
from src.data.splits import fingerprint_ids, load_split, stratified_subset
from src.evaluate import evaluate, save_results
from src.features.tfidf import get_tfidf_features
from src.features.vocab import get_encoded_splits
from src.models import REGISTRY, ModelSpec, build_model, get_model_spec, require_implemented
from src.models.transformer import train_transformer
from src.trainer import (
    BEST_FILE,
    ArrayDataset,
    ConfigMismatchError,
    FitResult,
    RunExistsError,
    Trainer,
    TrainingSettings,
)
from src.utils.logging import setup_logging
from src.utils.seed import seed_everything

logger = logging.getLogger(__name__)

DEFAULT_RUN_NAME = "default"
SPLIT_COLUMNS = ("id", "text", "label_id")
USAGE_ERROR_EXIT_CODE = 2


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Train one model on the frozen splits.")
    parser.add_argument("--model", required=True, choices=sorted(REGISTRY))
    parser.add_argument("--config", default=None, help="config YAML (default: configs/config.yaml)")
    parser.add_argument("--run-name", default=DEFAULT_RUN_NAME)
    parser.add_argument("--resume", choices=("auto", "none"), default="auto")
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--subset", type=int, default=None,
                        help="train on a stratified N-row sample of train (quick checks)")
    parser.add_argument("--force-new", action="store_true",
                        help="start a fresh run in a new timestamped folder")
    return parser.parse_args(argv)


def training_config(cfg: Mapping[str, Any], model: str,
                    max_epochs: int | None) -> TrainingSettings:
    """Defaults from `training`, overridden by `models.<model>.training`, then the CLI."""
    merged = dict(cfg["training"]) | dict(cfg["models"].get(model, {}).get("training", {}))
    settings = TrainingSettings.from_config(merged, seed=cfg["seed"])
    return replace(settings, max_epochs=max_epochs) if max_epochs is not None else settings


def model_params(cfg: Mapping[str, Any], model: str) -> dict[str, Any]:
    """The model's own hyperparameters (its config section without `training`)."""
    return {k: v for k, v in cfg["models"].get(model, {}).items() if k != "training"}


def build_datasets(spec: ModelSpec, splits: Mapping[str, pd.DataFrame],
                   cfg: Mapping[str, Any]) -> tuple[dict[str, ArrayDataset], dict[str, Any],
                                                    dict[str, Any]]:
    """Featurize the splits for this model; return datasets, cache info and model dims."""
    cache_root = get_path("splits_dir") / cfg["features"]["cache_subdir"]
    labels = {name: df["label_id"].to_numpy() for name, df in splits.items()}
    num_classes = len(cfg["classes"])
    if spec.input_kind == "tfidf":
        feats = get_tfidf_features(splits, cfg["features"]["tfidf"], cache_root)
        data = {n: ArrayDataset(feats.matrices[n], labels[n]) for n in splits}
        info = {"kind": "tfidf", "cache_key": feats.key, "n_features": feats.n_features}
        dims = {"input_dim": feats.n_features, "num_classes": num_classes}
    elif spec.input_kind == "tokens":
        enc = get_encoded_splits(splits, cfg["features"]["vocab"], cache_root)
        data = {n: ArrayDataset(enc.tensors[n], labels[n]) for n in splits}
        info = {"kind": "tokens", "cache_key": enc.key, "vocab_size": len(enc.vocab)}
        dims = {"vocab_size": len(enc.vocab), "pad_id": enc.vocab.pad_id,
                "max_len": cfg["features"]["vocab"]["max_len"], "num_classes": num_classes}
    else:
        raise ValueError(f"model {spec.name!r} has input kind {spec.input_kind!r}, which is "
                         "not trained by src.trainer")
    return data, info, dims


def evaluate_best(trainer: Trainer, result: FitResult, label: str, run_name: str,
                  subset: int | None) -> dict[str, Any]:
    """Load best.pt, evaluate it on val with the shared evaluate(), and save the results."""
    best = torch.load(result.checkpoint_dir / BEST_FILE, map_location="cpu", weights_only=True)
    trainer.model.load_state_dict(best["model"])
    y_true = trainer.val_data.labels.numpy()
    y_prob = trainer.predict_proba(trainer.val_data)
    metrics = evaluate(y_true, y_prob.argmax(axis=1), y_prob, trainer.class_names,
                       trainer.settings.human_label)
    save_results(metrics, label, "val", {
        "train_time_seconds": result.train_time_seconds,
        "num_parameters": trainer.num_parameters,
        "best_epoch": best["epoch"], "epochs_completed": result.epochs_completed,
        "status": result.status, "run_name": run_name, "subset": subset,
    }, y_true, y_prob, get_path("results_dir"))
    return metrics


def run(args: argparse.Namespace) -> int:
    """Train the requested model; return the process exit code."""
    use_config(args.config)  # get_path() must follow --config too, not just cfg
    cfg = load_config()
    spec = get_model_spec(args.model)
    if spec.input_kind == "hf":  # === TRANSFORMER HOOK: Hugging Face Trainer ===
        return train_transformer(cfg, run_name=args.run_name, resume=args.resume,
                                 max_epochs=args.max_epochs, subset=args.subset)
    require_implemented(spec)

    run_name = args.run_name
    if args.force_new:
        run_name = f"{run_name}-{datetime.now():%Y%m%d-%H%M%S}"
    checkpoint_dir = get_path("checkpoint_dir") / args.model / run_name
    run_results_dir = get_path("results_dir") / args.model / run_name
    setup_logging(run_results_dir / "train.log")
    logger.info("=== %s / %s ===", args.model, run_name)

    seed_everything(cfg["seed"])
    train_full = load_split("train", columns=SPLIT_COLUMNS)
    val = load_split("val", columns=SPLIT_COLUMNS)
    class_weights = get_class_weights(
        train_full, cfg["classes"], cfg["class_weights"]["method"],
        get_path("results_dir") / cfg["class_weights"]["file"])
    train = (stratified_subset(train_full, args.subset, label_column="label_id",
                               seed=cfg["seed"]) if args.subset else train_full)
    splits = {"train": train, "val": val}
    data, input_info, dims = build_datasets(spec, splits, cfg)

    params = model_params(cfg, args.model)
    settings = training_config(cfg, args.model, args.max_epochs)
    seed_everything(cfg["seed"])  # model init does not depend on feature-building randomness
    model = build_model(args.model, params, **dims)
    run_config = {"model": args.model, "model_params": params, "input": input_info,
                  "subset": args.subset, "train_fingerprint": fingerprint_ids(train["id"]),
                  "val_fingerprint": fingerprint_ids(val["id"]),
                  "class_weights": class_weights.tolist()}
    trainer = Trainer(model, data["train"], data["val"], settings=settings,
                      class_weights=class_weights, class_names=cfg["classes"],
                      run_config=run_config, checkpoint_dir=checkpoint_dir,
                      results_dir=run_results_dir)
    result = trainer.fit(resume=args.resume)

    if result.status == "stopped":
        print(f"Stopped: checkpoint saved in {checkpoint_dir}, safe to resume "
              "(rerun the same command).")
        return result.exit_code
    label = args.model if run_name == DEFAULT_RUN_NAME else f"{args.model}-{run_name}"
    metrics = evaluate_best(trainer, result, label, run_name, args.subset)
    print(f"\n{label}: {result.status} after {result.epochs_completed} epochs "
          f"(best epoch {result.best_epoch})")
    print(f"  val macro-F1  {metrics['macro_f1']:.4f}")
    print(f"  val accuracy  {metrics['accuracy']:.4f}")
    print(f"  human FPR     {metrics['human_fpr']}")
    print(f"  train time    {result.train_time_seconds:.1f}s, "
          f"{trainer.num_parameters:,} parameters")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point: turns expected user errors into a clear message and exit code 2."""
    args = parse_args(argv)
    setup_logging()
    try:
        return run(args)
    except (ConfigMismatchError, RunExistsError, NotImplementedError) as err:
        logger.error("%s", err)
        return USAGE_ERROR_EXIT_CODE


if __name__ == "__main__":
    sys.exit(main())

