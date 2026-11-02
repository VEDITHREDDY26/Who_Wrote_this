"""Hyperparameter sweep on the validation set.

    python -m src.tune --model {mlp,cnn,lstm,transformer} [--config PATH] [--runs NAME ...]

Runs every entry of `tuning.<model>` in the config (run name -> overrides merged into
models.<model>) through `python -m src.train`, one after another. Each run has its own run
name, so it resumes after an interruption and is skipped once finished; a run that crashes is
restarted from its last checkpoint up to `tuning.max_attempts` times. Finally all finished runs
(including the current config, run "default") are ranked by val macro-F1 and written to
results/tuning/<model>_tuning.json. The test split is never used.
"""

from __future__ import annotations

import argparse
import copy
import logging
import signal
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml

from src.config import active_config_path, get_path, load_config, use_config
from src.utils.io import atomic_write_bytes, atomic_write_json, safe_load_json
from src.utils.logging import setup_logging

logger = logging.getLogger(__name__)

DEFAULT_RUN = "default"
TUNING_SUBDIR = "tuning"
CONFIGS_SUBDIR = "configs"
# src.train exit codes that a restart cannot fix: finished, usage/config error, Ctrl+C.
NO_RETRY_EXIT_CODES = frozenset({0, 2, 130})
SIGINT_EXIT_CODE = 130


class TuningConfigError(ValueError):
    """The `tuning` section of the config is missing or inconsistent."""


def deep_merge(base: Mapping[str, Any], overrides: Mapping[str, Any]) -> dict[str, Any]:
    """Copy of `base` with `overrides` merged in (nested mappings are merged, not replaced)."""
    out = copy.deepcopy(dict(base))
    for key, value in overrides.items():
        if isinstance(value, Mapping) and isinstance(out.get(key), Mapping):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def get_grid(cfg: Mapping[str, Any], model: str,
             only: Sequence[str] | None = None) -> dict[str, dict[str, Any]]:
    """The model's tuning grid (run name -> overrides), optionally limited to `only`."""
    grid = cfg.get("tuning", {}).get(model)
    if not grid:
        raise TuningConfigError(f"no tuning grid for {model!r}: add `tuning.{model}` to the "
                                "config")
    if DEFAULT_RUN in grid:
        raise TuningConfigError(f"tuning.{model} may not define {DEFAULT_RUN!r}; that name is the "
                         "current config")
    if only:
        unknown = sorted(set(only) - set(grid))
        if unknown:
            raise TuningConfigError(f"unknown runs {unknown}; tuning.{model} has "
                                    f"{sorted(grid)}")
        grid = {name: grid[name] for name in only}
    return dict(grid)


def write_run_config(raw_cfg: Mapping[str, Any], model: str, overrides: Mapping[str, Any],
                     path: Path) -> Path:
    """Write the full config for one run: the base config with models.<model> overridden."""
    cfg = copy.deepcopy(dict(raw_cfg))
    cfg["models"] = dict(cfg["models"]) | {model: deep_merge(cfg["models"].get(model, {}),
                                                             overrides)}
    atomic_write_bytes(path, yaml.safe_dump(cfg, sort_keys=False).encode("utf-8"))
    return path


def wait_forwarding_interrupt(proc: subprocess.Popen) -> int:
    """Wait for `proc`; on Ctrl+C pass SIGINT on (it saves a checkpoint) and return 130."""
    try:
        return proc.wait()
    except KeyboardInterrupt:
        proc.send_signal(signal.SIGINT)
        proc.wait()
        return SIGINT_EXIT_CODE


def run_training(model: str, config_path: Path, run_name: str, max_attempts: int) -> int:
    """Train one run with src.train, restarting it after a crash; return the last exit code."""
    cmd = [sys.executable, "-m", "src.train", "--model", model, "--config", str(config_path),
           "--run-name", run_name]
    code = 1
    for attempt in range(1, max_attempts + 1):
        logger.info("=== %s / %s (attempt %d/%d) ===", model, run_name, attempt, max_attempts)
        code = wait_forwarding_interrupt(subprocess.Popen(cmd))
        if code in NO_RETRY_EXIT_CODES:
            break
        logger.warning("%s exited with code %d; restarting from its last checkpoint",
                       run_name, code)
    return code


def metrics_file(results_dir: Path, model: str, run_name: str) -> Path:
    """Where src.train writes a run's val metrics."""
    label = model if run_name == DEFAULT_RUN else f"{model}-{run_name}"
    return results_dir / f"{label}_val_metrics.json"


def summarize(results_dir: Path, model: str, run_overrides: Mapping[str, Any]) -> dict[str, Any]:
    """Rank every finished run by val macro-F1 (ties keep the grid order)."""
    rows = []
    for name, overrides in run_overrides.items():
        m = safe_load_json(metrics_file(results_dir, model, name), default=None)
        if m is None:
            continue
        rows.append({"run_name": name, "overrides": overrides, "val_macro_f1": m["macro_f1"],
                     "val_accuracy": m["accuracy"], "val_human_fpr": m["human_fpr"],
                     "val_log_loss": m["log_loss"], "best_epoch": m["extra"]["best_epoch"],
                     "epochs_completed": m["extra"]["epochs_completed"],
                     "train_time_seconds": m["extra"]["train_time_seconds"],
                     "num_parameters": m["extra"]["num_parameters"]})
    rows.sort(key=lambda r: -r["val_macro_f1"])
    return {"model": model, "selection": "highest val macro-F1",
            "chosen": rows[0]["run_name"] if rows else None, "runs": rows}


def print_table(summary: Mapping[str, Any]) -> None:
    """CLI summary of the sweep."""
    print(f"\n{summary['model']} tuning (val), best first:")
    print(f"  {'run':16s} {'macro-F1':>8s} {'acc':>7s} {'hFPR':>6s} {'best ep':>7s} "
          f"{'minutes':>7s}  overrides")
    for r in summary["runs"]:
        mark = "*" if r["run_name"] == summary["chosen"] else " "
        print(f"{mark} {r['run_name']:16s} {r['val_macro_f1']:8.4f} {r['val_accuracy']:7.4f} "
              f"{r['val_human_fpr'] or 0:6.3f} {r['best_epoch']!s:>7s} "
              f"{r['train_time_seconds'] / 60:7.1f}  {r['overrides']}")


def run(args: argparse.Namespace) -> int:
    """Run the sweep; return the process exit code."""
    use_config(args.config)
    cfg = load_config()
    grid = get_grid(cfg, args.model, args.runs)
    max_attempts = int(cfg["tuning"]["max_attempts"])
    raw_cfg = yaml.safe_load(active_config_path().read_text(encoding="utf-8"))
    results_dir = get_path("results_dir")
    configs_dir = results_dir / TUNING_SUBDIR / CONFIGS_SUBDIR

    failed: dict[str, int] = {}
    for name, overrides in grid.items():
        path = write_run_config(raw_cfg, args.model, overrides,
                                configs_dir / f"{args.model}-{name}.yaml")
        code = run_training(args.model, path, name, max_attempts)
        if code == SIGINT_EXIT_CODE:
            logger.warning("Stopped by Ctrl+C during %s; rerun the same command to continue",
                           name)
            return code
        if code != 0:
            failed[name] = code

    all_runs = {DEFAULT_RUN: {}} | dict(cfg["tuning"][args.model])
    summary = summarize(results_dir, args.model, all_runs)
    out = results_dir / TUNING_SUBDIR / f"{args.model}_tuning.json"
    atomic_write_json(out, summary)
    print_table(summary)
    print(f"\nSaved {out}")
    if failed:
        logger.error("Runs that did not finish: %s", failed)
        return 1
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Hyperparameter sweep on the val split.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--config", default=None, help="config YAML (default: configs/config.yaml)")
    parser.add_argument("--runs", nargs="+", default=None, help="only these runs of the grid")
    args = parser.parse_args(argv)
    setup_logging()
    try:
        return run(args)
    except TuningConfigError as err:
        logger.error("%s", err)
        return 2


if __name__ == "__main__":
    sys.exit(main())
