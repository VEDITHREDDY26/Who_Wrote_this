"""Tests of `python -m src.final_test` on a tiny fake dataset (subprocess, CPU)."""

import json
import os
import random
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from src.final_test import TEST_FLAG, parse_runs
from tests.test_train_cli import CLASSES, ROOT, fake_split, make_workspace, run_cli


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    """Fake train/val/test splits; a fresh one per test, since each test uses test once."""
    root = make_workspace(tmp_path)
    fake_split(4, "x", random.Random(1)).to_parquet(root / "data" / "splits" / "test.parquet",
                                                    index=False)
    return root


def run_final(config: Path, *args: str) -> subprocess.CompletedProcess:
    env = {k: v for k, v in os.environ.items() if not k.startswith("TD_")}
    proc = subprocess.run([sys.executable, "-m", "src.final_test", "--config", str(config),
                           *args], cwd=ROOT, env=env, capture_output=True, text=True,
                          timeout=240)
    proc.output = proc.stdout + proc.stderr
    return proc


def train_dummy(workspace: Path, *args: str) -> None:
    proc = run_cli(workspace / "config.yaml", "--model", "dummy", *args)
    assert proc.returncode == 0, proc.output


def write_tuning_choice(workspace: Path, model: str, run: str) -> None:
    path = workspace / "results" / "tuning" / f"{model}_tuning.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"model": model, "chosen": run, "runs": []}))


def test_parse_runs() -> None:
    assert parse_runs(["mlp=h512", "cnn=default"]) == {"mlp": "h512", "cnn": "default"}
    for bad in ("mlp", "=h512", "mlp="):
        with pytest.raises(ValueError, match="MODEL=RUN"):
            parse_runs([bad])


def test_refuses_without_the_confirmation_flag(workspace: Path) -> None:
    proc = run_final(workspace / "config.yaml", "--models", "dummy")
    assert proc.returncode == 2 and TEST_FLAG in proc.output
    assert not (workspace / "results" / "dummy_test_metrics.json").exists()


def test_scores_the_tuned_run_once(workspace: Path) -> None:
    train_dummy(workspace)
    write_tuning_choice(workspace, "dummy", "default")
    proc = run_final(workspace / "config.yaml", TEST_FLAG, "--models", "dummy")
    assert proc.returncode == 0, proc.output
    assert "val macro-F1 reproduced" in proc.output

    results = workspace / "results"
    metrics = json.loads((results / "dummy_test_metrics.json").read_text())
    assert metrics["split"] == "test" and metrics["n_samples"] == 24
    assert metrics["extra"]["run_name"] == "default"
    assert metrics["extra"]["num_parameters"] > 0
    assert metrics["extra"]["val_macro_f1"] == pytest.approx(
        metrics["extra"]["val_macro_f1_recomputed"], abs=1e-6)
    for name in ("dummy_test_confusion.png", "dummy_test_probs.npz", "final_test.log"):
        assert (results / name).exists(), name

    preds = pd.read_parquet(results / "dummy_test_predictions.parquet")
    assert len(preds) == 24 and preds["id"].str.startswith("x").all()
    assert list(preds.columns[:4]) == ["id", "domain", "label_id", "pred_id"]
    assert [c.removeprefix("prob_") for c in preds.columns[4:]] == CLASSES
    assert (preds["pred_id"] == preds[[f"prob_{c}" for c in CLASSES]].to_numpy()
            .argmax(axis=1)).all()
    summary = json.loads((results / "final_test_summary.json").read_text())
    assert [r["model"] for r in summary["models"]] == ["dummy"]

    before = (results / "dummy_test_metrics.json").read_bytes()
    again = run_final(workspace / "config.yaml", TEST_FLAG, "--models", "dummy")
    assert again.returncode == 0 and "already scored on test" in again.output
    assert (results / "dummy_test_metrics.json").read_bytes() == before


def test_run_flag_overrides_the_tuning_choice(workspace: Path) -> None:
    train_dummy(workspace, "--run-name", "other", "--max-epochs", "1")
    write_tuning_choice(workspace, "dummy", "missing-run")
    proc = run_final(workspace / "config.yaml", TEST_FLAG, "--models", "dummy",
                     "--run", "dummy=other")
    assert proc.returncode == 0, proc.output
    metrics = json.loads((workspace / "results" / "dummy_test_metrics.json").read_text())
    assert metrics["extra"]["run_name"] == "other"


def test_missing_choice_or_unfinished_run_is_not_scored(workspace: Path) -> None:
    no_choice = run_final(workspace / "config.yaml", TEST_FLAG, "--models", "dummy")
    assert no_choice.returncode == 1 and "--run dummy=<run name>" in no_choice.output
    unfinished = run_final(workspace / "config.yaml", TEST_FLAG, "--models", "dummy",
                           "--run", "dummy=never-trained")
    assert unfinished.returncode == 1 and "has not finished training" in unfinished.output
    assert not (workspace / "results" / "dummy_test_metrics.json").exists()


def test_subset_runs_are_refused(workspace: Path) -> None:
    train_dummy(workspace, "--run-name", "sub", "--subset", "30", "--max-epochs", "1")
    proc = run_final(workspace / "config.yaml", TEST_FLAG, "--models", "dummy",
                     "--run", "dummy=sub")
    assert proc.returncode == 1 and "30-row subset" in proc.output
    assert not (workspace / "results" / "dummy_test_metrics.json").exists()


def test_checkpoint_that_does_not_reproduce_val_is_refused(workspace: Path) -> None:
    train_dummy(workspace)
    path = workspace / "results" / "dummy_val_metrics.json"
    saved = json.loads(path.read_text())
    path.write_text(json.dumps(saved | {"macro_f1": saved["macro_f1"] + 0.1}))
    proc = run_final(workspace / "config.yaml", TEST_FLAG, "--models", "dummy",
                     "--run", "dummy=default")
    assert proc.returncode == 1 and "does not match its results" in proc.output
    assert not (workspace / "results" / "dummy_test_metrics.json").exists()


def test_length_baseline_is_scored(workspace: Path) -> None:
    env = {k: v for k, v in os.environ.items() if not k.startswith("TD_")}
    baseline = subprocess.run([sys.executable, "-m", "src.baselines.length_baseline",
                               "--config", str(workspace / "config.yaml")], cwd=ROOT, env=env,
                              capture_output=True, text=True, timeout=240)
    assert baseline.returncode == 0, baseline.stdout + baseline.stderr
    proc = run_final(workspace / "config.yaml", TEST_FLAG, "--models", "length_baseline")
    assert proc.returncode == 0, proc.output
    metrics = json.loads((workspace / "results" / "length_baseline_test_metrics.json")
                         .read_text())
    assert metrics["n_samples"] == 24 and metrics["extra"]["num_parameters"] == 6 * 3 + 6
