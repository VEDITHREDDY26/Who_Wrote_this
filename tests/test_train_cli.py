"""End-to-end tests of `python -m src.train` on a tiny fake dataset (subprocess, CPU)."""

import json
import os
import random
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
CLASSES = ["Human", "GPT", "Llama", "Mistral", "MPT", "Cohere"]


def fake_split(n_per_class: int, prefix: str, rng: random.Random) -> pd.DataFrame:
    rows = []
    for label_id, label in enumerate(CLASSES):
        for i in range(n_per_class):
            words = [f"w{label_id}{rng.randint(0, 4)}" for _ in range(8)]
            words += rng.choices(["the", "a", "of", "and", "text"], k=6)
            rng.shuffle(words)
            rows.append({"id": f"{prefix}{label_id}-{i}", "text": " ".join(words) + ".",
                         "label": label, "label_id": label_id, "domain": "news",
                         "group": f"{prefix}g{i}"})
    return pd.DataFrame(rows)


def write_config(workspace: Path, name: str, **training: object) -> Path:
    cfg = yaml.safe_load((ROOT / "configs" / "config.yaml").read_text(encoding="utf-8"))
    cfg["paths"] = {"data_dir": str(workspace / "data"), "raw_dir": "raw",
                    "splits_dir": "splits", "checkpoint_dir": str(workspace / "checkpoints"),
                    "results_dir": str(workspace / "results")}
    cfg["features"]["tfidf"]["word"] |= {"max_features": 60, "min_df": 1}
    cfg["features"]["tfidf"]["char"] |= {"max_features": 40, "min_df": 1}
    cfg["features"]["tfidf"]["n_jobs"] = 1
    cfg["training"] |= {"batch_size": 8, "eval_batch_size": 64, "max_epochs": 2,
                        "lr": 0.05} | training
    path = workspace / name
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return path


@pytest.fixture(scope="module")
def workspace(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("cli")
    splits = root / "data" / "splits"
    splits.mkdir(parents=True)
    rng = random.Random(0)
    fake_split(12, "t", rng).to_parquet(splits / "train.parquet", index=False)
    fake_split(4, "v", rng).to_parquet(splits / "val.parquet", index=False)
    write_config(root, "config.yaml")
    return root


def run_cli(config: Path, *args: str) -> subprocess.CompletedProcess:
    env = {k: v for k, v in os.environ.items() if not k.startswith("TD_")}
    proc = subprocess.run([sys.executable, "-m", "src.train", "--config", str(config), *args],
                          cwd=ROOT, env=env, capture_output=True, text=True, timeout=240)
    proc.output = proc.stdout + proc.stderr
    return proc


def test_dummy_trains_end_to_end_and_rerun_is_a_noop(workspace: Path) -> None:
    proc = run_cli(workspace / "config.yaml", "--model", "dummy")
    assert proc.returncode == 0, proc.output
    assert "72 train rows" in proc.output  # the fake split, not the real data
    assert f"results to {workspace / 'results'}" in proc.output
    results, ckpt = workspace / "results", workspace / "checkpoints" / "dummy" / "default"
    for path in (ckpt / "best.pt", ckpt / "last.pt", results / "class_weights.json",
                 results / "dummy" / "default" / "history.jsonl",
                 results / "dummy" / "default" / "train.log",
                 results / "dummy_val_metrics.json", results / "dummy_val_confusion.png",
                 results / "dummy_val_probs.npz"):
        assert path.exists(), path
    metrics = json.loads((results / "dummy_val_metrics.json").read_text())
    assert metrics["n_samples"] == 24 and metrics["extra"]["num_parameters"] > 0

    again = run_cli(workspace / "config.yaml", "--model", "dummy")
    assert again.returncode == 0 and "already finished" in again.output


@pytest.mark.parametrize("model, hint", [("mlp", "src/models/mlp.py"),
                                         ("cnn", "src/models/cnn.py"),
                                         ("transformer", "Hugging Face")])
def test_unimplemented_models_fail_with_instructions(workspace: Path, model: str,
                                                     hint: str) -> None:
    proc = run_cli(workspace / "config.yaml", "--model", model)
    assert proc.returncode == 2
    assert "not implemented yet" in proc.output and hint in proc.output


def test_config_change_is_refused_until_force_new(workspace: Path) -> None:
    first = run_cli(workspace / "config.yaml", "--model", "dummy", "--run-name", "cfg")
    assert first.returncode == 0, first.output
    changed = write_config(workspace, "changed.yaml", lr=0.01)
    refused = run_cli(changed, "--model", "dummy", "--run-name", "cfg")
    assert refused.returncode == 2
    assert "training.lr: 0.05 -> 0.01" in refused.output and "--force-new" in refused.output
    fresh = run_cli(changed, "--model", "dummy", "--run-name", "cfg", "--force-new")
    assert fresh.returncode == 0, fresh.output
    runs = sorted(p.name for p in (workspace / "checkpoints" / "dummy").iterdir())
    assert "cfg" in runs and any(r.startswith("cfg-") for r in runs)


def test_subset_trains_on_a_stratified_sample(workspace: Path) -> None:
    proc = run_cli(workspace / "config.yaml", "--model", "dummy", "--run-name", "sub",
                   "--subset", "30", "--max-epochs", "1")
    assert proc.returncode == 0, proc.output
    metrics = json.loads((workspace / "results" / "dummy-sub_val_metrics.json").read_text())
    assert metrics["extra"]["subset"] == 30 and metrics["extra"]["epochs_completed"] == 1
