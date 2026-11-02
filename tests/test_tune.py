"""Tests for the tuning sweep (src.tune) on the fake CLI workspace."""

import json
import os
import signal
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from src.tune import TuningConfigError, deep_merge, get_grid, run_training
from tests.test_train_cli import ROOT, make_workspace

GRID = {"d01": {"dropout": 0.1}, "lr01": {"training": {"lr": 0.01}}}


def test_deep_merge_merges_nested_blocks_without_touching_the_base() -> None:
    base = {"dropout": 0.5, "training": {"lr": 0.1, "batch_size": 8}}
    merged = deep_merge(base, {"training": {"lr": 0.2}, "hidden": [4]})
    assert merged == {"dropout": 0.5, "training": {"lr": 0.2, "batch_size": 8}, "hidden": [4]}
    assert base["training"]["lr"] == 0.1


def test_get_grid_checks_the_config() -> None:
    cfg = {"tuning": {"max_attempts": 1, "dummy": GRID}}
    assert list(get_grid(cfg, "dummy")) == ["d01", "lr01"]
    assert list(get_grid(cfg, "dummy", ["lr01"])) == ["lr01"]
    with pytest.raises(TuningConfigError, match="no tuning grid"):
        get_grid(cfg, "mlp")
    with pytest.raises(TuningConfigError, match="unknown runs"):
        get_grid(cfg, "dummy", ["nope"])
    with pytest.raises(TuningConfigError, match="default"):
        get_grid({"tuning": {"dummy": {"default": {}}}}, "dummy")


@pytest.mark.parametrize("codes, calls", [([1, 1, 0], 3), ([2], 1), ([130], 1), ([1, 1, 1], 3)])
def test_crashed_runs_are_restarted(monkeypatch: pytest.MonkeyPatch, codes: list[int],
                                    calls: int) -> None:
    returned = iter(codes)
    seen = []

    class FakeProcess:
        def __init__(self, cmd: list[str]) -> None:
            seen.append(cmd)

        def wait(self) -> int:
            return next(returned)

    monkeypatch.setattr("src.tune.subprocess.Popen", FakeProcess)
    assert run_training("dummy", Path("c.yaml"), "r", max_attempts=3) == codes[calls - 1]
    assert len(seen) == calls and seen[0][-2:] == ["--run-name", "r"]


def test_ctrl_c_is_passed_on_to_the_training_process() -> None:
    from src.tune import wait_forwarding_interrupt

    class Interrupted:
        sent: list[int] = []
        waits = 0

        def wait(self) -> int:
            Interrupted.waits += 1
            if Interrupted.waits == 1:
                raise KeyboardInterrupt
            return 130

        def send_signal(self, sig: int) -> None:
            Interrupted.sent.append(sig)

    assert wait_forwarding_interrupt(Interrupted()) == 130
    assert Interrupted.sent == [signal.SIGINT] and Interrupted.waits == 2


def run_tune(config: Path, *args: str) -> subprocess.CompletedProcess:
    env = {k: v for k, v in os.environ.items() if not k.startswith("TD_")}
    proc = subprocess.run([sys.executable, "-m", "src.tune", "--config", str(config), *args],
                          cwd=ROOT, env=env, capture_output=True, text=True, timeout=240)
    proc.output = proc.stdout + proc.stderr
    return proc


def test_sweep_trains_each_run_ranks_them_and_skips_finished_runs(tmp_path: Path) -> None:
    workspace = make_workspace(tmp_path)
    config = workspace / "config.yaml"
    cfg = yaml.safe_load(config.read_text())
    cfg["training"]["max_epochs"] = 1
    cfg["tuning"] = {"max_attempts": 2, "dummy": GRID}
    config.write_text(yaml.safe_dump(cfg))

    proc = run_tune(config, "--model", "dummy")
    assert proc.returncode == 0, proc.output
    results = workspace / "results"
    summary = json.loads((results / "tuning" / "dummy_tuning.json").read_text())
    assert {r["run_name"] for r in summary["runs"]} == {"d01", "lr01"}
    assert summary["chosen"] == summary["runs"][0]["run_name"]
    assert summary["runs"][0]["val_macro_f1"] >= summary["runs"][1]["val_macro_f1"]
    run_cfg = yaml.safe_load((results / "tuning" / "configs" / "dummy-lr01.yaml").read_text())
    assert run_cfg["models"]["dummy"] == {"dropout": 0.0, "training": {"lr": 0.01}}
    assert (workspace / "checkpoints" / "dummy" / "d01" / "best.pt").exists()

    again = run_tune(config, "--model", "dummy")
    assert again.returncode == 0 and again.output.count("already finished") == 2


def test_unknown_model_grid_is_a_usage_error(tmp_path: Path) -> None:
    config = make_workspace(tmp_path) / "config.yaml"
    proc = run_tune(config, "--model", "mlp")
    assert proc.returncode == 2 and "no tuning grid" in proc.output
