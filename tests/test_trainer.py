"""Trainer tests on CPU with tiny fake data: checkpoints, exact resume, guards, signals."""

import json
import logging
import os
import signal
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from torch import nn

from src.models.dummy import DummyLinear
from src.trainer import (
    BEST_FILE,
    HISTORY_FILE,
    LAST_FILE,
    PREV_FILE,
    TIMING_KEYS,
    ArrayDataset,
    Callback,
    ConfigMismatchError,
    NonFiniteLossError,
    RunExistsError,
    Trainer,
    TrainingSettings,
    epoch_permutation,
)

CLASS_NAMES = ["Human", "GPT", "Llama"]
N_TRAIN, N_VAL, DIM = 72, 24, 12
STEPS_PER_EPOCH = N_TRAIN // 8
SETTINGS = TrainingSettings(
    batch_size=8, eval_batch_size=16, max_epochs=3, lr=0.05, weight_decay=0.01,
    scheduler="cosine", warmup_ratio=0.1, max_grad_norm=1.0, amp=False, patience=10,
    min_delta=0.0, checkpoint_every_minutes=1e6, max_nonfinite_per_epoch=5,
    log_every_steps=1000, human_label="Human", seed=123)


def make_data() -> tuple[ArrayDataset, ArrayDataset]:
    g = torch.Generator().manual_seed(0)
    n = N_TRAIN + N_VAL
    y = torch.arange(n) % len(CLASS_NAMES)
    x = torch.randn(len(CLASS_NAMES), DIM, generator=g)[y] + 0.8 * torch.randn(n, DIM,
                                                                                generator=g)
    return ArrayDataset(x[:N_TRAIN], y[:N_TRAIN]), ArrayDataset(x[N_TRAIN:], y[N_TRAIN:])


def make_trainer(root: Path, name: str, *, model: nn.Module | None = None,
                 callbacks: tuple[Callback, ...] = (), init_seed: int = 0,
                 **overrides: object) -> Trainer:
    torch.manual_seed(init_seed)
    train, val = make_data()
    return Trainer(model or DummyLinear(DIM, len(CLASS_NAMES), dropout=0.2), train, val,
                   settings=replace(SETTINGS, **overrides),
                   class_weights=torch.tensor([2.0, 1.0, 1.0]), class_names=CLASS_NAMES,
                   run_config={"model": "dummy", "purpose": "test"},
                   checkpoint_dir=root / name / "ckpt", results_dir=root / name / "results",
                   device="cpu", callbacks=callbacks)


class RaiseAt(Callback):
    """Raise an exception after a given (0-based) epoch and batch."""

    def __init__(self, epoch: int, batch: int, exc: type[BaseException]) -> None:
        self.at, self.exc = (epoch, batch), exc

    def on_batch_end(self, trainer: Trainer, epoch: int, batch: int) -> None:
        if (epoch, batch) == self.at:
            raise self.exc("simulated crash")


class SignalAt(Callback):
    """Send a real signal to this process after a given (0-based) epoch and batch."""

    def __init__(self, epoch: int, batch: int, signum: int) -> None:
        self.at, self.signum = (epoch, batch), signum

    def on_batch_end(self, trainer: Trainer, epoch: int, batch: int) -> None:
        if (epoch, batch) == self.at:
            os.kill(os.getpid(), self.signum)


def weights(trainer: Trainer) -> dict[str, torch.Tensor]:
    return {k: v.detach().clone() for k, v in trainer.model.state_dict().items()}


def assert_same_weights(a: dict[str, torch.Tensor], b: dict[str, torch.Tensor]) -> None:
    assert a.keys() == b.keys()
    for key in a:
        assert torch.equal(a[key], b[key]), f"{key} differs"


def untimed(history: list[dict]) -> list[dict]:
    return [{k: v for k, v in rec.items() if k not in TIMING_KEYS} for rec in history]


@pytest.fixture(scope="module")
def straight_run(tmp_path_factory: pytest.TempPathFactory) -> Trainer:
    """Reference: 3 epochs with no interruption."""
    trainer = make_trainer(tmp_path_factory.mktemp("straight"), "a")
    assert trainer.fit().status == "finished"
    return trainer


def test_dummy_trains_two_epochs_and_writes_files(tmp_path: Path) -> None:
    trainer = make_trainer(tmp_path, "run", max_epochs=2)
    result = trainer.fit()
    assert result.status == "finished" and result.epochs_completed == 2
    for path in (trainer.checkpoint_dir / BEST_FILE, trainer.checkpoint_dir / LAST_FILE,
                 trainer.results_dir / HISTORY_FILE):
        assert path.exists(), path
    lines = (trainer.results_dir / HISTORY_FILE).read_text().splitlines()
    assert [json.loads(line)["epoch"] for line in lines] == [1, 2]
    best = torch.load(trainer.checkpoint_dir / BEST_FILE, weights_only=True)
    assert best["epoch"] == result.best_epoch
    assert result.train_time_seconds > 0


def test_kill_and_resume_matches_straight_run(tmp_path: Path, straight_run: Trainer) -> None:
    crashing = make_trainer(tmp_path, "b", callbacks=(RaiseAt(1, 4, KeyboardInterrupt),))
    with pytest.raises(KeyboardInterrupt):
        crashing.fit()
    saved = torch.load(crashing.checkpoint_dir / LAST_FILE, weights_only=True)
    assert (saved["epoch"], saved["batch_in_epoch"]) == (1, 5)

    resumed = make_trainer(tmp_path, "b", init_seed=999)  # different init: must load weights
    result = resumed.fit(resume="auto")
    assert result.status == "finished"
    assert_same_weights(weights(straight_run), weights(resumed))
    assert untimed(resumed.state.history) == untimed(straight_run.state.history)
    on_disk = [json.loads(x) for x in (resumed.results_dir / HISTORY_FILE).read_text()
               .splitlines()]
    assert untimed(on_disk) == untimed(straight_run.state.history)
    assert result.train_time_seconds >= saved["train_time_seconds"]


def test_corrupted_last_checkpoint_falls_back_to_previous(
        tmp_path: Path, straight_run: Trainer, caplog: pytest.LogCaptureFixture) -> None:
    crashing = make_trainer(tmp_path, "c", callbacks=(RaiseAt(1, 4, KeyboardInterrupt),))
    with pytest.raises(KeyboardInterrupt):
        crashing.fit()
    last = crashing.checkpoint_dir / LAST_FILE
    assert (crashing.checkpoint_dir / PREV_FILE).exists()
    last.write_bytes(last.read_bytes()[: last.stat().st_size // 2])  # truncate

    resumed = make_trainer(tmp_path, "c")
    with caplog.at_level(logging.WARNING, logger="src.trainer"):
        assert resumed.fit().status == "finished"
    assert "Could not read checkpoint" in caplog.text
    assert_same_weights(weights(straight_run), weights(resumed))
    assert untimed(resumed.state.history) == untimed(straight_run.state.history)


def test_all_checkpoints_corrupted_raises(tmp_path: Path) -> None:
    trainer = make_trainer(tmp_path, "d", callbacks=(RaiseAt(1, 2, KeyboardInterrupt),))
    with pytest.raises(KeyboardInterrupt):
        trainer.fit()
    for name in (LAST_FILE, PREV_FILE):
        (trainer.checkpoint_dir / name).write_bytes(b"garbage")
    with pytest.raises(RuntimeError, match="No readable checkpoint"):
        make_trainer(tmp_path, "d").fit()


def test_resume_with_changed_lr_is_refused(tmp_path: Path) -> None:
    trainer = make_trainer(tmp_path, "e", callbacks=(RaiseAt(0, 3, KeyboardInterrupt),))
    with pytest.raises(KeyboardInterrupt):
        trainer.fit()
    with pytest.raises(ConfigMismatchError, match=r"(?s)training\.lr: 0\.05 -> 0\.01.*--force-new"):
        make_trainer(tmp_path, "e", lr=0.01).fit()


def test_resume_safe_setting_change_is_allowed(tmp_path: Path, straight_run: Trainer) -> None:
    trainer = make_trainer(tmp_path, "f", callbacks=(RaiseAt(0, 3, KeyboardInterrupt),))
    with pytest.raises(KeyboardInterrupt):
        trainer.fit()
    resumed = make_trainer(tmp_path, "f", checkpoint_every_minutes=5, eval_batch_size=4)
    assert resumed.fit().status == "finished"
    assert_same_weights(weights(straight_run), weights(resumed))


def test_resume_none_refuses_existing_run(tmp_path: Path) -> None:
    make_trainer(tmp_path, "g", max_epochs=1).fit()
    with pytest.raises(RunExistsError, match="--force-new"):
        make_trainer(tmp_path, "g", max_epochs=1).fit(resume="none")


def test_finished_run_is_not_retrained(tmp_path: Path) -> None:
    make_trainer(tmp_path, "h", max_epochs=1).fit()
    assert make_trainer(tmp_path, "h", max_epochs=1).fit().status == "already_finished"


@pytest.mark.skipif(not hasattr(signal, "SIGUSR1"), reason="SIGUSR1 not available")
def test_sigusr1_saves_and_exits_zero_then_resumes(tmp_path: Path,
                                                   straight_run: Trainer) -> None:
    first = make_trainer(tmp_path, "i", callbacks=(SignalAt(1, 2, signal.SIGUSR1),))
    result = first.fit()
    assert result.status == "stopped" and result.exit_code == 0
    assert (first.checkpoint_dir / LAST_FILE).exists()
    resumed = make_trainer(tmp_path, "i")
    assert resumed.fit().status == "finished"
    assert_same_weights(weights(straight_run), weights(resumed))


def test_sigint_stops_gracefully_and_restores_handler(tmp_path: Path) -> None:
    before = signal.getsignal(signal.SIGINT)
    result = make_trainer(tmp_path, "j", callbacks=(SignalAt(0, 1, signal.SIGINT),)).fit()
    assert result.status == "stopped" and result.exit_code == 130
    assert signal.getsignal(signal.SIGINT) is before


class NaNOnCalls(nn.Module):
    """Wraps a model and returns NaN logits on chosen training forward calls (1-based)."""

    def __init__(self, inner: nn.Module, bad_calls: set[int] | None = None,
                 nan_from: int | None = None) -> None:
        super().__init__()
        self.inner, self.bad_calls, self.nan_from, self.calls = inner, bad_calls or set(), \
            nan_from, 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.inner(x)
        if self.training:
            self.calls += 1
            if self.calls in self.bad_calls or (self.nan_from and self.calls >= self.nan_from):
                return out * float("nan")
        return out


def test_nan_loss_batch_is_skipped(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    torch.manual_seed(0)
    model = NaNOnCalls(DummyLinear(DIM, len(CLASS_NAMES)), bad_calls={3})
    trainer = make_trainer(tmp_path, "k", model=model, max_epochs=2)
    with caplog.at_level(logging.WARNING, logger="src.trainer"):
        assert trainer.fit().status == "finished"
    assert "Non-finite loss" in caplog.text
    assert [r["skipped_batches"] for r in trainer.state.history] == [1, 0]
    assert all(torch.isfinite(p).all() for p in trainer.model.parameters())


def test_too_many_nan_losses_stop_and_keep_last_good_checkpoint(tmp_path: Path) -> None:
    torch.manual_seed(0)
    model = NaNOnCalls(DummyLinear(DIM, len(CLASS_NAMES)), nan_from=STEPS_PER_EPOCH + 1)
    trainer = make_trainer(tmp_path, "l", model=model)
    with pytest.raises(NonFiniteLossError, match="non-finite losses in epoch 2"):
        trainer.fit()
    last = torch.load(trainer.checkpoint_dir / LAST_FILE, weights_only=True)
    assert (last["epoch"], last["batch_in_epoch"], last["finished"]) == (1, 0, False)


class OOMAbove(nn.Module):
    """Raises a (simulated) out-of-memory error for training batches larger than `limit`."""

    def __init__(self, inner: nn.Module, limit: int) -> None:
        super().__init__()
        self.inner, self.limit = inner, limit

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training and x.shape[0] > self.limit:
            raise torch.OutOfMemoryError("CUDA out of memory (simulated)")
        return self.inner(x)


def test_out_of_memory_halves_micro_batch_with_same_result(tmp_path: Path) -> None:
    torch.manual_seed(0)
    reference = make_trainer(tmp_path, "m1", model=OOMAbove(DummyLinear(DIM, 3), limit=999))
    reference.fit()
    torch.manual_seed(0)
    small = make_trainer(tmp_path, "m2", model=OOMAbove(DummyLinear(DIM, 3), limit=3))
    assert small.fit().status == "finished"
    assert small.state.grad_accum == 4  # 8 -> 4 -> 2 per micro-batch
    for (name, a), (_, b) in zip(reference.model.state_dict().items(),
                                 small.model.state_dict().items(), strict=True):
        assert torch.allclose(a, b, atol=1e-5), name


def test_early_stopping(tmp_path: Path) -> None:
    trainer = make_trainer(tmp_path, "n", lr=0.0, patience=1, max_epochs=5)
    result = trainer.fit()
    assert result.status == "early_stopped"
    assert result.epochs_completed == 2 and result.best_epoch == 1


def test_epoch_order_is_deterministic_and_differs_per_epoch() -> None:
    assert torch.equal(epoch_permutation(50, 7, 2), epoch_permutation(50, 7, 2))
    assert not torch.equal(epoch_permutation(50, 7, 2), epoch_permutation(50, 7, 3))
    assert sorted(epoch_permutation(50, 7, 2).tolist()) == list(range(50))


def test_torn_history_file_is_repaired_on_resume(tmp_path: Path) -> None:
    trainer = make_trainer(tmp_path, "o", callbacks=(RaiseAt(1, 2, KeyboardInterrupt),))
    with pytest.raises(KeyboardInterrupt):
        trainer.fit()
    with open(trainer.results_dir / HISTORY_FILE, "a") as f:
        f.write('{"epoch": 2, "trunc')
    resumed = make_trainer(tmp_path, "o")
    resumed.fit()
    lines = (resumed.results_dir / HISTORY_FILE).read_text().splitlines()
    assert [json.loads(x)["epoch"] for x in lines] == [1, 2, 3]


def test_settings_reject_unknown_and_missing_keys() -> None:
    with pytest.raises(KeyError, match="unknown keys"):
        TrainingSettings.from_config({"batch_size": 8, "typo": 1}, seed=1)
    with pytest.raises(ValueError, match="scheduler"):
        replace(SETTINGS, scheduler="exponential")
