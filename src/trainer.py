"""Crash-safe training loop for the PyTorch models (MLP, TextCNN, BiLSTM).

Guarantees:
- Deterministic batch order: epoch e is shuffled by a generator seeded from (seed, e).
- Resumable at any moment: last.pt is written atomically every `checkpoint_every_minutes`,
  at every epoch end and on SIGINT / SIGTERM / SIGUSR1. It holds the model, optimizer,
  scheduler and AMP scaler state, the position inside the epoch, early-stopping state, RNG
  states and the history, so an interrupted run resumed with `resume="auto"` produces exactly
  the same weights as an uninterrupted one.
- The previous checkpoint is kept as last.prev.pt; a corrupted newest file falls back to it.
- Resuming with a different config is refused (see ConfigMismatchError).
- Non-finite losses are skipped (and abort after too many); CUDA/MPS out-of-memory halves the
  micro-batch and accumulates gradients so the effective batch size stays the same.

The Transformer does not use this class; it uses Hugging Face's Trainer (see
src/models/transformer.py).
"""

from __future__ import annotations

import gc
import hashlib
import json
import logging
import math
import os
import pickle
import signal
import threading
import time
import zipfile
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field, fields
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import scipy.sparse as sp
import torch
from torch import nn

from src.config import get_device
from src.evaluate import evaluate
from src.utils.io import atomic_open, atomic_torch_save
from src.utils.seed import get_rng_state, set_rng_state

logger = logging.getLogger(__name__)

CHECKPOINT_VERSION = 1
LAST_FILE = "last.pt"
PREV_FILE = "last.prev.pt"
NEXT_FILE = "last.next.pt"
BEST_FILE = "best.pt"
HISTORY_FILE = "history.jsonl"
SCHEDULERS = ("none", "linear", "cosine")
# Settings that may change between resume attempts without changing the result.
RESUME_SAFE_SETTINGS = frozenset({"checkpoint_every_minutes", "log_every_steps",
                                  "eval_batch_size"})
# History fields that depend on wall-clock time (excluded when comparing runs).
TIMING_KEYS = frozenset({"epoch_seconds", "train_time_seconds", "eta_seconds"})
STOP_SIGNALS = tuple(s for s in (getattr(signal, n, None) for n in
                                 ("SIGINT", "SIGTERM", "SIGUSR1")) if s is not None)
EXIT_ZERO_SIGNALS = frozenset(s for s in (getattr(signal, n, None) for n in
                                          ("SIGTERM", "SIGUSR1")) if s is not None)
SIGINT_EXIT_CODE = 130
CHECKPOINT_READ_ERRORS = (RuntimeError, EOFError, OSError, ValueError, KeyError,
                          pickle.UnpicklingError, zipfile.BadZipFile)
REQUIRED_CHECKPOINT_KEYS = frozenset({
    "version", "config_hash", "config", "model", "optimizer", "scheduler", "scaler", "epoch",
    "batch_in_epoch", "global_step", "best_metric", "best_epoch", "bad_epochs", "history",
    "train_time_seconds", "grad_accum", "epoch_stats", "finished", "status", "rng"})


class ConfigMismatchError(RuntimeError):
    """The checkpoint was produced with a different configuration."""


class RunExistsError(RuntimeError):
    """resume="none" was requested but the run folder already holds checkpoints."""


class NonFiniteLossError(RuntimeError):
    """Too many NaN/Inf losses in one epoch."""


@dataclass(frozen=True)
class TrainingSettings:
    """Validated training hyperparameters (the `training` config section plus the seed)."""

    batch_size: int
    eval_batch_size: int
    max_epochs: int
    lr: float
    weight_decay: float
    scheduler: str
    warmup_ratio: float
    max_grad_norm: float
    amp: bool
    patience: int
    min_delta: float
    checkpoint_every_minutes: float
    max_nonfinite_per_epoch: int
    log_every_steps: int
    human_label: str
    seed: int

    def __post_init__(self) -> None:
        if self.scheduler not in SCHEDULERS:
            raise ValueError(f"scheduler must be one of {SCHEDULERS}, got {self.scheduler!r}")
        for name in ("batch_size", "eval_batch_size", "max_epochs", "patience",
                     "log_every_steps"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be >= 1, got {getattr(self, name)}")
        if not 0.0 <= self.warmup_ratio < 1.0:
            raise ValueError(f"warmup_ratio must be in [0, 1), got {self.warmup_ratio}")

    @classmethod
    def from_config(cls, training: Mapping[str, Any], seed: int) -> TrainingSettings:
        """Build from a config mapping; unknown or missing keys are errors."""
        names = {f.name for f in fields(cls)} - {"seed"}
        unknown, missing = set(training) - names, names - set(training)
        if unknown or missing:
            raise KeyError(f"training config: unknown keys {sorted(unknown)}, "
                           f"missing keys {sorted(missing)}")
        return cls(**dict(training), seed=seed)

    def hashed(self) -> dict[str, Any]:
        """The settings that must match for a checkpoint to be resumable."""
        return {k: v for k, v in asdict(self).items() if k not in RESUME_SAFE_SETTINGS}


class ArrayDataset:
    """Model inputs (tensor, ndarray or scipy sparse matrix) plus integer labels."""

    def __init__(self, inputs: torch.Tensor | np.ndarray | sp.spmatrix,
                 labels: Sequence[int] | np.ndarray | torch.Tensor) -> None:
        self.inputs = inputs.tocsr() if sp.issparse(inputs) else inputs
        if isinstance(labels, torch.Tensor):
            self.labels = labels.detach().clone().long()
        else:  # copy: pandas 3 hands out read-only arrays
            self.labels = torch.tensor(np.array(labels, copy=True), dtype=torch.long)
        if self.inputs.shape[0] != len(self.labels):
            raise ValueError(f"inputs have {self.inputs.shape[0]} rows but there are "
                             f"{len(self.labels)} labels")

    def __len__(self) -> int:
        return len(self.labels)

    def batch(self, idx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (inputs, labels) for the given row indices, as CPU tensors."""
        rows = idx.numpy()
        if sp.issparse(self.inputs):
            x = torch.from_numpy(self.inputs[rows].toarray())
        elif isinstance(self.inputs, np.ndarray):
            x = torch.from_numpy(self.inputs[rows])
        else:
            x = self.inputs[idx]
        return x, self.labels[idx]


class Callback:
    """Hooks called by the Trainer. Subclass and override what you need."""

    def on_batch_end(self, trainer: Trainer, epoch: int, batch: int) -> None:
        """Called after each optimizer step (or skipped batch); `epoch` is 0-based."""

    def on_epoch_end(self, trainer: Trainer, epoch: int, record: Mapping[str, Any]) -> None:
        """Called after validation and checkpointing at the end of an epoch."""


@dataclass
class FitResult:
    """Outcome of Trainer.fit()."""

    status: str  # finished | early_stopped | stopped | already_finished
    epochs_completed: int
    best_metric: float | None
    best_epoch: int | None
    train_time_seconds: float
    checkpoint_dir: Path
    stop_signal: int | None = None

    @property
    def exit_code(self) -> int:
        """Process exit code: 0 except after a Ctrl+C stop (130)."""
        if self.status == "stopped" and self.stop_signal not in EXIT_ZERO_SIGNALS:
            return SIGINT_EXIT_CODE
        return 0


@dataclass
class _State:
    epoch: int = 0                 # 0-based epoch currently being trained
    batch_in_epoch: int = 0        # batches of this epoch already done
    global_step: int = 0
    best_metric: float | None = None
    best_epoch: int | None = None  # 1-based
    bad_epochs: int = 0
    history: list[dict[str, Any]] = field(default_factory=list)
    train_time_seconds: float = 0.0
    grad_accum: int = 1
    epoch_stats: dict[str, float] = field(default_factory=lambda: {
        "loss_sum": 0.0, "batches": 0, "nonfinite": 0})
    finished: bool = False
    status: str | None = None


def epoch_seed(seed: int, epoch: int) -> int:
    """Derive a 63-bit seed for one epoch's shuffle from (seed, epoch)."""
    digest = hashlib.sha256(f"{seed}:{epoch}".encode()).digest()
    return int.from_bytes(digest[:8], "little") >> 1


def epoch_permutation(n: int, seed: int, epoch: int) -> torch.Tensor:
    """The row order for one epoch; identical every time for the same (n, seed, epoch)."""
    generator = torch.Generator().manual_seed(epoch_seed(seed, epoch))
    return torch.randperm(n, generator=generator)


def _jsonable(obj: Any) -> Any:
    return json.loads(json.dumps(obj, sort_keys=True, default=str))


def config_hash(config: Mapping[str, Any]) -> str:
    """Stable hash of a JSON-serializable config."""
    payload = json.dumps(config, sort_keys=True, default=str).encode()
    return hashlib.sha256(payload).hexdigest()[:16]


def _flatten(obj: Any, prefix: str = "") -> dict[str, Any]:
    if isinstance(obj, Mapping):
        out: dict[str, Any] = {}
        for k, v in obj.items():
            out.update(_flatten(v, f"{prefix}{k}."))
        return out
    return {prefix.rstrip("."): obj}


def config_diff(old: Mapping[str, Any], new: Mapping[str, Any]) -> list[str]:
    """Human-readable list of differences, e.g. ["training.lr: 0.001 -> 0.002"]."""
    a, b = _flatten(old), _flatten(new)
    return [f"{k}: {a.get(k, '<missing>')!r} -> {b.get(k, '<missing>')!r}"
            for k in sorted(set(a) | set(b)) if a.get(k) != b.get(k)]


def make_lr_lambda(scheduler: str, total_steps: int, warmup_steps: int) -> Callable[[int], float]:
    """LR multiplier: linear warmup, then constant / linear decay / cosine decay to 0."""
    def factor(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        if scheduler == "none":
            return 1.0
        progress = min(1.0, (step - warmup_steps) / max(1, total_steps - warmup_steps))
        if scheduler == "linear":
            return 1.0 - progress
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return factor


def _is_out_of_memory(err: BaseException) -> bool:
    return isinstance(err, torch.OutOfMemoryError) or (
        isinstance(err, RuntimeError) and "out of memory" in str(err).lower())


def _free_device_memory(device: torch.device) -> None:
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps":
        torch.mps.empty_cache()


def _format_seconds(seconds: float) -> str:
    seconds = int(round(seconds))
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    return f"{hours}h{minutes:02d}m{secs:02d}s" if hours else f"{minutes}m{secs:02d}s"


def _append_jsonl(path: Path, record: Mapping[str, Any]) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
        f.flush()
        os.fsync(f.fileno())


class Trainer:
    """Trains one model with checkpointing, resume, early stopping and safety guards.

    Args:
        model: Module mapping a batch of inputs to logits.
        train_data / val_data: Datasets; the test split is never passed here.
        settings: Training hyperparameters.
        class_weights: Per-class loss weights (from results/class_weights.json).
        class_names: Class names in label-id order.
        run_config: Everything else that defines the run (model params, features, data
            fingerprints, subset); hashed together with `settings` to detect config changes.
        checkpoint_dir: Folder for last.pt / last.prev.pt / best.pt.
        results_dir: Folder for history.jsonl.
        device: Torch device; auto-detected (cuda > mps > cpu) when None.
        callbacks: Optional hooks (used by tests to simulate crashes).
    """

    def __init__(self, model: nn.Module, train_data: ArrayDataset, val_data: ArrayDataset, *,
                 settings: TrainingSettings, class_weights: torch.Tensor,
                 class_names: Sequence[str], run_config: Mapping[str, Any],
                 checkpoint_dir: Path, results_dir: Path,
                 device: str | torch.device | None = None,
                 callbacks: Sequence[Callback] = ()) -> None:
        if len(class_weights) != len(class_names):
            raise ValueError(f"{len(class_weights)} class weights for {len(class_names)} classes")
        self.settings = settings
        self.device = torch.device(device) if device is not None else get_device()
        self.model = model.to(self.device)
        self.train_data, self.val_data = train_data, val_data
        self.class_names = list(class_names)
        self.class_weights = class_weights.float().cpu()
        self.callbacks = list(callbacks)
        self.checkpoint_dir, self.results_dir = Path(checkpoint_dir), Path(results_dir)

        self.criterion = nn.CrossEntropyLoss(weight=self.class_weights.to(self.device),
                                             reduction="none")
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=settings.lr,
                                           weight_decay=settings.weight_decay)
        self.steps_per_epoch = math.ceil(len(train_data) / settings.batch_size)
        total_steps = self.steps_per_epoch * settings.max_epochs
        warmup_steps = round(settings.warmup_ratio * total_steps)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, make_lr_lambda(settings.scheduler, total_steps, warmup_steps))
        self.use_amp = settings.amp and self.device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)

        self.full_config = _jsonable({"run": dict(run_config), "training": settings.hashed()})
        self.config_hash = config_hash(self.full_config)
        self.state = _State()
        self._stop_signal: int | None = None
        self._unsafe_to_save = False  # True while state is mid-update
        self._segment_start = time.monotonic()
        self._time_base = 0.0
        self._last_save = time.monotonic()

    # ----------------------------------------------------------------------------- public API
    @property
    def num_parameters(self) -> int:
        """Number of trainable parameters."""
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def fit(self, resume: str = "auto") -> FitResult:
        """Train until max_epochs or early stopping, resuming from last.pt if present.

        Args:
            resume: "auto" continues from the newest valid checkpoint; "none" requires an
                empty run folder (raises RunExistsError otherwise).
        """
        if resume not in ("auto", "none"):
            raise ValueError(f"resume must be 'auto' or 'none', got {resume!r}")
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        candidates = self._checkpoint_candidates()
        if candidates and resume == "none":
            raise RunExistsError(f"{self.checkpoint_dir} already has checkpoints; use "
                                 "--resume auto to continue it, or --force-new for a new run")
        if candidates:
            self._resume(candidates)
        self._sync_history_file()
        if self.state.finished:
            logger.info("Run already finished (%s); nothing to do", self.state.status)
            return self._result("already_finished")

        self._segment_start = self._last_save = time.monotonic()
        self._time_base = self.state.train_time_seconds
        logger.info("Training on %s: %d train rows, %d batches/epoch, %d parameters%s",
                    self.device, len(self.train_data), self.steps_per_epoch,
                    self.num_parameters, ", mixed precision" if self.use_amp else "")
        with self._signal_handlers():
            try:
                status = self._train_loop()
            except KeyboardInterrupt:
                if self._unsafe_to_save:
                    logger.warning("Interrupted mid-update; keeping the last saved checkpoint")
                else:
                    self._save_last()
                    logger.warning("Interrupted: checkpoint saved, safe to resume")
                raise
        return self._result(status)

    def predict_proba(self, data: ArrayDataset) -> np.ndarray:
        """Class probabilities for every row of `data`, in order."""
        self.model.eval()
        batch_size = self.settings.eval_batch_size
        order = torch.arange(len(data))
        outputs, start = [], 0
        with torch.no_grad():
            while start < len(data):
                idx = order[start:start + batch_size]
                try:
                    x, _ = data.batch(idx)
                    with torch.autocast(self.device.type, dtype=torch.float16,
                                        enabled=self.use_amp):
                        logits = self.model(x.to(self.device))
                    outputs.append(torch.softmax(logits.float(), dim=-1).cpu())
                    start += len(idx)
                except RuntimeError as err:
                    if not _is_out_of_memory(err) or batch_size == 1:
                        raise
                    _free_device_memory(self.device)
                    batch_size = max(1, batch_size // 2)
                    logger.warning("Out of memory during evaluation; eval batch -> %d",
                                   batch_size)
        return torch.cat(outputs).numpy()

    # ----------------------------------------------------------------------------- training
    def _train_loop(self) -> str:
        s = self.state
        while s.epoch < self.settings.max_epochs:
            epoch_start = time.monotonic()
            batches = epoch_permutation(len(self.train_data), self.settings.seed,
                                        s.epoch).split(self.settings.batch_size)
            if s.batch_in_epoch:
                logger.info("Resuming epoch %d at batch %d/%d", s.epoch + 1, s.batch_in_epoch,
                            len(batches))
            for b in range(s.batch_in_epoch, len(batches)):
                self._unsafe_to_save = True
                self._train_step(batches[b])
                self._unsafe_to_save = False
                for cb in self.callbacks:
                    cb.on_batch_end(self, s.epoch, b)
                if self._stop_signal is not None:
                    return self._stop_now()
                if (time.monotonic() - self._last_save
                        >= self.settings.checkpoint_every_minutes * 60):
                    self._save_last()
                if s.global_step % self.settings.log_every_steps == 0:
                    logger.info("epoch %d batch %d/%d  loss %.4f  lr %.2e", s.epoch + 1, b + 1,
                                len(batches), s.epoch_stats["loss_sum"]
                                / max(1, s.epoch_stats["batches"]),
                                self.optimizer.param_groups[0]["lr"])
            self._unsafe_to_save = True
            record = self._end_epoch(epoch_start)
            self._unsafe_to_save = False
            early_stop = s.bad_epochs >= self.settings.patience
            if early_stop or s.epoch >= self.settings.max_epochs:
                s.finished, s.status = True, "early_stopped" if early_stop else "finished"
            self._save_last()
            for cb in self.callbacks:
                cb.on_epoch_end(self, s.epoch - 1, record)
            if early_stop:
                logger.info("Early stopping: no val macro-F1 gain for %d epochs (best %.4f at "
                            "epoch %d)", s.bad_epochs, s.best_metric, s.best_epoch)
                return "early_stopped"
            if self._stop_signal is not None and not s.finished:
                return self._stop_now()
        s.finished, s.status = True, "finished"
        return "finished"

    def _stop_now(self) -> str:
        self._save_last()
        logger.warning("Stopped by %s: checkpoint saved, safe to resume",
                       signal.Signals(self._stop_signal).name)
        return "stopped"

    def _train_step(self, idx: torch.Tensor) -> None:
        s = self.state
        inputs, labels = self.train_data.batch(idx)
        total_weight = float(self.class_weights[labels].sum())
        while True:
            try:
                loss = self._forward_backward(inputs, labels, total_weight)
                break
            except RuntimeError as err:
                if not _is_out_of_memory(err):
                    raise
                self._recover_from_oom(len(labels), err)

        if loss is None:
            s.epoch_stats["nonfinite"] += 1
            logger.warning("Non-finite loss in epoch %d batch %d; batch skipped (%d this "
                           "epoch)", s.epoch + 1, s.batch_in_epoch + 1,
                           s.epoch_stats["nonfinite"])
            if s.epoch_stats["nonfinite"] > self.settings.max_nonfinite_per_epoch:
                raise NonFiniteLossError(
                    f"{s.epoch_stats['nonfinite']} non-finite losses in epoch {s.epoch + 1} "
                    f"(limit {self.settings.max_nonfinite_per_epoch}); stopping. The last "
                    f"good checkpoint in {self.checkpoint_dir} is kept. Try a lower lr.")
        else:
            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(self.model.parameters(), self.settings.max_grad_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            s.epoch_stats["loss_sum"] += loss
            s.epoch_stats["batches"] += 1
        self.optimizer.zero_grad(set_to_none=True)
        self.scheduler.step()
        s.batch_in_epoch += 1
        s.global_step += 1

    def _forward_backward(self, inputs: torch.Tensor, labels: torch.Tensor,
                          total_weight: float) -> float | None:
        """Accumulate gradients over micro-batches; return the batch loss or None if NaN/Inf.

        Each micro-batch loss is its weighted sum divided by the weight of the WHOLE batch,
        so the summed gradient equals the full-batch weighted mean regardless of splitting.
        """
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        micro = math.ceil(len(labels) / self.state.grad_accum)
        total = 0.0
        for start in range(0, len(labels), micro):
            x = inputs[start:start + micro].to(self.device, non_blocking=True)
            y = labels[start:start + micro].to(self.device, non_blocking=True)
            with torch.autocast(self.device.type, dtype=torch.float16, enabled=self.use_amp):
                logits = self.model(x)
            loss = self.criterion(logits.float(), y).sum() / total_weight
            if not torch.isfinite(loss):
                self.optimizer.zero_grad(set_to_none=True)
                return None
            self.scaler.scale(loss).backward()
            total += loss.item()
        return total

    def _recover_from_oom(self, batch_size: int, err: RuntimeError) -> None:
        self.optimizer.zero_grad(set_to_none=True)
        _free_device_memory(self.device)
        old_micro = math.ceil(batch_size / self.state.grad_accum)
        if old_micro <= 1:
            raise RuntimeError("Out of memory even with a micro-batch of 1; use a smaller "
                               "model or max_len") from err
        self.state.grad_accum *= 2
        new_micro = math.ceil(batch_size / self.state.grad_accum)
        logger.warning("Out of memory: micro-batch %d -> %d with %dx gradient accumulation "
                       "(effective batch stays %d)", old_micro, new_micro,
                       self.state.grad_accum, self.settings.batch_size)

    def _end_epoch(self, epoch_start: float) -> dict[str, Any]:
        s = self.state
        y_prob = self.predict_proba(self.val_data)
        metrics = evaluate(self.val_data.labels.numpy(), y_prob.argmax(axis=1), y_prob,
                           self.class_names, self.settings.human_label)
        improved = (s.best_metric is None
                    or metrics["macro_f1"] > s.best_metric + self.settings.min_delta)
        if improved:
            s.best_metric, s.best_epoch, s.bad_epochs = metrics["macro_f1"], s.epoch + 1, 0
            self._save_best(metrics)
        else:
            s.bad_epochs += 1
        self._update_train_time()
        epoch_seconds = time.monotonic() - epoch_start
        stats = s.epoch_stats
        record = {
            "epoch": s.epoch + 1,
            "global_step": s.global_step,
            "train_loss": stats["loss_sum"] / stats["batches"] if stats["batches"] else None,
            "val_macro_f1": metrics["macro_f1"],
            "val_accuracy": metrics["accuracy"],
            "val_human_fpr": metrics["human_fpr"],
            "val_log_loss": metrics["log_loss"],
            "lr": self.optimizer.param_groups[0]["lr"],
            "skipped_batches": int(stats["nonfinite"]),
            "grad_accum": s.grad_accum,
            "is_best": improved,
            "epoch_seconds": epoch_seconds,
            "train_time_seconds": s.train_time_seconds,
            "eta_seconds": epoch_seconds * (self.settings.max_epochs - s.epoch - 1),
        }
        s.history.append(record)
        _append_jsonl(self.results_dir / HISTORY_FILE, record)
        logger.info("Epoch %d/%d  train_loss %.4f  val_macro_f1 %.4f%s  human_fpr %s  | %s, "
                    "ETA %s", s.epoch + 1, self.settings.max_epochs, record["train_loss"] or 0.0,
                    metrics["macro_f1"], " (best)" if improved else "",
                    "n/a" if metrics["human_fpr"] is None else f"{metrics['human_fpr']:.4f}",
                    _format_seconds(epoch_seconds), _format_seconds(record["eta_seconds"]))
        s.epoch += 1
        s.batch_in_epoch = 0
        s.epoch_stats = {"loss_sum": 0.0, "batches": 0, "nonfinite": 0}
        return record

    # ----------------------------------------------------------------------------- checkpoints
    def _update_train_time(self) -> None:
        self.state.train_time_seconds = self._time_base + (time.monotonic()
                                                           - self._segment_start)

    def _checkpoint(self) -> dict[str, Any]:
        s = self.state
        return {
            "version": CHECKPOINT_VERSION,
            "config_hash": self.config_hash,
            "config": self.full_config,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "scaler": self.scaler.state_dict(),
            "epoch": s.epoch,
            "batch_in_epoch": s.batch_in_epoch,
            "global_step": s.global_step,
            "best_metric": s.best_metric,
            "best_epoch": s.best_epoch,
            "bad_epochs": s.bad_epochs,
            "history": s.history,
            "train_time_seconds": s.train_time_seconds,
            "grad_accum": s.grad_accum,
            "epoch_stats": dict(s.epoch_stats),
            "finished": s.finished,
            "status": s.status,
            "rng": get_rng_state(),
            "saved_at": datetime.now(UTC).isoformat(timespec="seconds"),
        }

    def _save_last(self) -> None:
        """Write last.pt atomically, keeping the previous one as last.prev.pt.

        Order: complete last.next.pt -> rename last.pt to last.prev.pt -> rename last.next.pt
        to last.pt. A crash at any point leaves at least one complete checkpoint.
        """
        self._update_train_time()
        nxt, last, prev = (self.checkpoint_dir / n for n in (NEXT_FILE, LAST_FILE, PREV_FILE))
        atomic_torch_save(self._checkpoint(), nxt)
        if last.exists():
            os.replace(last, prev)
        os.replace(nxt, last)
        self._last_save = time.monotonic()

    def _save_best(self, metrics: Mapping[str, Any]) -> None:
        atomic_torch_save({
            "model": self.model.state_dict(),
            "epoch": self.state.epoch + 1,
            "val_macro_f1": metrics["macro_f1"],
            "val_accuracy": metrics["accuracy"],
            "val_human_fpr": metrics["human_fpr"],
            "config_hash": self.config_hash,
            "config": self.full_config,
        }, self.checkpoint_dir / BEST_FILE)

    def _checkpoint_candidates(self) -> list[Path]:
        names = (NEXT_FILE, LAST_FILE, PREV_FILE)  # newest first
        return [p for p in (self.checkpoint_dir / n for n in names) if p.exists()]

    @staticmethod
    def _read_checkpoint(path: Path) -> dict[str, Any]:
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(ckpt, dict) or not REQUIRED_CHECKPOINT_KEYS <= set(ckpt):
            raise ValueError(f"{path} is not a complete training checkpoint")
        if ckpt["version"] != CHECKPOINT_VERSION:
            raise ValueError(f"{path} has checkpoint version {ckpt['version']}, expected "
                             f"{CHECKPOINT_VERSION}")
        return ckpt

    def _resume(self, candidates: Sequence[Path]) -> None:
        ckpt, used = None, None
        for path in candidates:
            try:
                ckpt, used = self._read_checkpoint(path), path
                break
            except CHECKPOINT_READ_ERRORS as err:
                logger.warning("Could not read checkpoint %s (%s: %s); trying an older one",
                               path, type(err).__name__, err)
        if ckpt is None:
            raise RuntimeError(f"No readable checkpoint in {self.checkpoint_dir} (tried "
                               f"{[p.name for p in candidates]}). Move the folder aside to "
                               "start over.")
        if ckpt["config_hash"] != self.config_hash:
            changes = "\n  ".join(config_diff(ckpt["config"], self.full_config))
            raise ConfigMismatchError(
                f"Refusing to resume from {used}: the config changed since it was saved:\n  "
                f"{changes}\nRerun with the original settings to continue, or pass --force-new "
                "to start a fresh run in a new folder.")

        self.model.load_state_dict(ckpt["model"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.scheduler.load_state_dict(ckpt["scheduler"])
        if self.use_amp and ckpt["scaler"]:
            self.scaler.load_state_dict(ckpt["scaler"])
        self.state = _State(
            epoch=ckpt["epoch"], batch_in_epoch=ckpt["batch_in_epoch"],
            global_step=ckpt["global_step"], best_metric=ckpt["best_metric"],
            best_epoch=ckpt["best_epoch"], bad_epochs=ckpt["bad_epochs"],
            history=list(ckpt["history"]), train_time_seconds=ckpt["train_time_seconds"],
            grad_accum=ckpt["grad_accum"], epoch_stats=dict(ckpt["epoch_stats"]),
            finished=ckpt["finished"], status=ckpt["status"])
        set_rng_state(ckpt["rng"])
        logger.info("Resumed from %s (saved %s): epoch %d, batch %d/%d, step %d", used.name,
                    ckpt.get("saved_at"), self.state.epoch + 1, self.state.batch_in_epoch,
                    self.steps_per_epoch, self.state.global_step)

    def _sync_history_file(self) -> None:
        """Make history.jsonl match the checkpoint's history (fixes torn or extra lines)."""
        path = self.results_dir / HISTORY_FILE
        lines = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
        try:
            on_disk = [json.loads(line) for line in lines]
        except json.JSONDecodeError:
            on_disk = None
        if on_disk == self.state.history:
            return
        if lines:
            logger.info("Rewriting %s to match the checkpoint (%d epochs)", path.name,
                        len(self.state.history))
        with atomic_open(path) as f:
            f.write("".join(json.dumps(r) + "\n" for r in self.state.history).encode("utf-8"))

    @contextmanager
    def _signal_handlers(self) -> Iterator[None]:
        if threading.current_thread() is not threading.main_thread():
            yield
            return
        previous = {sig: signal.signal(sig, self._on_signal) for sig in STOP_SIGNALS}
        try:
            yield
        finally:
            for sig, handler in previous.items():
                signal.signal(sig, handler)

    def _on_signal(self, signum: int, frame: object) -> None:
        if self._stop_signal is not None and signum == signal.SIGINT:
            logger.warning("Second Ctrl+C: stopping now; the last saved checkpoint is kept")
            raise KeyboardInterrupt
        self._stop_signal = signum
        logger.warning("Received %s: finishing the current batch, then saving a checkpoint",
                       signal.Signals(signum).name)

    def _result(self, status: str) -> FitResult:
        s = self.state
        return FitResult(status=status, epochs_completed=len(s.history),
                         best_metric=s.best_metric, best_epoch=s.best_epoch,
                         train_time_seconds=s.train_time_seconds,
                         checkpoint_dir=self.checkpoint_dir, stop_signal=self._stop_signal)
