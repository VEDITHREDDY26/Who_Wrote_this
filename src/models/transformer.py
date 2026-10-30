"""Fine-tuned Transformer classifier (DistilBERT by default, RoBERTa-base optional).

Trained with Hugging Face's Trainer instead of src.trainer.Trainer, with the same guarantees:
- Same splits, seed and class weights (weighted cross-entropy), and the shared evaluate() /
  save_results() for the final val metrics.
- Resumable: a checkpoint is saved at every epoch end, every `checkpoint_every_minutes` and on
  SIGINT / SIGTERM / SIGUSR1. Rerunning the same command resumes from the newest complete
  checkpoint (one whose trainer_state.json, which Hugging Face writes last, parses).
- Resuming with a different config is refused (ConfigMismatchError).
- Early stopping on val macro-F1; the best epoch's weights are exported to best_model/.
- Mixed precision on CUDA (bf16 when supported, otherwise fp16); a CUDA out-of-memory error
  retries with a smaller batch (auto_find_batch_size).
"""

from __future__ import annotations

import json
import logging
import signal
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, fields
from pathlib import Path
from types import FrameType
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    EvalPrediction,
    PreTrainedTokenizerBase,
    PrinterCallback,
    Trainer,
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)

from src.config import get_device, get_path
from src.evaluate import evaluate, save_results
from src.trainer import (
    HISTORY_FILE,
    STOP_SIGNALS,
    ConfigMismatchError,
    FitResult,
    RunExistsError,
    TrainingSettings,
    config_diff,
    config_hash,
    to_jsonable,
)
from src.utils.io import atomic_open, atomic_write_json, safe_load_json

logger = logging.getLogger(__name__)

RUN_CONFIG_FILE = "run_config.json"
TIMING_FILE = "timing.json"
DONE_FILE = "done.json"
BEST_MODEL_DIR = "best_model"
CHECKPOINT_PREFIX = "checkpoint-"
TRAINER_STATE_FILE = "trainer_state.json"
METRIC_NAME = "macro_f1"
SAVE_TOTAL_LIMIT = 2  # the best checkpoint and the newest one
HF_SCHEDULERS = {"none": "constant_with_warmup", "linear": "linear", "cosine": "cosine"}


@dataclass(frozen=True)
class TransformerParams:
    """The `models.transformer` config section (without its `training` block)."""

    pretrained_name: str
    max_len: int

    @classmethod
    def from_config(cls, params: Mapping[str, Any]) -> TransformerParams:
        """Validate the config section; unknown or missing keys are errors."""
        names = {f.name for f in fields(cls)}
        unknown, missing = set(params) - names, names - set(params)
        if unknown or missing:
            raise KeyError(f"models.transformer: unknown keys {sorted(unknown)}, "
                           f"missing keys {sorted(missing)}")
        out = cls(pretrained_name=str(params["pretrained_name"]), max_len=int(params["max_len"]))
        if out.max_len < 2:
            raise ValueError(f"models.transformer.max_len must be >= 2, got {out.max_len}")
        return out


@dataclass
class TransformerOutcome:
    """What train_transformer() returns to the CLI."""

    result: FitResult
    metrics: dict[str, Any] | None  # val metrics, None when training was stopped
    num_parameters: int


class EncodedTexts(Dataset):
    """Tokenized texts (unpadded) plus integer labels, one dict per row for the collator."""

    def __init__(self, encodings: Mapping[str, list[list[int]]], labels: np.ndarray) -> None:
        lengths = {len(v) for v in encodings.values()}
        if lengths != {len(labels)}:
            raise ValueError(f"encodings have lengths {sorted(lengths)} but there are "
                             f"{len(labels)} labels")
        self.encodings = dict(encodings)
        self.labels = np.asarray(labels, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, i: int) -> dict[str, Any]:
        return {k: v[i] for k, v in self.encodings.items()} | {"labels": int(self.labels[i])}


def encode_texts(tokenizer: PreTrainedTokenizerBase, texts: Sequence[str], labels: np.ndarray,
                 max_len: int) -> EncodedTexts:
    """Tokenize and truncate to `max_len` tokens; padding happens per batch in the collator."""
    enc = tokenizer(list(texts), truncation=True, max_length=max_len)
    return EncodedTexts({k: enc[k] for k in enc}, labels)


def complete_checkpoints(output_dir: Path) -> list[Path]:
    """Checkpoint folders whose trainer_state.json parses, newest first.

    Hugging Face writes trainer_state.json last, so a folder without a valid one was cut off
    mid-save and is skipped.
    """
    found: list[tuple[int, Path]] = []
    for path in output_dir.glob(f"{CHECKPOINT_PREFIX}*"):
        step = path.name.removeprefix(CHECKPOINT_PREFIX)
        if not (path.is_dir() and step.isdigit()):
            continue
        try:
            safe_load_json(path / TRAINER_STATE_FILE)
        except (FileNotFoundError, ValueError) as err:
            logger.warning("Ignoring incomplete checkpoint %s (%s)", path.name, err)
            continue
        found.append((int(step), path))
    return [path for _, path in sorted(found, reverse=True)]


def mixed_precision(device: torch.device, enabled: bool) -> dict[str, bool]:
    """TrainingArguments flags: bf16 on CUDA when supported, else fp16; off elsewhere."""
    if not enabled or device.type != "cuda":
        return {"bf16": False, "fp16": False}
    bf16 = torch.cuda.is_bf16_supported()
    return {"bf16": bf16, "fp16": not bf16}


def softmax(logits: np.ndarray) -> np.ndarray:
    """Row-wise softmax in float64."""
    return torch.softmax(torch.as_tensor(logits, dtype=torch.float64), dim=-1).numpy()


class WeightedLossTrainer(Trainer):
    """Hugging Face Trainer with the shared class-weighted cross-entropy loss."""

    def __init__(self, *args: Any, class_weights: Sequence[float], **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.class_weights = torch.tensor(list(class_weights), dtype=torch.float32)

    def compute_loss(self, model: torch.nn.Module, inputs: dict[str, Any],
                     return_outputs: bool = False,
                     num_items_in_batch: torch.Tensor | int | None = None) -> Any:
        """Weighted mean cross-entropy over the batch (same loss as src.trainer)."""
        outputs = model(**{k: v for k, v in inputs.items() if k != "labels"})
        weight = self.class_weights.to(outputs.logits.device)
        loss = F.cross_entropy(outputs.logits.float(), inputs["labels"], weight=weight)
        return (loss, outputs) if return_outputs else loss


def make_compute_metrics(class_names: Sequence[str],
                         human_label: str) -> Callable[[EvalPrediction], dict[str, float]]:
    """Per-epoch validation metrics for the Trainer, computed with the shared evaluate()."""
    def compute(pred: EvalPrediction) -> dict[str, float]:
        logits = pred.predictions[0] if isinstance(pred.predictions, tuple) else pred.predictions
        prob = softmax(logits)
        m = evaluate(pred.label_ids, prob.argmax(axis=1), prob, class_names, human_label)
        out = {METRIC_NAME: m["macro_f1"], "accuracy": m["accuracy"]}
        if m["human_fpr"] is not None:
            out["human_fpr"] = m["human_fpr"]
        return out
    return compute


class TimedCheckpointCallback(TrainerCallback):
    """Ask for a checkpoint every `minutes` of training, on top of the epoch-end saves."""

    def __init__(self, minutes: float) -> None:
        self.interval = minutes * 60
        self._last = time.monotonic()

    def on_step_end(self, args: TrainingArguments, state: TrainerState,
                    control: TrainerControl, **kwargs: Any) -> TrainerControl:
        if time.monotonic() - self._last >= self.interval:
            control.should_save = True
        return control

    def on_save(self, args: TrainingArguments, state: TrainerState,
                control: TrainerControl, **kwargs: Any) -> None:
        self._last = time.monotonic()


class TrainTimeCallback(TrainerCallback):
    """Training wall time summed over resumes, kept in <output_dir>/timing.json."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.base = float(safe_load_json(path, default={}).get("train_time_seconds", 0.0))
        self._start: float | None = None

    @property
    def total(self) -> float:
        """Seconds trained so far, including earlier attempts."""
        return self.base + (time.monotonic() - self._start if self._start is not None else 0.0)

    def on_train_begin(self, args: TrainingArguments, state: TrainerState,
                       control: TrainerControl, **kwargs: Any) -> None:
        self._start = time.monotonic()

    def on_save(self, args: TrainingArguments, state: TrainerState,
                control: TrainerControl, **kwargs: Any) -> None:
        atomic_write_json(self.path, {"train_time_seconds": self.total})


class StopOnSignalCallback(TrainerCallback):
    """On SIGINT / SIGTERM / SIGUSR1: finish the current step, save a checkpoint and stop."""

    def __init__(self) -> None:
        self.signal: int | None = None
        self._previous: dict[int, Any] = {}

    def _handle(self, signum: int, frame: FrameType | None) -> None:
        self.signal = signum
        logger.warning("Received %s: saving a checkpoint and stopping after this step",
                       signal.Signals(signum).name)

    def on_train_begin(self, args: TrainingArguments, state: TrainerState,
                       control: TrainerControl, **kwargs: Any) -> None:
        if threading.current_thread() is threading.main_thread():
            self._previous = {s: signal.signal(s, self._handle) for s in STOP_SIGNALS}

    def on_step_end(self, args: TrainingArguments, state: TrainerState,
                    control: TrainerControl, **kwargs: Any) -> TrainerControl:
        if self.signal is not None:
            control.should_save = True
            control.should_training_stop = True
        return control

    def on_train_end(self, args: TrainingArguments, state: TrainerState,
                     control: TrainerControl, **kwargs: Any) -> None:
        for s, handler in self._previous.items():
            signal.signal(s, handler)
        self._previous = {}


class LogCallback(TrainerCallback):
    """Send the Trainer's loss / eval logs to our logger (and so to train.log)."""

    def on_log(self, args: TrainingArguments, state: TrainerState, control: TrainerControl,
               logs: Mapping[str, Any] | None = None, **kwargs: Any) -> None:
        if logs:
            shown = {k: round(v, 4) if isinstance(v, float) else v for k, v in logs.items()}
            logger.info("step %d/%d  epoch %.2f  %s", state.global_step, state.max_steps,
                        state.epoch or 0.0, shown)


def prepare_run(checkpoint_dir: Path, full_config: Mapping[str, Any], resume: str) -> None:
    """Refuse to reuse a run folder with another config, or at all with resume="none"."""
    if resume not in ("auto", "none"):
        raise ValueError(f"resume must be 'auto' or 'none', got {resume!r}")
    used = bool(complete_checkpoints(checkpoint_dir)) or (checkpoint_dir / DONE_FILE).exists()
    if used and resume == "none":
        raise RunExistsError(f"{checkpoint_dir} already has checkpoints; use --resume auto to "
                             "continue it, or --force-new for a new run")
    path = checkpoint_dir / RUN_CONFIG_FILE
    saved = safe_load_json(path, default=None)
    if saved is None:
        atomic_write_json(path, {"config_hash": config_hash(full_config), "config": full_config})
    elif saved["config_hash"] != config_hash(full_config):
        changes = "\n  ".join(config_diff(saved["config"], full_config))
        raise ConfigMismatchError(
            f"Refusing to resume {checkpoint_dir}: the config changed since it was started:\n  "
            f"{changes}\nRerun with the original settings to continue, or pass --force-new "
            "to start a fresh run in a new folder.")


def eval_history(state: TrainerState) -> list[dict[str, Any]]:
    """One record per epoch-end evaluation, from the Trainer's log history."""
    return [{"epoch": round(r["epoch"]), "global_step": r["step"],
             "val_macro_f1": r[f"eval_{METRIC_NAME}"], "val_accuracy": r.get("eval_accuracy"),
             "val_human_fpr": r.get("eval_human_fpr"), "val_loss": r.get("eval_loss")}
            for r in state.log_history if f"eval_{METRIC_NAME}" in r]


def build_training_args(checkpoint_dir: Path, settings: TrainingSettings,
                        device: torch.device) -> TrainingArguments:
    """Map the shared TrainingSettings onto Hugging Face TrainingArguments."""
    return TrainingArguments(
        output_dir=str(checkpoint_dir),
        per_device_train_batch_size=settings.batch_size,
        per_device_eval_batch_size=settings.eval_batch_size,
        num_train_epochs=settings.max_epochs,
        learning_rate=settings.lr,
        weight_decay=settings.weight_decay,
        lr_scheduler_type=HF_SCHEDULERS[settings.scheduler],
        warmup_steps=float(settings.warmup_ratio),  # a float in [0, 1) is a ratio
        max_grad_norm=settings.max_grad_norm,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=SAVE_TOTAL_LIMIT,
        load_best_model_at_end=True,
        metric_for_best_model=METRIC_NAME,
        greater_is_better=True,
        logging_steps=settings.log_every_steps,
        seed=settings.seed,
        data_seed=settings.seed,
        use_cpu=device.type == "cpu",
        auto_find_batch_size=device.type == "cuda",
        report_to="none",
        disable_tqdm=True,
        dataloader_pin_memory=device.type == "cuda",
        **mixed_precision(device, settings.amp),
    )


def train_transformer(splits: Mapping[str, pd.DataFrame], *, params: Mapping[str, Any],
                      settings: TrainingSettings, class_weights: Sequence[float],
                      class_names: Sequence[str], run_config: Mapping[str, Any],
                      checkpoint_dir: Path, results_dir: Path, label: str, run_name: str,
                      resume: str, subset: int | None) -> TransformerOutcome:
    """Fine-tune the Transformer on splits["train"], pick the best epoch on splits["val"].

    Writes checkpoints to `checkpoint_dir`, history.jsonl to `results_dir`, and the val
    metrics (via save_results) as results/<label>_val_*.
    """
    hp = TransformerParams.from_config(params)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    full_config = to_jsonable({"run": dict(run_config), "training": settings.hashed()})
    prepare_run(checkpoint_dir, full_config, resume)

    done = safe_load_json(checkpoint_dir / DONE_FILE, default=None)
    if done is not None:
        logger.info("Run already finished (%s); nothing to do", done["status"])
        result = FitResult("already_finished", done["epochs_completed"], done["best_metric"],
                           done["best_epoch"], done["train_time_seconds"], checkpoint_dir)
        return TransformerOutcome(result, done["metrics"], done["num_parameters"])

    device = get_device()
    tokenizer = AutoTokenizer.from_pretrained(hp.pretrained_name)
    logger.info("Tokenizing %s (max_len %d)", ", ".join(f"{n}: {len(df)}"
                                                       for n, df in splits.items()), hp.max_len)
    data = {name: encode_texts(tokenizer, df["text"].tolist(), df["label_id"].to_numpy(),
                               hp.max_len) for name, df in splits.items()}
    model = AutoModelForSequenceClassification.from_pretrained(
        hp.pretrained_name, num_labels=len(class_names), id2label=dict(enumerate(class_names)),
        label2id={name: i for i, name in enumerate(class_names)})
    num_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    timer = TrainTimeCallback(checkpoint_dir / TIMING_FILE)
    stopper = StopOnSignalCallback()
    trainer = WeightedLossTrainer(
        model=model, args=build_training_args(checkpoint_dir, settings, device),
        data_collator=DataCollatorWithPadding(tokenizer), train_dataset=data["train"],
        eval_dataset=data["val"], processing_class=tokenizer,
        compute_metrics=make_compute_metrics(class_names, settings.human_label),
        callbacks=[EarlyStoppingCallback(settings.patience, settings.min_delta),
                   TimedCheckpointCallback(settings.checkpoint_every_minutes), timer, stopper,
                   LogCallback()],
        class_weights=class_weights)
    trainer.remove_callback(PrinterCallback)  # LogCallback already logs every step

    checkpoints = complete_checkpoints(checkpoint_dir)
    resume_from = str(checkpoints[0]) if checkpoints else None
    logger.info("Training %s on %s: %d train rows, %d parameters%s%s", hp.pretrained_name,
                device, len(data["train"]), num_parameters,
                ", mixed precision" if any(mixed_precision(device, settings.amp).values()) else "",
                f", resuming from {checkpoints[0].name}" if resume_from else "")
    trainer.train(resume_from_checkpoint=resume_from)

    history = eval_history(trainer.state)
    with atomic_open(results_dir / HISTORY_FILE) as f:
        f.write("".join(json.dumps(r) + "\n" for r in history).encode("utf-8"))
    best = max(history, key=lambda r: r["val_macro_f1"]) if history else None
    best_metric, best_epoch = (best["val_macro_f1"], best["epoch"]) if best else (None, None)
    if stopper.signal is not None:
        result = FitResult("stopped", len(history), best_metric, best_epoch, timer.total,
                           checkpoint_dir, stop_signal=stopper.signal)
        return TransformerOutcome(result, None, num_parameters)

    # load_best_model_at_end: trainer.model now holds the best epoch's weights.
    status = "finished" if len(history) >= settings.max_epochs else "early_stopped"
    trainer.save_model(str(checkpoint_dir / BEST_MODEL_DIR))
    y_true = splits["val"]["label_id"].to_numpy()
    y_prob = softmax(trainer.predict(data["val"]).predictions)
    metrics = evaluate(y_true, y_prob.argmax(axis=1), y_prob, class_names, settings.human_label)
    train_time = timer.total
    save_results(metrics, label, "val", {
        "train_time_seconds": train_time, "num_parameters": num_parameters,
        "best_epoch": best_epoch, "epochs_completed": len(history), "status": status,
        "run_name": run_name, "subset": subset, "pretrained_name": hp.pretrained_name,
    }, y_true, y_prob, get_path("results_dir"))
    result = FitResult(status, len(history), best_metric, best_epoch, train_time, checkpoint_dir)
    atomic_write_json(checkpoint_dir / DONE_FILE, {
        "status": status, "epochs_completed": len(history), "best_metric": best_metric,
        "best_epoch": best_epoch, "train_time_seconds": train_time,
        "num_parameters": num_parameters, "metrics": metrics})
    return TransformerOutcome(result, metrics, num_parameters)

