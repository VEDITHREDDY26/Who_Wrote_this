"""Fine-tuned DistilBERT classifier (RoBERTa-base optional), max_len 256.

=== TRANSFORMER HOOK ===
The Transformer does NOT go through src.trainer.Trainer. It will use Hugging Face's Trainer,
which already has crash-safe checkpointing; mirror our guarantees with it:
  - TrainingArguments(save_strategy="steps", save_steps=..., save_total_limit=2,
    eval_strategy="epoch", load_best_model_at_end=True, metric_for_best_model="macro_f1",
    seed=cfg["seed"], data_seed=cfg["seed"]) and trainer.train(resume_from_checkpoint=True
    when a checkpoint exists in the run folder).
  - Class-weighted loss with the shared weights from results/class_weights.json
    (override compute_loss), the same train/val splits, and src.evaluate.evaluate() +
    save_results() for the final val metrics.
  - Check the installed transformers 5.x API (help(TrainingArguments)) before writing it.
"""

from __future__ import annotations

from typing import Any


def train_transformer(cfg: dict[str, Any], *, run_name: str, resume: str,
                      max_epochs: int | None, subset: int | None) -> int:
    """Fine-tune the Transformer and return a process exit code. Not implemented yet."""
    raise NotImplementedError(
        "Model 'transformer' is not implemented yet. Implement train_transformer() in "
        "src/models/transformer.py with Hugging Face's Trainer (see the module docstring).")
