"""Model registry: maps a model name to its input type and builder.

To add a model, write `build_<name>(params, *, input_dim | vocab_size, num_classes, ...)` in
src/models/<name>.py and point its ModelSpec.build at it. The training CLI (src/train.py)
does the rest.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from torch import nn

from src.models.cnn import build_cnn
from src.models.dummy import build_dummy
from src.models.lstm import build_lstm

INPUT_KINDS = ("tfidf", "tokens", "hf")


@dataclass(frozen=True)
class ModelSpec:
    """How to build one model and which features it consumes."""

    name: str
    input_kind: str  # "tfidf" (sparse TF-IDF), "tokens" (word ids), "hf" (Hugging Face)
    build: Callable[..., nn.Module] | None
    how_to_add: str = ""

    @property
    def implemented(self) -> bool:
        """True once a builder (or, for "hf", a training function) exists."""
        return self.build is not None


REGISTRY: dict[str, ModelSpec] = {
    "dummy": ModelSpec("dummy", "tfidf", build_dummy),
    "mlp": ModelSpec("mlp", "tfidf", None,
                     "write build_mlp(params, *, input_dim, num_classes) in src/models/mlp.py "
                     "and register it here"),
    "cnn": ModelSpec("cnn", "tokens", build_cnn),
    "lstm": ModelSpec("lstm", "tokens", build_lstm),
    # Transformer hook: trained by Hugging Face's Trainer, not src.trainer.Trainer.
    # See src/models/transformer.py:train_transformer.
    "transformer": ModelSpec("transformer", "hf", None,
                             "implement train_transformer() in src/models/transformer.py "
                             "(Hugging Face Trainer with its own checkpointing)"),
}


def get_model_spec(name: str) -> ModelSpec:
    """Look up a model by name."""
    if name not in REGISTRY:
        raise KeyError(f"unknown model {name!r}; choose from {sorted(REGISTRY)}")
    return REGISTRY[name]


def require_implemented(spec: ModelSpec) -> None:
    """Raise NotImplementedError with instructions if the model has no builder yet."""
    if not spec.implemented:
        raise NotImplementedError(f"Model {spec.name!r} is not implemented yet. To add it: "
                                  f"{spec.how_to_add}.")


def build_model(name: str, params: dict[str, Any], **dims: Any) -> nn.Module:
    """Build a registered PyTorch model; `dims` carries input_dim / vocab_size / num_classes."""
    spec = get_model_spec(name)
    require_implemented(spec)
    if spec.input_kind == "hf":
        raise TypeError(f"{name!r} is trained by Hugging Face's Trainer, not built here")
    return spec.build(params, **dims)
