"""A tiny linear classifier used to exercise the training pipeline end to end."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import nn


class DummyLinear(nn.Module):
    """Input dropout followed by a single linear layer (logistic regression)."""

    def __init__(self, input_dim: int, num_classes: int, dropout: float = 0.0) -> None:
        super().__init__()
        if input_dim < 1 or num_classes < 2:
            raise ValueError(f"need input_dim >= 1 and num_classes >= 2, got {input_dim}, "
                             f"{num_classes}")
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(input_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return logits of shape (batch, num_classes)."""
        return self.linear(self.dropout(x))


def build_dummy(params: Mapping[str, Any], *, input_dim: int, num_classes: int,
                **_: Any) -> DummyLinear:
    """Build the dummy model from its config section (`models.dummy`)."""
    return DummyLinear(input_dim, num_classes, dropout=float(params.get("dropout", 0.0)))
