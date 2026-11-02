"""MLP classifier on TF-IDF features."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch
from torch import nn


class MLP(nn.Module):
    """Fully connected layers with ReLU and dropout, then a linear layer to the class logits."""

    def __init__(self, input_dim: int, num_classes: int, *, hidden_sizes: Sequence[int],
                 dropout: float) -> None:
        super().__init__()
        if input_dim < 1 or num_classes < 2:
            raise ValueError(f"need input_dim >= 1 and num_classes >= 2, got {input_dim}, "
                             f"{num_classes}")
        if not hidden_sizes or min(hidden_sizes) < 1:
            raise ValueError(f"hidden_sizes must be a non-empty list of sizes >= 1, got "
                             f"{list(hidden_sizes)}")
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {dropout}")
        layers: list[nn.Module] = []
        in_dim = input_dim
        for size in hidden_sizes:
            layers += [nn.Linear(in_dim, size), nn.ReLU(), nn.Dropout(dropout)]
            in_dim = size
        self.hidden = nn.Sequential(*layers)
        self.fc = nn.Linear(in_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return logits of shape (batch, num_classes) for features of shape (batch, input_dim)."""
        return self.fc(self.hidden(x.float()))


def build_mlp(params: Mapping[str, Any], *, input_dim: int, num_classes: int,
              **_: Any) -> MLP:
    """Build the MLP from its config section (`models.mlp`)."""
    return MLP(input_dim, num_classes, hidden_sizes=[int(size) for size in params["hidden_sizes"]],
               dropout=float(params["dropout"]))
