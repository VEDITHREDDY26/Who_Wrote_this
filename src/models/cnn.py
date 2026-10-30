"""TextCNN classifier on word-index sequences (Kim, 2014)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch
from torch import nn


class TextCNN(nn.Module):
    """Embedding, parallel 1D convolutions with ReLU, max-over-time pooling, dropout, linear."""

    def __init__(self, vocab_size: int, num_classes: int, *, pad_id: int, embed_dim: int,
                 filter_sizes: Sequence[int], num_filters: int, dropout: float) -> None:
        super().__init__()
        if vocab_size < 2 or num_classes < 2:
            raise ValueError(f"need vocab_size >= 2 and num_classes >= 2, got {vocab_size}, "
                             f"{num_classes}")
        if embed_dim < 1 or num_filters < 1:
            raise ValueError(f"need embed_dim >= 1 and num_filters >= 1, got {embed_dim}, "
                             f"{num_filters}")
        if not filter_sizes or min(filter_sizes) < 1:
            raise ValueError(f"filter_sizes must be a non-empty list of sizes >= 1, got "
                             f"{list(filter_sizes)}")
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {dropout}")
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_id)
        self.convs = nn.ModuleList(nn.Conv1d(embed_dim, num_filters, size)
                                   for size in filter_sizes)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(len(filter_sizes) * num_filters, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return logits of shape (batch, num_classes) for token ids of shape (batch, seq_len)."""
        emb = self.embedding(x.long()).transpose(1, 2)  # (batch, embed_dim, seq_len) for Conv1d
        pooled = [torch.relu(conv(emb)).amax(dim=2) for conv in self.convs]
        return self.fc(self.dropout(torch.cat(pooled, dim=1)))


def build_cnn(params: Mapping[str, Any], *, vocab_size: int, pad_id: int, num_classes: int,
              **_: Any) -> TextCNN:
    """Build the TextCNN from its config section (`models.cnn`)."""
    return TextCNN(vocab_size, num_classes, pad_id=pad_id, embed_dim=int(params["embed_dim"]),
                   filter_sizes=[int(size) for size in params["filter_sizes"]],
                   num_filters=int(params["num_filters"]), dropout=float(params["dropout"]))
