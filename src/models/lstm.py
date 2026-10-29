"""BiLSTM classifier on word-index sequences."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence


class BiLSTM(nn.Module):
    """Embedding, bidirectional LSTM over the packed (unpadded) sequence, dropout, linear.

    The text is represented by the last layer's final forward and backward hidden states.
    Packing means padding never reaches the LSTM, so extra padding does not change the output.
    """

    def __init__(self, vocab_size: int, num_classes: int, *, pad_id: int, embed_dim: int,
                 hidden_size: int, num_layers: int, dropout: float) -> None:
        super().__init__()
        if vocab_size < 2 or num_classes < 2:
            raise ValueError(f"need vocab_size >= 2 and num_classes >= 2, got {vocab_size}, "
                             f"{num_classes}")
        if embed_dim < 1 or hidden_size < 1 or num_layers < 1:
            raise ValueError(f"need embed_dim, hidden_size and num_layers >= 1, got "
                             f"{embed_dim}, {hidden_size}, {num_layers}")
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {dropout}")
        self.pad_id = pad_id
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_id)
        # Dropout between LSTM layers only exists when there is more than one layer.
        self.lstm = nn.LSTM(embed_dim, hidden_size, num_layers, batch_first=True,
                            bidirectional=True, dropout=dropout if num_layers > 1 else 0.0)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(2 * hidden_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return logits of shape (batch, num_classes) for token ids of shape (batch, seq_len)."""
        x = x.long()
        # Texts are right-padded; an all-padding row counts as length 1 so packing accepts it.
        lengths = (x != self.pad_id).sum(dim=1).clamp(min=1)
        packed = pack_padded_sequence(self.embedding(x), lengths.cpu(), batch_first=True,
                                      enforce_sorted=False)
        _, (h_n, _) = self.lstm(packed)  # h_n: (num_layers * 2, batch, hidden_size)
        final = torch.cat([h_n[-2], h_n[-1]], dim=1)  # last layer: forward, backward
        return self.fc(self.dropout(final))


def build_lstm(params: Mapping[str, Any], *, vocab_size: int, pad_id: int, num_classes: int,
               **_: Any) -> BiLSTM:
    """Build the BiLSTM from its config section (`models.lstm`)."""
    return BiLSTM(vocab_size, num_classes, pad_id=pad_id, embed_dim=int(params["embed_dim"]),
                  hidden_size=int(params["hidden_size"]), num_layers=int(params["num_layers"]),
                  dropout=float(params["dropout"]))
