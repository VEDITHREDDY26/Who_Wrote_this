"""Word vocabulary and integer encoding for TextCNN / BiLSTM.

Lowercase + regex tokenizer; vocabulary = <pad>=0, <unk>=1, then the top 50K train tokens
(ties broken alphabetically, so the order never depends on row order). Encoded splits are
cached in <splits_dir>/features/tokens-<key>/ like the TF-IDF matrices.
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from src.data.splits import fingerprint_ids
from src.features.cache import cache_key, split_file_name
from src.utils.io import atomic_torch_save, atomic_write_json, safe_load_json

logger = logging.getLogger(__name__)

KIND = "tokens"
VERSION = 1
PAD_ID = 0
UNK_ID = 1
VOCAB_FILE = "vocab.json"
CACHE_DTYPE = torch.int32  # halves disk use; ids are returned as int64


@dataclass
class Vocab:
    """Token <-> id mapping plus the tokenizer settings it was built with."""

    itos: list[str]
    token_pattern: str
    lowercase: bool
    stoi: dict[str, int] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if len(self.itos) < 2:
            raise ValueError("vocab needs at least the <pad> and <unk> tokens")
        self.stoi = {tok: i for i, tok in enumerate(self.itos)}
        if len(self.stoi) != len(self.itos):
            raise ValueError("vocab contains duplicate tokens")
        self._regex = re.compile(self.token_pattern)

    def __len__(self) -> int:
        return len(self.itos)

    @property
    def pad_id(self) -> int:
        """Id of the padding token (always 0)."""
        return PAD_ID

    @property
    def unk_id(self) -> int:
        """Id of the unknown-word token (always 1)."""
        return UNK_ID

    def tokenize(self, text: str) -> list[str]:
        """Split text into tokens with the vocab's regex (after optional lowercasing)."""
        return self._regex.findall(text.lower() if self.lowercase else text)

    @classmethod
    def build(cls, texts: Iterable[str], *, max_words: int, token_pattern: str, lowercase: bool,
              pad_token: str, unk_token: str) -> Vocab:
        """Build a vocabulary of the `max_words` most frequent tokens in `texts`."""
        if max_words < 1:
            raise ValueError(f"max_words must be >= 1, got {max_words}")
        regex = re.compile(token_pattern)
        counts: Counter[str] = Counter()
        for text in texts:
            counts.update(regex.findall(text.lower() if lowercase else text))
        for special in (pad_token, unk_token):
            counts.pop(special, None)
        ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:max_words]
        return cls([pad_token, unk_token, *(tok for tok, _ in ranked)], token_pattern, lowercase)

    def encode(self, texts: Sequence[str], max_len: int) -> torch.Tensor:
        """Encode texts as an int64 tensor of shape (n, max_len), right-padded with <pad>.

        Texts longer than `max_len` tokens are truncated (the first `max_len` tokens are kept).
        """
        if max_len < 1:
            raise ValueError(f"max_len must be >= 1, got {max_len}")
        out = np.full((len(texts), max_len), PAD_ID, dtype=np.int64)
        for row, text in enumerate(texts):
            ids = [self.stoi.get(tok, UNK_ID) for tok in self.tokenize(text)[:max_len]]
            out[row, :len(ids)] = ids
        return torch.from_numpy(out)

    def save(self, path: Path) -> None:
        """Atomically save the vocabulary as JSON."""
        atomic_write_json(path, {"tokens": self.itos, "token_pattern": self.token_pattern,
                                 "lowercase": self.lowercase})

    @classmethod
    def load(cls, path: Path) -> Vocab:
        """Load a vocabulary written by `save`."""
        data = safe_load_json(path)
        missing = {"tokens", "token_pattern", "lowercase"} - set(data)
        if missing:
            raise ValueError(f"{path} is not a vocab file; missing keys {sorted(missing)}")
        return cls(data["tokens"], data["token_pattern"], data["lowercase"])


@dataclass
class EncodedSplits:
    """A vocabulary plus one (n, max_len) int64 tensor per split."""

    vocab: Vocab
    tensors: dict[str, torch.Tensor]
    cache_dir: Path
    key: str
    from_cache: bool


def get_encoded_splits(splits: Mapping[str, pd.DataFrame], settings: Mapping[str, Any],
                       cache_root: Path, *, train_split: str = "train") -> EncodedSplits:
    """Return encoded tensors for every split, building the vocab on `train_split` only.

    Args:
        splits: Split name -> DataFrame with `id` and `text` columns. Must include train.
        settings: The `features.vocab` config section.
        cache_root: Folder that holds the cache folders (<splits_dir>/features).
        train_split: Which entry of `splits` the vocabulary is built from.
    """
    if train_split not in splits:
        raise KeyError(f"splits must include {train_split!r} to build the vocab; "
                       f"got {list(splits)}")
    train_fp = fingerprint_ids(splits[train_split]["id"])
    key = cache_key(KIND, VERSION, train_fp, settings)
    cache_dir = cache_root / f"{KIND}-{key}"
    vocab_path = cache_dir / VOCAB_FILE

    from_cache = vocab_path.exists()
    if from_cache:
        vocab = Vocab.load(vocab_path)
    else:
        logger.info("Building vocabulary from %d %s texts", len(splits[train_split]), train_split)
        vocab = Vocab.build(splits[train_split]["text"], max_words=settings["max_words"],
                            token_pattern=settings["token_pattern"],
                            lowercase=settings["lowercase"], pad_token=settings["pad_token"],
                            unk_token=settings["unk_token"])
        vocab.save(vocab_path)

    tensors = {}
    for name, df in splits.items():
        path = cache_dir / split_file_name(name, fingerprint_ids(df["id"]), ".pt")
        if path.exists():
            tensors[name] = torch.load(path, weights_only=True).long()
        else:
            from_cache = False
            logger.info("Encoding %d %s texts", len(df), name)
            tensors[name] = vocab.encode(list(df["text"]), settings["max_len"])
            atomic_torch_save(tensors[name].to(CACHE_DTYPE), path)
        expected = (len(df), settings["max_len"])
        if tuple(tensors[name].shape) != expected:
            raise ValueError(f"cached tensor {path} has shape {tuple(tensors[name].shape)}, "
                             f"expected {expected}; delete {cache_dir} and rerun")
    return EncodedSplits(vocab, tensors, cache_dir, key, from_cache)
