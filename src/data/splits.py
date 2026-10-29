"""Read access to the frozen splits.

The test split is locked: `load_split("test")` raises unless `allow_test=True`, which only
src/final_test.py may pass (behind its --i-understand-this-uses-the-test-set flag).
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Sequence
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import get_path

SPLITS = ("train", "val", "test")
TEST_SPLIT = "test"
FINGERPRINT_CHARS = 16


class LockedTestSplitError(PermissionError):
    """Raised when code tries to read the test split without explicit permission."""


def fingerprint_ids(ids: Iterable[str]) -> str:
    """Return a short, order-independent sha256 fingerprint of a set of row ids."""
    digest = hashlib.sha256("\n".join(sorted(map(str, ids))).encode("utf-8")).hexdigest()
    return digest[:FINGERPRINT_CHARS]


def load_split(name: str, splits_dir: Path | None = None, *, columns: Sequence[str] | None = None,
               allow_test: bool = False) -> pd.DataFrame:
    """Load one frozen split from <splits_dir>/<name>.parquet.

    Args:
        name: "train", "val" or "test".
        splits_dir: Folder holding the parquet files; defaults to paths.splits_dir.
        columns: Optional subset of columns to read.
        allow_test: Must be True to read the test split. Only src/final_test.py sets it.
    """
    if name not in SPLITS:
        raise ValueError(f"unknown split {name!r}; expected one of {SPLITS}")
    if name == TEST_SPLIT and not allow_test:
        raise LockedTestSplitError(
            "The test split is used once, at the very end, and only through "
            "src/final_test.py --i-understand-this-uses-the-test-set. Tune on 'val' instead.")
    path = (splits_dir or get_path("splits_dir")) / f"{name}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; build the splits with "
                                "`python -m src.data.build_splits`")
    return pd.read_parquet(path, columns=list(columns) if columns else None)


def stratified_subset(df: pd.DataFrame, n: int, *, label_column: str, seed: int) -> pd.DataFrame:
    """Return a stratified sample of `n` rows (deterministic for a given seed)."""
    n_classes = df[label_column].nunique()
    if n < n_classes:
        raise ValueError(f"subset size {n} is smaller than the number of classes ({n_classes})")
    if n >= len(df):
        return df.reset_index(drop=True)
    subset, _ = train_test_split(df, train_size=n, stratify=df[label_column], random_state=seed)
    return subset.sort_values("id", kind="stable").reset_index(drop=True)
