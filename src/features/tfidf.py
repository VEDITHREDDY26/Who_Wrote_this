"""TF-IDF features for the MLP (and the dummy model).

Word 1-2 grams (15K features) + char_wb 3-5 grams (5K features), sublinear tf, joined with a
FeatureUnion and fit on train only. The fitted vectorizer and every split's sparse matrix are
cached in <splits_dir>/features/tfidf-<key>/, where the key hashes the train fingerprint and
the settings, so they are built once and rebuilt automatically when either changes.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion

from src.data.splits import fingerprint_ids
from src.features.cache import cache_key, split_file_name
from src.utils.io import atomic_open, atomic_write_json

logger = logging.getLogger(__name__)

KIND = "tfidf"
VERSION = 1                      # bump when the meaning of the cached files changes
PARTS = ("word", "char")
RUNTIME_ONLY_SETTINGS = ("n_jobs",)  # do not change the result, so not part of the cache key
VECTORIZER_FILE = "vectorizer.joblib"
META_FILE = "meta.json"


@dataclass
class TfidfFeatures:
    """A fitted vectorizer plus one CSR matrix per split."""

    vectorizer: FeatureUnion
    matrices: dict[str, sp.csr_matrix]
    cache_dir: Path
    key: str
    from_cache: bool

    @property
    def n_features(self) -> int:
        """Number of columns in every matrix."""
        return next(iter(self.matrices.values())).shape[1]


def _part_params(part: Mapping[str, Any], dtype: str) -> dict[str, Any]:
    params = dict(part)
    params["ngram_range"] = tuple(params["ngram_range"])
    params["dtype"] = np.dtype(dtype).type
    return params


def build_tfidf(train_texts: Sequence[str], settings: Mapping[str, Any]) -> FeatureUnion:
    """Fit the word + char TF-IDF union on the train texts only."""
    missing = [p for p in PARTS if p not in settings]
    if missing:
        raise KeyError(f"tfidf settings missing parts {missing}; expected {PARTS}")
    union = FeatureUnion(
        [(p, TfidfVectorizer(**_part_params(settings[p], settings["dtype"]))) for p in PARTS],
        n_jobs=settings.get("n_jobs"))
    union.fit(list(train_texts))
    return union


def transform(vectorizer: FeatureUnion, texts: Sequence[str], dtype: str) -> sp.csr_matrix:
    """Turn texts into a CSR matrix with the fitted vectorizer."""
    return sp.csr_matrix(vectorizer.transform(list(texts)), dtype=np.dtype(dtype))


def save(vectorizer: FeatureUnion, path: Path) -> None:
    """Atomically save a fitted vectorizer with joblib."""
    with atomic_open(path) as f:
        joblib.dump(vectorizer, f)


def load(path: Path) -> FeatureUnion:
    """Load a vectorizer saved by `save` (only load files this project wrote)."""
    vectorizer = joblib.load(path)
    if not isinstance(vectorizer, FeatureUnion):
        raise TypeError(f"{path} holds a {type(vectorizer).__name__}, expected a FeatureUnion")
    return vectorizer


def _save_matrix(matrix: sp.csr_matrix, path: Path) -> None:
    with atomic_open(path) as f:
        sp.save_npz(f, matrix)


def get_tfidf_features(splits: Mapping[str, pd.DataFrame], settings: Mapping[str, Any],
                       cache_root: Path, *, train_split: str = "train") -> TfidfFeatures:
    """Return TF-IDF matrices for every split, fitting on `train_split` only, using the cache.

    Args:
        splits: Split name -> DataFrame with `id` and `text` columns. Must include train.
        settings: The `features.tfidf` config section.
        cache_root: Folder that holds the cache folders (<splits_dir>/features).
        train_split: Which entry of `splits` the vectorizer is fit on.
    """
    if train_split not in splits:
        raise KeyError(f"splits must include {train_split!r} to fit TF-IDF; got {list(splits)}")
    key_settings = {k: v for k, v in settings.items() if k not in RUNTIME_ONLY_SETTINGS}
    train_fp = fingerprint_ids(splits[train_split]["id"])
    key = cache_key(KIND, VERSION, train_fp, key_settings)
    cache_dir = cache_root / f"{KIND}-{key}"
    vec_path = cache_dir / VECTORIZER_FILE

    from_cache = vec_path.exists()
    if from_cache:
        vectorizer = load(vec_path)
        logger.info("Loaded cached TF-IDF vectorizer from %s", cache_dir)
    else:
        logger.info("Fitting TF-IDF on %d %s texts", len(splits[train_split]), train_split)
        vectorizer = build_tfidf(splits[train_split]["text"], settings)
        save(vectorizer, vec_path)
        atomic_write_json(cache_dir / META_FILE, {
            "kind": KIND, "version": VERSION, "train_fingerprint": train_fp,
            "settings": key_settings, "n_features": len(vectorizer.get_feature_names_out())})

    matrices = {}
    for name, df in splits.items():
        path = cache_dir / split_file_name(name, fingerprint_ids(df["id"]), ".npz")
        if path.exists():
            matrices[name] = sp.load_npz(path).tocsr()
        else:
            from_cache = False
            logger.info("Transforming %d %s texts to TF-IDF", len(df), name)
            matrices[name] = transform(vectorizer, df["text"], settings["dtype"])
            _save_matrix(matrices[name], path)
        if matrices[name].shape[0] != len(df):
            raise ValueError(f"cached matrix {path} has {matrices[name].shape[0]} rows but the "
                             f"{name} split has {len(df)}; delete {cache_dir} and rerun")
    return TfidfFeatures(vectorizer, matrices, cache_dir, key, from_cache)
