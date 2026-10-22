"""Integrity tests for the frozen splits, plus unit tests for the cleaning rules.

The split tests read <splits_dir>/*.parquet and are skipped if the splits are not built.
"""

import itertools

import pandas as pd
import pytest

from src.config import get_path, load_config
from src.data.build_splits import OUT_COLUMNS, SPLITS, clean, normalize_text

PAIRS = list(itertools.combinations(SPLITS, 2))


@pytest.fixture(scope="module")
def cfg():
    return load_config()


@pytest.fixture(scope="module")
def splits():
    paths = {s: get_path("splits_dir") / f"{s}.parquet" for s in SPLITS}
    if not all(p.exists() for p in paths.values()):
        pytest.skip("splits not built; run `python -m src.data.build_splits`")
    return {s: pd.read_parquet(p) for s, p in paths.items()}


@pytest.fixture(scope="module")
def norm_texts(splits, cfg):
    return {s: set(normalize_text(df["text"], cfg)) for s, df in splits.items()}


@pytest.mark.parametrize("a,b", PAIRS)
def test_no_source_id_in_two_splits(splits, a, b):
    assert not set(splits[a]["group"]) & set(splits[b]["group"])


@pytest.mark.parametrize("a,b", PAIRS)
def test_no_normalized_text_in_two_splits(norm_texts, a, b):
    assert not norm_texts[a] & norm_texts[b]


@pytest.mark.parametrize("name", SPLITS)
def test_all_labels_present(splits, cfg, name):
    assert set(splits[name]["label"]) == set(cfg["classes"])


def test_split_sizes(splits, cfg):
    assert len(splits["train"]) > cfg["split"]["min_train_rows"]
    assert len(splits["test"]) > cfg["split"]["min_test_rows"]


@pytest.mark.parametrize("name", SPLITS)
def test_columns_and_fixed_label_order(splits, name):
    df = splits[name]
    assert list(df.columns) == OUT_COLUMNS
    order = {"Human": 0, "GPT": 1, "Llama": 2, "Mistral": 3, "MPT": 4, "Cohere": 5}
    assert (df["label"].map(order) == df["label_id"]).all()


def test_cleaning_rules(cfg):
    ten = "one two three four five six seven eight nine ten"
    df = pd.DataFrame({
        "id":     ["a", "b", "c", "d", "e", "f", "g"],
        "text":   ["​ ﻿", "too short", ten + " x", "​" + ten + " x ",
                   ten + " y", ten + " y", ten + " z"],
        "family": ["Mistral", "GPT", "GPT", "GPT", "Mistral", "MPT", "Human"],
    })
    kept, removed = clean(df, cfg)
    assert removed["empty"]["total"] == 1                     # a
    assert removed["under_10_words"]["total"] == 1            # b
    assert removed["duplicate_conflicting_families"]["total"] == 2  # e, f
    assert removed["duplicate_same_family"]["total"] == 1     # d (c has the smaller id)
    assert sorted(kept["id"]) == ["c", "g"]
    assert kept.loc[kept["id"] == "c", "text"].item() == ten + " x"  # original text kept
