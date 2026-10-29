"""Tests for the vocabulary, TF-IDF features, their caches, class weights and split access."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

import src.features.tfidf as tfidf
from src.data.class_weights import compute_class_weights, get_class_weights
from src.data.splits import LockedTestSplitError, load_split, stratified_subset
from src.features.vocab import PAD_ID, UNK_ID, Vocab, get_encoded_splits

VOCAB_SETTINGS = {"max_words": 50, "max_len": 256, "lowercase": True,
                  "token_pattern": r"\w+(?:'\w+)?|[^\w\s]", "pad_token": "<pad>",
                  "unk_token": "<unk>"}
TFIDF_SETTINGS = {
    "word": {"analyzer": "word", "ngram_range": [1, 2], "max_features": 40,
             "token_pattern": r"(?u)\b\w+\b", "sublinear_tf": True, "lowercase": True,
             "min_df": 1},
    "char": {"analyzer": "char_wb", "ngram_range": [3, 5], "max_features": 30,
             "sublinear_tf": True, "lowercase": True, "min_df": 1},
    "dtype": "float32", "n_jobs": 1}


def frame(texts: list[str], prefix: str) -> pd.DataFrame:
    return pd.DataFrame({"id": [f"{prefix}{i}" for i in range(len(texts))], "text": texts})


@pytest.fixture
def splits() -> dict[str, pd.DataFrame]:
    return {"train": frame(["The cat sat on the mat.", "A dog sat.", "The cat ran!"], "t"),
            "val": frame(["The zebra sat.", "A cat."], "v")}


# ---------------------------------------------------------------------------- vocabulary
def test_vocab_specials_and_frequency_order() -> None:
    vocab = Vocab.build(["the cat sat", "The dog sat.", "a cat"], max_words=3,
                        token_pattern=VOCAB_SETTINGS["token_pattern"], lowercase=True,
                        pad_token="<pad>", unk_token="<unk>")
    assert vocab.itos[:2] == ["<pad>", "<unk>"]
    assert (vocab.pad_id, vocab.unk_id) == (PAD_ID, UNK_ID) == (0, 1)
    assert vocab.itos[2:] == ["cat", "sat", "the"]  # count 2 each, ties alphabetical
    assert len(vocab) == 5


def test_encode_shape_padding_unknown_and_truncation() -> None:
    vocab = Vocab.build(["the cat sat"], max_words=10, token_pattern=r"\w+", lowercase=True,
                        pad_token="<pad>", unk_token="<unk>")
    long_text = " ".join(["cat"] * 300)
    encoded = vocab.encode(["The CAT jumped", long_text], max_len=256)
    assert encoded.shape == (2, 256) and encoded.dtype == torch.int64
    assert encoded[0, :3].tolist() == [vocab.stoi["the"], vocab.stoi["cat"], UNK_ID]
    assert (encoded[0, 3:] == PAD_ID).all()
    assert (encoded[1] == vocab.stoi["cat"]).all()  # truncated to the first 256 tokens


def test_vocab_save_load_roundtrip(tmp_path: Path) -> None:
    vocab = Vocab.build(["hello world", "hello there"], **{
        k: VOCAB_SETTINGS[k] for k in ("max_words", "token_pattern", "lowercase", "pad_token",
                                       "unk_token")})
    vocab.save(tmp_path / "vocab.json")
    loaded = Vocab.load(tmp_path / "vocab.json")
    assert loaded.itos == vocab.itos
    assert torch.equal(loaded.encode(["hello world"], 8), vocab.encode(["hello world"], 8))


def test_encoded_splits_cache_is_reused(tmp_path: Path,
                                        splits: dict[str, pd.DataFrame]) -> None:
    first = get_encoded_splits(splits, VOCAB_SETTINGS, tmp_path)
    second = get_encoded_splits(splits, VOCAB_SETTINGS, tmp_path)
    assert not first.from_cache and second.from_cache
    assert first.cache_dir == second.cache_dir
    for name in splits:
        assert second.tensors[name].shape == (len(splits[name]), 256)
        assert torch.equal(first.tensors[name], second.tensors[name])
    assert "zebra" not in first.vocab.stoi  # val-only word: vocab is built on train only
    other = get_encoded_splits(splits, VOCAB_SETTINGS | {"max_len": 16}, tmp_path)
    assert other.cache_dir != first.cache_dir and not other.from_cache


# ---------------------------------------------------------------------------- TF-IDF
def test_tfidf_fits_on_train_only(tmp_path: Path, splits: dict[str, pd.DataFrame]) -> None:
    feats = tfidf.get_tfidf_features(splits, TFIDF_SETTINGS, tmp_path)
    names = feats.vectorizer.get_feature_names_out()
    assert not any("zebra" in n for n in names if n.startswith("word__"))
    reference = tfidf.build_tfidf(splits["train"]["text"], TFIDF_SETTINGS)
    expected = tfidf.transform(reference, splits["val"]["text"], "float32")
    assert (feats.matrices["val"] != expected).nnz == 0
    assert feats.matrices["train"].shape == (3, feats.n_features)
    assert feats.matrices["val"].dtype == np.float32


def test_tfidf_cache_is_reused(tmp_path: Path, splits: dict[str, pd.DataFrame],
                               monkeypatch: pytest.MonkeyPatch) -> None:
    first = tfidf.get_tfidf_features(splits, TFIDF_SETTINGS, tmp_path)

    def fail(*args: object, **kwargs: object) -> None:
        raise AssertionError("vectorizer was refit instead of loaded from cache")

    monkeypatch.setattr(tfidf, "build_tfidf", fail)
    monkeypatch.setattr(tfidf, "transform", fail)
    second = tfidf.get_tfidf_features(splits, TFIDF_SETTINGS | {"n_jobs": 4}, tmp_path)
    assert second.from_cache and second.cache_dir == first.cache_dir
    for name in splits:
        assert (first.matrices[name] != second.matrices[name]).nnz == 0


def test_tfidf_rebuilds_when_settings_or_train_change(tmp_path: Path,
                                                      splits: dict[str, pd.DataFrame]) -> None:
    base = tfidf.get_tfidf_features(splits, TFIDF_SETTINGS, tmp_path)
    changed = dict(TFIDF_SETTINGS, char=TFIDF_SETTINGS["char"] | {"max_features": 10})
    assert tfidf.get_tfidf_features(splits, changed, tmp_path).cache_dir != base.cache_dir
    new_train = dict(splits, train=frame(["Totally different text here."], "x"))
    assert tfidf.get_tfidf_features(new_train, TFIDF_SETTINGS, tmp_path).key != base.key


# ---------------------------------------------------------------------------- class weights
def test_balanced_class_weights_formula() -> None:
    labels = np.array([0, 0, 0, 0, 1, 1, 2, 2])
    weights = compute_class_weights(labels, 3, "balanced")
    assert np.allclose(weights, [8 / (3 * 4), 8 / (3 * 2), 8 / (3 * 2)])
    with pytest.raises(ValueError, match="no examples"):
        compute_class_weights(np.array([0, 1]), 3, "balanced")


def test_class_weights_are_saved_once_and_checked(tmp_path: Path) -> None:
    train = pd.DataFrame({"id": ["a", "b", "c", "d"], "label_id": [0, 0, 0, 1]})
    path = tmp_path / "class_weights.json"
    first = get_class_weights(train, ["Human", "GPT"], "balanced", path)
    assert path.exists()
    assert torch.equal(get_class_weights(train, ["Human", "GPT"], "balanced", path), first)
    other = pd.DataFrame({"id": ["a", "b", "c", "e"], "label_id": [0, 0, 1, 1]})
    with pytest.raises(ValueError, match="different train split"):
        get_class_weights(other, ["Human", "GPT"], "balanced", path)


# ---------------------------------------------------------------------------- split access
def test_test_split_is_locked(tmp_path: Path) -> None:
    with pytest.raises(LockedTestSplitError, match="i-understand-this-uses-the-test-set"):
        load_split("test", tmp_path)


def test_stratified_subset_is_deterministic_and_balanced() -> None:
    df = pd.DataFrame({"id": [f"r{i}" for i in range(600)], "label_id": [i % 6 for i in
                                                                         range(600)]})
    a = stratified_subset(df, 120, label_column="label_id", seed=42)
    b = stratified_subset(df, 120, label_column="label_id", seed=42)
    assert a["id"].tolist() == b["id"].tolist()
    assert a["label_id"].value_counts().tolist() == [20] * 6


def test_train_cli_never_reads_the_test_split() -> None:
    source = (Path(__file__).resolve().parents[1] / "src" / "train.py").read_text()
    assert "allow_test" not in source and 'load_split("test"' not in source
