"""Tests for the shared evaluate() and save_results()."""

import json
from pathlib import Path

import numpy as np
import pytest
from sklearn.metrics import f1_score

from src.evaluate import METRIC_KEYS, evaluate, save_results

NAMES = ["Human", "GPT", "Llama"]
Y_TRUE = [0, 0, 0, 0, 1, 1, 2, 2]
Y_PRED = [0, 1, 2, 0, 1, 1, 2, 0]


def probs_for(pred: list[int], n_classes: int = 3) -> np.ndarray:
    prob = np.full((len(pred), n_classes), 0.1)
    prob[np.arange(len(pred)), pred] = 1 - 0.1 * (n_classes - 1)
    return prob


def test_evaluate_hand_made_example() -> None:
    m = evaluate(Y_TRUE, Y_PRED, probs_for(Y_PRED), NAMES)
    assert set(METRIC_KEYS) <= set(m)
    assert m["n_samples"] == 8
    assert m["accuracy"] == pytest.approx(5 / 8)
    assert m["human_fpr"] == pytest.approx(2 / 4)  # 2 of 4 human texts called AI
    assert m["macro_f1"] == pytest.approx(f1_score(Y_TRUE, Y_PRED, average="macro"))
    assert m["confusion_matrix"] == [[2, 1, 1], [0, 2, 0], [1, 0, 1]]
    assert m["per_class"]["GPT"] == {"precision": pytest.approx(2 / 3), "recall": 1.0,
                                     "f1": pytest.approx(0.8), "support": 2}


def test_human_fpr_is_none_without_human_rows() -> None:
    m = evaluate([1, 2], [1, 2], probs_for([1, 2]), NAMES)
    assert m["human_fpr"] is None


@pytest.mark.parametrize("kwargs, message", [
    ({"y_pred": [0, 1]}, "y_pred has 2"),
    ({"y_prob": np.ones((8, 2)) / 2}, "y_prob must have shape"),
    ({"y_prob": np.ones((8, 3))}, "sum to 1"),
    ({"y_pred": [0, 1, 2, 3, 0, 1, 2, 0]}, "outside"),
    ({"human_label": "Person"}, "human_label"),
])
def test_evaluate_rejects_bad_input(kwargs: dict, message: str) -> None:
    args = {"y_true": Y_TRUE, "y_pred": Y_PRED, "y_prob": probs_for(Y_PRED),
            "class_names": NAMES} | kwargs
    with pytest.raises(ValueError, match=message):
        evaluate(**args)


def test_save_results_writes_metrics_figure_and_probs(tmp_path: Path) -> None:
    prob = probs_for(Y_PRED)
    m = evaluate(Y_TRUE, Y_PRED, prob, NAMES)
    paths = save_results(m, "dummy", "val", {"train_time_seconds": 1.5, "num_parameters": 42},
                         Y_TRUE, prob, tmp_path)
    assert {p.name for p in paths.values()} == {
        "dummy_val_metrics.json", "dummy_val_confusion.png", "dummy_val_probs.npz"}
    saved = json.loads(paths["metrics"].read_text())
    assert saved["macro_f1"] == m["macro_f1"] and saved["extra"]["num_parameters"] == 42
    assert paths["confusion"].read_bytes()[:4] == b"\x89PNG"
    with np.load(paths["probs"]) as npz:
        assert npz["y_true"].tolist() == Y_TRUE
        assert np.allclose(npz["y_prob"], prob)
        assert npz["class_names"].tolist() == NAMES


def test_save_results_requires_extra_fields(tmp_path: Path) -> None:
    m = evaluate(Y_TRUE, Y_PRED, probs_for(Y_PRED), NAMES)
    with pytest.raises(KeyError, match="num_parameters"):
        save_results(m, "dummy", "val", {"train_time_seconds": 1.0}, Y_TRUE,
                     probs_for(Y_PRED), tmp_path)
    with pytest.raises(ValueError, match="may only contain"):
        save_results(m, "../evil", "val", {"train_time_seconds": 1, "num_parameters": 1},
                     Y_TRUE, probs_for(Y_PRED), tmp_path)
