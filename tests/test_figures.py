"""Tests for the report figures and tables (src.analysis) on a small fake results folder."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import src.analysis.data as analysis_data
from src.analysis.data import (
    ResultsError,
    bucket_labels,
    expected_calibration_error,
    length_buckets,
    load_analysis_data,
    load_tuning_runs,
    parse_train_log,
    reliability_curve,
)
from src.analysis.figures import (
    FIGURES,
    FigureSettings,
    build_tables,
    check_predictions,
    make_all,
)
from src.evaluate import evaluate, save_results
from tests.test_train_cli import CLASSES, make_workspace

SETTINGS = FigureSettings("figures", "tables", ("png",), 40, 5, (5, 10, 20))
MODELS = {"transformer": None, "mlp": "h8", "length_baseline": None}  # model -> tested run
LOG = """\
2026-10-30 12:47:31 INFO    x: step 100/200  epoch 0.50  {'loss': 1.5, 'learning_rate': 1e-05}
2026-10-30 12:47:38 INFO    x: step 100/200  epoch 1.00  {'eval_loss': 0.9, 'eval_macro_f1': 0.7}
2026-10-30 12:47:45 INFO    x: step 200/200  epoch 2.00  {'loss': 0.8, 'epoch': 2.0}
not a step line
"""


def fake_probs(labels: np.ndarray, rng: np.random.Generator, skill: float) -> np.ndarray:
    logits = rng.normal(size=(len(labels), len(CLASSES)))
    logits[np.arange(len(labels)), labels] += skill
    exp = np.exp(logits)
    return exp / exp.sum(axis=1, keepdims=True)


def write_results(workspace: Path) -> Path:
    """Fake versions of everything src.final_test, src.train and src.tune leave behind."""
    results = workspace / "results"
    results.mkdir()
    rng = np.random.default_rng(0)
    val_labels = pd.read_parquet(workspace / "data" / "splits" / "val.parquet")["label_id"]
    y_test = np.repeat(np.arange(len(CLASSES)), 10)
    rows = []
    for k, (model, run) in enumerate(MODELS.items()):
        prob = fake_probs(y_test, rng, skill=3.0 - k)
        metrics = evaluate(y_test, prob.argmax(axis=1), prob, CLASSES, "Human")
        extra = {"train_time_seconds": 60.0 * (k + 1), "num_parameters": 1000 * (k + 1)}
        save_results(metrics, model, "test", extra, y_test, prob, results)
        pd.DataFrame({"id": [f"x{i}" for i in range(len(y_test))],
                      "domain": np.array(["news", "wiki"] * (len(y_test) // 2)),
                      "label_id": y_test, "pred_id": prob.argmax(axis=1)}
                     | {f"prob_{c}": prob[:, i] for i, c in enumerate(CLASSES)}
                     ).to_parquet(results / f"{model}_test_predictions.parquet", index=False)
        rows.append({"model": model, "run_name": run, "test_macro_f1": metrics["macro_f1"],
                     "test_accuracy": metrics["accuracy"],
                     "test_human_fpr": metrics["human_fpr"], "val_macro_f1": 0.5 - 0.1 * k,
                     **extra})
        if model == "length_baseline":
            continue
        label = model if run is None else f"{model}-{run}"
        np.savez(results / f"{label}_val_probs.npz", y_true=val_labels.to_numpy(),
                 y_prob=fake_probs(val_labels.to_numpy(), rng, skill=3.0 - k))
        run_dir = results / model / (run or "default")
        run_dir.mkdir(parents=True)
        history = [{"epoch": e, "val_macro_f1": 0.4 + 0.1 * e} for e in (1, 2, 3)]
        if model == "mlp":  # src.trainer histories also hold losses
            history = [r | {"train_loss": 1 / r["epoch"], "val_log_loss": 0.5} for r in history]
        else:
            (run_dir / "train.log").write_text(LOG, encoding="utf-8")
        (run_dir / "history.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in history), encoding="utf-8")
    (results / "final_test_summary.json").write_text(json.dumps({"models": rows}))
    tuning = results / "tuning"
    tuning.mkdir()
    (tuning / "mlp_tuning.json").write_text(json.dumps({"runs": [
        {"run_name": "h8", "val_macro_f1": 0.4}, {"run_name": "h4", "val_macro_f1": 0.35}]}))
    pd.DataFrame([{"model": "transformer", "run": "lr", "val_macro_f1": 0.45}]).to_csv(
        tuning / "from_logs.csv", index=False)
    (results / "split_stats.json").write_text(json.dumps({
        "sampling": {c: {"available": 100, "kept": 50} for c in CLASSES},
        "cleaning": {"empty": {"total": 1}, "under_10_words": {"total": 2},
                     "rows_after_cleaning": {"total": 300}}}))
    return results


@pytest.fixture(scope="module")
def fake(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    workspace = make_workspace(tmp_path_factory.mktemp("figures"))
    return workspace, write_results(workspace)


def test_every_figure_and_table_is_written_without_loading_test(
        fake: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch) -> None:
    workspace, results = fake
    loaded = []
    real_load_split = analysis_data.load_split

    def spy(name: str, *args: object, **kwargs: object) -> pd.DataFrame:
        loaded.append(name)
        return real_load_split(name, *args, **kwargs)

    monkeypatch.setattr(analysis_data, "load_split", spy)
    data = load_analysis_data(results, CLASSES, "Human", workspace / "data" / "splits")
    assert "test" not in loaded
    assert [m.name for m in data.models][0] == "transformer" and data.baseline is not None

    written = make_all(data, SETTINGS, results)
    figures = {p.stem for p in written if p.suffix == ".png"}
    expected = {name for name, _, _ in FIGURES} | {
        f"16_confusion_{i + 1}_{m.name}" for i, m in enumerate(data.models)}
    assert figures == expected
    captions = (results / "figures" / "captions.txt").read_text(encoding="utf-8")
    assert len(captions.splitlines()) == len(expected)
    summary = pd.read_csv(results / "tables" / "summary.csv")
    assert len(summary) == len(MODELS)
    tuning = pd.read_csv(results / "tables" / "tuning_runs.csv")
    assert set(zip(tuning["model"], tuning["run"], strict=True)) == {
        ("mlp", "h8"), ("mlp", "h4"), ("transformer", "lr")}


def test_only_draws_the_matching_figures(fake: tuple[Path, Path], tmp_path: Path) -> None:
    workspace, results = fake
    data = load_analysis_data(results, CLASSES, "Human", workspace / "data" / "splits")
    written = make_all(data, SETTINGS, results, only=["10_model"])
    assert {p.stem for p in written if p.suffix == ".png"} == {"10_model_comparison"}


def test_predictions_must_reproduce_the_saved_metrics(fake: tuple[Path, Path]) -> None:
    workspace, results = fake
    data = load_analysis_data(results, CLASSES, "Human", workspace / "data" / "splits")
    tables = build_tables(data, SETTINGS)
    check_predictions(tables)
    tables["macro_f1_check"].loc[0, "saved"] += 0.1
    with pytest.raises(ValueError, match="do not match"):
        check_predictions(tables)


def test_missing_summary_is_a_clear_error(tmp_path: Path) -> None:
    with pytest.raises(ResultsError, match="src.final_test"):
        load_analysis_data(tmp_path, CLASSES, "Human", tmp_path)


def test_length_buckets() -> None:
    assert bucket_labels((10, 100, 200)) == ["10-99", "100-199", "200+"]
    words = np.array([5, 10, 99, 100, 250])
    assert length_buckets(words, (10, 100, 200)).tolist() == [
        "", "10-99", "10-99", "100-199", "200+"]
    with pytest.raises(ValueError, match="increasing"):
        bucket_labels((100, 10))


def test_calibration_of_a_perfectly_confident_correct_model_is_zero() -> None:
    y = np.array([0, 1, 2, 2])
    prob = np.eye(3)[y]
    assert expected_calibration_error(y, prob, 10) == 0
    curve = reliability_curve(y, np.full((4, 3), 1 / 3), 10)
    assert curve["count"].sum() == 4 and curve["accuracy"].iloc[0] == 0.25


def test_tuning_runs_first_source_wins_and_subset_runs_are_skipped(tmp_path: Path) -> None:
    def metrics(f1: float, subset: int | None = None) -> str:
        return json.dumps({"macro_f1": f1, "extra": {"subset": subset}})

    (tmp_path / "mlp_val_metrics.json").write_text(metrics(0.7))
    (tmp_path / "mlp-a_val_metrics.json").write_text(metrics(0.6))
    (tmp_path / "mlp-quick_val_metrics.json").write_text(metrics(0.9, subset=100))
    (tmp_path / "mlpx_val_metrics.json").write_text(metrics(0.1))  # another model's name
    (tmp_path / "tuning").mkdir()
    (tmp_path / "tuning" / "mlp_tuning.json").write_text(json.dumps({"runs": [
        {"run_name": "a", "val_macro_f1": 0.0}, {"run_name": "b", "val_macro_f1": 0.5}]}))
    runs = load_tuning_runs(tmp_path, ["mlp"]).set_index("run")["val_macro_f1"].to_dict()
    assert runs == {"default": 0.7, "a": 0.6, "b": 0.5}


def test_parse_train_log(tmp_path: Path) -> None:
    path = tmp_path / "train.log"
    path.write_text(LOG, encoding="utf-8")
    log = parse_train_log(path)
    assert log["kind"].tolist() == ["train", "eval", "train"]
    assert log.loc[log["kind"] == "eval", "macro_f1"].item() == 0.7


def test_settings_validation() -> None:
    good = {"figures_subdir": "f", "tables_subdir": "t", "formats": ["png"], "dpi": 50,
            "calibration_bins": 5, "length_bins": [1, 2]}
    assert FigureSettings.from_config(good).formats == ("png",)
    with pytest.raises(KeyError, match="unknown"):
        FigureSettings.from_config(good | {"extra": 1})
    with pytest.raises(ValueError, match="increasing"):
        FigureSettings.from_config(good | {"length_bins": [5]})
