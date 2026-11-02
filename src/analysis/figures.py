"""Report and presentation figures, plus the tables behind them, from the saved results.

    python -m src.analysis.figures [--config PATH] [--only NAME ...]

Writes every figure as PNG (slides) and PDF (report) to <results_dir>/figures/, a caption per
figure to figures/captions.txt, and the numbers behind the figures as CSV tables to
<results_dir>/tables/. Nothing is re-evaluated: the test-set figures use the per-text test
predictions saved by src.final_test (the test split itself is never loaded), and the text-length
figure uses the val split.
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from matplotlib.transforms import offset_copy  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    average_precision_score,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from src.analysis.data import (  # noqa: E402
    AnalysisData,
    ModelResults,
    bucket_labels,
    expected_calibration_error,
    length_buckets,
    load_analysis_data,
    macro_f1_by_group,
    parse_train_log,
    per_class_f1_by_group,
    reliability_curve,
)
from src.config import get_path, load_config, use_config  # noqa: E402
from src.utils.io import atomic_open, atomic_write_bytes  # noqa: E402
from src.utils.logging import setup_logging  # noqa: E402

logger = logging.getLogger(__name__)
logging.getLogger("fontTools").setLevel(logging.WARNING)  # PDF font subsetting is chatty

# --- Visual style -------------------------------------------------------------------------
# Categorical slots 1-4 of the validated default palette (dataviz skill), checked on white:
# CVD and normal-vision separation pass; aqua and yellow are below 3:1 contrast, so every
# chart that uses them carries direct labels and its numbers are also written as CSV tables.
MODEL_STYLE: dict[str, tuple[str, str, str]] = {  # name -> (display name, color, marker)
    "transformer": ("DistilBERT", "#2a78d6", "o"),
    "mlp": ("MLP (TF-IDF)", "#eb6834", "s"),
    "lstm": ("BiLSTM", "#1baf7a", "^"),
    "cnn": ("TextCNN", "#eda100", "D"),
    "length_baseline": ("Length baseline", "#898781", "X"),
}
SURFACE, INK, SECONDARY, MUTED = "#ffffff", "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
BLUES = LinearSegmentedColormap.from_list("blues", [
    "#f4f8fe", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"])
STYLE: dict[str, Any] = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 10, "axes.titlesize": 11, "axes.titlelocation": "left",
    "axes.titlecolor": INK, "axes.labelcolor": SECONDARY, "axes.edgecolor": AXIS,
    "axes.spines.top": False, "axes.spines.right": False, "axes.axisbelow": True,
    "axes.facecolor": SURFACE, "figure.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "grid.color": GRID, "grid.linewidth": 0.8, "xtick.major.size": 0, "ytick.major.size": 0,
    "xtick.color": SECONDARY, "ytick.color": SECONDARY, "legend.frameon": False,
    "legend.fontsize": 9, "lines.linewidth": 2, "lines.solid_capstyle": "round",
    "pdf.fonttype": 42,  # editable text in the PDFs
}
BAR = 0.6  # bar thickness as a share of its slot: thin bars, the rest is air
HEAT_DARK = 0.55  # cells darker than this share of the color range get white text


@dataclass(frozen=True)
class FigureSettings:
    """The `figures` section of the config."""

    figures_subdir: str
    tables_subdir: str
    formats: tuple[str, ...]
    dpi: int
    calibration_bins: int
    length_bins: tuple[int, ...]

    @classmethod
    def from_config(cls, section: Mapping[str, Any]) -> FigureSettings:
        """Build from the config, failing loudly on missing or unknown keys."""
        expected = set(cls.__dataclass_fields__)
        missing, unknown = expected - set(section), set(section) - expected
        if missing or unknown:
            raise KeyError(f"figures config: missing {sorted(missing)}, unknown "
                           f"{sorted(unknown)}")
        bucket_labels(section["length_bins"])  # validates the edges
        return cls(section["figures_subdir"], section["tables_subdir"],
                   tuple(section["formats"]), int(section["dpi"]),
                   int(section["calibration_bins"]), tuple(section["length_bins"]))


# --- Helpers ------------------------------------------------------------------------------
def style_of(model: str) -> tuple[str, str, str]:
    """(display name, color, marker) of a model."""
    if model not in MODEL_STYLE:
        raise KeyError(f"no figure style for model {model!r}; add it to MODEL_STYLE")
    return MODEL_STYLE[model]


def name_of(model: ModelResults) -> str:
    """Display name of a model."""
    return style_of(model.name)[0]


def color_of(model: ModelResults) -> str:
    """Series color of a model."""
    return style_of(model.name)[1]


def add_titles(fig: Figure, title: str, subtitle: str | None = None) -> None:
    """Left-aligned title (and subtitle) above the whole figure."""
    if subtitle:
        fig.text(0, 1, subtitle, ha="left", va="bottom", fontsize=9.5, color=SECONDARY,
                 transform=offset_copy(fig.transFigure, fig=fig, y=8, units="points"))
    fig.text(0, 1, title, ha="left", va="bottom", fontsize=13.5, weight="bold",
             color=INK, transform=offset_copy(fig.transFigure, fig=fig,
                                              y=26 if subtitle else 8, units="points"))


def add_note(fig: Figure, note: str) -> None:
    """Small footnote under the figure."""
    fig.text(0, 0, note, ha="left", va="top", fontsize=8, color=MUTED,
             transform=offset_copy(fig.transFigure, fig=fig, y=-6, units="points"))


def heatmap(ax: plt.Axes, values: np.ndarray, rows: Sequence[str], cols: Sequence[str],
            fmt: Callable[[float], str], vmin: float | None = None,
            vmax: float | None = None) -> None:
    """Annotated heatmap on the sequential blue ramp, with a white gap between cells.

    NaN cells are left blank (surface color) and unlabeled.
    """
    values = np.asarray(values, dtype=float)
    lo = np.nanmin(values) if vmin is None else vmin
    hi = np.nanmax(values) if vmax is None else vmax
    ax.imshow(np.ma.masked_invalid(values), cmap=BLUES, vmin=lo, vmax=hi, aspect="auto")
    span = (hi - lo) or 1.0
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            if np.isnan(values[i, j]):
                continue
            dark = (values[i, j] - lo) / span > HEAT_DARK
            ax.text(j, i, fmt(values[i, j]), ha="center", va="center", fontsize=8.5,
                    color="white" if dark else INK)
    ax.set_xticks(range(len(cols)), cols)
    ax.set_yticks(range(len(rows)), rows)
    ax.set_xticks(np.arange(-0.5, len(cols)), minor=True)
    ax.set_yticks(np.arange(-0.5, len(rows)), minor=True)
    ax.grid(which="minor", color=SURFACE, linewidth=2)
    ax.tick_params(which="both", length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)


def value_grid(ax: plt.Axes, axis: str = "y") -> None:
    """Recessive hairline grid on the value axis only."""
    ax.grid(True, axis=axis)
    ax.grid(False, axis="x" if axis == "y" else "y")


def model_legend(ax_or_fig: Any, models: Sequence[ModelResults], **kwargs: Any) -> None:
    """Legend with one line-and-marker key per model."""
    handles = [Line2D([], [], color=color_of(m), marker=style_of(m.name)[2], markersize=7,
                      label=name_of(m)) for m in models]
    ax_or_fig.legend(handles=handles, **kwargs)


def ranked(data: AnalysisData, with_baseline: bool = False) -> list[ModelResults]:
    """Models best first, optionally followed by the length baseline."""
    extra = [data.baseline] if with_baseline and data.baseline is not None else []
    return [*data.models, *extra]


def fmt_pct(v: float) -> str:
    """0.834 -> '83%'."""
    return f"{v:.0%}"


def fmt_3(v: float) -> str:
    """0.8341 -> '0.834'."""
    return f"{v:.3f}"


def row_normalize(matrix: np.ndarray) -> np.ndarray:
    """Each row divided by its sum (rows that sum to 0 stay 0)."""
    matrix = np.asarray(matrix, dtype=float)
    sums = matrix.sum(axis=1, keepdims=True)
    return np.divide(matrix, sums, out=np.zeros_like(matrix), where=sums > 0)


def per_class(model: ModelResults, key: str, classes: Sequence[str]) -> list[float]:
    """Per-class test precision, recall or F1."""
    return [model.test_metrics["per_class"][c][key] for c in classes]


# --- Dataset figures ----------------------------------------------------------------------
def fig_dataset_sampling(data: AnalysisData, s: FigureSettings) -> Figure | None:
    """Texts available after cleaning vs kept by sampling, per class."""
    if not data.split_stats or "sampling" not in data.split_stats:
        return None
    sampling = data.split_stats["sampling"]
    classes = data.class_names
    available = [sampling[c]["available"] for c in classes]
    kept = [sampling[c]["kept"] for c in classes]
    fig, ax = plt.subplots(figsize=(9, 4.6), layout="constrained")
    x, w = np.arange(len(classes)), 0.36
    a = ax.bar(x - w / 2, available, w, color=GRID, edgecolor=SURFACE, linewidth=2,
               label="Available after cleaning")
    k = ax.bar(x + w / 2, kept, w, color="#2a78d6", edgecolor=SURFACE, linewidth=2,
               label="Kept (all Human, ≤40K per AI family)")
    for bars in (a, k):
        ax.bar_label(bars, labels=[f"{v / 1000:.1f}K" for v in bars.datavalues], padding=3,
                     fontsize=8, color=SECONDARY)
    ax.set_xticks(x, classes)
    ax.yaxis.set_major_formatter(lambda v, _: f"{v / 1000:.0f}K" if v else "0")
    ax.set_ylabel("Texts")
    value_grid(ax)
    ax.legend(loc="upper right")
    add_titles(fig, "Sampling: every human text, up to 40K per AI family",
               "RAID train_none after cleaning; the sampled texts are split 72/8/20 into "
               "train/val/test by source document")
    return fig


def fig_dataset_cleaning(data: AnalysisData, s: FigureSettings) -> Figure | None:
    """Rows removed by each cleaning rule."""
    if not data.split_stats or "cleaning" not in data.split_stats:
        return None
    cleaning = data.split_stats["cleaning"]
    rules = {"empty": "Empty text", "under_10_words": "Fewer than 10 words",
             "duplicate_conflicting_families": "Duplicate across families (all copies)",
             "duplicate_same_family": "Duplicate within a family (extra copies)"}
    rules = {k: v for k, v in rules.items() if k in cleaning}
    totals = [cleaning[k]["total"] for k in rules]
    fig, ax = plt.subplots(figsize=(9, 3.4), layout="constrained")
    y = np.arange(len(rules))[::-1]
    bars = ax.barh(y, totals, height=BAR, color="#2a78d6")
    ax.bar_label(bars, labels=[f"{v:,}" for v in totals], padding=4, fontsize=9, color=INK)
    ax.set_yticks(y, list(rules.values()))
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:,.0f}")
    ax.set_xlabel("Rows removed")
    ax.set_xlim(0, max(totals) * 1.15)
    value_grid(ax, "x")
    kept = cleaning.get("rows_after_cleaning", {}).get("total")
    add_titles(fig, "Cleaning removed duplicates and near-empty texts",
               f"{sum(totals):,} rows removed before sampling"
               + (f"; {kept:,} rows remain" if kept else ""))
    return fig


def fig_dataset_domains(data: AnalysisData, s: FigureSettings) -> Figure | None:
    """Train texts per domain and class."""
    if data.train_meta is None:
        return None
    counts = pd.crosstab(data.train_meta["domain"], data.train_meta["label_id"])
    counts = counts.reindex(columns=range(len(data.class_names)), fill_value=0)
    counts = counts.loc[counts.sum(axis=1).sort_values(ascending=False).index]
    fig, ax = plt.subplots(figsize=(9, 0.45 * len(counts) + 1.6), layout="constrained")
    heatmap(ax, counts.to_numpy(), list(counts.index), data.class_names, lambda v: f"{v:,.0f}",
            vmin=0)
    add_titles(fig, "Every class covers all eight domains",
               f"Train texts per domain and class ({int(counts.to_numpy().sum()):,} texts)")
    return fig


# --- Overall results ----------------------------------------------------------------------
def fig_model_comparison(data: AnalysisData, s: FigureSettings) -> Figure:
    """Test macro-F1 per model (bars) with the validation score (tick)."""
    models = ranked(data, with_baseline=True)
    fig, ax = plt.subplots(figsize=(9, 0.62 * len(models) + 1.4), layout="constrained")
    y = np.arange(len(models))[::-1]
    test = [m.summary["test_macro_f1"] for m in models]
    val = [m.summary["val_macro_f1"] for m in models]
    bars = ax.barh(y, test, height=BAR, color=[color_of(m) for m in models])
    ax.bar_label(bars, labels=[fmt_3(v) for v in test], padding=10, fontsize=10, color=INK,
                 weight="bold")
    ax.scatter(val, y, marker="|", s=420, linewidths=2.2, color=INK, zorder=3)
    ax.set_yticks(y, [name_of(m) for m in models])
    ax.set_xlim(0, 1)
    ax.set_xlabel("Macro-F1")
    value_grid(ax, "x")
    ax.legend(handles=[Patch(color=MUTED, label="Test (bar)"),
                       Line2D([], [], color=INK, marker="|", markersize=14, linestyle="none",
                              markeredgewidth=2.2, label="Validation (tick)")],
              loc="lower right", ncols=2, bbox_to_anchor=(1, 1.0))
    n = len(data.models[0].y_true)
    gap = max(abs(t - v) for t, v in zip(test, val, strict=True))
    add_titles(fig, f"{name_of(data.models[0])} is the best authorship classifier",
               f"Macro-F1 over {len(data.class_names)} classes on {n:,} held-out test texts; "
               f"test and validation scores agree within {gap:.3f}")
    return fig


def fig_metrics_overview(data: AnalysisData, s: FigureSettings) -> Figure:
    """Macro-F1, accuracy and human false-positive rate side by side."""
    models = ranked(data, with_baseline=True)
    panels = [("test_macro_f1", "Macro-F1 (higher is better)"),
              ("test_accuracy", "Accuracy (higher is better)"),
              ("test_human_fpr", "Human texts flagged as AI (lower is better)")]
    fig, axes = plt.subplots(1, 3, figsize=(13, 0.55 * len(models) + 1.6), sharey=True,
                             layout="constrained")
    y = np.arange(len(models))[::-1]
    for ax, (key, title) in zip(axes, panels, strict=True):
        values = [m.summary[key] for m in models]
        bars = ax.barh(y, values, height=BAR, color=[color_of(m) for m in models])
        ax.bar_label(bars, labels=[fmt_3(v) for v in values], padding=4, fontsize=9,
                     color=INK)
        ax.set_xlim(0, 1.08)
        ax.set_title(title)
        value_grid(ax, "x")
    axes[0].set_yticks(y, [name_of(m) for m in models])
    add_titles(fig, "Test results at a glance",
               "Human FPR: share of human-written test texts that a model assigns to any AI "
               "family")
    return fig


def fig_per_class_f1_heatmap(data: AnalysisData, s: FigureSettings) -> Figure:
    """Per-class test F1 for each model."""
    models = ranked(data)
    values = np.array([per_class(m, "f1", data.class_names) for m in models])
    fig, ax = plt.subplots(figsize=(9, 0.6 * len(models) + 1.5), layout="constrained")
    heatmap(ax, values, [name_of(m) for m in models], data.class_names, fmt_3)
    order = np.argsort(values.mean(axis=0))
    easiest, hardest = data.class_names[order[-1]], [data.class_names[i] for i in order[:2]]
    add_titles(fig, f"{easiest} is the easiest class to recognize; "
               f"{hardest[0]} and {hardest[1]} the hardest",
               "Test F1 per class (ranked by the mean over models)")
    return fig


def fig_per_class_f1_bars(data: AnalysisData, s: FigureSettings) -> Figure:
    """Per-class test F1 as grouped bars."""
    models = ranked(data)
    fig, ax = plt.subplots(figsize=(11, 4.8), layout="constrained")
    x = np.arange(len(data.class_names))
    width = 0.8 / len(models)
    for i, m in enumerate(models):
        offset = (i - (len(models) - 1) / 2) * width
        ax.bar(x + offset, per_class(m, "f1", data.class_names), width, color=color_of(m),
               edgecolor=SURFACE, linewidth=1.5, label=name_of(m))
    ax.set_xticks(x, data.class_names)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Test F1")
    value_grid(ax)
    ax.legend(ncols=len(models), loc="lower right", bbox_to_anchor=(1, 1.0))
    add_titles(fig, "Model ranking holds for every class",
               "Test F1 per class; exact values in tables/per_class_test.csv")
    return fig


def fig_precision_recall(data: AnalysisData, s: FigureSettings) -> Figure:
    """Per-class test precision and recall heatmaps."""
    models = ranked(data)
    fig, axes = plt.subplots(1, 2, figsize=(14, 0.6 * len(models) + 1.7), sharey=True,
                             layout="constrained")
    for ax, key, title in zip(axes, ("precision", "recall"), ("Precision", "Recall"),
                              strict=True):
        values = np.array([per_class(m, key, data.class_names) for m in models])
        heatmap(ax, values, [name_of(m) for m in models], data.class_names, fmt_3,
                vmin=0.4, vmax=1.0)
        ax.set_title(title)
    add_titles(fig, "Precision and recall per class (test)",
               "Precision: of the texts given a label, the share that truly have it. "
               "Recall: of the texts of a class, the share found")
    return fig


def draw_confusion(ax: plt.Axes, matrix: np.ndarray, classes: Sequence[str],
                   show_counts: bool) -> None:
    """Row-normalized confusion matrix (rows: true class), optionally with counts."""
    counts = np.asarray(matrix)
    share = row_normalize(counts)
    if show_counts:
        def fmt(i: int, j: int) -> str:
            return f"{share[i, j]:.0%}\n{counts[i, j]:,}"
    else:
        def fmt(i: int, j: int) -> str:
            return f"{share[i, j]:.0%}"
    heatmap(ax, share, classes, classes, lambda v: "", vmin=0, vmax=1)
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, fmt(i, j), ha="center", va="center", fontsize=8.5,
                    color="white" if share[i, j] > HEAT_DARK else INK)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")


def fig_confusion_all(data: AnalysisData, s: FigureSettings) -> Figure:
    """Row-normalized test confusion matrices of all models."""
    models = ranked(data)
    cols = 2
    rows = int(np.ceil(len(models) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(12, 5.2 * rows), layout="constrained",
                             squeeze=False)
    for ax, m in zip(axes.flat, models, strict=False):
        draw_confusion(ax, m.test_metrics["confusion_matrix"], data.class_names, False)
        ax.set_title(f"{name_of(m)}  ·  macro-F1 {m.summary['test_macro_f1']:.3f}")
    for ax in list(axes.flat)[len(models):]:
        ax.set_visible(False)
    add_titles(fig, "Where each model goes wrong (test)",
               "Each row is a true class and sums to 100%; the diagonal is recall")
    return fig


def confusion_figure(model: ModelResults, classes: Sequence[str]) -> Figure:
    """One model's test confusion matrix, with shares and counts."""
    fig, ax = plt.subplots(figsize=(7.5, 6.4), layout="constrained")
    draw_confusion(ax, model.test_metrics["confusion_matrix"], classes, True)
    add_titles(fig, f"{name_of(model)}: test confusion matrix",
               f"Macro-F1 {model.summary['test_macro_f1']:.3f} · each row is a true class "
               "and sums to 100% · counts below the shares")
    return fig


def fig_human_outcomes(data: AnalysisData, s: FigureSettings) -> Figure:
    """Where human texts go, and what is labelled Human."""
    models = ranked(data)
    h = data.human_id
    went = np.array([row_normalize(m.test_metrics["confusion_matrix"])[h] for m in models])
    cm = [np.asarray(m.test_metrics["confusion_matrix"], dtype=float) for m in models]
    came = np.array([c[:, h] / c[:, h].sum() if c[:, h].sum() else c[:, h] for c in cm])
    fig, axes = plt.subplots(1, 2, figsize=(14, 0.6 * len(models) + 1.9), sharey=True,
                             layout="constrained")
    heatmap(axes[0], went, [name_of(m) for m in models], data.class_names, fmt_pct, 0, 1)
    axes[0].set_title("Human-written texts were predicted as …")
    axes[0].set_xlabel("Predicted class")
    heatmap(axes[1], came, [name_of(m) for m in models], data.class_names, fmt_pct, 0, 1)
    axes[1].set_title("Texts predicted Human were actually …")
    axes[1].set_xlabel("True class")
    missed = went.copy()
    missed[:, h] = 0
    sink = data.class_names[int(missed.sum(axis=0).argmax())]
    add_titles(fig, f"Missed human texts are most often mistaken for {sink}",
               "Left rows: recall of Human and where the rest went. Right rows: precision of "
               "Human and which families get mistaken for people")
    return fig


def human_vs_ai(model: ModelResults, human_id: int) -> dict[str, Any]:
    """Binary human-vs-AI view of a 6-way model: AI score = 1 - P(Human)."""
    is_ai = model.y_true != human_id
    score = 1 - model.y_prob[:, human_id]
    fpr, tpr, _ = roc_curve(is_ai, score)
    predicted_ai = model.y_pred != human_id
    return {"fpr": fpr, "tpr": tpr, "auc": roc_auc_score(is_ai, score),
            "op_fpr": float(predicted_ai[~is_ai].mean()),
            "op_tpr": float(predicted_ai[is_ai].mean())}


def fig_human_vs_ai_roc(data: AnalysisData, s: FigureSettings) -> Figure:
    """ROC of the binary human-vs-AI decision implied by each model."""
    models = ranked(data)
    fig, ax = plt.subplots(figsize=(7.2, 6.4), layout="constrained")
    ax.plot([0, 1], [0, 1], color=AXIS, linewidth=1)
    handles = []
    for m in models:
        r = human_vs_ai(m, data.human_id)
        ax.plot(r["fpr"], r["tpr"], color=color_of(m))
        ax.scatter([r["op_fpr"]], [r["op_tpr"]], s=70, color=color_of(m),
                   marker=style_of(m.name)[2], edgecolors=SURFACE, linewidths=2, zorder=3)
        handles.append(Line2D([], [], color=color_of(m), marker=style_of(m.name)[2],
                              label=f"{name_of(m)}  AUC {r['auc']:.3f}"))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.01)
    ax.set_xlabel("Human texts flagged as AI (false-positive rate)")
    ax.set_ylabel("AI texts flagged as AI (true-positive rate)")
    ax.grid(True)
    ax.legend(handles=handles, loc="lower right")
    add_titles(fig, "As a human-vs-AI detector, every model ranks texts well",
               "AI score = 1 − P(Human); markers show where each model's 6-way decision lands. "
               "A stricter threshold would flag fewer human texts")
    return fig


def fig_pr_per_class(data: AnalysisData, s: FigureSettings) -> Figure:
    """One-vs-rest precision-recall curves per class (test)."""
    models = ranked(data)
    classes = data.class_names
    cols = 3
    rows = int(np.ceil(len(classes) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(13, 4 * rows), sharex=True, sharey=True,
                             layout="constrained", squeeze=False)
    for k, (ax, c) in enumerate(zip(axes.flat, classes, strict=False)):
        best = 0.0
        for m in models:
            truth = m.y_true == k
            precision, recall, _ = precision_recall_curve(truth, m.y_prob[:, k])
            ax.plot(recall, precision, color=color_of(m))
            best = max(best, average_precision_score(truth, m.y_prob[:, k]))
        ax.set_title(f"{c}  ·  best average precision {best:.3f}")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.02)
        ax.grid(True)
    for ax in axes[-1]:
        ax.set_xlabel("Recall")
    for ax in axes[:, 0]:
        ax.set_ylabel("Precision")
    model_legend(fig, models, loc="outside lower center", ncols=len(models))
    add_titles(fig, "Precision-recall per class (test, one class vs the rest)",
               "Curves higher and further right are better; average precision per model in "
               "tables/average_precision.csv")
    return fig


# --- Breakdowns ---------------------------------------------------------------------------
def domain_f1(data: AnalysisData) -> pd.DataFrame:
    """Test macro-F1 per model (rows) and domain (columns), best domain first."""
    n = len(data.class_names)
    table = pd.DataFrame({name_of(m): macro_f1_by_group(m.y_true, m.y_pred, m.domains, n)
                          for m in ranked(data)}).T
    return table[table.iloc[0].sort_values(ascending=False).index]


def fig_f1_by_domain(data: AnalysisData, s: FigureSettings) -> Figure:
    """Test macro-F1 within each domain."""
    table = domain_f1(data)
    counts = pd.Series(data.models[0].domains).value_counts()
    cols = [f"{d}\n{counts[d]:,}" for d in table.columns]
    fig, ax = plt.subplots(figsize=(12, 0.6 * len(table) + 1.8), layout="constrained")
    heatmap(ax, table.to_numpy(), list(table.index), cols, fmt_3)
    add_titles(fig, f"Performance depends on the domain: {table.columns[0]} easiest, "
               f"{table.columns[-1]} hardest",
               "Test macro-F1 within each domain (test texts per domain under the name)")
    return fig


def fig_domain_class_best(data: AnalysisData, s: FigureSettings) -> Figure:
    """Best model's test F1 per class within each domain."""
    best = data.models[0]
    table = per_class_f1_by_group(best.y_true, best.y_pred, best.domains, data.class_names)
    table = table[domain_f1(data).columns]
    fig, ax = plt.subplots(figsize=(12, 4.6), layout="constrained")
    heatmap(ax, table.to_numpy(), list(table.index), list(table.columns), fmt_3)
    add_titles(fig, f"{name_of(best)}: F1 per class and domain (test)",
               "Which families are hard to tell apart depends on the kind of text")
    return fig


def length_f1(data: AnalysisData, s: FigureSettings) -> pd.DataFrame:
    """Val macro-F1 per model (rows) and length bucket (columns)."""
    buckets = length_buckets(data.val_words, s.length_bins)
    n = len(data.class_names)
    rows = {}
    for m in ranked(data):
        if m.val_prob is None:
            continue
        keep = buckets != ""
        scores = macro_f1_by_group(m.val_true[keep], m.val_prob[keep].argmax(axis=1),
                                   buckets[keep], n)
        rows[name_of(m)] = scores
    return pd.DataFrame(rows).T.reindex(columns=bucket_labels(s.length_bins))  # NaN if empty


def fig_f1_by_length(data: AnalysisData, s: FigureSettings) -> Figure | None:
    """Val macro-F1 by text length."""
    if data.val_words is None:
        return None
    table = length_f1(data, s)
    buckets = length_buckets(data.val_words, s.length_bins)
    counts = pd.Series(buckets[buckets != ""]).value_counts()
    fig, ax = plt.subplots(figsize=(10, 5), layout="constrained")
    x = np.arange(len(table.columns))
    for m in ranked(data):
        if name_of(m) not in table.index:
            continue
        values = table.loc[name_of(m)].to_numpy()
        ax.plot(x, values, color=color_of(m), marker=style_of(m.name)[2], markersize=7,
                markeredgecolor=SURFACE, markeredgewidth=1.5)
        last = np.flatnonzero(~np.isnan(values))[-1]  # empty buckets are NaN
        ax.annotate(f"{name_of(m)} ", (x[last], values[last]), xytext=(8, 0),
                    textcoords="offset points", va="center", fontsize=9, color=INK)
    ax.set_xticks(x, [f"{c}\n{counts.get(c, 0):,}" for c in table.columns])
    ax.set_xlim(-0.3, len(x) - 1 + 1.2)
    ax.set_xlabel("Words per text (validation texts per bucket under the range)")
    ax.set_ylabel("Validation macro-F1")
    value_grid(ax)
    model_legend(ax, ranked(data), loc="lower right", bbox_to_anchor=(1, 1.0),
                 ncols=len(data.models))
    mean = table.mean(axis=0)
    add_titles(fig, f"Texts of {mean.idxmax()} words are easiest; {mean.idxmin()} words "
               "hardest",
               "Validation macro-F1 per length bucket (the test texts are never reloaded); "
               "DistilBERT, CNN and LSTM read only the first 256 tokens/words")
    return fig


# --- Training and tuning ------------------------------------------------------------------
def fig_learning_curves(data: AnalysisData, s: FigureSettings) -> Figure:
    """Val macro-F1 after each epoch; the ringed point is the epoch that was kept."""
    fig, ax = plt.subplots(figsize=(10, 5.2), layout="constrained")
    for m in ranked(data):
        if not m.history:
            continue
        epochs = [r["epoch"] for r in m.history]
        f1 = [r["val_macro_f1"] for r in m.history]
        ax.plot(epochs, f1, color=color_of(m), marker=style_of(m.name)[2], markersize=5)
        best = int(np.argmax(f1))
        ax.scatter([epochs[best]], [f1[best]], s=150, facecolors="none",
                   edgecolors=color_of(m), linewidths=2, zorder=3)
        ax.annotate(f" {name_of(m)} {max(f1):.3f}", (epochs[-1], f1[-1]), xytext=(6, 0),
                    textcoords="offset points", va="center", fontsize=9, color=INK)
    max_epoch = max(r["epoch"] for m in data.models for r in m.history)
    ax.set_xticks(range(1, max_epoch + 1))
    ax.set_xlim(0.7, max_epoch + 2.2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation macro-F1")
    value_grid(ax)
    model_legend(ax, [m for m in data.models if m.history], loc="lower right",
                 bbox_to_anchor=(1, 1.0), ncols=len(data.models))
    add_titles(fig, "Learning curves",
               "Validation macro-F1 after every epoch; the ringed point is the checkpoint kept")
    return fig


def fig_loss_curves(data: AnalysisData, s: FigureSettings) -> Figure | None:
    """Train vs val loss per epoch for the models trained with src.trainer."""
    models = [m for m in ranked(data) if m.history and "train_loss" in m.history[0]]
    if not models:
        return None
    fig, axes = plt.subplots(1, len(models), figsize=(4.4 * len(models), 4), sharey=True,
                             layout="constrained", squeeze=False)
    lowest = []
    for ax, m in zip(axes[0], models, strict=True):
        epochs = [r["epoch"] for r in m.history]
        train = [r["train_loss"] for r in m.history]
        val = [r["val_log_loss"] for r in m.history]
        ax.plot(epochs, train, color=SECONDARY)
        ax.plot(epochs, val, color=color_of(m))
        for values, label in ((train, "train"), (val, "validation")):
            ax.annotate(label, (epochs[-1], values[-1]), xytext=(6, 0),
                        textcoords="offset points", va="center", fontsize=9, color=INK)
        best = int(np.argmin(val))
        ax.scatter([epochs[best]], [val[best]], s=60, color=color_of(m), edgecolors=SURFACE,
                   linewidths=2, zorder=3)
        lowest.append(epochs[best])
        ax.set_title(f"{name_of(m)}  ·  lowest validation loss at epoch {epochs[best]}")
        ax.set_xlabel("Epoch")
        ax.set_xticks(epochs)
        ax.set_xlim(epochs[0] - 0.4, epochs[-1] + 2)
        value_grid(ax)
    axes[0, 0].set_ylabel("Loss")
    add_titles(fig, f"Validation loss bottoms out by epoch {max(lowest)} while training loss "
               "keeps falling",
               "Train: class-weighted cross-entropy; validation: log loss. Macro-F1 kept "
               "improving after the minimum (see the learning curves): the models grow "
               "overconfident rather than less accurate")
    return fig


def fig_transformer_training(data: AnalysisData, s: FigureSettings) -> Figure | None:
    """DistilBERT training loss per logged step, with val loss at the end of each epoch."""
    model = next((m for m in data.models if m.name == "transformer" and m.train_log), None)
    if model is None:
        return None
    log = parse_train_log(model.train_log)
    if log.empty:
        return None
    train, evals = log[log["kind"] == "train"], log[log["kind"] == "eval"]
    fig, ax = plt.subplots(figsize=(11, 4.6), layout="constrained")
    ax.plot(train["step"], train["loss"], color=color_of(model), linewidth=1.5,
            label="Training loss (every 200 steps)")
    ax.scatter(evals["step"], evals["loss"], s=70, color=INK, edgecolors=SURFACE,
               linewidths=2, zorder=3, label="Validation loss (end of epoch)")
    for epoch, row in enumerate(evals.itertuples(), start=1):  # one evaluation per epoch
        ax.axvline(row.step, color=GRID, linewidth=1, zorder=0)
        ax.annotate(f"epoch {epoch}\nval macro-F1 {row.macro_f1:.3f}",
                    (row.step, ax.get_ylim()[1]), xytext=(-4, -4), textcoords="offset points",
                    ha="right", va="top", fontsize=8, color=SECONDARY)
    ax.set_xlabel("Optimizer step (batch of 32)")
    ax.set_ylabel("Class-weighted cross-entropy")
    value_grid(ax)
    ax.legend(loc="lower left")
    add_titles(fig, "DistilBERT fine-tuning",
               "Training loss keeps falling while validation loss rises after epoch 2: the "
               "model grows overconfident, yet macro-F1 still edges up")
    return fig


def fig_tuning(data: AnalysisData, s: FigureSettings) -> Figure | None:
    """Val macro-F1 of every tuning run; the larger dot is the run that was tested."""
    if data.tuning.empty:
        return None
    models = [m for m in ranked(data) if (data.tuning["model"] == m.name).any()]
    fig, ax = plt.subplots(figsize=(10, 0.8 * len(models) + 1.6), layout="constrained")
    y = np.arange(len(models))[::-1]
    labels, spreads = [], []
    for yi, m in zip(y, models, strict=True):
        # A run recorded under two names (e.g. "default" and its sweep name) has the
        # identical score; count each setting once.
        runs = data.tuning[data.tuning["model"] == m.name].drop_duplicates("val_macro_f1")
        spreads.append(runs["val_macro_f1"].max() - runs["val_macro_f1"].min())
        ax.scatter(runs["val_macro_f1"], np.full(len(runs), yi), s=45, color=MUTED,
                   alpha=0.7, edgecolors=SURFACE, linewidths=1)
        chosen = m.summary["val_macro_f1"]
        ax.scatter([chosen], [yi], s=150, color=color_of(m), marker=style_of(m.name)[2],
                   edgecolors=SURFACE, linewidths=2, zorder=3)
        ax.annotate(f"{chosen:.3f}", (chosen, yi), xytext=(0, 11), textcoords="offset points",
                    ha="center", fontsize=9, color=INK)
        labels.append(f"{name_of(m)}\n{len(runs)} runs, range {spreads[-1]:.3f}")
    ax.set_yticks(y, labels)
    ax.set_ylim(y.min() - 0.6, y.max() + 0.6)
    ax.set_xlabel("Validation macro-F1")
    value_grid(ax, "x")
    chosen = [m.summary["val_macro_f1"] for m in models]
    add_titles(fig, f"Tuning moved scores by up to {max(spreads) * 100:.0f} points; the choice "
               f"of model by {(max(chosen) - min(chosen)) * 100:.0f}",
               "Each gray dot is one hyperparameter setting; the colored dot is the run tested. "
               "All runs in tables/tuning_runs.csv")
    return fig


def fig_efficiency(data: AnalysisData, s: FigureSettings, key: str) -> Figure:
    """Test macro-F1 against model size or training time."""
    models = ranked(data)
    f1 = [m.summary["test_macro_f1"] for m in models]
    best = models[0]
    if key == "num_parameters":
        values = [m.summary["num_parameters"] / 1e6 for m in models]
        xlabel = "Trainable parameters (millions, log scale)"
        title = "Accuracy vs model size"
        if len(models) > 1:
            runner = models[1]
            ratio = best.summary["num_parameters"] / runner.summary["num_parameters"]
            if ratio > 1:
                title = (f"{name_of(runner)} reaches {f1[1] / f1[0]:.0%} of {name_of(best)}'s "
                         f"score with 1/{ratio:.0f} of the parameters")
        note = ("Length baseline (24 parameters, macro-F1 "
                f"{data.baseline.summary['test_macro_f1']:.2f}) left out of the axis range."
                if data.baseline else "")
        suffix = "M"
    else:
        values = [m.summary["train_time_seconds"] / 60 for m in models]
        xlabel = "Training time (minutes, log scale)"
        slowest = max(models, key=lambda m: m.summary["train_time_seconds"])
        title = f"Accuracy vs training time: {name_of(slowest)} is the slowest to train"
        note = ("Wall-clock time on the machine each model was trained on (DistilBERT on a "
                "Colab A100 GPU, the others on different machines); not a controlled "
                "speed benchmark.")
        suffix = " min"
    fig, ax = plt.subplots(figsize=(8.5, 5.2), layout="constrained")
    for m, v, f in zip(models, values, f1, strict=True):
        ax.scatter([v], [f], s=160, color=color_of(m), marker=style_of(m.name)[2],
                   edgecolors=SURFACE, linewidths=2, zorder=3)
        ax.annotate(f"{name_of(m)}\n{v:,.1f}{suffix} · {f:.3f}", (v, f), xytext=(10, 0),
                    textcoords="offset points", va="center", fontsize=9, color=INK)
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:g}")  # 10, 100 instead of 10^1, 10^2
    lo, hi = min(values), max(values)
    ax.set_xlim(lo / 1.8, hi * 4)
    ax.set_ylim(min(f1) - 0.04, max(f1) + 0.04)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Test macro-F1")
    ax.grid(True)
    add_titles(fig, title, "Up and to the left is better")
    if note:
        add_note(fig, note)
    return fig


# --- Confidence and agreement -------------------------------------------------------------
def fig_calibration(data: AnalysisData, s: FigureSettings) -> Figure:
    """Reliability diagram of the top-class confidence (test)."""
    models = ranked(data)
    fig, ax = plt.subplots(figsize=(7.2, 6.4), layout="constrained")
    ax.plot([0, 1], [0, 1], color=AXIS, linewidth=1)
    handles, eces = [], {}
    for m in models:
        curve = reliability_curve(m.y_true, m.y_prob, s.calibration_bins)
        ece = eces[name_of(m)] = expected_calibration_error(m.y_true, m.y_prob,
                                                           s.calibration_bins)
        ax.plot(curve["confidence"], curve["accuracy"], color=color_of(m),
                marker=style_of(m.name)[2], markersize=6, markeredgecolor=SURFACE)
        handles.append(Line2D([], [], color=color_of(m), marker=style_of(m.name)[2],
                              label=f"{name_of(m)}  ECE {ece:.3f}"))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Confidence (probability of the predicted class)")
    ax.set_ylabel("Accuracy")
    ax.grid(True)
    ax.legend(handles=handles, loc="upper left")
    ax.annotate("perfectly calibrated", (0.62, 0.6), rotation=40, fontsize=8, color=MUTED)
    add_titles(fig, f"All models are overconfident; {min(eces, key=eces.get)} is the best "
               "calibrated",
               f"Test texts in {s.calibration_bins} confidence bins; below the diagonal = "
               "overconfident. ECE = expected calibration error")
    return fig


def fig_confidence(data: AnalysisData, s: FigureSettings) -> Figure:
    """Confidence of correct vs wrong predictions, per model (test)."""
    models = ranked(data)
    cols = 2
    rows = int(np.ceil(len(models) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(11, 3.6 * rows), sharex=True,
                             layout="constrained", squeeze=False)
    bins = np.linspace(0, 1, 21)
    for ax, m in zip(axes.flat, models, strict=False):
        confidence = m.y_prob.max(axis=1)
        correct = m.y_pred == m.y_true
        for mask, color, label in ((correct, color_of(m), "correct"),
                                   (~correct, SECONDARY, "wrong")):
            weights = np.full(mask.sum(), 1 / len(confidence))
            ax.hist(confidence[mask], bins=bins, weights=weights, histtype="stepfilled",
                    color=color, alpha=0.15)
            ax.hist(confidence[mask], bins=bins, weights=weights, histtype="step",
                    color=color, linewidth=2, label=label)
        ax.set_title(f"{name_of(m)}  ·  median confidence when wrong "
                     f"{np.median(confidence[~correct]):.2f}")
        ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
        value_grid(ax)
        ax.legend(loc="upper left")
    for ax in list(axes.flat)[len(models):]:
        ax.set_visible(False)
    for ax in axes[-1]:
        ax.set_xlabel("Confidence (probability of the predicted class)")
    add_titles(fig, "How sure are the models when they are right vs wrong?",
               "Share of all test texts per confidence bin; wrong answers with high confidence "
               "are the dangerous ones")
    return fig


def fig_models_correct(data: AnalysisData, s: FigureSettings) -> Figure:
    """How many of the models get each test text right."""
    models = ranked(data)
    correct = np.sum([m.y_pred == m.y_true for m in models], axis=0)
    share = np.bincount(correct, minlength=len(models) + 1) / len(correct)
    fig, ax = plt.subplots(figsize=(8.5, 4.6), layout="constrained")
    bars = ax.bar(range(len(share)), share, width=BAR, color="#2a78d6")
    ax.bar_label(bars, labels=[fmt_pct(v) if v >= 0.005 else f"{v:.1%}" for v in share],
                 padding=3, fontsize=9.5, color=INK)
    ax.set_xticks(range(len(share)), [f"{k} of {len(models)}" for k in range(len(share))])
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.set_xlabel("Models that classify the text correctly")
    ax.set_ylabel("Test texts")
    value_grid(ax)
    add_titles(fig, f"{share[-1]:.0%} of test texts are solved by every model; "
               f"{share[0]:.0%} by none",
               "The texts no model gets right are the natural place to look for label noise "
               "or genuinely ambiguous writing")
    return fig


def fig_agreement(data: AnalysisData, s: FigureSettings) -> Figure:
    """Pairwise agreement of predictions and of errors (test)."""
    models = ranked(data)
    names = [name_of(m) for m in models]
    k = len(models)
    agree, both_wrong = np.zeros((k, k)), np.zeros((k, k))
    for i, a in enumerate(models):
        for j, b in enumerate(models):
            agree[i, j] = np.mean(a.y_pred == b.y_pred)
            wrong_a, wrong_b = a.y_pred != a.y_true, b.y_pred != b.y_true
            union = np.sum(wrong_a | wrong_b)
            both_wrong[i, j] = np.sum(wrong_a & wrong_b) / union if union else 0
    np.fill_diagonal(agree, np.nan)  # a model always agrees with itself
    np.fill_diagonal(both_wrong, np.nan)
    fig, axes = plt.subplots(1, 2, figsize=(13, 0.75 * k + 2.2), layout="constrained")
    heatmap(axes[0], agree, names, names, fmt_pct, vmin=0.5, vmax=1)
    axes[0].set_title("Same prediction (share of test texts)")
    heatmap(axes[1], both_wrong, names, names, fmt_pct, vmin=0, vmax=1)
    axes[1].set_title("Shared errors (both wrong ÷ either wrong)")
    for ax in axes:
        ax.tick_params(axis="x", rotation=20)
    add_titles(fig, f"The models share only {np.nanmin(both_wrong):.0%}–"
               f"{np.nanmax(both_wrong):.0%} of their errors",
               "They fail on different texts, so combining them (an ensemble) could help")
    return fig


# --- Registry -----------------------------------------------------------------------------
FigureFn = Callable[[AnalysisData, FigureSettings], Figure | None]
FIGURES: list[tuple[str, FigureFn, str]] = [
    ("01_dataset_sampling", fig_dataset_sampling,
     "Texts per class available after cleaning and kept by sampling."),
    ("02_dataset_cleaning", fig_dataset_cleaning, "Rows removed by each cleaning rule."),
    ("03_dataset_domains", fig_dataset_domains, "Train texts per domain and class."),
    ("10_model_comparison", fig_model_comparison,
     "Headline result: test macro-F1 per model, validation score as a tick."),
    ("11_metrics_overview", fig_metrics_overview,
     "Test macro-F1, accuracy and human false-positive rate side by side."),
    ("12_per_class_f1_heatmap", fig_per_class_f1_heatmap, "Test F1 per model and class."),
    ("13_per_class_f1_bars", fig_per_class_f1_bars, "Test F1 per class as grouped bars."),
    ("14_precision_recall", fig_precision_recall,
     "Test precision and recall per model and class."),
    ("15_confusion_matrices", fig_confusion_all,
     "Row-normalized test confusion matrices of all four models."),
    ("20_human_outcomes", fig_human_outcomes,
     "Where human texts are sent, and which families are mistaken for human."),
    ("21_human_vs_ai_roc", fig_human_vs_ai_roc,
     "ROC of the binary human-vs-AI decision (AI score = 1 - P(Human))."),
    ("22_pr_curves_per_class", fig_pr_per_class,
     "One-vs-rest precision-recall curves per class."),
    ("30_f1_by_domain", fig_f1_by_domain, "Test macro-F1 within each domain."),
    ("31_domain_class_best_model", fig_domain_class_best,
     "Best model's test F1 per class and domain."),
    ("32_f1_by_length_val", fig_f1_by_length,
     "Validation macro-F1 by text length (validation split)."),
    ("40_learning_curves", fig_learning_curves, "Validation macro-F1 after every epoch."),
    ("41_loss_curves", fig_loss_curves, "Train vs validation loss for MLP, BiLSTM, TextCNN."),
    ("42_distilbert_training", fig_transformer_training,
     "DistilBERT training loss per step with validation loss per epoch."),
    ("43_tuning_runs", fig_tuning, "Validation macro-F1 of every tuning run."),
    ("50_efficiency_parameters", lambda d, s: fig_efficiency(d, s, "num_parameters"),
     "Test macro-F1 vs number of parameters."),
    ("51_efficiency_train_time", lambda d, s: fig_efficiency(d, s, "train_time_seconds"),
     "Test macro-F1 vs training time."),
    ("60_calibration", fig_calibration, "Reliability diagram and expected calibration error."),
    ("61_confidence_correct_vs_wrong", fig_confidence,
     "Confidence distribution of correct vs wrong predictions."),
    ("62_models_correct_per_text", fig_models_correct,
     "How many models classify each test text correctly."),
    ("63_model_agreement", fig_agreement, "Pairwise agreement of predictions and errors."),
]


def all_figures(data: AnalysisData,
                s: FigureSettings) -> Iterator[tuple[str, FigureFn, str]]:
    """The registry plus one large confusion matrix per model."""
    yield from FIGURES
    for i, m in enumerate(ranked(data)):
        def one(d: AnalysisData, st: FigureSettings, model: ModelResults = m) -> Figure:
            return confusion_figure(model, d.class_names)
        yield (f"16_confusion_{i + 1}_{m.name}", one,
               f"{name_of(m)} test confusion matrix with shares and counts.")


# --- Tables -------------------------------------------------------------------------------
def build_tables(data: AnalysisData, s: FigureSettings) -> dict[str, pd.DataFrame]:
    """The numbers behind the figures, one table per CSV file."""
    models = ranked(data, with_baseline=True)
    summary = pd.DataFrame([{
        "model": name_of(m), "run": m.run, "test_macro_f1": m.summary["test_macro_f1"],
        "test_accuracy": m.summary["test_accuracy"],
        "test_human_fpr": m.summary["test_human_fpr"],
        "val_macro_f1": m.summary["val_macro_f1"],
        "test_log_loss": m.test_metrics.get("log_loss"),
        "test_ece": expected_calibration_error(m.y_true, m.y_prob, s.calibration_bins),
        "human_vs_ai_auc": human_vs_ai(m, data.human_id)["auc"],
        "num_parameters": m.summary["num_parameters"],
        "train_minutes": m.summary["train_time_seconds"] / 60} for m in models])
    per_class_rows = [{"model": name_of(m), "class": c, **m.test_metrics["per_class"][c]}
                      for m in models for c in data.class_names]
    ap_rows = [{"model": name_of(m), "class": c,
                "average_precision": average_precision_score(m.y_true == k, m.y_prob[:, k])}
               for m in ranked(data) for k, c in enumerate(data.class_names)]
    domain = domain_f1(data).rename_axis("model").reset_index()
    tables = {"summary": summary, "per_class_test": pd.DataFrame(per_class_rows),
              "average_precision": pd.DataFrame(ap_rows), "domain_macro_f1_test": domain,
              "tuning_runs": data.tuning}
    best = data.models[0]
    domain_class = per_class_f1_by_group(best.y_true, best.y_pred, best.domains,
                                         data.class_names)
    tables["domain_class_f1_best_model_test"] = domain_class.rename_axis("class").reset_index()
    if data.val_words is not None:
        tables["length_macro_f1_val"] = length_f1(data, s).rename_axis("model").reset_index()
    correct = np.sum([m.y_pred == m.y_true for m in data.models], axis=0)
    tables["models_correct_per_text"] = pd.DataFrame({
        "models_correct": range(len(data.models) + 1),
        "share_of_test_texts": np.bincount(correct, minlength=len(data.models) + 1)
        / len(correct)})
    tables["macro_f1_check"] = pd.DataFrame([{
        "model": name_of(m), "saved": m.summary["test_macro_f1"],
        "from_predictions": f1_score(m.y_true, m.y_pred, average="macro")} for m in models])
    return tables


# --- Output -------------------------------------------------------------------------------
def save_figure(fig: Figure, out_dir: Path, name: str, s: FigureSettings) -> list[Path]:
    """Write the figure in every configured format (atomically) and close it."""
    paths = []
    for fmt in s.formats:
        path = out_dir / f"{name}.{fmt}"
        with atomic_open(path) as f:
            fig.savefig(f, format=fmt, dpi=s.dpi, bbox_inches="tight", pad_inches=0.25)
        paths.append(path)
    plt.close(fig)
    return paths


def write_tables(tables: Mapping[str, pd.DataFrame], out_dir: Path) -> list[Path]:
    """Write each table as <name>.csv (atomically)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, table in tables.items():
        path = out_dir / f"{name}.csv"
        with atomic_open(path) as f:
            table.to_csv(f, index=False, float_format="%.4f")
        paths.append(path)
    return paths


def check_predictions(tables: Mapping[str, pd.DataFrame]) -> None:
    """Fail loudly if the saved predictions do not reproduce the saved test macro-F1."""
    check = tables["macro_f1_check"]
    bad = check[(check["saved"] - check["from_predictions"]).abs() > 1e-9]
    if not bad.empty:
        raise ValueError(f"saved test predictions do not match the saved metrics:\n{bad}")


def make_all(data: AnalysisData, s: FigureSettings, results_dir: Path,
             only: Sequence[str] | None = None) -> list[Path]:
    """Draw every figure (or only those whose name contains one of `only`) and the tables."""
    figures_dir = results_dir / s.figures_subdir
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables = build_tables(data, s)
    check_predictions(tables)
    written = write_tables(tables, results_dir / s.tables_subdir)
    captions = []
    with matplotlib.rc_context(STYLE):
        for name, fn, caption in all_figures(data, s):
            if only and not any(o in name for o in only):
                continue
            fig = fn(data, s)
            if fig is None:
                logger.warning("%s skipped: its input files are missing", name)
                continue
            written += save_figure(fig, figures_dir, name, s)
            captions.append(f"{name}: {caption}")
            logger.info("Wrote %s", name)
    if not only:
        atomic_write_bytes(figures_dir / "captions.txt",
                           ("\n".join(captions) + "\n").encode("utf-8"))
    return written


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Report and slide figures from the results.")
    parser.add_argument("--config", default=None, help="config YAML (default: configs/config.yaml)")
    parser.add_argument("--only", nargs="+", default=None,
                        help="only figures whose name contains one of these strings")
    args = parser.parse_args(argv)
    setup_logging()
    use_config(args.config)
    cfg = load_config()
    settings = FigureSettings.from_config(cfg["figures"])
    results_dir = get_path("results_dir")
    data = load_analysis_data(results_dir, cfg["classes"], cfg["training"]["human_label"])
    written = make_all(data, settings, results_dir, args.only)
    figures = sorted({p.stem for p in written if p.parent.name == settings.figures_subdir})
    print(f"\n{len(figures)} figures ({', '.join(settings.formats)}) in "
          f"{results_dir / settings.figures_subdir}")
    print(f"Tables in {results_dir / settings.tables_subdir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
