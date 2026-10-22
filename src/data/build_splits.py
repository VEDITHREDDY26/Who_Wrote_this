"""Build the frozen train/val/test splits from raw RAID train_none.csv.

Pipeline (settings from configs/config.yaml):
  1. Normalize text for checks only: drop zero-width chars, strip whitespace.
     The original `generation` text stays the model input.
  2. Drop rows whose normalized text is empty.
  3. Drop rows with fewer than `min_words` words.
  4. Duplicates by normalized text: if copies span more than one family, drop every copy;
     otherwise keep the copy with the smallest id and drop the rest.
  5. Map raw `model` -> 6-class label (fixed order Human=0 ... Cohere=5).
  6. Sample: every Human row, up to 40,000 rows per AI family (seed 42).
  7. GroupShuffleSplit on group (= source_id, fallback id): 80/20 trainval/test,
     then 90/10 train/val (seed 42).

Writes <splits_dir>/{train,val,test}.parquet (id, text, label, label_id, domain, group),
<results_dir>/split_stats.json and two figures in <results_dir>/figures/.
Splits are frozen: the script refuses to overwrite them unless --force is given.

Usage: python -m src.data.build_splits [--force]
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import re

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from src.config import get_path, load_config

SPLITS = ("train", "val", "test")
OUT_COLUMNS = ["id", "text", "label", "label_id", "domain", "group"]


def normalize_text(texts: pd.Series, cfg: dict | None = None) -> pd.Series:
    """Text used for emptiness/length/duplicate checks only (never fed to a model)."""
    cfg = cfg or load_config()
    zero_width = "[" + "".join(map(re.escape, cfg["cleaning"]["zero_width_chars"])) + "]"
    return texts.fillna("").astype(str).str.replace(zero_width, "", regex=True).str.strip()


def load_raw(cfg: dict) -> pd.DataFrame:
    """Read only the allowed columns and attach `family` and `group`."""
    ds = cfg["dataset"]
    text_col, model_col = ds["text_column"], ds["label_source_column"]
    group_col, fallback_col = ds["group_column"], ds["group_fallback_column"]
    path = get_path("raw_dir") / "train_none.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run `python -m src.data.download` first")

    df = pd.read_csv(path, usecols=[*ds["keep_columns"], "attack"], dtype=str,
                     keep_default_na=False, na_values=[""])
    if not (df["attack"] == "none").all():
        raise ValueError(f"expected only attack == 'none', got {df['attack'].unique()}")

    unknown = set(df[model_col].dropna()) - set(cfg["label_map"])
    if unknown or df[model_col].isna().any():
        raise ValueError(f"model values missing from label_map: {sorted(unknown)}")

    sid = df[group_col]
    sid_missing = sid.isna() | (sid.str.strip() == "")
    return pd.DataFrame({
        "id": df[fallback_col],
        "text": df[text_col].fillna(""),
        "family": df[model_col].map(cfg["label_map"]),
        "domain": df["domain"],
        "group": sid.where(~sid_missing, df[fallback_col]),
    })


def clean(df: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, dict]:
    """Apply cleaning rules 1-4 in order; return the kept rows and removals per family."""
    classes = cfg["classes"]
    min_words = cfg["cleaning"]["min_words"]
    removed: dict[str, dict[str, int]] = {}

    def drop(frame: pd.DataFrame, mask: pd.Series, rule: str) -> pd.DataFrame:
        counts = frame.loc[mask, "family"].value_counts()
        removed[rule] = {c: int(counts.get(c, 0)) for c in classes} | {"total": int(mask.sum())}
        return frame[~mask]

    df = df.assign(norm=normalize_text(df["text"], cfg))                                  # 1
    df = drop(df, df["norm"] == "", "empty")                                              # 2
    df = df.assign(n_words=df["norm"].str.split().str.len())
    df = drop(df, df["n_words"] < min_words, f"under_{min_words}_words")                  # 3
    n_families = df.groupby("norm")["family"].transform("nunique")
    df = drop(df, n_families > 1, "duplicate_conflicting_families")                       # 4a
    df = df.sort_values("id", kind="stable")
    df = drop(df, df.duplicated("norm", keep="first"), "duplicate_same_family")           # 4b

    counts = df["family"].value_counts()
    removed["rows_after_cleaning"] = {c: int(counts.get(c, 0)) for c in classes} | {
        "total": len(df)}
    return df.reset_index(drop=True), removed


def sample(df: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, dict]:
    """Keep every Human row; sample up to max_per_ai_family rows of each AI family."""
    s = cfg["sampling"]
    human = cfg["label_map"]["human"]
    parts, stats = [], {}
    for family in cfg["classes"]:
        rows = df[df["family"] == family].sort_values("id", kind="stable")
        keep_all = family == human and s["keep_all_human"]
        if not keep_all and len(rows) > s["max_per_ai_family"]:
            rows = rows.sample(n=s["max_per_ai_family"], random_state=s["seed"])
        parts.append(rows)
        stats[family] = {"available": int((df["family"] == family).sum()), "kept": len(rows)}
    out = pd.concat(parts).sort_values("id", kind="stable").reset_index(drop=True)
    out["label"] = out["family"]  # step 5: family name is the label
    out["label_id"] = out["label"].map({c: i for i, c in enumerate(cfg["classes"])})
    return out, stats


def split(df: pd.DataFrame, cfg: dict) -> dict[str, pd.DataFrame]:
    """Two GroupShuffleSplits: trainval/test, then train/val inside trainval."""
    s = cfg["split"]
    outer = GroupShuffleSplit(n_splits=1, test_size=s["test_size"], random_state=s["seed"])
    tv_idx, test_idx = next(outer.split(df, groups=df["group"]))
    trainval, test = df.iloc[tv_idx], df.iloc[test_idx]
    inner = GroupShuffleSplit(n_splits=1, test_size=s["val_size"], random_state=s["seed"])
    train_idx, val_idx = next(inner.split(trainval, groups=trainval["group"]))
    parts = {"train": trainval.iloc[train_idx], "val": trainval.iloc[val_idx], "test": test}
    return {k: v.reset_index(drop=True) for k, v in parts.items()}


def check_splits(splits: dict[str, pd.DataFrame], cfg: dict) -> None:
    """Integrity checks (mirrored in tests/test_splits.py). Raises AssertionError."""
    norms = {k: set(normalize_text(v["text"], cfg)) for k, v in splits.items()}
    for a, b in itertools.combinations(splits, 2):
        shared = set(splits[a]["group"]) & set(splits[b]["group"])
        assert not shared, f"{len(shared)} source_id groups appear in both {a} and {b}"
        shared = norms[a] & norms[b]
        assert not shared, f"{len(shared)} normalized texts appear in both {a} and {b}"
    for name, frame in splits.items():
        missing = set(cfg["classes"]) - set(frame["label"])
        assert not missing, f"{name} is missing labels {sorted(missing)}"
    s = cfg["split"]
    assert len(splits["train"]) > s["min_train_rows"], f"train too small: {len(splits['train'])}"
    assert len(splits["test"]) > s["min_test_rows"], f"test too small: {len(splits['test'])}"


def compute_stats(cleaning, sampling, splits, cfg) -> dict:
    classes = cfg["classes"]
    everything = pd.concat(splits.values())
    words = {}
    for c in classes:
        n = everything.loc[everything["label"] == c, "n_words"].to_numpy()
        words[c] = {"mean": round(float(n.mean()), 1), "median": float(np.median(n)),
                    "p95": float(np.percentile(n, 95))}
    return {
        "seed": cfg["seed"],
        "label_order": {c: i for i, c in enumerate(classes)},
        "cleaning": cleaning,
        "sampling": sampling,
        "rows_per_split": {k: len(v) for k, v in splits.items()} | {"total": len(everything)},
        "groups_per_split": {k: int(v["group"].nunique()) for k, v in splits.items()},
        "rows_per_class_per_split": {
            k: {c: int((v["label"] == c).sum()) for c in classes} for k, v in splits.items()},
        "words_per_text": words,
        # sha256 of sorted ids: compare across machines to confirm identical splits
        "fingerprint": {k: hashlib.sha256("\n".join(sorted(v["id"])).encode()).hexdigest()[:16]
                        for k, v in splits.items()},
    }


def plot_figures(splits: dict[str, pd.DataFrame], cfg: dict, out_dir) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    classes = cfg["classes"]
    series = {"train": "#2a78d6", "val": "#eb6834", "test": "#1baf7a"}  # validated slots 1-3
    ink, muted, grid = "#0b0b0b", "#52514e", "#e4e3df"
    thousands = FuncFormatter(lambda v, _: f"{v / 1000:.0f}K" if v else "0")

    def style(ax):
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color(grid)
        ax.tick_params(colors=muted, length=0)
        ax.yaxis.grid(True, color=grid, linewidth=0.8)
        ax.set_axisbelow(True)

    # Grouped bars: rows per class per split
    fig, ax = plt.subplots(figsize=(10, 5), dpi=150)
    x, width = np.arange(len(classes)), 0.27
    for i, (name, color) in enumerate(series.items()):
        counts = [int((splits[name]["label"] == c).sum()) for c in classes]
        bars = ax.bar(x + (i - 1) * width, counts, width, color=color, label=name,
                      edgecolor="white", linewidth=2)
        ax.bar_label(bars, labels=[f"{c / 1000:.1f}K" for c in counts], padding=2,
                     fontsize=7, color=muted)
    ax.set_xticks(x, classes, color=ink)
    ax.yaxis.set_major_formatter(thousands)
    ax.set_ylabel("Rows", color=muted)
    ax.set_title("Rows per class in each split", loc="left", color=ink, fontsize=12, pad=22)
    ax.legend(frameon=False, ncols=3, loc="lower right", bbox_to_anchor=(1, 1.0),
              labelcolor=ink, fontsize=9)
    style(ax)
    fig.tight_layout()
    fig.savefig(out_dir / "class_counts.png")
    plt.close(fig)

    # Box plot: words per text per class (all splits together)
    everything = pd.concat(splits.values())
    data = [everything.loc[everything["label"] == c, "n_words"].to_numpy() for c in classes]
    fig, ax = plt.subplots(figsize=(10, 5), dpi=150)
    ax.boxplot(data, tick_labels=classes, showfliers=False, widths=0.5, patch_artist=True,
               boxprops=dict(facecolor="#2a78d6", edgecolor="#2a78d6", alpha=0.35),
               medianprops=dict(color=ink, linewidth=2),
               whiskerprops=dict(color=muted), capprops=dict(color=muted))
    ax.axhline(256, color=muted, linestyle="--", linewidth=1)
    ax.annotate("CNN/LSTM\nmax_len 256", xy=(1.01, 256), xycoords=("axes fraction", "data"),
                va="center", ha="left", fontsize=8, color=muted)
    ax.set_ylabel("Words per text", color=muted)
    ax.tick_params(axis="x", colors=ink)
    ax.set_title("Words per text by class (whiskers 1.5×IQR, outliers hidden)",
                 loc="left", color=ink, fontsize=12)
    style(ax)
    fig.tight_layout()
    fig.savefig(out_dir / "text_length.png")
    plt.close(fig)


def build(force: bool = False) -> dict:
    cfg = load_config()
    splits_dir = get_path("splits_dir", create=True)
    paths = {s: splits_dir / f"{s}.parquet" for s in SPLITS}
    existing = [p.name for p in paths.values() if p.exists()]
    if existing and not force:
        raise SystemExit(f"Splits are frozen and already exist in {splits_dir} ({existing}). "
                         "Use --force only if you really mean to rebuild them.")

    raw = load_raw(cfg)
    print(f"Loaded {len(raw):,} raw rows")
    cleaned, cleaning = clean(raw, cfg)
    print(f"After cleaning: {len(cleaned):,} rows")
    sampled, sampling = sample(cleaned, cfg)
    print(f"After sampling: {len(sampled):,} rows")
    splits = split(sampled, cfg)
    check_splits(splits, cfg)
    print("Integrity checks passed")

    for name, frame in splits.items():
        frame[OUT_COLUMNS].to_parquet(paths[name], index=False)
        print(f"  {name:5s} {len(frame):>8,} rows -> {paths[name]}")

    stats = compute_stats(cleaning, sampling, splits, cfg)
    results_dir = get_path("results_dir", create=True)
    (results_dir / "split_stats.json").write_text(json.dumps(stats, indent=2))
    figures = get_path("results_dir") / "figures"
    figures.mkdir(exist_ok=True)
    plot_figures(splits, cfg, figures)
    print(f"Wrote {results_dir / 'split_stats.json'} and figures in {figures}")
    return stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--force", action="store_true", help="overwrite existing frozen splits")
    build(force=parser.parse_args().force)
