"""Score the original and the Llama-rewritten passages with the final models; tables and figures.

    python -m src.experiments.paraphrase_eval [--config PATH] [--generator MODEL_ID]

Uses the runs that were scored on test (results/final_test_summary.json), reloaded the same way
as src.final_test. Before scoring, each reloaded model must reproduce its saved val predictions
on the full source texts (at least `check_agreement` of them), so the experiment measures exactly
the models in the results. Only passages whose rewrite is usable (status "ok") are scored, and
both versions of each passage are scored, so every comparison is paired.

Writes <results_dir>/<output_subdir>/<generator>/scores.parquet (one row per passage, model and
version), the tables paraphrase_{shift,by_domain,quality}_<generator> to <results_dir>/tables/,
and figures 70-71 (PNG + PDF, suffixed with the generator) with their captions to
<results_dir>/figures/, so the runs of several rewriters sit side by side.
"""

from __future__ import annotations

import argparse
import dataclasses
import logging
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from src.analysis.figures import (  # noqa: E402
    AXIS,
    BAR,
    INK,
    MODEL_STYLE,
    STYLE,
    FigureSettings,
    add_note,
    add_titles,
    fmt_pct,
    heatmap,
    save_figure,
    write_tables,
)
from src.config import get_device, get_path, load_config, use_config  # noqa: E402
from src.data.splits import fingerprint_ids, load_split  # noqa: E402
from src.experiments.paraphrase import (  # noqa: E402
    REWRITES_FILE,
    GeneratorSettings,
    experiment_dir,
)
from src.features.tfidf import get_tfidf_features, transform  # noqa: E402
from src.features.vocab import get_encoded_splits  # noqa: E402
from src.final_test import SUMMARY_FILE, FinalTestError, check_run_config  # noqa: E402
from src.models import REGISTRY, build_model, get_model_spec  # noqa: E402
from src.models.transformer import (  # noqa: E402
    RUN_CONFIG_FILE,
    TransformerParams,
    load_best_model,
    predict_proba_texts,
)
from src.train import DEFAULT_RUN_NAME, training_config  # noqa: E402
from src.trainer import BEST_FILE, ArrayDataset, predict_proba  # noqa: E402
from src.utils.io import atomic_open, atomic_write_bytes, safe_load_json  # noqa: E402
from src.utils.logging import setup_logging  # noqa: E402

logger = logging.getLogger(__name__)

SCORES_FILE = "scores.parquet"
CAPTIONS_FILE = "paraphrase_captions_{slug}.txt"
VERSIONS = ("original", "rewrite")
VERSION_LABELS = {"original": "Before", "rewrite": "After"}
OTHER = "Other AI"
PROB_PREFIX = "prob_"
USAGE_ERROR_EXIT_CODE = 2
# Class colors for these figures: the validated categorical slots 1-3 plus a neutral, by role.
# Models are only named on the axis here, never colored, so the slots cannot be misread.
SOURCE_COLOR, REWRITER_COLOR, HUMAN_COLOR, OTHER_COLOR = "#2a78d6", "#eb6834", "#1baf7a", AXIS
LABEL_MIN_SHARE = 0.07  # segments narrower than this carry no percentage label

Scorer = Callable[[Sequence[str]], np.ndarray]


class ParaphraseEvalError(RuntimeError):
    """The inputs of the evaluation are missing or do not match the saved results."""


# --- Loading the final models -------------------------------------------------------------
def final_runs(results_dir: Path) -> list[tuple[str, str]]:
    """(model, run) of every neural model scored on test, in the figure order."""
    summary = safe_load_json(results_dir / SUMMARY_FILE, default=None)
    if summary is None:
        raise ParaphraseEvalError(f"{results_dir / SUMMARY_FILE} not found; the experiment uses "
                                  "the models scored by src.final_test")
    runs = {row["model"]: row["run_name"] or DEFAULT_RUN_NAME for row in summary["models"]}
    return [(m, runs[m]) for m in MODEL_STYLE if m in runs and m in REGISTRY]


def trainer_scorer(model: str, run: str, train: pd.DataFrame,
                   cfg: Mapping[str, Any]) -> Scorer:
    """Scorer for a src.trainer run (MLP, CNN, LSTM), with its train-fitted features."""
    path = get_path("checkpoint_dir") / model / run / BEST_FILE
    if not path.exists():
        raise ParaphraseEvalError(f"{model}: {path} not found")
    best = torch.load(path, map_location="cpu", weights_only=True)
    run_cfg = best["config"]["run"]
    check_run_config(model, run_cfg, fingerprint_ids(train["id"]), path)
    cache_root = get_path("splits_dir") / cfg["features"]["cache_subdir"]
    num_classes = len(cfg["classes"])
    if get_model_spec(model).input_kind == "tfidf":
        feats = get_tfidf_features({"train": train}, cfg["features"]["tfidf"], cache_root)
        key, dims = feats.key, {"input_dim": feats.n_features, "num_classes": num_classes}
        dtype = cfg["features"]["tfidf"]["dtype"]

        def featurize(texts: Sequence[str]) -> Any:
            return transform(feats.vectorizer, texts, dtype)
    else:
        enc = get_encoded_splits({"train": train}, cfg["features"]["vocab"], cache_root)
        max_len = cfg["features"]["vocab"]["max_len"]
        key, dims = enc.key, {"vocab_size": len(enc.vocab), "pad_id": enc.vocab.pad_id,
                              "max_len": max_len, "num_classes": num_classes}

        def featurize(texts: Sequence[str]) -> Any:
            return enc.vocab.encode(list(texts), max_len)
    if key != run_cfg["input"]["cache_key"]:
        raise ParaphraseEvalError(f"{model}: the feature settings changed since run {run!r} "
                                  f"was trained ({run_cfg['input']['cache_key']} -> {key})")
    net = build_model(model, run_cfg["model_params"], **dims)
    net.load_state_dict(best["model"])
    device = get_device()
    net.to(device)
    settings = training_config(cfg, model, None)

    def score(texts: Sequence[str]) -> np.ndarray:
        data = ArrayDataset(featurize(texts), np.zeros(len(texts), dtype=np.int64))
        return predict_proba(net, data, device=device, batch_size=settings.eval_batch_size,
                             use_amp=settings.amp and device.type == "cuda")
    return score


def transformer_scorer(run: str, train: pd.DataFrame, cfg: Mapping[str, Any]) -> Scorer:
    """Scorer for a Transformer run's best_model/."""
    checkpoint_dir = get_path("checkpoint_dir") / "transformer" / run
    saved = safe_load_json(checkpoint_dir / RUN_CONFIG_FILE, default=None)
    if saved is None:
        raise ParaphraseEvalError(f"transformer: {checkpoint_dir / RUN_CONFIG_FILE} not found")
    run_cfg = saved["config"]["run"]
    check_run_config("transformer", run_cfg, fingerprint_ids(train["id"]), checkpoint_dir)
    hp = TransformerParams.from_config(run_cfg["model_params"])
    net, tokenizer = load_best_model(checkpoint_dir)
    settings = training_config(cfg, "transformer", None)
    device = get_device()

    def score(texts: Sequence[str]) -> np.ndarray:
        return predict_proba_texts(net, tokenizer, list(texts), max_len=hp.max_len,
                                   batch_size=settings.eval_batch_size, device=device,
                                   use_amp=settings.amp)
    return score


def load_scorer(model: str, run: str, train: pd.DataFrame, cfg: Mapping[str, Any]) -> Scorer:
    """Scorer for one final model."""
    if get_model_spec(model).input_kind == "hf":
        return transformer_scorer(run, train, cfg)
    return trainer_scorer(model, run, train, cfg)


def check_reproduces(model: str, run: str, score: Scorer, val: pd.DataFrame,
                     ids: Sequence[str], results_dir: Path, min_agreement: float) -> float:
    """The reloaded model must repeat its saved val predictions on these texts."""
    label = model if run == DEFAULT_RUN_NAME else f"{model}-{run}"
    path = results_dir / f"{label}_val_probs.npz"
    if not path.exists():
        raise ParaphraseEvalError(f"{model}: {path} not found")
    saved = np.load(path)["y_prob"]
    if len(saved) != len(val):
        raise ParaphraseEvalError(f"{model}: {path} has {len(saved)} rows but the val split has "
                                  f"{len(val)}")
    rows = val.index[val["id"].isin(set(ids))].to_numpy()
    now = score(val["text"].to_numpy()[rows].tolist())
    agreement = float((now.argmax(axis=1) == saved[rows].argmax(axis=1)).mean())
    if agreement < min_agreement:
        raise ParaphraseEvalError(f"{model}: the reloaded model repeats only {agreement:.1%} of "
                                  f"its saved val predictions (need {min_agreement:.0%}); it "
                                  "does not match the results")
    logger.info("%s: repeats %.1f%% of its saved val predictions", model, 100 * agreement)
    return agreement


def score_versions(model: str, score: Scorer, rewrites: pd.DataFrame,
                   class_names: Sequence[str]) -> pd.DataFrame:
    """Long table: one row per passage and version, with the prediction and probabilities."""
    frames = []
    for version, column in zip(VERSIONS, ("passage", "rewrite"), strict=True):
        prob = score(rewrites[column].tolist())
        frame = rewrites[["id", "group", "source", "domain"]].assign(
            model=model, version=version, pred=np.asarray(class_names)[prob.argmax(axis=1)])
        for i, name in enumerate(class_names):
            frame[f"{PROB_PREFIX}{name}"] = prob[:, i].astype(np.float32)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


# --- Tables -------------------------------------------------------------------------------
def outcome(pred: pd.Series, source: str, rewriter: str, human: str) -> pd.Series:
    """Collapse predictions to the source class, the rewriter, Human or 'Other AI'."""
    keep = {source, rewriter, human}
    return pred.where(pred.isin(keep), OTHER)


def outcome_order(source: str, rewriter: str, human: str) -> list[str]:
    """Stacking order: source, rewriter, Human, Other AI (each once)."""
    return list(dict.fromkeys([source, rewriter, human, OTHER]))


def role_of(name: str, source: str, rewriter: str, human: str) -> str:
    """The role of an outcome in a group: source, rewriter, human or other."""
    if name == source:
        return "source"
    if name == rewriter:
        return "rewriter"
    return "human" if name == human else "other"


def shift_table(scores: pd.DataFrame, rewriter: str, human: str) -> pd.DataFrame:
    """Per group, model and version: share of passages given to the source class, the
    rewriter, Human and other AI classes, and the mean probability of source and rewriter.

    When the rewriter (or Human) is the source class, its share is the source share.
    """
    rows = []
    for (group, model, version), df in scores.groupby(["group", "model", "version"],
                                                      sort=False):
        source = df["source"].iloc[0]
        named = outcome(df["pred"], source, rewriter, human)
        row = {"group": group, "source": source, "rewriter": rewriter, "model": model,
               "version": version, "n": len(df)}
        for name, column in ((source, "source"), (rewriter, "rewriter"), (human, "human"),
                             (OTHER, "other")):
            row[f"share_{column}"] = float((named == name).mean())
        row["mean_prob_source"] = float(df[f"{PROB_PREFIX}{source}"].mean())
        row["mean_prob_rewriter"] = float(df[f"{PROB_PREFIX}{rewriter}"].mean())
        rows.append(row)
    return pd.DataFrame(rows)


def domain_table(scores: pd.DataFrame, group: str, rewriter: str) -> pd.DataFrame:
    """Share of one group's rewrites given to the rewriter, per model (rows) and domain."""
    after = scores[(scores["group"] == group) & (scores["version"] == "rewrite")]
    table = (after.assign(hit=after["pred"] == rewriter)
             .pivot_table(index="model", columns="domain", values="hit", aggfunc="mean"))
    order = table.mean().sort_values(ascending=False).index
    return table[order].reindex([m for m in MODEL_STYLE if m in table.index])


def quality_table(rewrites: pd.DataFrame) -> pd.DataFrame:
    """Per group: how many rewrites are usable, and how different they are from the source."""
    rows = []
    for group, df in rewrites.groupby("group", sort=False):
        ok = df[df["status"] == "ok"]
        rows.append({"group": group, "rewritten": len(df), "usable": len(ok),
                     **{s: int((df["status"] == s).sum())
                        for s in ("refused", "too_short", "unchanged")},
                     "preamble_removed": float(df["removed_lead"].mean()),
                     "note_removed": float(df["removed_note"].mean()),
                     "median_length_ratio": float(ok["length_ratio"].median()),
                     "median_word_overlap": float(ok["word_overlap"].median()),
                     "seconds_per_passage": float(df["seconds"].mean())})
    return pd.DataFrame(rows)


# --- Figures ------------------------------------------------------------------------------
def outcome_color(name: str, source: str, rewriter: str, human: str) -> str:
    """Color of an outcome by role (the rewriter keeps its color when it is also the source)."""
    if name == rewriter:
        return REWRITER_COLOR
    if name == human:
        return HUMAN_COLOR
    return SOURCE_COLOR if name == source else OTHER_COLOR


def model_name(model: str) -> str:
    """Display name of a model."""
    return MODEL_STYLE[model][0]


def fig_shift(shift: pd.DataFrame, rewriter: str, human: str,
              groups: Mapping[str, Mapping[str, Any]],
              rewriter_model: str) -> tuple[Figure, str]:
    """Before/after stacked bars of who each model names, one panel per group."""
    names = [g for g in groups if g in set(shift["group"])]
    models = [m for m in MODEL_STYLE if m in set(shift["model"])]
    fig, axes = plt.subplots(len(names), 1, figsize=(8.6, len(names) * (0.62 * len(models)
                                                                      + 1.1)), squeeze=False)
    for ax, group in zip(axes[:, 0], names, strict=True):
        df = shift[shift["group"] == group]
        source = df["source"].iloc[0]
        order = outcome_order(source, rewriter, human)
        labels, y = [], 0.0
        for model in models:
            for version in VERSIONS:
                row = df[(df["model"] == model) & (df["version"] == version)].iloc[0]
                left = 0.0
                for o in order:
                    share = row[f"share_{role_of(o, source, rewriter, human)}"]
                    color = outcome_color(o, source, rewriter, human)
                    ax.barh(-y, share, left=left, height=BAR, color=color,
                            edgecolor="white", linewidth=1.5)
                    if share >= LABEL_MIN_SHARE:
                        ax.text(left + share / 2, -y, fmt_pct(share), ha="center",
                                va="center", fontsize=8,
                                color=INK if color == OTHER_COLOR else "white")
                    left += share
                labels.append((-y, f"{model_name(model)}  {VERSION_LABELS[version]}"))
                y += 1
            y += 0.4
        ax.set_yticks([p for p, _ in labels], [t for _, t in labels], fontsize=8.5)
        ax.set_xlim(0, 1)
        ax.set_xticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_title(f"{source} texts rewritten by {rewriter_model}  (n = "
                     f"{int(df['n'].iloc[0])} per "
                     "model)", fontsize=10.5)
        handles = [Patch(color=outcome_color(o, source, rewriter, human),
                         label=f"{o} (rewriter)" if o == rewriter else o) for o in order]
        ax.legend(handles=handles, loc="center left", bbox_to_anchor=(1.01, 0.5),
                  fontsize=8.5, handlelength=1.2)
    fig.tight_layout(h_pad=2.4)
    main = shift[(shift["group"] == names[0]) & (shift["model"] == models[0])]
    source = main["source"].iloc[0]
    before, after = (main.set_index("version").loc[v, "share_rewriter"] for v in VERSIONS)
    kept = main.set_index("version").loc["rewrite", "share_source"]
    title = (f"After a {rewriter_model} rewrite, {model_name(models[0])} names {rewriter} for "
             f"{fmt_pct(after)} of {source} texts (was {fmt_pct(before)}); {fmt_pct(kept)} "
             f"still read as {source}")
    add_titles(fig, title, "Share of passages each model assigns to each author, before and "
                           "after the rewrite (same passages)")
    add_note(fig, "Passages: val split, cut to at most 200 words, stratified by domain. Other AI "
                  "= Mistral, MPT, Cohere, or GPT when GPT is not the source.")
    return fig, title


def fig_by_domain(table: pd.DataFrame, group: str, source: str, rewriter: str,
                  rewriter_model: str) -> tuple[Figure, str]:
    """Heatmap: share of rewrites given to the rewriter, per model and domain."""
    fig, ax = plt.subplots(figsize=(8.6, 0.6 * len(table) + 1.4))
    heatmap(ax, table.to_numpy(), [model_name(m) for m in table.index], list(table.columns),
            fmt_pct, vmin=0, vmax=1)
    mean = table.mean()
    title = (f"The switch to {rewriter} is strongest for {mean.index[0]} "
             f"({fmt_pct(mean.iloc[0])}) and weakest for {mean.index[-1]} "
             f"({fmt_pct(mean.iloc[-1])})")
    add_titles(fig, title, f"Share of {source} texts named {rewriter} after the "
                           f"{rewriter_model} rewrite, per domain (group {group})")
    return fig, title


# --- CLI ----------------------------------------------------------------------------------
def load_rewrites(out_dir: Path,
                  settings: GeneratorSettings) -> tuple[pd.DataFrame, pd.DataFrame]:
    """All rewrites of the configured generator, and the usable ones (status ok)."""
    path = out_dir / settings.slug / REWRITES_FILE
    if not path.exists():
        raise ParaphraseEvalError(f"{path} not found; run `python -m src.experiments.paraphrase "
                                  "generate` (or collect) and copy its folder here")
    rewrites = pd.read_parquet(path)
    usable = rewrites[rewrites["status"] == "ok"].reset_index(drop=True)
    if usable.empty:
        raise ParaphraseEvalError(f"{path} has no usable rewrites")
    return rewrites, usable


def run(cfg: Mapping[str, Any], generator: str | None = None) -> list[Path]:
    """Score both versions with every final model, then write tables and figures.

    `generator` overrides paraphrase.generator.model_name (which rewrites to evaluate).
    """
    section = cfg["paraphrase"]
    settings = GeneratorSettings.from_config(section["generator"])
    if generator:
        settings = dataclasses.replace(settings, model_name=generator)
    slug = settings.slug
    results_dir = get_path("results_dir")
    out_dir = experiment_dir(section)
    all_rewrites, rewrites = load_rewrites(out_dir, settings)
    runs = final_runs(results_dir)
    train = load_split("train", columns=("id", "text"))
    val = load_split(section["source_split"], columns=("id", "text"))
    rewriter, human = section["rewriter_class"], cfg["training"]["human_label"]

    frames = []
    for model, run_name in runs:
        logger.info("Scoring with %s (%s)", model, run_name)
        score = load_scorer(model, run_name, train, cfg)
        check_reproduces(model, run_name, score, val, rewrites["id"], results_dir,
                         section["check_agreement"])
        frames.append(score_versions(model, score, rewrites, cfg["classes"]))
    scores = pd.concat(frames, ignore_index=True)
    with atomic_open(out_dir / settings.slug / SCORES_FILE) as f:
        scores.to_parquet(f, index=False)

    fig_settings = FigureSettings.from_config(cfg["figures"])
    shift = shift_table(scores, rewriter, human)
    main_group = next(iter(section["groups"]))
    by_domain = domain_table(scores, main_group, rewriter)
    tables = {f"paraphrase_shift_{slug}": shift,
              f"paraphrase_by_domain_{slug}": by_domain.reset_index(),
              f"paraphrase_quality_{slug}": quality_table(all_rewrites)}
    written = write_tables(tables, results_dir / fig_settings.tables_subdir)

    fig_dir = results_dir / fig_settings.figures_subdir
    captions = []
    with plt.rc_context(STYLE):
        for name, (fig, caption) in {
            f"70_paraphrase_shift_{slug}": fig_shift(shift, rewriter, human,
                                                     section["groups"], settings.label),
            f"71_paraphrase_by_domain_{slug}": fig_by_domain(
                by_domain, main_group, section["groups"][main_group]["source"], rewriter,
                settings.label),
        }.items():
            written += save_figure(fig, fig_dir, name, fig_settings)
            captions.append(f"{name}: {caption}")
    atomic_write_bytes(fig_dir / CAPTIONS_FILE.format(slug=slug),
                       ("\n".join(captions) + "\n").encode("utf-8"))
    print(shift.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    return written


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Score the paraphrase experiment.")
    parser.add_argument("--config", default=None, help="config YAML (default: configs/config.yaml)")
    parser.add_argument("--generator", default=None, metavar="MODEL_ID",
                        help="evaluate this rewriter's outputs instead of "
                             "paraphrase.generator.model_name")
    args = parser.parse_args(argv)
    use_config(args.config)
    cfg = load_config()
    setup_logging(experiment_dir(cfg["paraphrase"]) / "paraphrase_eval.log")
    try:
        written = run(cfg, args.generator)
    except (ParaphraseEvalError, FinalTestError, FileNotFoundError) as err:
        logger.error("%s", err)
        return USAGE_ERROR_EXIT_CODE
    print(f"\nWrote {len(written)} files")
    return 0


if __name__ == "__main__":
    sys.exit(main())
