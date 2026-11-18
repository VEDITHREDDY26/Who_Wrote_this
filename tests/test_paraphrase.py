"""Tests for the paraphrase experiment (src.experiments) with a fake rewriter instead of Llama."""

import dataclasses
import random
from collections.abc import Iterator, Sequence
from pathlib import Path

import pandas as pd
import pytest
import yaml

import src.experiments.paraphrase_eval as paraphrase_eval
from src.config import load_config, use_config
from src.experiments.paraphrase import (
    PARTS_SUBDIR,
    GeneratorSettings,
    ParaphraseError,
    check_generator,
    clean_rewrite,
    collect,
    count_words,
    cut_passage,
    experiment_dir,
    load_parts,
    rewrite_all,
    run_collect,
    run_sample,
    sample_sources,
    word_overlap,
)
from tests.test_train_cli import CLASSES, make_workspace, run_cli

LEAD = "Sure! Here is a paraphrased version of the text:\n\n"


def reverse_words(passages: Sequence[str]) -> list[str]:
    """Stands in for Llama: the same words backwards, wrapped like a chat answer."""
    return [LEAD + " ".join(reversed(p.split())) for p in passages]


def sources_frame(n: int) -> pd.DataFrame:
    passages = [f"passage {i} has some words in it." * 2 for i in range(n)]
    return pd.DataFrame({"id": [f"s{i}" for i in range(n)], "group": "g", "source": "GPT",
                         "domain": "news", "passage": passages,
                         "n_words": [count_words(p) for p in passages]})


def test_cut_passage_keeps_spacing_and_ends_at_a_sentence() -> None:
    assert cut_passage("  short text.  ", 10) == "short text."
    text = "One two three.\nFour five six. Seven eight nine ten"
    assert cut_passage(text, 8) == "One two three.\nFour five six."
    assert cut_passage("a b c d e f", 3) == "a b c"  # no sentence end: plain cut
    with pytest.raises(ValueError, match="max_words"):
        cut_passage(text, 0)


@pytest.mark.parametrize(("raw", "text", "lead", "note", "refused"), [
    (LEAD + "The cat sat.\n\nNote: I kept the meaning.", "The cat sat.", True, True, False),
    ("Here's the rewritten text: The cat sat on a mat.", "The cat sat on a mat.", True, False,
     False),
    ("Sure!\n\nThe cat sat.", "The cat sat.", True, False, False),
    ('"The cat sat."', "The cat sat.", False, False, False),
    ("The cat sat.\n\nThe dog ran.", "The cat sat.\n\nThe dog ran.", False, False, False),
    ("I apologize, but I cannot rewrite this text.", "I apologize, but I cannot rewrite this "
     "text.", False, False, True),
])
def test_clean_rewrite(raw: str, text: str, lead: bool, note: bool, refused: bool) -> None:
    cleaned = clean_rewrite(raw)
    assert (cleaned.text, cleaned.removed_lead, cleaned.removed_note, cleaned.refused) == (
        text, lead, note, refused)


def test_word_overlap() -> None:
    assert word_overlap("the cat sat", "The cat sat") == 1.0
    assert word_overlap("the cat sat", "a dog ran") == 0.0
    assert word_overlap("x", "") == 0.0


def test_sample_sources_is_stratified_and_deterministic() -> None:
    rng = random.Random(0)
    split = pd.DataFrame([{"id": f"{label}-{domain}-{i}", "label": label, "domain": domain,
                           "text": " ".join(f"w{rng.randint(0, 9)}" for _ in range(30)) + "."}
                          for label in ("GPT", "Human") for domain in ("news", "wiki")
                          for i in range(20)])
    groups = {"a": {"source": "GPT", "n": 10}, "b": {"source": "Human", "n": 6}}
    first = sample_sources(split, groups, CLASSES, seed=1, max_words=12, min_words=5)
    again = sample_sources(split, groups, CLASSES, seed=1, max_words=12, min_words=5)
    pd.testing.assert_frame_equal(first, again)
    assert first.groupby("group").size().to_dict() == {"a": 10, "b": 6}
    assert first[first["group"] == "a"]["domain"].value_counts().to_dict() == {"news": 5,
                                                                              "wiki": 5}
    assert first["n_words"].max() <= 12
    with pytest.raises(ParaphraseError, match="not one of the classes"):
        sample_sources(split, {"x": {"source": "Bard", "n": 4}}, CLASSES, seed=1,
                       max_words=12, min_words=5)


def test_rewrite_all_resumes_without_repeating_work(tmp_path: Path) -> None:
    sources = sources_frame(10)
    seen: list[str] = []

    def flaky(passages: Sequence[str]) -> list[str]:
        if len(seen) >= 6:
            raise RuntimeError("job killed")
        seen.extend(passages)
        return reverse_words(passages)

    with pytest.raises(RuntimeError, match="killed"):
        rewrite_all(sources, flaky, tmp_path, batch_size=3)
    assert len(load_parts(tmp_path)) == 6  # two whole batches were saved

    def counting(passages: Sequence[str]) -> list[str]:
        seen.extend(passages)
        return reverse_words(passages)

    assert rewrite_all(sources, counting, tmp_path, batch_size=4) == 4
    assert sorted(seen) == sorted(sources["passage"])  # every passage exactly once
    assert rewrite_all(sources, counting, tmp_path, batch_size=4) == 0


def test_generator_settings_and_mixing_are_checked(tmp_path: Path) -> None:
    settings = GeneratorSettings.from_config({"model_name": "meta-llama/Llama-2-7b-chat-hf",
                                              "dtype": "bfloat16", "batch_size": 2,
                                              "max_new_tokens": 8})
    assert (settings.slug, settings.label) == ("llama-2-7b-chat-hf", "Llama-2-7b-chat")
    llama3 = dataclasses.replace(settings, model_name="meta-llama/Meta-Llama-3-70B-Instruct")
    assert (llama3.slug, llama3.label) == ("meta-llama-3-70b-instruct", "Llama-3-70B-Instruct")
    with pytest.raises(ValueError, match="dtype"):
        GeneratorSettings.from_config({"model_name": "m", "dtype": "fp99", "batch_size": 1,
                                       "max_new_tokens": 1})
    check_generator(tmp_path, {"model_name": "a"})
    check_generator(tmp_path, {"model_name": "a"})  # same settings: fine
    with pytest.raises(ParaphraseError, match="model_name"):
        check_generator(tmp_path, {"model_name": "b"})


def test_collect_sets_a_status_per_rewrite() -> None:
    sources = pd.DataFrame({"id": ["a", "b", "c", "d", "e"], "group": "g", "source": "GPT",
                            "domain": "news", "n_words": 12, "passage": [
                                "one two three four five six seven eight nine ten.",
                                "same text here with enough words to pass the floor.",
                                "short one gets cut down to almost nothing at all.",
                                "a normal passage that will be refused by the model.",
                                "I can't hold myself together, the poem starts like this."]})
    parts = pd.DataFrame({"id": list("abcde"), "seconds": 1.0, "raw": [
        LEAD + "ten nine eight seven six five four three two one.",
        sources["passage"][1], "too short.", "I cannot help with that request, sorry.",
        "I can't keep myself together, this poem begins so."]})
    out = collect(sources, parts, min_words=5).set_index("id")
    assert out["status"].to_dict() == {"a": "ok", "b": "unchanged", "c": "too_short",
                                       "d": "refused", "e": "ok"}
    assert bool(out.loc["a", "removed_lead"]) and out.loc["a", "word_overlap"] == 1.0
    with pytest.raises(ParaphraseError, match="twice"):
        collect(sources, pd.concat([parts, parts]), min_words=5)


@pytest.fixture
def trained_workspace(tmp_path: Path) -> Iterator[Path]:
    """Fake splits, a trained dummy model and a paraphrase config pointing into tmp_path."""
    workspace = make_workspace(tmp_path)
    config = workspace / "config.yaml"
    cfg = yaml.safe_load(config.read_text(encoding="utf-8"))
    cfg["paraphrase"] |= {"groups": {"gpt_by_llama": {"source": "GPT", "n": 4},
                                     "human_by_llama": {"source": "Human", "n": 4}},
                          "min_words": 5, "check_agreement": 1.0}
    cfg["paraphrase"]["generator"]["model_name"] = "test/fake-rewriter"
    cfg["figures"] |= {"formats": ["png"], "dpi": 40}
    config.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    proc = run_cli(config, "--model", "dummy")
    assert proc.returncode == 0, proc.output
    (workspace / "results" / "final_test_summary.json").write_text(
        '{"models": [{"model": "dummy", "run_name": "default"}]}')
    use_config(config)
    yield workspace
    use_config(None)


def test_end_to_end_with_a_fake_rewriter(trained_workspace: Path,
                                         monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(paraphrase_eval.MODEL_STYLE, "dummy", ("Dummy", "#2a78d6", "o"))
    cfg = load_config()
    settings = GeneratorSettings.from_config(cfg["paraphrase"]["generator"])
    out_dir = experiment_dir(cfg["paraphrase"])
    run_sample(cfg, out_dir)
    sources = pd.read_parquet(out_dir / "sources.parquet")
    rewrite_all(sources, reverse_words, out_dir / settings.slug / PARTS_SUBDIR, batch_size=3)
    run_collect(cfg, out_dir, settings)

    written = paraphrase_eval.run(cfg)
    names = {p.name for p in written}
    assert {"70_paraphrase_shift_fake-rewriter.png", "71_paraphrase_by_domain_fake-rewriter.png",
            "paraphrase_shift_fake-rewriter.csv", "paraphrase_quality_fake-rewriter.csv"} <= names
    scores = pd.read_parquet(out_dir / settings.slug / "scores.parquet")
    assert len(scores) == 2 * len(sources)  # both versions of every passage
    shift = pd.read_csv(trained_workspace / "results" / "tables" /
                        "paraphrase_shift_fake-rewriter.csv")
    shares = shift[["share_source", "share_rewriter", "share_human", "share_other"]]
    gpt = shift["group"] == "gpt_by_llama"
    assert (shares[gpt].sum(axis=1).round(6) == 1).all()  # GPT, Llama, Human, other: no overlap


def test_missing_rewrites_are_a_clear_error(trained_workspace: Path) -> None:
    with pytest.raises(paraphrase_eval.ParaphraseEvalError, match="generate"):
        paraphrase_eval.run(load_config())
