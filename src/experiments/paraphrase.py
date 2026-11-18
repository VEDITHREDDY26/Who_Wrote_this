"""Rewrite val passages with Llama for the paraphrase experiment (the professor's question).

    python -m src.experiments.paraphrase sample   [--config PATH]
    python -m src.experiments.paraphrase generate [--config PATH] [--limit N]
    python -m src.experiments.paraphrase collect  [--config PATH]

sample (any machine): picks the source texts from the val split, stratified by domain, cuts each
one to a passage of at most `max_words` words (at a sentence end) and writes
<results_dir>/<output_subdir>/sources.parquet. Only this file is needed on the GPU machine.

generate (GPU machine, e.g. Newton): rewrites every passage with the configured chat model
(greedy decoding) and saves each batch atomically under <output_subdir>/<generator>/parts/.
Rerunning the command continues where it stopped, so a killed job loses at most one batch; when
every passage is done it runs collect.

collect: cleans the raw outputs (chat preambles such as "Sure! Here is the paraphrased text:",
closing notes, refusals), adds quality measures and writes <generator>/rewrites.parquet.
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import torch

from src.config import get_device, get_path, load_config, use_config
from src.data.splits import fingerprint_ids, load_split, stratified_subset
from src.utils.io import atomic_open, atomic_write_json, safe_load_json
from src.utils.logging import setup_logging

logger = logging.getLogger(__name__)

SOURCES_FILE = "sources.parquet"
PARTS_SUBDIR = "parts"
GENERATOR_FILE = "generator.json"
REWRITES_FILE = "rewrites.parquet"
LOG_FILE = "paraphrase.log"
SOURCE_COLUMNS = ("id", "text", "label", "domain")
USAGE_ERROR_EXIT_CODE = 2

Rewriter = Callable[[Sequence[str]], list[str]]


class ParaphraseError(RuntimeError):
    """The experiment's inputs or saved outputs are inconsistent."""

# Chat models wrap their answer: "Sure! Here is a paraphrased version of the text:" before it,
# "Note: I kept ..." after it. Both would give the rewriter away, so they are removed.
LEAD_KEYWORDS = re.compile(r"paraphras|rewr|reword|version|own words|text", re.I)
LEAD_WORDS = 30
BARE_LEAD = re.compile(r"^(?:sure|certainly|of course|okay)[!.,]?$", re.I)
INLINE_LEAD = re.compile(r"^(?:(?:sure|certainly|of course|okay)[!.,]?\s+)?"
                         r"here(?:'s| is| are)\b[^:\n]{0,100}:\s*", re.I)
TRAILING_NOTE = re.compile(r"^\(?(?:note\b|please note|i hope|let me know|i have (?:paraphras|"
                           r"rewrit|kept|tried|maintain)|in this (?:paraphras|rewrit)|this "
                           r"(?:paraphras|rewrit))", re.I)
REFUSAL = re.compile(r"^(?:i cannot|i can't|i can not|i apologi[sz]e|i'm just an ai|"
                     r"as a responsible|i'm not able to|i am not able to|i must inform)", re.I)
PARAGRAPH_BREAK = re.compile(r"\n\s*\n")
SENTENCE_END = re.compile(r"[.!?][\"')\]]*(?=\s|$)")
WORD = re.compile(r"\w+")


@dataclass(frozen=True)
class GeneratorSettings:
    """The `paraphrase.generator` section of the config."""

    model_name: str
    dtype: str
    batch_size: int
    max_new_tokens: int

    @classmethod
    def from_config(cls, section: Mapping[str, Any]) -> GeneratorSettings:
        """Build from the config, failing loudly on missing or unknown keys."""
        expected = set(cls.__dataclass_fields__)
        missing, unknown = expected - set(section), set(section) - expected
        if missing or unknown:
            raise KeyError(f"paraphrase.generator config: missing {sorted(missing)}, unknown "
                           f"{sorted(unknown)}")
        if not isinstance(getattr(torch, section["dtype"], None), torch.dtype):
            raise ValueError(f"paraphrase.generator.dtype {section['dtype']!r} is not a torch "
                             "dtype (e.g. bfloat16, float16)")
        return cls(str(section["model_name"]), str(section["dtype"]),
                   int(section["batch_size"]), int(section["max_new_tokens"]))

    @property
    def slug(self) -> str:
        """Folder name for this generator's outputs, e.g. 'llama-2-70b-chat-hf'."""
        return self.model_name.rstrip("/").split("/")[-1].lower()


@dataclass(frozen=True)
class Cleaned:
    """A raw model output after removing the chat wrapping around the rewrite."""

    text: str
    removed_lead: bool
    removed_note: bool
    refused: bool


# --- Passages and sources -----------------------------------------------------------------
def count_words(text: str) -> int:
    """Whitespace-separated words, as in the cleaning rules."""
    return len(text.split())


def cut_passage(text: str, max_words: int) -> str:
    """The first `max_words` words of `text` (original spacing kept), ending at a sentence end
    when one leaves at least half of them."""
    if max_words < 1:
        raise ValueError(f"max_words must be >= 1, got {max_words}")
    words = list(re.finditer(r"\S+", text))
    if len(words) <= max_words:
        return text.strip()
    head = text[:words[max_words - 1].end()]
    ends = list(SENTENCE_END.finditer(head))
    if ends and count_words(head[:ends[-1].end()]) >= max_words // 2:
        head = head[:ends[-1].end()]
    return head.strip()


def sample_sources(split: pd.DataFrame, groups: Mapping[str, Mapping[str, Any]],
                   class_names: Sequence[str], *, seed: int, max_words: int,
                   min_words: int) -> pd.DataFrame:
    """Source passages for every group: `n` texts of its source class, stratified by domain.

    Returns one row per passage with id, group, source, domain, passage and n_words.
    """
    frames = []
    for group, spec in groups.items():
        if spec["source"] not in class_names:
            raise ParaphraseError(f"paraphrase group {group!r}: source {spec['source']!r} is "
                                  f"not one of the classes {list(class_names)}")
        pool = split[split["label"] == spec["source"]]
        picked = stratified_subset(pool, int(spec["n"]), label_column="domain", seed=seed)
        passages = picked["text"].map(lambda t: cut_passage(t, max_words))
        frames.append(pd.DataFrame({"id": picked["id"], "group": group,
                                    "source": spec["source"], "domain": picked["domain"],
                                    "passage": passages, "n_words": passages.map(count_words)}))
    sources = pd.concat(frames, ignore_index=True)
    if sources["id"].duplicated().any():
        raise ParaphraseError("a text was picked by two paraphrase groups; give each group its "
                              "own source class")
    return sources[sources["n_words"] >= min_words].reset_index(drop=True)


# --- Cleaning and quality -----------------------------------------------------------------
def clean_rewrite(raw: str) -> Cleaned:
    """Remove a chat preamble and closing notes around a rewrite; flag refusals."""
    paragraphs = [p.strip() for p in PARAGRAPH_BREAK.split(raw.strip()) if p.strip()]
    removed_lead = removed_note = False
    if paragraphs:
        first = paragraphs[0]
        if BARE_LEAD.match(first) or (count_words(first) <= LEAD_WORDS and first.endswith(":")
                                      and LEAD_KEYWORDS.search(first)):
            paragraphs, removed_lead = paragraphs[1:], True
        elif INLINE_LEAD.match(first):
            paragraphs[0], removed_lead = INLINE_LEAD.sub("", first, count=1).strip(), True
    while len(paragraphs) > 1 and TRAILING_NOTE.match(paragraphs[-1]):
        paragraphs, removed_note = paragraphs[:-1], True
    text = "\n\n".join(p for p in paragraphs if p)
    if len(text) > 1 and text[0] == text[-1] == '"' and text.count('"') == 2:
        text = text[1:-1].strip()
    return Cleaned(text, removed_lead, removed_note, bool(REFUSAL.match(text)))


def word_overlap(original: str, rewrite: str) -> float:
    """Share of the rewrite's words that also occur in the original (1.0 = no new words)."""
    words = WORD.findall(rewrite.lower())
    if not words:
        return 0.0
    vocabulary = set(WORD.findall(original.lower()))
    return sum(w in vocabulary for w in words) / len(words)


# --- Resumable generation -----------------------------------------------------------------
def part_path(parts_dir: Path, ids: Sequence[str]) -> Path:
    """File for one batch of rewrites, named by the batch's ids."""
    return parts_dir / f"part-{fingerprint_ids(ids)}.json"


def load_parts(parts_dir: Path) -> pd.DataFrame:
    """Every saved rewrite record (id, raw, seconds)."""
    records = [r for path in sorted(parts_dir.glob("part-*.json"))
               for r in safe_load_json(path)["records"]]
    return pd.DataFrame(records, columns=["id", "raw", "seconds"])


def check_generator(run_dir: Path, identity: Mapping[str, Any]) -> None:
    """Record which model and prompts made the rewrites in `run_dir`; refuse to mix two."""
    path = run_dir / GENERATOR_FILE
    saved = safe_load_json(path, default=None)
    if saved is None:
        atomic_write_json(path, dict(identity))
    elif saved != dict(identity):
        changed = sorted(k for k in set(saved) | set(identity) if saved.get(k) != identity.get(k))
        raise ParaphraseError(f"{run_dir} holds rewrites made with other settings (changed: "
                              f"{changed}); restore them or move that folder away to start "
                              "over")


def rewrite_all(sources: pd.DataFrame, rewrite: Rewriter, parts_dir: Path, *, batch_size: int,
                limit: int | None = None) -> int:
    """Rewrite every passage not yet saved in `parts_dir`, one atomic file per batch.

    Longest passages go first, so a batch that does not fit in memory fails early. Returns the
    number of passages rewritten by this call.
    """
    done = set(load_parts(parts_dir)["id"])
    todo = sources[~sources["id"].isin(done)].sort_values(["n_words", "id"],
                                                          ascending=[False, True])
    if limit is not None:
        todo = todo.head(limit)
    logger.info("%d of %d passages already rewritten; %d to go", len(done & set(sources["id"])),
                len(sources), len(todo))
    start, written = time.monotonic(), 0
    for i in range(0, len(todo), batch_size):
        batch = todo.iloc[i:i + batch_size]
        t0 = time.monotonic()
        outputs = rewrite(batch["passage"].tolist())
        if len(outputs) != len(batch):
            raise RuntimeError(f"the rewriter returned {len(outputs)} texts for a batch of "
                               f"{len(batch)}")
        seconds = (time.monotonic() - t0) / len(batch)
        atomic_write_json(part_path(parts_dir, batch["id"].tolist()), {"records": [
            {"id": i_, "raw": out, "seconds": seconds}
            for i_, out in zip(batch["id"], outputs, strict=True)]})
        written += len(batch)
        rate = (time.monotonic() - start) / written
        logger.info("rewrote %d/%d (%.1f s per passage, about %.0f min left)", written,
                    len(todo), rate, rate * (len(todo) - written) / 60)
    return written


def collect(sources: pd.DataFrame, parts: pd.DataFrame, *, min_words: int) -> pd.DataFrame:
    """Join the rewrites to their sources, clean them and add quality columns and a status."""
    if parts["id"].duplicated().any():
        raise ParaphraseError("a passage was rewritten twice in the parts folder; delete "
                              "the duplicate part files")
    out = sources.merge(parts, on="id", how="inner", validate="one_to_one")
    cleaned = out["raw"].map(clean_rewrite)
    out["rewrite"] = [c.text for c in cleaned]
    out["removed_lead"] = [c.removed_lead for c in cleaned]
    out["removed_note"] = [c.removed_note for c in cleaned]
    out["n_words_rewrite"] = out["rewrite"].map(count_words)
    out["length_ratio"] = out["n_words_rewrite"] / out["n_words"]
    out["word_overlap"] = [word_overlap(p, r) for p, r in zip(out["passage"], out["rewrite"],
                                                               strict=True)]
    status = pd.Series("ok", index=out.index)
    status[out["rewrite"].str.strip() == out["passage"].str.strip()] = "unchanged"
    status[out["n_words_rewrite"] < min_words] = "too_short"
    # A refusal-like start only counts when the passage itself does not start that way.
    passage_like = out["passage"].map(lambda p: bool(REFUSAL.match(p.strip())))
    refused = pd.Series([c.refused for c in cleaned], index=out.index)
    status[refused & ~passage_like] = "refused"
    out["status"] = status
    return out


# --- The Hugging Face chat model ----------------------------------------------------------
class ChatRewriter:
    """Greedy rewrites with a Hugging Face chat model spread over all visible GPUs."""

    def __init__(self, settings: GeneratorSettings, system_prompt: str,
                 user_prompt: str) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer  # only on the GPU machine

        self.settings, self.system_prompt, self.user_prompt = settings, system_prompt, user_prompt
        self.tokenizer = AutoTokenizer.from_pretrained(settings.model_name)
        if self.tokenizer.chat_template is None:
            raise ParaphraseError(f"{settings.model_name} has no chat template; use a chat model")
        self.tokenizer.padding_side = "left"  # generation continues at the right end
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        on_cuda = torch.cuda.is_available()
        self.model = AutoModelForCausalLM.from_pretrained(
            settings.model_name, dtype=getattr(torch, settings.dtype),
            device_map="auto" if on_cuda else None)
        if not on_cuda:
            self.model.to(get_device())
        self.model.eval()

    def prompt(self, passage: str) -> str:
        """The chat-formatted prompt asking for a rewrite of `passage`."""
        messages = [{"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": self.user_prompt.format(text=passage)}]
        return self.tokenizer.apply_chat_template(messages, tokenize=False,
                                                  add_generation_prompt=True)

    def __call__(self, passages: Sequence[str]) -> list[str]:
        """Rewrite a batch; on out-of-memory, retry it in two halves."""
        try:
            return self._generate(passages)
        except torch.cuda.OutOfMemoryError:
            if len(passages) == 1:
                raise
            torch.cuda.empty_cache()
            half = len(passages) // 2
            logger.warning("Out of memory on a batch of %d; retrying in halves", len(passages))
            return self(passages[:half]) + self(passages[half:])

    def _generate(self, passages: Sequence[str]) -> list[str]:
        enc = self.tokenizer([self.prompt(p) for p in passages], return_tensors="pt",
                             padding=True, add_special_tokens=False).to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(**enc, do_sample=False,
                                      max_new_tokens=self.settings.max_new_tokens,
                                      pad_token_id=self.tokenizer.pad_token_id)
        return self.tokenizer.batch_decode(out[:, enc["input_ids"].shape[1]:],
                                           skip_special_tokens=True)


# --- CLI ----------------------------------------------------------------------------------
def experiment_dir(section: Mapping[str, Any]) -> Path:
    """<results_dir>/<output_subdir>."""
    return get_path("results_dir") / section["output_subdir"]


def load_sources(out_dir: Path) -> pd.DataFrame:
    """The sources written by `sample`."""
    path = out_dir / SOURCES_FILE
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run `python -m src.experiments.paraphrase "
                                "sample` first (and copy it to this machine)")
    return pd.read_parquet(path)


def run_sample(cfg: Mapping[str, Any], out_dir: Path) -> None:
    """Pick the source passages and write sources.parquet."""
    section = cfg["paraphrase"]
    split = load_split(section["source_split"], columns=SOURCE_COLUMNS)
    sources = sample_sources(split, section["groups"], cfg["classes"], seed=cfg["seed"],
                             max_words=section["max_words"], min_words=section["min_words"])
    with atomic_open(out_dir / SOURCES_FILE) as f:
        sources.to_parquet(f, index=False)
    counts = sources.groupby("group").size().to_dict()
    print(f"Wrote {len(sources)} passages to {out_dir / SOURCES_FILE}: {counts}")


def run_collect(cfg: Mapping[str, Any], out_dir: Path, settings: GeneratorSettings) -> None:
    """Clean the saved rewrites and write rewrites.parquet."""
    run_dir = out_dir / settings.slug
    sources = load_sources(out_dir)
    rewrites = collect(sources, load_parts(run_dir / PARTS_SUBDIR),
                       min_words=cfg["paraphrase"]["min_words"])
    rewrites.insert(0, "generator", settings.model_name)
    with atomic_open(run_dir / REWRITES_FILE) as f:
        rewrites.to_parquet(f, index=False)
    missing = len(sources) - len(rewrites)
    print(f"Wrote {len(rewrites)} rewrites to {run_dir / REWRITES_FILE}"
          + (f" ({missing} passages not rewritten yet)" if missing else ""))
    print(rewrites.groupby(["group", "status"]).size().to_string())


def run_generate(cfg: Mapping[str, Any], out_dir: Path, settings: GeneratorSettings,
                 limit: int | None) -> None:
    """Rewrite every passage (resumable), then collect."""
    section = cfg["paraphrase"]
    run_dir = out_dir / settings.slug
    sources = load_sources(out_dir)
    check_generator(run_dir, {"model_name": settings.model_name, "dtype": settings.dtype,
                              "max_new_tokens": settings.max_new_tokens,
                              "system_prompt": section["system_prompt"],
                              "user_prompt": section["user_prompt"]})
    logger.info("Loading %s (%s)", settings.model_name, settings.dtype)
    rewriter = ChatRewriter(settings, section["system_prompt"], section["user_prompt"])
    rewrite_all(sources, rewriter, run_dir / PARTS_SUBDIR, batch_size=settings.batch_size,
                limit=limit)
    run_collect(cfg, out_dir, settings)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Rewrite val passages with a chat model.")
    parser.add_argument("step", choices=("sample", "generate", "collect"))
    parser.add_argument("--config", default=None, help="config YAML (default: configs/config.yaml)")
    parser.add_argument("--limit", type=int, default=None,
                        help="generate: rewrite at most N more passages (quick check)")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point."""
    args = parse_args(argv)
    use_config(args.config)
    cfg = load_config()
    out_dir = experiment_dir(cfg["paraphrase"])
    setup_logging(out_dir / LOG_FILE)
    settings = GeneratorSettings.from_config(cfg["paraphrase"]["generator"])
    try:
        if args.step == "sample":
            run_sample(cfg, out_dir)
        elif args.step == "generate":
            run_generate(cfg, out_dir, settings, args.limit)
        else:
            run_collect(cfg, out_dir, settings)
    except (FileNotFoundError, ParaphraseError) as err:
        logger.error("%s", err)
        return USAGE_ERROR_EXIT_CODE
    return 0


if __name__ == "__main__":
    sys.exit(main())
