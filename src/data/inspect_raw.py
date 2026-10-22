"""Inspect raw RAID train_none.csv before building splits.

Reads only id, generation, model, domain, source_id, attack, in chunks, and reports:
row counts per model (exact spellings) and domain, attack values, human source_id
coverage and overlap with AI rows, unique source_ids, empty/duplicate generations, and
word-count stats per model. Prints a summary and writes <results_dir>/raw_inspection.json.

Usage: python -m src.data.inspect_raw [--chunksize 100000]
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.config import get_path

COLUMNS = ["id", "generation", "model", "domain", "source_id", "attack"]
HUMAN = "human"
INVISIBLE = r"[\s​‌‍⁠﻿]+"
SHORT_WORDS = 10


def _missing(s: pd.Series) -> pd.Series:
    return s.isna() | (s.astype(str).str.strip() == "")


def inspect(chunksize: int = 100_000) -> dict:
    path = get_path("raw_dir") / "train_none.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run `python -m src.data.download` first")

    model_counts, domain_counts, attack_counts = Counter(), Counter(), Counter()
    human_missing_sid = 0
    human_sids, ai_sids = set(), set()
    all_sids = set()
    missing_sid_total = 0
    empty_gen = short_gen = 0
    gen_hashes = Counter()
    word_counts = defaultdict(list)
    total = 0

    reader = pd.read_csv(path, usecols=COLUMNS, chunksize=chunksize, dtype=str,
                         keep_default_na=False, na_values=[""])
    for chunk in tqdm(reader, desc="chunks", unit="chunk"):
        total += len(chunk)
        model_counts.update(chunk["model"].fillna("<NA>"))
        domain_counts.update(chunk["domain"].fillna("<NA>"))
        attack_counts.update(chunk["attack"].fillna("<NA>"))

        sid_missing = _missing(chunk["source_id"])
        missing_sid_total += int(sid_missing.sum())
        sids = chunk.loc[~sid_missing, "source_id"]
        all_sids.update(sids)
        is_human = chunk["model"] == HUMAN
        human_missing_sid += int((is_human & sid_missing).sum())
        human_sids.update(chunk.loc[is_human & ~sid_missing, "source_id"])
        ai_sids.update(chunk.loc[~is_human & ~sid_missing, "source_id"])

        gen = chunk["generation"].fillna("")
        # strip() misses zero-width chars; some generations are only "​"
        empty = gen.str.replace(INVISIBLE, "", regex=True) == ""
        empty_gen += int(empty.sum())
        # 64-bit hashes instead of full strings keep memory flat
        gen_hashes.update(pd.util.hash_pandas_object(gen[~empty], index=False).tolist())

        wc = gen.str.split().str.len().fillna(0).astype(np.int32)
        wc[empty] = 0
        short_gen += int((wc < SHORT_WORDS).sum())
        for model, counts in wc.groupby(chunk["model"].fillna("<NA>")):
            word_counts[model].append(counts.to_numpy())

    dup_rows = sum(c - 1 for c in gen_hashes.values() if c > 1)
    dup_texts = sum(1 for c in gen_hashes.values() if c > 1)

    wc_stats = {}
    for model, parts in sorted(word_counts.items()):
        arr = np.concatenate(parts)
        wc_stats[model] = {
            "n": int(arr.size),
            "mean": round(float(arr.mean()), 1),
            "median": float(np.median(arr)),
            "p95": float(np.percentile(arr, 95)),
            "min": int(arr.min()),
            "max": int(arr.max()),
        }

    shared = human_sids & ai_sids
    n_human = model_counts.get(HUMAN, 0)
    return {
        "file": str(path),
        "total_rows": total,
        "rows_per_model": dict(model_counts.most_common()),
        "rows_per_domain": dict(domain_counts.most_common()),
        "attack_values": dict(attack_counts.most_common()),
        "source_id": {
            "unique_values": len(all_sids),
            "missing_total": missing_sid_total,
            "human_rows": n_human,
            "human_missing": human_missing_sid,
            "human_unique": len(human_sids),
            "ai_unique": len(ai_sids),
            "human_ai_shared": len(shared),
            "frac_human_sids_shared_with_ai": round(len(shared) / max(len(human_sids), 1), 4),
            "frac_ai_sids_shared_with_human": round(len(shared) / max(len(ai_sids), 1), 4),
        },
        "generation": {
            "empty": empty_gen,                  # blank or only invisible chars
            f"under_{SHORT_WORDS}_words": short_gen,  # includes empty
            "duplicate_rows": dup_rows,          # rows that repeat an earlier text
            "texts_with_duplicates": dup_texts,  # distinct texts appearing 2+ times
        },
        "word_count_per_model": wc_stats,
    }


def print_report(r: dict) -> None:
    print(f"\nFile: {r['file']}\nTotal rows: {r['total_rows']:,}")
    print("\nRows per model:")
    for k, v in r["rows_per_model"].items():
        print(f"  {k!r:20s} {v:>10,}")
    print("\nRows per domain:")
    for k, v in r["rows_per_domain"].items():
        print(f"  {k!r:20s} {v:>10,}")
    print(f"\nAttack values: {r['attack_values']}")
    print("\nsource_id:")
    for k, v in r["source_id"].items():
        print(f"  {k:32s} {v:,}" if isinstance(v, int) else f"  {k:32s} {v}")
    print("\ngeneration:")
    for k, v in r["generation"].items():
        print(f"  {k:32s} {v:,}")
    print("\nWord count per model:")
    print(f"  {'model':16s} {'n':>9s} {'mean':>7s} {'median':>7s} {'p95':>7s} {'min':>5s} {'max':>6s}")
    for m, s in r["word_count_per_model"].items():
        print(f"  {m:16s} {s['n']:>9,} {s['mean']:>7.1f} {s['median']:>7.0f} "
              f"{s['p95']:>7.0f} {s['min']:>5d} {s['max']:>6d}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--chunksize", type=int, default=100_000)
    report = inspect(parser.parse_args().chunksize)
    print_report(report)
    out = get_path("results_dir", create=True) / "raw_inspection.json"
    out.write_text(json.dumps(report, indent=2))
    print(f"\nSaved {out}")
