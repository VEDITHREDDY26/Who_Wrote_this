"""Download RAID's clean (non-attacked) labeled training split to <raw_dir>/train_none.csv.

Sources are tried in order, stopping at the first that works:
  (a) direct CSV from https://dataset.raid-bench.xyz/train_none.csv (resumable)
  (b) the `raid-bench` pip package (raid.utils.load_data), filtered to attack == "none"
  (c) Hugging Face `liamdugan/raid` (official, by the RAID authors), streamed and
      filtered to attack == "none"

Skips everything if the file already exists. Usage: python -m src.data.download [--force]
"""

from __future__ import annotations

import argparse
import csv
import shutil
import urllib.request
from pathlib import Path

from tqdm import tqdm

from src.config import get_path

URL = "https://dataset.raid-bench.xyz/train_none.csv"
HF_REPO = "liamdugan/raid"
FILENAME = "train_none.csv"
CHUNK = 1 << 20  # 1 MiB


def from_url(out: Path) -> None:
    """(a) Stream the CSV over HTTP into a .part file, resuming if a partial file exists."""
    part = out.with_suffix(out.suffix + ".part")
    done = part.stat().st_size if part.exists() else 0

    req = urllib.request.Request(URL, headers={"User-Agent": "cap5610-turing-detectives"})
    if done:
        req.add_header("Range", f"bytes={done}-")
    with urllib.request.urlopen(req, timeout=60) as resp:
        if done and resp.status != 206:  # server ignored Range: start over
            done = 0
        total = done + int(resp.headers.get("Content-Length", 0)) or None
        mode = "ab" if done else "wb"
        with open(part, mode) as f, tqdm(
            total=total, initial=done, unit="B", unit_scale=True, desc=FILENAME
        ) as bar:
            while chunk := resp.read(CHUNK):
                f.write(chunk)
                bar.update(len(chunk))

    if total and part.stat().st_size != total:
        raise IOError(f"incomplete download: {part.stat().st_size} of {total} bytes")
    part.rename(out)


def from_raid_package(out: Path) -> None:
    """(b) Use the raid-bench package's loader, keep only non-attacked rows."""
    from raid.utils import load_data  # pip install raid-bench

    df = load_data(split="train", include_adversarial=False)
    df = df[df["attack"] == "none"]
    tmp = out.with_suffix(".tmp")
    df.to_csv(tmp, index=False)
    tmp.rename(out)


def from_huggingface(out: Path) -> None:
    """(c) Stream the official HF copy (all attacks, ~10 GB) and write only attack == "none"."""
    from datasets import load_dataset

    ds = load_dataset(HF_REPO, "raid", split="train", streaming=True)
    tmp = out.with_suffix(".tmp")
    kept = 0
    with open(tmp, "w", newline="") as f:
        writer = None
        for row in tqdm(ds, desc="HF rows scanned", unit="row"):
            if row.get("attack") != "none":
                continue
            if writer is None:
                writer = csv.DictWriter(f, fieldnames=list(row.keys()))
                writer.writeheader()
            writer.writerow(row)
            kept += 1
    if kept == 0:
        raise ValueError("no attack == 'none' rows found")
    tmp.rename(out)


SOURCES = [("direct URL", from_url), ("raid-bench package", from_raid_package),
           ("Hugging Face", from_huggingface)]


def download(force: bool = False) -> Path:
    out = get_path("raw_dir", create=True) / FILENAME
    if out.exists() and not force:
        print(f"Already present, skipping: {out} ({out.stat().st_size / 1e6:.0f} MB)")
        return out

    free_gb = shutil.disk_usage(out.parent).free / 1e9
    if free_gb < 2:
        raise OSError(f"only {free_gb:.1f} GB free in {out.parent}; need ~1 GB")

    errors = []
    for name, fn in SOURCES:
        print(f"Trying source: {name}")
        try:
            fn(out)
            print(f"Saved {out} ({out.stat().st_size / 1e6:.0f} MB) via {name}")
            return out
        except Exception as e:  # fall through to the next source
            print(f"  failed: {type(e).__name__}: {e}")
            errors.append(f"{name}: {e}")
    raise RuntimeError("all download sources failed:\n  " + "\n  ".join(errors))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--force", action="store_true", help="re-download even if present")
    download(force=parser.parse_args().force)
