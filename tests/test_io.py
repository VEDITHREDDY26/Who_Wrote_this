"""Tests for src.utils.io: atomic writes survive exceptions and hard kills mid-write."""

import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
import torch

from src.utils.io import (
    atomic_open,
    atomic_torch_save,
    atomic_write_bytes,
    atomic_write_json,
    safe_load_json,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OLD = b"old complete content"


def _tmp_files(folder: Path) -> list[Path]:
    return sorted(folder.glob(".*.tmp"))


@pytest.fixture
def target(tmp_path: Path) -> Path:
    path = tmp_path / "out.bin"
    path.write_bytes(OLD)
    return path


def test_write_bytes_creates_file_and_parent_dirs(tmp_path: Path) -> None:
    path = tmp_path / "a" / "b" / "file.bin"
    atomic_write_bytes(path, b"hello")
    assert path.read_bytes() == b"hello"
    assert _tmp_files(path.parent) == []


def test_write_bytes_replaces_existing(target: Path) -> None:
    atomic_write_bytes(target, b"new")
    assert target.read_bytes() == b"new"


def test_write_bytes_rejects_str(target: Path) -> None:
    with pytest.raises(TypeError, match="expects bytes"):
        atomic_write_bytes(target, "not bytes")  # type: ignore[arg-type]
    assert target.read_bytes() == OLD


def test_exception_mid_write_keeps_old_file_and_removes_tmp(target: Path) -> None:
    with pytest.raises(RuntimeError, match="crash"):
        with atomic_open(target) as f:
            f.write(b"partial new conte")
            raise RuntimeError("simulated crash")
    assert target.read_bytes() == OLD
    assert _tmp_files(target.parent) == []


def test_hard_kill_mid_write_keeps_old_file(target: Path) -> None:
    """os._exit skips every cleanup handler, like SIGKILL or a Colab disconnect."""
    script = textwrap.dedent(f"""
        import os
        from src.utils.io import atomic_open
        with atomic_open({str(target)!r}) as f:
            f.write(b"NEW" * 100_000)
            f.flush()
            os._exit(1)
    """)
    result = subprocess.run([sys.executable, "-c", script], cwd=PROJECT_ROOT, check=False)
    assert result.returncode == 1
    assert target.read_bytes() == OLD
    assert len(_tmp_files(target.parent)) == 1  # orphaned temp file, never read


def test_json_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "stats.json"
    data = {"macro_f1": 0.91, "labels": ["Human", "GPT"], "note": "naïve ✓"}
    atomic_write_json(path, data)
    assert safe_load_json(path) == data
    assert path.read_text(encoding="utf-8").endswith("\n")


def test_json_unserializable_raises_and_keeps_old(tmp_path: Path) -> None:
    path = tmp_path / "stats.json"
    atomic_write_json(path, {"ok": 1})
    with pytest.raises(TypeError, match="cannot write"):
        atomic_write_json(path, {"bad": object()})
    assert safe_load_json(path) == {"ok": 1}


def test_torch_save_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "ckpt.pt"
    state = {"step": 7, "weights": torch.arange(5.0)}
    atomic_torch_save(state, path)
    loaded = torch.load(path, weights_only=True)
    assert loaded["step"] == 7
    assert torch.equal(loaded["weights"], state["weights"])


def test_torch_save_failure_keeps_old_checkpoint(target: Path) -> None:
    class Unpicklable:
        def __reduce__(self):
            raise RuntimeError("cannot pickle")

    with pytest.raises(RuntimeError, match="cannot pickle"):
        atomic_torch_save({"model": Unpicklable()}, target)
    assert target.read_bytes() == OLD
    assert _tmp_files(target.parent) == []


def test_safe_load_json_missing_returns_default(tmp_path: Path) -> None:
    assert safe_load_json(tmp_path / "nope.json", default=None) is None
    assert safe_load_json(tmp_path / "nope.json", default={}) == {}


def test_safe_load_json_missing_without_default_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="does not exist"):
        safe_load_json(tmp_path / "nope.json")


def test_safe_load_json_corrupt_raises(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text('{"truncated": ')
    with pytest.raises(ValueError, match="not valid JSON"):
        safe_load_json(path, default={})
