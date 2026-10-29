"""Crash-safe file writes.

Every writer goes through `atomic_open`: data is written to a temporary file in the target's
folder, flushed and fsynced, then moved onto the final name with `os.replace`. A crash at
any moment leaves either the complete old file or the complete new file, never a partial
one. A hard kill (SIGKILL, power loss) can leave a stray `.<name>.*.tmp` file behind; it is
harmless and never read.
"""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import IO, Any

DEFAULT_FILE_MODE = 0o644
TMP_SUFFIX = ".tmp"
_MISSING = object()


def _fsync_dir(folder: Path) -> None:
    """Persist a rename by fsyncing its folder (POSIX only; a no-op on Windows)."""
    if os.name == "nt":
        return
    fd = os.open(folder, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


@contextmanager
def atomic_open(path: str | Path) -> Iterator[IO[bytes]]:
    """Open a binary temp file that atomically replaces `path` when the block exits cleanly.

    If the block raises, the temp file is deleted and `path` is left untouched.

    Example:
        with atomic_open("results/metrics.csv") as f:
            df.to_csv(f, index=False)
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=target.parent, prefix=f".{target.name}.",
                                    suffix=TMP_SUFFIX)
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as f:
            yield f
            f.flush()
            os.fsync(f.fileno())
        os.chmod(tmp, DEFAULT_FILE_MODE)
        os.replace(tmp, target)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    _fsync_dir(target.parent)


def atomic_write_bytes(path: str | Path, data: bytes) -> None:
    """Atomically write raw bytes to `path`."""
    if not isinstance(data, bytes | bytearray):
        raise TypeError(f"atomic_write_bytes expects bytes, got {type(data).__name__}")
    with atomic_open(path) as f:
        f.write(data)


def atomic_write_json(path: str | Path, obj: Any, indent: int = 2) -> None:
    """Atomically write `obj` as UTF-8 JSON. Raises TypeError if `obj` is not serializable."""
    try:
        text = json.dumps(obj, indent=indent, ensure_ascii=False)
    except TypeError as e:
        raise TypeError(f"cannot write {path} as JSON: {e}") from e
    atomic_write_bytes(path, (text + "\n").encode("utf-8"))


def atomic_torch_save(obj: Any, path: str | Path) -> None:
    """Atomically save `obj` with torch.save (same argument order as torch.save)."""
    import torch

    with atomic_open(path) as f:
        torch.save(obj, f)


def safe_load_json(path: str | Path, default: Any = _MISSING) -> Any:
    """Load JSON from `path`.

    Returns `default` if the file does not exist and a default was given; otherwise raises
    FileNotFoundError. A file that exists but is not valid JSON always raises ValueError,
    because silently ignoring it could hide a real problem.
    """
    target = Path(path)
    if not target.exists():
        if default is _MISSING:
            raise FileNotFoundError(f"expected a JSON file at {target}, but it does not exist")
        return default
    try:
        return json.loads(target.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise ValueError(f"{target} is not valid JSON (line {e.lineno}, col {e.colno}): "
                         f"{e.msg}") from e
