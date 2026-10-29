"""Cache keys shared by the feature builders.

A cache folder is named `<kind>-<key>`, where the key hashes the train-split fingerprint, the
feature settings and a code version. Change any of them and a new folder is built; the old
one is simply no longer used.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any

KEY_CHARS = 16


def cache_key(kind: str, version: int, train_fingerprint: str,
              settings: Mapping[str, Any]) -> str:
    """Return a short hash identifying one feature build."""
    payload = json.dumps({"kind": kind, "version": version, "train": train_fingerprint,
                          "settings": settings}, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:KEY_CHARS]


def split_file_name(split: str, fingerprint: str, suffix: str) -> str:
    """File name for one split's cached matrix, e.g. `val-6c517d46f22dda76.npz`."""
    return f"{split}-{fingerprint}{suffix}"
