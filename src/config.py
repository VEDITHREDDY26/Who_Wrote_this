"""Project configuration: loads configs/config.yaml, applies env-var overrides, picks a device.

Path resolution order (highest wins):
  1. Env var for that path (e.g. TD_DATA_DIR, TD_CHECKPOINT_DIR)
  2. Value in config.yaml
Relative paths are resolved against the project root, except raw_dir and splits_dir,
which resolve against data_dir (so setting TD_DATA_DIR alone moves all data). The config
file itself can be swapped with TD_CONFIG=/path/to/other.yaml. TD_ROOT overrides the
project root.
"""

from __future__ import annotations

import os
from functools import cache
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(os.environ.get("TD_ROOT", Path(__file__).resolve().parents[1]))
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"

# config key under `paths:` -> env var that overrides it
PATH_ENV_VARS = {
    "data_dir": "TD_DATA_DIR",
    "raw_dir": "TD_RAW_DIR",
    "splits_dir": "TD_SPLITS_DIR",
    "checkpoint_dir": "TD_CHECKPOINT_DIR",
    "results_dir": "TD_RESULTS_DIR",
}


# paths that resolve relative to data_dir instead of the project root
DATA_SUBDIRS = {"raw_dir", "splits_dir"}


def _resolve(p: str | os.PathLike, base: Path = PROJECT_ROOT) -> Path:
    path = Path(os.path.expandvars(os.path.expanduser(str(p))))
    return path if path.is_absolute() else (base / path).resolve()


@cache
def load_config(config_path: str | None = None) -> dict[str, Any]:
    """Load the YAML config, apply env overrides, and resolve `paths` to absolute Paths."""
    path = Path(config_path or os.environ.get("TD_CONFIG", DEFAULT_CONFIG_PATH))
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}

    paths = cfg.setdefault("paths", {})
    for key, env_var in PATH_ENV_VARS.items():  # data_dir first, so subdirs can use it
        value = os.environ.get(env_var, paths.get(key))
        if value is not None:
            base = paths["data_dir"] if key in DATA_SUBDIRS else PROJECT_ROOT
            paths[key] = _resolve(value, base)

    if "TD_SEED" in os.environ:
        cfg["seed"] = int(os.environ["TD_SEED"])
    return cfg


def get_path(key: str, create: bool = False) -> Path:
    """Return a resolved path from the `paths` section, optionally creating the directory."""
    path = load_config()["paths"][key]
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path


def get_device(prefer: str | None = None):
    """Return a torch.device, auto-detected as cuda > mps > cpu.

    `prefer` (or env var TD_DEVICE) forces a specific device, e.g. "cpu" for debugging.
    """
    import torch  # imported lazily so config loading doesn't require torch

    forced = prefer or os.environ.get("TD_DEVICE")
    if forced:
        return torch.device(forced)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


if __name__ == "__main__":
    cfg = load_config()
    for k, v in cfg["paths"].items():
        print(f"{k:15s} {v}")
    print(f"{'seed':15s} {cfg['seed']}")
    try:
        print(f"{'device':15s} {get_device()}")
    except ImportError:
        print(f"{'device':15s} (torch not installed)")
