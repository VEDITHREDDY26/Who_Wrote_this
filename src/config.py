"""Project configuration: loads configs/config.yaml, applies env-var overrides, picks a device.

Path resolution order (highest wins):
  1. Env var for that path (e.g. TD_DATA_DIR, TD_CHECKPOINT_DIR)
  2. Value in config.yaml
Relative paths are resolved against the project root, except raw_dir and splits_dir,
which resolve against data_dir (so setting TD_DATA_DIR alone moves all data). The config
file itself can be swapped with TD_CONFIG=/path/to/other.yaml, or per process with
use_config(path) (what the --config CLI flags do). TD_ROOT overrides the project root.
"""

from __future__ import annotations

import os
from functools import cache
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

if TYPE_CHECKING:
    import torch

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

_active_config: Path | None = None  # set by use_config(); None means TD_CONFIG or default


def _resolve(p: str | os.PathLike, base: Path = PROJECT_ROOT) -> Path:
    path = Path(os.path.expandvars(os.path.expanduser(str(p))))
    return path if path.is_absolute() else (base / path).resolve()


def use_config(config_path: str | os.PathLike | None) -> None:
    """Make `config_path` the config that load_config() and get_path() use in this process.

    Without this, get_path() would silently keep reading the default config even when a
    script was given --config. Passing None restores the default (TD_CONFIG or config.yaml).
    """
    global _active_config
    if config_path is not None and not Path(config_path).is_file():
        raise FileNotFoundError(f"config file {config_path} does not exist")
    _active_config = Path(config_path).resolve() if config_path is not None else None
    load_config.cache_clear()


def active_config_path() -> Path:
    """The config file in use: use_config() > TD_CONFIG > configs/config.yaml."""
    return Path(_active_config or os.environ.get("TD_CONFIG", DEFAULT_CONFIG_PATH))


@cache
def load_config(config_path: str | None = None) -> dict[str, Any]:
    """Load the YAML config, apply env overrides, and resolve `paths` to absolute Paths.

    With no argument this is the active config: use_config() > TD_CONFIG > configs/config.yaml.
    """
    path = Path(config_path) if config_path else active_config_path()
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


def get_device(prefer: str | None = None) -> torch.device:
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
