"""Tests for src.config: a chosen config file must control get_path() too."""

from collections.abc import Iterator
from pathlib import Path

import pytest
import yaml

from src.config import DEFAULT_CONFIG_PATH, get_path, load_config, use_config


@pytest.fixture(autouse=True)
def restore_default_config() -> Iterator[None]:
    yield
    use_config(None)


def write_config(tmp_path: Path) -> Path:
    cfg = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
    cfg["paths"] = {"data_dir": str(tmp_path / "d"), "raw_dir": "raw", "splits_dir": "s",
                    "checkpoint_dir": str(tmp_path / "c"), "results_dir": str(tmp_path / "r")}
    path = tmp_path / "other.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return path


def test_use_config_redirects_get_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    for var in ("TD_DATA_DIR", "TD_SPLITS_DIR", "TD_RESULTS_DIR", "TD_CHECKPOINT_DIR"):
        monkeypatch.delenv(var, raising=False)
    default_results = get_path("results_dir")
    use_config(write_config(tmp_path))
    assert get_path("results_dir") == tmp_path / "r"
    assert get_path("splits_dir") == tmp_path / "d" / "s"
    assert load_config()["paths"]["checkpoint_dir"] == tmp_path / "c"
    use_config(None)
    assert get_path("results_dir") == default_results


def test_use_config_rejects_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="does not exist"):
        use_config(tmp_path / "missing.yaml")
