"""Tests for src.utils.seed: identical seeds give identical streams; RNG state round-trips."""

import logging
import random
from pathlib import Path

import numpy as np
import pytest
import torch

from src.utils.io import atomic_torch_save
from src.utils.seed import get_rng_state, seed_everything, set_rng_state

MPS = getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available()


def _draw() -> tuple[float, list[float], list[float]]:
    return random.random(), np.random.rand(3).tolist(), torch.rand(3).tolist()


def test_same_seed_same_numbers() -> None:
    seed_everything(42)
    first = _draw()
    seed_everything(42)
    assert _draw() == first


def test_different_seed_different_numbers() -> None:
    seed_everything(42)
    first = _draw()
    seed_everything(43)
    assert _draw() != first


@pytest.mark.skipif(not MPS, reason="MPS not available")
def test_same_seed_same_numbers_on_mps() -> None:
    seed_everything(42)
    first = torch.rand(3, device="mps").cpu()
    seed_everything(42)
    assert torch.equal(torch.rand(3, device="mps").cpu(), first)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_same_seed_same_numbers_on_cuda() -> None:
    seed_everything(42)
    first = torch.rand(3, device="cuda").cpu()
    seed_everything(42)
    assert torch.equal(torch.rand(3, device="cuda").cpu(), first)


@pytest.mark.parametrize("bad", ["42", 4.2, True, None])
def test_seed_must_be_int(bad: object) -> None:
    with pytest.raises(TypeError, match="seed must be an int"):
        seed_everything(bad)  # type: ignore[arg-type]


def test_rng_state_roundtrip_continues_same_stream() -> None:
    seed_everything(7)
    _draw()
    state = get_rng_state()
    expected = _draw()
    _draw()  # move every generator further on
    set_rng_state(state)
    assert _draw() == expected


def test_rng_state_survives_checkpoint_with_weights_only(tmp_path: Path) -> None:
    seed_everything(7)
    path = tmp_path / "rng.pt"
    atomic_torch_save(get_rng_state(), path)
    expected = _draw()
    set_rng_state(torch.load(path, weights_only=True))
    assert _draw() == expected


def test_set_rng_state_missing_keys_raises() -> None:
    with pytest.raises(KeyError, match="missing keys"):
        set_rng_state({"python": random.getstate()})


def test_cuda_state_without_cuda_warns(monkeypatch: pytest.MonkeyPatch,
                                       caplog: pytest.LogCaptureFixture) -> None:
    state = get_rng_state()
    state["cuda"] = [torch.zeros(1, dtype=torch.uint8)]
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with caplog.at_level(logging.WARNING, logger="src.utils.seed"):
        set_rng_state(state)
    assert "CUDA is unavailable" in caplog.text
