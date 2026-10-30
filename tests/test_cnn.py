"""TextCNN tests on CPU with tiny sizes: shapes, registry, parameter checks, one training step."""

from typing import Any

import pytest
import torch
from torch import nn

from src.models import build_model
from src.models.cnn import TextCNN, build_cnn

PARAMS = {"embed_dim": 8, "filter_sizes": [3, 4, 5], "num_filters": 4, "dropout": 0.5}
DIMS = {"vocab_size": 100, "pad_id": 0, "max_len": 256, "num_classes": 6}


def random_ids(batch: int, seq_len: int = 256) -> torch.Tensor:
    return torch.randint(2, DIMS["vocab_size"], (batch, seq_len))


def test_output_shape() -> None:
    torch.manual_seed(0)
    model = build_cnn(PARAMS, **DIMS)
    assert model(random_ids(4)).shape == (4, 6)


def test_builds_through_the_registry() -> None:
    model = build_model("cnn", PARAMS, **DIMS)
    assert isinstance(model, TextCNN)
    assert len(model.convs) == 3 and model.fc.in_features == 3 * 4


def test_accepts_int32_ids_and_padding_rows() -> None:
    torch.manual_seed(0)
    model = build_cnn(PARAMS, **DIMS).eval()
    x = random_ids(3).to(torch.int32)  # the vocab cache stores ids as int32
    x[0, 10:] = 0
    x[1] = 0  # all padding
    logits = model(x)
    assert logits.shape == (3, 6) and torch.isfinite(logits).all()


@pytest.mark.parametrize("change", [{"embed_dim": 0}, {"num_filters": 0}, {"filter_sizes": []},
                                    {"filter_sizes": [0, 3]}, {"dropout": 1.0},
                                    {"dropout": -0.1}])
def test_bad_parameters_raise(change: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        build_cnn(PARAMS | change, **DIMS)


def test_missing_parameter_raises() -> None:
    with pytest.raises(KeyError):
        build_cnn({k: v for k, v in PARAMS.items() if k != "num_filters"}, **DIMS)


def test_one_training_step_lowers_the_loss() -> None:
    torch.manual_seed(0)
    model = build_cnn(PARAMS | {"dropout": 0.0}, **DIMS)
    x, y = random_ids(12, seq_len=32), torch.arange(12) % 6
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    before = loss_fn(model(x), y)
    before.backward()
    optimizer.step()
    with torch.no_grad():
        after = loss_fn(model(x), y)
    assert after < before
