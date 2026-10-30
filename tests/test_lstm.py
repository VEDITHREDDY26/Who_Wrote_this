"""BiLSTM tests on CPU with tiny sizes: shapes, registry, padding, parameter checks, training."""

from typing import Any

import pytest
import torch
from torch import nn

from src.models import build_model
from src.models.lstm import BiLSTM, build_lstm

PARAMS = {"embed_dim": 8, "hidden_size": 6, "num_layers": 2, "dropout": 0.3}
DIMS = {"vocab_size": 100, "pad_id": 0, "max_len": 256, "num_classes": 6}


def random_ids(batch: int, seq_len: int = 256) -> torch.Tensor:
    return torch.randint(2, DIMS["vocab_size"], (batch, seq_len))


def test_output_shape() -> None:
    torch.manual_seed(0)
    model = build_lstm(PARAMS, **DIMS)
    assert model(random_ids(4)).shape == (4, 6)


def test_builds_through_the_registry() -> None:
    model = build_model("lstm", PARAMS, **DIMS)
    assert isinstance(model, BiLSTM)
    assert model.lstm.bidirectional and model.lstm.num_layers == 2
    assert model.fc.in_features == 2 * 6


def test_single_layer_has_no_inter_layer_dropout() -> None:
    model = build_lstm(PARAMS | {"num_layers": 1}, **DIMS)
    assert model.lstm.dropout == 0.0


def test_extra_padding_does_not_change_the_output() -> None:
    torch.manual_seed(0)
    model = build_lstm(PARAMS, **DIMS).eval()
    x = random_ids(2, seq_len=20)
    x[0, 7:] = 0  # one short text, one full-length text
    longer = torch.cat([x, torch.zeros(2, 30, dtype=torch.long)], dim=1)
    with torch.no_grad():
        assert torch.allclose(model(x), model(longer), atol=1e-6)


def test_accepts_int32_ids_and_padding_rows() -> None:
    torch.manual_seed(0)
    model = build_lstm(PARAMS, **DIMS).eval()
    x = random_ids(3).to(torch.int32)  # the vocab cache stores ids as int32
    x[0, 10:] = 0
    x[1] = 0  # all padding
    logits = model(x)
    assert logits.shape == (3, 6) and torch.isfinite(logits).all()


@pytest.mark.parametrize("change", [{"embed_dim": 0}, {"hidden_size": 0}, {"num_layers": 0},
                                    {"dropout": 1.0}, {"dropout": -0.1}])
def test_bad_parameters_raise(change: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        build_lstm(PARAMS | change, **DIMS)


def test_missing_parameter_raises() -> None:
    with pytest.raises(KeyError):
        build_lstm({k: v for k, v in PARAMS.items() if k != "hidden_size"}, **DIMS)


def test_one_training_step_lowers_the_loss() -> None:
    torch.manual_seed(0)
    model = build_lstm(PARAMS | {"dropout": 0.0}, **DIMS)
    x, y = random_ids(12, seq_len=32), torch.arange(12) % 6
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    before = loss_fn(model(x), y)
    before.backward()
    optimizer.step()
    with torch.no_grad():
        after = loss_fn(model(x), y)
    assert after < before
