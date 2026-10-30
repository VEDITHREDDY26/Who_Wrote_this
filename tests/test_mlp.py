"""MLP tests on CPU with tiny sizes: shapes, registry, parameter checks, one training step."""

from typing import Any

import numpy as np
import pytest
import scipy.sparse as sp
import torch
from torch import nn

from src.models import build_model
from src.models.mlp import MLP, build_mlp
from src.trainer import ArrayDataset

PARAMS = {"hidden_sizes": [16, 8], "dropout": 0.3}
DIMS = {"input_dim": 40, "num_classes": 6}


def test_output_shape() -> None:
    torch.manual_seed(0)
    model = build_mlp(PARAMS, **DIMS)
    assert model(torch.rand(4, 40)).shape == (4, 6)


def test_builds_through_the_registry() -> None:
    model = build_model("mlp", PARAMS, **DIMS)
    assert isinstance(model, MLP)
    assert model.hidden[0].in_features == 40 and model.fc.in_features == 8


def test_accepts_sparse_tfidf_batches() -> None:
    torch.manual_seed(0)
    model = build_mlp(PARAMS, **DIMS).eval()
    rows = sp.random(5, 40, density=0.1, format="csr", dtype=np.float32, random_state=0)
    x, _ = ArrayDataset(rows, np.zeros(5, dtype=np.int64)).batch(torch.arange(5))
    logits = model(x)
    assert logits.shape == (5, 6) and torch.isfinite(logits).all()


@pytest.mark.parametrize("change", [{"hidden_sizes": []}, {"hidden_sizes": [16, 0]},
                                    {"dropout": 1.0}, {"dropout": -0.1}])
def test_bad_parameters_raise(change: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        build_mlp(PARAMS | change, **DIMS)


def test_missing_parameter_raises() -> None:
    with pytest.raises(KeyError):
        build_mlp({"dropout": 0.3}, **DIMS)


def test_one_training_step_lowers_the_loss() -> None:
    torch.manual_seed(0)
    model = build_mlp(PARAMS | {"dropout": 0.0}, **DIMS)
    x, y = torch.rand(12, 40), torch.arange(12) % 6
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    before = loss_fn(model(x), y)
    before.backward()
    optimizer.step()
    with torch.no_grad():
        after = loss_fn(model(x), y)
    assert after < before
