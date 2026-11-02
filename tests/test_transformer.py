"""Transformer tests with a tiny randomly initialised DistilBERT saved locally (no downloads)."""

import json
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F
import yaml
from transformers import (
    DistilBertConfig,
    DistilBertForSequenceClassification,
    DistilBertTokenizerFast,
)

from src.models.transformer import (
    BEST_MODEL_DIR,
    DONE_FILE,
    TransformerParams,
    WeightedLossTrainer,
    complete_checkpoints,
    encode_texts,
    mixed_precision,
)
from tests.test_train_cli import CLASSES, make_workspace, run_cli

WORDS = [f"w{c}{i}" for c in range(6) for i in range(5)] + ["the", "a", "of", "and", "text"]


@pytest.fixture(scope="module")
def tiny_model(tmp_path_factory: pytest.TempPathFactory) -> Path:
    folder = tmp_path_factory.mktemp("tiny-distilbert")
    vocab = folder / "vocab.txt"
    vocab.write_text("\n".join(["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "."] + WORDS))
    tokenizer = DistilBertTokenizerFast(vocab_file=str(vocab), do_lower_case=False)
    config = DistilBertConfig(vocab_size=6 + len(WORDS), dim=16, n_layers=1, n_heads=2,
                              hidden_dim=32, max_position_embeddings=64, num_labels=6)
    torch.manual_seed(0)
    DistilBertForSequenceClassification(config).save_pretrained(folder)
    tokenizer.save_pretrained(folder)
    return folder


def transformer_config(workspace: Path, tiny_model: Path, name: str, **training: object) -> Path:
    cfg = yaml.safe_load((workspace / "config.yaml").read_text(encoding="utf-8"))
    cfg["models"]["transformer"] = {
        "pretrained_name": str(tiny_model), "max_len": 32,
        "training": {"batch_size": 8, "eval_batch_size": 16, "max_epochs": 2, "lr": 1e-3,
                     "patience": 2} | training}
    path = workspace / name
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return path


@pytest.fixture(scope="module")
def workspace(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return make_workspace(tmp_path_factory.mktemp("tf-cli"))


def test_params_validation() -> None:
    assert TransformerParams.from_config({"pretrained_name": "x", "max_len": 8}).max_len == 8
    with pytest.raises(KeyError, match="missing"):
        TransformerParams.from_config({"pretrained_name": "x"})
    with pytest.raises(KeyError, match="unknown"):
        TransformerParams.from_config({"pretrained_name": "x", "max_len": 8, "lr": 1})
    with pytest.raises(ValueError, match="max_len"):
        TransformerParams.from_config({"pretrained_name": "x", "max_len": 1})


def test_mixed_precision_only_on_cuda() -> None:
    assert mixed_precision(torch.device("cpu"), True) == {"bf16": False, "fp16": False}
    assert mixed_precision(torch.device("mps"), True) == {"bf16": False, "fp16": False}
    if torch.cuda.is_available():
        assert sum(mixed_precision(torch.device("cuda"), True).values()) == 1


def test_encode_texts_truncates_and_keeps_labels(tiny_model: Path) -> None:
    tokenizer = DistilBertTokenizerFast.from_pretrained(tiny_model)
    data = encode_texts(tokenizer, ["the text", " ".join(["a"] * 100)], np.array([2, 5]), 16)
    assert len(data) == 2 and data[0]["labels"] == 2 and data[1]["labels"] == 5
    assert len(data[1]["input_ids"]) == 16


def test_weighted_loss_matches_shared_definition(tiny_model: Path, tmp_path: Path) -> None:
    from transformers import TrainingArguments

    model = DistilBertForSequenceClassification.from_pretrained(tiny_model)
    weights = [2.0, 1.0, 1.0, 1.0, 1.0, 0.5]
    trainer = WeightedLossTrainer(model=model, args=TrainingArguments(
        output_dir=str(tmp_path), report_to="none", use_cpu=True), class_weights=weights)
    inputs = {"input_ids": torch.tensor([[2, 10, 3], [2, 11, 3]]),
              "attention_mask": torch.ones(2, 3, dtype=torch.long),
              "labels": torch.tensor([0, 5])}
    loss, outputs = trainer.compute_loss(model, inputs, return_outputs=True)
    expected = F.cross_entropy(outputs.logits, inputs["labels"], weight=torch.tensor(weights))
    assert torch.allclose(loss, expected)


def test_incomplete_checkpoints_are_skipped(tmp_path: Path) -> None:
    for step, state in [(10, "{}"), (20, "{not json"), (30, None)]:
        folder = tmp_path / f"checkpoint-{step}"
        folder.mkdir()
        if state is not None:
            (folder / "trainer_state.json").write_text(state)
    (tmp_path / "checkpoint-tmp").mkdir()
    assert [p.name for p in complete_checkpoints(tmp_path)] == ["checkpoint-10"]


def test_transformer_trains_end_to_end_and_rerun_is_a_noop(workspace: Path,
                                                           tiny_model: Path) -> None:
    config = transformer_config(workspace, tiny_model, "tf.yaml")
    proc = run_cli(config, "--model", "transformer")
    assert proc.returncode == 0, proc.output
    assert "72 train rows" in proc.output  # the fake split, not the real data
    ckpt = workspace / "checkpoints" / "transformer" / "default"
    assert (ckpt / BEST_MODEL_DIR / "config.json").exists() and (ckpt / DONE_FILE).exists()
    results = workspace / "results"
    metrics = json.loads((results / "transformer_val_metrics.json").read_text())
    assert metrics["n_samples"] == 24 and metrics["class_names"] == CLASSES
    assert metrics["extra"]["epochs_completed"] == 2 and metrics["extra"]["num_parameters"] > 0
    history = (results / "transformer" / "default" / "history.jsonl").read_text().splitlines()
    assert [json.loads(line)["epoch"] for line in history] == [1, 2]

    again = run_cli(config, "--model", "transformer")
    assert again.returncode == 0 and "already finished" in again.output


def test_transformer_resumes_from_a_checkpoint(workspace: Path, tiny_model: Path) -> None:
    one = transformer_config(workspace, tiny_model, "tf-res.yaml")
    first = run_cli(one, "--model", "transformer", "--run-name", "res", "--max-epochs", "1")
    assert first.returncode == 0, first.output
    # Pretend the run was cut off after epoch 1: drop the finished marker, keep the checkpoint.
    ckpt = workspace / "checkpoints" / "transformer" / "res"
    (ckpt / DONE_FILE).unlink()
    refused = run_cli(one, "--model", "transformer", "--run-name", "res")
    assert refused.returncode == 2 and "training.max_epochs: 1 -> 2" in refused.output
    resumed = run_cli(one, "--model", "transformer", "--run-name", "res", "--max-epochs", "1")
    assert resumed.returncode == 0, resumed.output
    assert "resuming from checkpoint-" in resumed.output


def test_config_change_is_refused(workspace: Path, tiny_model: Path) -> None:
    first = run_cli(transformer_config(workspace, tiny_model, "tf-a.yaml"),
                    "--model", "transformer", "--run-name", "cfg", "--max-epochs", "1")
    assert first.returncode == 0, first.output
    changed = transformer_config(workspace, tiny_model, "tf-b.yaml", lr=5e-4)
    refused = run_cli(changed, "--model", "transformer", "--run-name", "cfg",
                      "--max-epochs", "1")
    assert refused.returncode == 2 and "training.lr: 0.001 -> 0.0005" in refused.output
