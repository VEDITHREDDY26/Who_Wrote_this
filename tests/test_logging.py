"""Tests for src.utils.logging.setup_logging."""

import logging
import re
from collections.abc import Iterator
from pathlib import Path

import pytest

from src.utils.logging import setup_logging

TIMESTAMPED = re.compile(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} INFO\s+test\.module: hello$",
                         re.MULTILINE)


@pytest.fixture(autouse=True)
def restore_root_logger() -> Iterator[None]:
    root = logging.getLogger()
    handlers, level = list(root.handlers), root.level
    yield
    for handler in list(root.handlers):
        if handler not in handlers:
            root.removeHandler(handler)
            handler.close()
    root.setLevel(level)


def test_logs_to_stdout_and_file_with_timestamps(tmp_path: Path,
                                                 capsys: pytest.CaptureFixture[str]) -> None:
    log_file = tmp_path / "logs" / "run.log"
    setup_logging(log_file)
    logging.getLogger("test.module").info("hello")
    assert TIMESTAMPED.search(capsys.readouterr().out)
    assert TIMESTAMPED.search(log_file.read_text())


def test_repeated_setup_does_not_duplicate_handlers(tmp_path: Path,
                                                    capsys: pytest.CaptureFixture[str]) -> None:
    setup_logging(tmp_path / "run.log")
    setup_logging(tmp_path / "run.log")
    logging.getLogger("test.module").info("hello")
    assert capsys.readouterr().out.count("hello") == 1


def test_file_is_appended_across_setups(tmp_path: Path) -> None:
    log_file = tmp_path / "run.log"
    setup_logging(log_file)
    logging.getLogger("test.module").info("first")
    setup_logging(log_file)
    logging.getLogger("test.module").info("second")
    text = log_file.read_text()
    assert "first" in text and "second" in text


def test_level_filters_messages(capsys: pytest.CaptureFixture[str]) -> None:
    setup_logging(level="WARNING")
    logging.getLogger("test.module").info("hidden")
    logging.getLogger("test.module").warning("shown")
    out = capsys.readouterr().out
    assert "shown" in out and "hidden" not in out
