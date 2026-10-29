"""Project-wide logging configuration.

Call `setup_logging()` once at the start of a script or job. Every module then uses its own
logger via `logger = logging.getLogger(__name__)`.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

LOG_FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
_HANDLER_MARK = "_turing_detectives_handler"


def setup_logging(log_file: str | Path | None = None, level: int | str = logging.INFO) -> None:
    """Configure the root logger to write timestamped lines to stdout and optionally a file.

    Safe to call more than once: handlers added by a previous call are replaced, never
    duplicated. The log file is opened in append mode so a resumed job keeps its history.

    Args:
        log_file: Optional path of a log file; parent folders are created if needed.
        level: Logging level name or number, e.g. "INFO" or logging.DEBUG.
    """
    root = logging.getLogger()
    for handler in list(root.handlers):
        if getattr(handler, _HANDLER_MARK, False):
            root.removeHandler(handler)
            handler.close()

    formatter = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if log_file is not None:
        path = Path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(path, mode="a", encoding="utf-8"))

    for handler in handlers:
        handler.setFormatter(formatter)
        setattr(handler, _HANDLER_MARK, True)
        root.addHandler(handler)
    root.setLevel(level)
