"""Logging. One console stream plus one log file per run."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

_FORMAT = "%(asctime)s %(levelname)-7s %(name)-24s %(message)s"
_DATEFMT = "%H:%M:%S"


def setup(log_path: str | Path | None = None, verbose: bool = False) -> logging.Logger:
    """Configure the root logger. Safe to call more than once."""
    root = logging.getLogger()
    root.setLevel(logging.DEBUG if verbose else logging.INFO)
    for handler in list(root.handlers):
        root.removeHandler(handler)

    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.DEBUG if verbose else logging.INFO)
    console.setFormatter(logging.Formatter(_FORMAT, datefmt=_DATEFMT))
    root.addHandler(console)

    if log_path is not None:
        path = Path(log_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(path)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter(_FORMAT, datefmt=_DATEFMT))
        root.addHandler(file_handler)

    # These libraries are loud and say nothing useful at INFO.
    for noisy in ("matplotlib", "PIL", "urllib3", "filelock", "h5py", "absl"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    return root


def get(name: str) -> logging.Logger:
    return logging.getLogger(name)
