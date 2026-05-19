"""Logger that prints to stdout AND tees into a per-experiment log file."""
from __future__ import annotations
import logging
import sys
from pathlib import Path


def build_logger(name: str, out_dir: str | Path | None = None,
                 level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(level)
    fmt = logging.Formatter("[%(asctime)s][%(levelname)s][%(name)s] %(message)s",
                             datefmt="%Y-%m-%d %H:%M:%S")
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    if out_dir is not None:
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(Path(out_dir) / f"{name}.log")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger
