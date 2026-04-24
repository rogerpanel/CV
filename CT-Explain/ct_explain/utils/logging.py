"""Unified logger factory. One handler per process; sub-loggers inherit."""
from __future__ import annotations

import logging
import os
import sys

_FMT = "%(asctime)s %(levelname)-7s %(name)s :: %(message)s"
_ROOT = "ct_explain"


def _configure_root() -> None:
    root = logging.getLogger(_ROOT)
    if root.handlers:
        return
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter(_FMT))
    root.addHandler(handler)
    level = os.getenv("CT_EXPLAIN_LOG", "INFO").upper()
    root.setLevel(getattr(logging, level, logging.INFO))
    root.propagate = False


def get_logger(name: str) -> logging.Logger:
    _configure_root()
    if not name.startswith(_ROOT):
        name = f"{_ROOT}.{name}"
    return logging.getLogger(name)
