"""Minimal YAML config loader with ``defaults`` composition.

A config file may list other files to merge under the ``defaults`` key, e.g.

    defaults: [sode_guard.yaml]
    experiment: {name: smoke}

The composition is shallow-merge per top-level key with the current file
taking precedence over its defaults.
"""
from __future__ import annotations
from pathlib import Path
from typing import Any
import yaml


class ConfigDict(dict):
    """Dict with attribute access and recursive ConfigDict materialisation."""

    def __init__(self, data: dict[str, Any]):
        super().__init__()
        for k, v in data.items():
            self[k] = ConfigDict(v) if isinstance(v, dict) else v

    def __getattr__(self, key: str) -> Any:
        try:
            return self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value


def _deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path: str | Path) -> ConfigDict:
    path = Path(path)
    with path.open() as f:
        cfg = yaml.safe_load(f) or {}
    defaults = cfg.pop("defaults", [])
    merged: dict[str, Any] = {}
    for d in defaults:
        merged = _deep_merge(merged, load_config(path.parent / d))
    merged = _deep_merge(merged, cfg)
    return ConfigDict(merged)
