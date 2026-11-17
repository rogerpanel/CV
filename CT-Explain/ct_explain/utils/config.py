"""YAML / JSON config I/O with dot-attribute access."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import yaml


class ConfigDict(dict):
    """Dict subclass whose keys are also accessible as attributes."""

    def __getattr__(self, name: str) -> Any:
        try:
            v = self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc
        return ConfigDict(v) if isinstance(v, dict) else v

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value


def _wrap(obj: Any) -> Any:
    if isinstance(obj, dict):
        return ConfigDict({k: _wrap(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_wrap(v) for v in obj]
    return obj


def load_config(path: str | Path) -> ConfigDict:
    p = Path(path)
    text = p.read_text()
    if p.suffix in {".yml", ".yaml"}:
        data = yaml.safe_load(text) or {}
    elif p.suffix == ".json":
        data = json.loads(text)
    else:
        raise ValueError(f"Unsupported config format: {p.suffix}")
    return _wrap(data)


def save_config(cfg: Mapping[str, Any], path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.suffix in {".yml", ".yaml"}:
        p.write_text(yaml.safe_dump(dict(cfg), sort_keys=False))
    elif p.suffix == ".json":
        p.write_text(json.dumps(dict(cfg), indent=2))
    else:
        raise ValueError(f"Unsupported config format: {p.suffix}")
