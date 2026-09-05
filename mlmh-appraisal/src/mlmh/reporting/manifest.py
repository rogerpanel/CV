"""Every run writes a manifest: git SHA, config hash, seeds, versions, checksums."""
from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

import yaml

PACKAGES = ["numpy", "pandas", "scikit-learn", "xgboost", "imbalanced-learn", "scipy", "matplotlib", "torch"]


def git_sha(cwd: Path | None = None) -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=cwd, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def git_dirty(cwd: Path | None = None) -> bool:
    try:
        out = subprocess.check_output(["git", "status", "--porcelain"], cwd=cwd, text=True, stderr=subprocess.DEVNULL)
        return bool(out.strip())
    except Exception:
        return True


def config_hash(cfg: dict) -> str:
    return hashlib.sha256(yaml.safe_dump(cfg, sort_keys=True).encode()).hexdigest()[:16]


def versions() -> dict[str, str]:
    v = {}
    for p in PACKAGES:
        try:
            v[p] = metadata.version(p)
        except metadata.PackageNotFoundError:
            v[p] = "not installed"
    return v


def write_manifest(out_dir: Path, cfg: dict, extra: dict | None = None, checksums_path: Path | None = None) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    checksums = {}
    if checksums_path and Path(checksums_path).exists():
        checksums = json.loads(Path(checksums_path).read_text())
    m = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_sha": git_sha(),
        "git_dirty": git_dirty(),
        "config_hash": config_hash(cfg),
        "config": cfg,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "package_versions": versions(),
        "dataset_checksums": checksums,
        "synthetic": bool(cfg.get("synthetic", False)),
    }
    if extra:
        m.update(extra)
    path = out_dir / "manifest.json"
    path.write_text(json.dumps(m, indent=2, default=str))
    return path
