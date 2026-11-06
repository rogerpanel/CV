"""Train MambaGuard end-to-end.

Usage:
    python -m scripts.train --config configs/default.yaml --dataset iis3d \
        --data-root data/raw/iis3d \
        --output-dir outputs/run_$(date +%Y%m%d_%H%M%S) --seed 42

Writes ``config.yaml``, ``metrics.json``, ``train.log`` and checkpoints under
``--output-dir``.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

try:
    from omegaconf import OmegaConf  # type: ignore

    _HAS_OMEGACONF = True
except Exception:  # pragma: no cover
    _HAS_OMEGACONF = False

import yaml


def _read_yaml(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def _load_config(path: str) -> dict[str, Any]:
    """Load a config and compose its Hydra-style ``defaults`` list.

    ``defaults: [{model: mambaguard}, ...]`` pulls ``<config dir>/model/mambaguard.yaml``
    into ``cfg["model"]``; keys in the top-level file override composed ones.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found: {path}")
    top = _read_yaml(path)
    base = os.path.dirname(os.path.abspath(path))
    cfg: dict[str, Any] = {}
    for entry in top.pop("defaults", []) or []:
        if isinstance(entry, dict):
            for group, name in entry.items():
                sub = os.path.join(base, group, f"{name}.yaml")
                if os.path.exists(sub):
                    cfg[group] = _read_yaml(sub)
    for k, v in top.items():
        if isinstance(v, dict) and isinstance(cfg.get(k), dict):
            cfg[k].update(v)
        else:
            cfg[k] = v
    return cfg


def _save_config(cfg: dict[str, Any], path: Path) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(cfg, fh, sort_keys=False)


def _setup_logger(log_path: Path) -> logging.Logger:
    logger = logging.getLogger("mambaguard.train")
    logger.setLevel(logging.INFO)
    for h in list(logger.handlers):
        logger.removeHandler(h)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    fh = logging.FileHandler(log_path)
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


_MODEL_KEY_MAP = {"d_metadata": "d_mu", "n_layers": "n_blocks", "n_heads": "gat_heads"}


def _model_kwargs(cfg: dict[str, Any]) -> dict[str, Any]:
    """Map the model YAML onto :class:`MambaGuardConfig` fields."""
    from dataclasses import fields

    from mambaguard.models import MambaGuardConfig

    known = {f.name for f in fields(MambaGuardConfig)}
    out: dict[str, Any] = {}
    for k, v in (cfg.get("model", {}) or {}).items():
        k = _MODEL_KEY_MAP.get(k, k)
        if k in known:
            out[k] = v
    graph = (cfg.get("model", {}) or {}).get("graph") or {}
    if "n_layers" in graph:
        out["n_gat_layers"] = int(graph["n_layers"]) if graph.get("enabled", True) else 0
    return out


def _training_cfg(cfg: dict[str, Any]) -> dict[str, Any]:
    tcfg = dict(cfg.get("training", {}) or {})
    # Map keys to TrainerConfig fields used by MambaGuardTrainer._cfg_get.
    out: dict[str, Any] = {}
    if "epochs" in tcfg:
        out["epochs"] = int(tcfg["epochs"])
    if "grad_clip" in tcfg:
        out["grad_clip"] = float(tcfg["grad_clip"])
    if "batch_size" in tcfg:
        out["batch_size"] = int(tcfg["batch_size"])
    if "amp_dtype" in tcfg:
        out["amp_dtype"] = str(tcfg["amp_dtype"])
    fl = tcfg.get("focal_loss", {}) or {}
    if "alpha" in fl:
        out["focal_alpha"] = float(fl["alpha"])
    if "gamma" in fl:
        out["focal_gamma"] = float(fl["gamma"])
    lr_reg = tcfg.get("lipschitz_reg", {}) or {}
    if "lambda_L" in lr_reg:
        out["lipschitz_lambda"] = float(lr_reg["lambda_L"])
    opt = tcfg.get("optimizer", {}) or {}
    if "lr" in opt:
        out["lr"] = float(opt["lr"])
    if "weight_decay" in opt:
        out["weight_decay"] = float(opt["weight_decay"])
    out.setdefault("num_classes", int(cfg.get("model", {}).get("num_classes", 34)))
    return out


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train MambaGuard.")
    p.add_argument("--config", type=str, default="configs/default.yaml")
    p.add_argument("--dataset", type=str, default="iis3d")
    p.add_argument(
        "--data-root",
        type=str,
        default=None,
        help="Filesystem root containing the extracted dataset.",
    )
    p.add_argument(
        "--output-dir",
        type=str,
        default=f"outputs/run_{time.strftime('%Y%m%d_%H%M%S')}",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--epochs", type=int, default=None, help="override training.epochs")
    p.add_argument("--batch-size", type=int, default=None, help="override training.batch_size")
    p.add_argument("--embedder", choices=["minilm", "hash"], default="minilm",
                   help="payload encoder: the paper's frozen MiniLM, or 'hash' for offline smoke tests only")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = _setup_logger(out_dir / "train.log")
    logger.info("Loading config from %s", args.config)
    cfg = _load_config(args.config)
    cfg["dataset"] = args.dataset
    cfg["seed"] = args.seed
    if args.epochs is not None:
        cfg.setdefault("training", {})["epochs"] = args.epochs
    if args.batch_size is not None:
        cfg.setdefault("training", {})["batch_size"] = args.batch_size
    _save_config(cfg, out_dir / "config.yaml")

    from mambaguard.data import (  # noqa: WPS433 - lazy
        DATASET_REGISTRY,
        ProtocolMessageDataset,
        protocol_collate,
    )
    from mambaguard.models import MambaGuard
    from mambaguard.training import MambaGuardTrainer
    from mambaguard.utils.seed import set_seed

    set_seed(args.seed)
    if args.dataset not in DATASET_REGISTRY:
        raise KeyError(
            f"Unknown dataset '{args.dataset}'. "
            f"Known: {sorted(DATASET_REGISTRY)}"
        )
    if args.data_root is None:
        raise SystemExit(
            "--data-root is required (path to the extracted dataset root)."
        )
    loader_fn = DATASET_REGISTRY[args.dataset]
    logger.info("Loading dataset %s from %s", args.dataset, args.data_root)
    # Every registered loader implements its own leak-free train/val/test split.
    train_msgs = list(loader_fn(args.data_root, split="train"))
    val_msgs = list(loader_fn(args.data_root, split="val"))

    from mambaguard.data.canonicalize import DEFAULT_METADATA_SCHEMA

    model_cfg = cfg.setdefault("model", {})
    model_cfg["d_metadata"] = len(DEFAULT_METADATA_SCHEMA)  # width produced by protocol_collate
    model_cfg["edge_dim"] = 5 + len(DEFAULT_METADATA_SCHEMA)  # edge type one-hot + metadata
    labels = {m.label for m in train_msgs + val_msgs if m.label is not None}
    if labels and labels <= {0, 1}:
        model_cfg["num_classes"] = 2  # binary attack/benign benchmarks
    logger.info("Building model (num_classes=%s)", model_cfg.get("num_classes"))
    model = MambaGuard.from_config(_model_kwargs(cfg))
    from mambaguard.data.canonicalize import make_canonicaliser

    embedder = make_canonicaliser(args.embedder)
    if args.embedder == "hash":
        logger.warning("Using the hashing embedder: smoke test only, do not report these numbers.")
    train_ds = ProtocolMessageDataset(train_msgs, canonicaliser=embedder, compute_embeddings=True)
    val_ds = ProtocolMessageDataset(val_msgs, canonicaliser=embedder, compute_embeddings=True)

    import torch
    from torch.utils.data import DataLoader

    bs = int(cfg.get("training", {}).get("batch_size", 256))
    train_loader = DataLoader(
        train_ds,
        batch_size=bs,
        shuffle=True,
        collate_fn=protocol_collate,
        num_workers=int(cfg.get("training", {}).get("num_workers", 0)) if torch.cuda.is_available() else 0,
    )
    val_loader = DataLoader(
        val_ds, batch_size=bs, shuffle=False, collate_fn=protocol_collate
    )

    trainer_cfg = _training_cfg(cfg)
    trainer_cfg["out_dir"] = str(out_dir / "checkpoints")
    device = args.device if torch.cuda.is_available() else "cpu"
    trainer_cfg["model_config"] = _model_kwargs(cfg)
    trainer_cfg["embedder"] = args.embedder
    trainer = MambaGuardTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        cfg=trainer_cfg,
        device=device,
    )

    if args.resume:
        logger.info("Resuming from %s", args.resume)
        ckpt = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(ckpt["model"])

    logger.info("Starting training")
    state = trainer.train()
    metrics = {
        "epoch": state.epoch,
        "global_step": state.global_step,
        "best_metric": state.best_metric,
        "best_epoch": state.best_epoch,
    }
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2, default=str)
    logger.info("Training complete. State: %s", metrics)
    return 0


if __name__ == "__main__":
    sys.exit(main())
