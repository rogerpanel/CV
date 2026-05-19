"""Single-seed training loop.

Invoked as ``python -m src.training.train --config configs/sode_guard.yaml``;
will loop over ``experiment.seeds`` and over ``data.benchmarks`` independently,
writing checkpoints under ``experiment.output_dir/<benchmark>/seed-<n>/``.
"""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

from ..models.sode_guard import SODEGuard, SODEGuardConfig
from ..data.registry import get_loader, DATASET_REGISTRY
from ..utils import set_global_seed, load_config, build_logger
from ..utils.metrics import aggregate_metrics
from .loss import CrossEntropyWithAC


def build_model(cfg) -> SODEGuard:
    m = SODEGuard(SODEGuardConfig(
        feature_dim=cfg.data.feature_dim,
        hidden_dim=cfg.encoder.hidden_dim,
        num_classes=cfg.data.label_space,
        drift_hidden=cfg.sde.drift.hidden_dim,
        diff_hidden=cfg.sde.diffusion.hidden_dim,
        drift_layers=cfg.sde.drift.num_layers,
        diff_layers=cfg.sde.diffusion.num_layers,
        noise_dim=cfg.sde.diffusion.noise_dim,
        horizon=cfg.sde.integrator.horizon,
        dt=cfg.sde.integrator.dt,
        ellipticity_floor=cfg.sde.diffusion.ellipticity_floor,
        spectral_norm=bool(cfg.regularizers.spectral_norm),
        mc_paths_eval=cfg.sde.monte_carlo.eval_paths,
        encoder_layers=cfg.encoder.num_layers,
        encoder_dropout=cfg.encoder.dropout,
        activation=cfg.sde.drift.activation,
        virtual_brownian=cfg.sde.integrator.virtual_brownian_tree,
    ))
    return m


def build_optimizer(model, cfg):
    return Adam(model.parameters(),
                lr=cfg.optim.lr,
                betas=tuple(cfg.optim.betas),
                weight_decay=cfg.optim.weight_decay)


def build_scheduler(opt, cfg, steps_per_epoch: int):
    return CosineAnnealingLR(opt, T_max=cfg.training.epochs * steps_per_epoch)


@torch.no_grad()
def evaluate(model, loader, device) -> dict[str, float]:
    model.eval()
    all_y, all_p, all_probs = [], [], []
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        probs = model.forward_mc(x)
        all_probs.append(probs.cpu().numpy())
        all_p.append(probs.argmax(-1).cpu().numpy())
        all_y.append(y.cpu().numpy())
    y_true = np.concatenate(all_y); y_pred = np.concatenate(all_p)
    return aggregate_metrics(y_true, y_pred, np.concatenate(all_probs))


def train_one_run(cfg, benchmark: str, seed: int) -> dict:
    out_dir = Path(cfg.experiment.output_dir) / benchmark / f"seed-{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    log = build_logger(f"sode_guard.{benchmark}.s{seed}", out_dir)
    set_global_seed(seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info("Loading dataset %s on %s", benchmark, device)
    train_loader, stats = get_loader(benchmark, split="train",
                                     batch_size=cfg.training.batch_size,
                                     num_workers=cfg.experiment.num_workers,
                                     seed=seed, return_stats=True)
    val_loader = get_loader(benchmark, split="val",
                            batch_size=cfg.training.batch_size,
                            num_workers=cfg.experiment.num_workers, seed=seed)

    cfg.data.label_space = max(cfg.data.label_space, stats["num_classes"])
    model = build_model(cfg).to(device)
    opt = build_optimizer(model, cfg)
    sched = build_scheduler(opt, cfg, len(train_loader))
    loss_fn = CrossEntropyWithAC(
        ac_weight=cfg.regularizers.anti_concentration.weight,
        chaos_degree=cfg.regularizers.anti_concentration.chaos_degree,
        beta_grid=tuple(cfg.regularizers.anti_concentration.beta_grid),
        label_smoothing=cfg.training.get("label_smoothing", 0.0),
        n_ac_paths=max(2, cfg.sde.monte_carlo.eval_paths // 2),
    )

    scaler = torch.cuda.amp.GradScaler(enabled=cfg.experiment.amp and device == "cuda")
    best = {"macro_f1": -1.0}
    for epoch in range(cfg.training.epochs):
        model.train()
        t0 = time.time()
        for step, (x, y) in enumerate(train_loader):
            x = x.to(device, non_blocking=True); y = y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
                loss, parts = loss_fn(model, x, y)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.optim.grad_clip)
            scaler.step(opt); scaler.update(); sched.step()
            if step % cfg.experiment.log_every == 0:
                log.info("epoch=%d step=%d/%d loss=%.4f ce=%.4f ac=%.4f lr=%.2e",
                         epoch, step, len(train_loader),
                         parts["loss"], parts["ce"], parts["ac"], sched.get_last_lr()[0])
        log.info("epoch=%d elapsed=%.1fs", epoch, time.time() - t0)

        val = evaluate(model, val_loader, device)
        log.info("val %s", json.dumps(val))
        if val["macro_f1"] > best["macro_f1"]:
            best = val | {"epoch": epoch}
            torch.save({"model": model.state_dict(), "cfg": dict(cfg),
                        "epoch": epoch, "val": val},
                       out_dir / "best.pt")

    (out_dir / "best_metrics.json").write_text(json.dumps(best, indent=2))
    return best


def _cli():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--benchmark", default=None,
                   help="Override cfg.data.benchmarks to a single benchmark")
    p.add_argument("--seed", type=int, default=None,
                   help="Override cfg.experiment.seeds to a single seed")
    args = p.parse_args()

    cfg = load_config(args.config)
    benches = [args.benchmark] if args.benchmark else cfg.data.benchmarks
    seeds = [args.seed] if args.seed is not None else cfg.experiment.seeds

    summary = {}
    for b in benches:
        if b not in DATASET_REGISTRY:
            raise SystemExit(f"Unknown benchmark '{b}'")
        for s in seeds:
            summary[f"{b}:{s}"] = train_one_run(cfg, b, s)
    out = Path(cfg.experiment.output_dir) / "summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    _cli()
