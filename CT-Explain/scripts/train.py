"""CLI: train CT-TGNN / SDE-TGNN on one of the six benchmark datasets."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from ct_explain.data.datasets import get_dataset
from ct_explain.models.ct_tgnn import CTTemporalGNN
from ct_explain.models.sde_tgnn import SDETemporalGNN
from ct_explain.training.losses import CTExplainLoss
from ct_explain.training.trainer import CTExplainTrainer
from ct_explain.utils.config import load_config
from ct_explain.utils.logging import get_logger
from ct_explain.utils.seed import set_global_seed

log = get_logger(__name__)


def _build_model(cfg) -> torch.nn.Module:
    cls = SDETemporalGNN if cfg.model.backbone == "sde_tgnn" else CTTemporalGNN
    kwargs = dict(
        node_feat_dim=cfg.model.node_feat_dim,
        edge_feat_dim=cfg.model.edge_feat_dim,
        hidden_dim=cfg.model.hidden_dim,
        num_classes=cfg.model.num_classes,
        num_heads=cfg.model.num_heads,
        time_dim=cfg.model.time_dim,
        dropout=cfg.model.dropout,
        ode_solver=cfg.model.ode_solver,
        rtol=float(cfg.model.rtol),
        atol=float(cfg.model.atol),
        adjoint=cfg.model.adjoint,
        integration_window=cfg.model.integration_window,
    )
    if cls is SDETemporalGNN:
        kwargs["diffusion_scale"] = cfg.model.diffusion_scale
    return cls(**kwargs)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--dataset", default="cic-iot-2023")
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--synthetic", action="store_true",
                    help="Use synthetic tiny frames (smoke test without real data).")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="runs/latest")
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_global_seed(int(cfg.get("seed", 20260101)))

    ds = get_dataset(args.dataset, root=args.data_root, synthetic=args.synthetic)
    df = ds.load_training_frame()
    log.info("Loaded %d rows for %s.", len(df), args.dataset)

    graphs = []
    events_per = int(cfg.training.events_per_graph)
    for k in range(int(cfg.training.batch_graphs_per_epoch)):
        chunk = df.iloc[k * events_per:(k + 1) * events_per]
        if chunk.empty:
            break
        graphs.append(ds.build_graph(chunk))

    model = _build_model(cfg)
    loss_fn = CTExplainLoss(
        use_focal=cfg.training.loss.use_focal,
        focal_gamma=cfg.training.loss.focal_gamma,
        lambda_faith=cfg.training.loss.lambda_faith,
        lambda_cost=cfg.training.loss.lambda_cost,
        lambda_width=cfg.training.loss.lambda_width,
    )
    trainer = CTExplainTrainer(
        model,
        lr=float(cfg.training.lr),
        weight_decay=float(cfg.training.weight_decay),
        grad_clip=float(cfg.training.grad_clip),
        loss_fn=loss_fn,
        device=args.device,
        out_dir=args.out,
    )
    state = trainer.fit(graphs, epochs=int(cfg.training.epochs))
    log.info("Training finished — best val metric: %.4f", state.best_val)
    log.info("Checkpoint: %s", Path(args.out) / "best.pt")


if __name__ == "__main__":
    main()
