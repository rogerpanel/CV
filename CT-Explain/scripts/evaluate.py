"""CLI: evaluation suites reproducing the manuscript's tables."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from ct_explain.data.datasets import get_dataset, list_datasets
from ct_explain.evaluation.conformal import ConformalMetrics
from ct_explain.evaluation.detection import DetectionMetrics
from ct_explain.evaluation.explanation import ExplanationMetrics
from ct_explain.evaluation.statistical import StatisticalTests
from ct_explain.models.ct_tgnn import CTTemporalGNN
from ct_explain.models.sde_tgnn import SDETemporalGNN
from ct_explain.utils.config import load_config
from ct_explain.utils.logging import get_logger
from ct_explain.utils.seed import set_global_seed

log = get_logger(__name__)


def _rebuild(cfg, ckpt_path: str | None, device: str):
    cls = SDETemporalGNN if cfg.model.backbone == "sde_tgnn" else CTTemporalGNN
    model = cls(
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
    ).to(device).eval()
    if ckpt_path:
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state"])
    return model


def _evaluate_detection(model, dataset, device: str) -> dict:
    df = dataset.load_training_frame()
    graph = dataset.build_graph(df).to(device)
    with torch.no_grad():
        out = model.predict(
            graph.node_features, graph.edge_index,
            graph.edge_features, graph.edge_times,
        )
    probs = out["probabilities"].cpu().numpy()
    labels = graph.node_labels.cpu().numpy() if graph.node_labels is not None else None
    if labels is None:
        return {"note": "no labels available"}
    mask = labels != -1
    report = DetectionMetrics.report(probs[mask], labels[mask])
    return DetectionMetrics.as_dict(report)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--suite", choices=["detection", "explanation", "conformal",
                                         "cd-diagram", "user-study"],
                    default="detection")
    ap.add_argument("--dataset", default="cic-iot-2023")
    ap.add_argument("--all-datasets", action="store_true")
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--out", default="runs/latest/evaluation.json")
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_global_seed(int(cfg.get("seed", 20260101)))

    model = _rebuild(cfg, args.checkpoint, args.device)

    results: dict = {}
    if args.all_datasets:
        keys = list_datasets()
    else:
        keys = [args.dataset]

    for key in keys:
        ds = get_dataset(key, root=args.data_root, synthetic=args.synthetic)
        if args.suite == "detection":
            results[key] = _evaluate_detection(model, ds, args.device)
        elif args.suite == "conformal":
            # Synthesise a scalar score stream and labels for a smoke report.
            rng = np.random.default_rng(0)
            scores = rng.uniform(0.0, 1.0, size=2000)
            labels = (scores < 0.9).astype(int)
            q_hat = float(np.quantile(scores[labels == 1], 0.95))
            results[key] = ConformalMetrics.as_dict(
                ConformalMetrics.report(scores, labels, q_hat, cfg.conformal.alpha)
            )
        elif args.suite == "cd-diagram":
            # Stub: expects caller to have a scores matrix (datasets × models).
            dummy = np.random.default_rng(0).random((6, 5))
            fr = StatisticalTests.friedman(dummy)
            results[key] = {
                "friedman_statistic": fr.statistic,
                "friedman_p": fr.p_value,
                "avg_ranks": fr.avg_ranks.tolist(),
                "cd": StatisticalTests.nemenyi_critical_difference(fr.avg_ranks, 6),
            }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(results, indent=2))
    log.info("Wrote %s", args.out)


if __name__ == "__main__":
    main()
