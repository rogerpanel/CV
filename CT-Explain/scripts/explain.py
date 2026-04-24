"""CLI: produce all four explanations + calibrated intervals for one alert."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from ct_explain.data.datasets import get_dataset
from ct_explain.models.ct_tgnn import CTTemporalGNN
from ct_explain.models.sde_tgnn import SDETemporalGNN
from ct_explain.soc.human_ai import HumanAICollaboration
from ct_explain.utils.config import load_config
from ct_explain.utils.logging import get_logger
from ct_explain.utils.seed import set_global_seed

log = get_logger(__name__)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--dataset", default="cic-iot-2023")
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--target", type=int, default=0,
                    help="Target node index (alert) to explain.")
    ap.add_argument("--render-dir", default=None)
    ap.add_argument("--out", default="runs/latest/explanation.json")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_global_seed(int(cfg.get("seed", 20260101)))

    cls = SDETemporalGNN if cfg.model.backbone == "sde_tgnn" else CTTemporalGNN
    model = cls(
        node_feat_dim=cfg.model.node_feat_dim,
        edge_feat_dim=cfg.model.edge_feat_dim,
        hidden_dim=cfg.model.hidden_dim,
        num_classes=cfg.model.num_classes,
    ).to(args.device).eval()
    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location=args.device)
        model.load_state_dict(ckpt["model_state"])

    ds = get_dataset(args.dataset, root=args.data_root, synthetic=args.synthetic)
    df = ds.load_training_frame().head(cfg.training.events_per_graph)
    graph = ds.build_graph(df).to(args.device)

    class_names = [f"class_{i}" for i in range(cfg.model.num_classes)]
    collab = HumanAICollaboration(model, class_names=class_names)
    alert = collab.triage(graph=graph, target_node=args.target,
                          alert_id=f"alert_{args.target}", source=args.dataset)
    panel = collab.investigate(graph=graph, alert=alert, render_dir=args.render_dir)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(panel, indent=2, default=str))
    log.info("Wrote %s", args.out)


if __name__ == "__main__":
    main()
