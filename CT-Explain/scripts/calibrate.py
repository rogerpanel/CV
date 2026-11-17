"""CLI: calibrate ConformalGuard on a held-out set."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from ct_explain.conformal.conformal_guard import ConformalGuard
from ct_explain.conformal.execution_graph import (
    DynamicExecutionGraph,
    EdgeKind,
    NodeKind,
)
from ct_explain.utils.config import load_config
from ct_explain.utils.logging import get_logger
from ct_explain.utils.seed import set_global_seed

log = get_logger(__name__)


def _demo_calibration(n: int = 200):
    """Synthetic calibration set: 200 workflows, each with a single action.

    Use a real execution graph + analyst-labelled safety flags in
    production — see `ct_explain.conformal.execution_graph`.
    """
    import random
    random.seed(0)
    calib = []
    for i in range(n):
        g = DynamicExecutionGraph()
        action_id = f"a_{i}"
        g.add_node(action_id, NodeKind.ACTION, timestamp=i, attrs={"agent_id": "agent0"})
        # Add 2 neighbours for cascade risk.
        for j, kind in enumerate([NodeKind.TOOL, NodeKind.OBSERVATION]):
            nid = f"n_{i}_{j}"
            g.add_node(nid, kind, timestamp=i - 1, attrs={})
            g.add_edge(action_id, nid, EdgeKind.INVOKES, timestamp=i)
        safe_base = random.random()
        p_safe = {action_id: safe_base, **{
            f"n_{i}_{j}": max(0.0, safe_base + random.gauss(0.0, 0.05))
            for j in range(2)
        }}
        y = 1 if safe_base > 0.2 else 0
        calib.append((action_id, g, (lambda m=p_safe: (lambda nid: m.get(nid, 1.0)))(), y))
    return calib


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--alpha", type=float, default=None)
    ap.add_argument("--out", default="runs/latest/conformal.json")
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_global_seed(int(cfg.get("seed", 20260101)))
    alpha = float(args.alpha if args.alpha is not None else cfg.conformal.alpha)

    guard = ConformalGuard(
        alpha=alpha,
        k_hops=int(cfg.conformal.k_hops),
        lambda_cascade=float(cfg.conformal.lambda_cascade),
        martingale_lr=float(cfg.conformal.martingale_lr),
    )
    calib = _demo_calibration()
    q_hat = guard.calibrate(calib)
    log.info("Calibrated ConformalGuard: q_hat = %.4f (n=%d, α=%.3f)",
             q_hat, len(calib), alpha)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({
        "q_hat": q_hat,
        "alpha": alpha,
        "n_calibration": len(calib),
        "k_hops": cfg.conformal.k_hops,
        "lambda_cascade": cfg.conformal.lambda_cascade,
    }, indent=2))


if __name__ == "__main__":
    main()
