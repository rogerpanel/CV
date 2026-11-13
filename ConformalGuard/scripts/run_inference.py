"""Streaming check of adaptive conformal inference under four shift regimes.

This is a *synthetic* sanity experiment on Gaussian score streams: it exercises
the exact update of Algorithm 1 and verifies Proposition 2 at every horizon.
It is not the deployment-stream experiment of the paper; use
``scripts/make_fig2.py`` with logged scores for that.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from conformalguard.adaptive.gibbs_candes import aci_bound, run_stream
from conformalguard.utils.config import load_config
from conformalguard.utils.logging import get_logger


def _stream(regime: str, n: int, rng: np.random.Generator) -> np.ndarray:
    if regime == "stationary":
        loc = np.zeros(n)
    elif regime == "mild_drift":
        loc = np.linspace(0.0, 0.4, n)
    elif regime == "abrupt_shift":
        loc = np.where(np.arange(n) >= n // 2, 1.2, 0.0)
    elif regime == "recovery":
        loc = np.concatenate([np.linspace(0.0, 1.0, n // 2),
                              np.linspace(1.0, 0.0, n - n // 2)])
    else:
        raise ValueError(regime)
    return rng.normal(loc, 1.0)


def main():
    log = get_logger("conformalguard.online")
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/adaptive.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)

    alpha = float(cfg["adaptive"]["target_alpha"])
    gamma = float(cfg["adaptive"]["gamma"])
    n = int(cfg["streaming"]["steps_per_episode"]) * int(cfg["streaming"]["n_episodes"])
    rng = np.random.default_rng(7)
    cal = rng.normal(0.0, 1.0, 2000)

    out = []
    for regime in cfg["streaming"]["shift_schedule"]:
        scores = _stream(regime, n, rng)
        aci = run_stream(cal, scores, alpha, gamma)
        static = run_stream(cal, scores, alpha, gamma, adaptive=False)
        T = np.arange(1, n + 1)
        band = np.array([aci_bound(int(t), gamma, alpha) for t in T])
        ok = bool(np.all(np.abs(aci["cum_err"] - alpha) <= band + 1e-12))
        log.info(f"regime={regime:13s} ACI err={aci['cum_err'][-1]:.4f} "
                 f"static err={static['cum_err'][-1]:.4f} "
                 f"Prop.2 holds at every T: {ok}")
        out.append({"regime": regime,
                    "aci_miscoverage": float(aci["cum_err"][-1]),
                    "static_miscoverage": float(static["cum_err"][-1]),
                    "prop2_bound_at_T": float(band[-1]),
                    "prop2_holds_at_every_T": ok})

    Path("runs/conformalguard").mkdir(parents=True, exist_ok=True)
    Path("runs/conformalguard/online.json").write_text(json.dumps(out, indent=2))
    log.info("Wrote runs/conformalguard/online.json")


if __name__ == "__main__":
    main()
