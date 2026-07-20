#!/usr/bin/env python3
"""Render the mission-completion figure and emit TikZ coordinates.

Produces:
  <out>/uav_mission_completion.pdf   matplotlib rendering (sanity check)
  <out>/uav_mission_completion.png
  <out>/tikz_coordinates.txt         paste-ready coords for the LaTeX figure

Usage:
  python scripts/make_figure.py --artifact ./artifact --out ./artifact
"""

from __future__ import annotations

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(__file__), "..")))
from uavbench.config import load_config

_STYLE = {
    "no_def":        dict(color="#c0392b", marker="^", label="No-Def (PX4 baseline, undefended)"),
    "caf_cnn":       dict(color="#e67e22", marker="s", label="CAF-CNN + PX4"),
    "seq2seq_tr":    dict(color="#16a085", marker="D", label="Seq2Seq Tr. + PX4"),
    "ours_m1m4m6m7": dict(color="#2c3e50", marker="o", label="Ours: M1+M4+M6+M7 (Phase A)"),
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact", default="./artifact")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out = args.out or args.artifact
    os.makedirs(out, exist_ok=True)

    cfg = load_config()
    rp = pd.read_csv(os.path.join(args.artifact, "report_points.csv"))

    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    order = ["no_def", "caf_cnn", "seq2seq_tr", "ours_m1m4m6m7"]
    for did in order:
        sub = rp[rp.defense == did].sort_values("js_db")
        if sub.empty:
            continue
        st = _STYLE[did]
        yerr = np.clip(
            [sub.completion_mean - sub.ci_low, sub.ci_high - sub.completion_mean],
            0.0, None,
        )
        ax.errorbar(
            sub.js_db, sub.completion_mean, yerr=yerr,
            color=st["color"], marker=st["marker"], markersize=5,
            linewidth=2, capsize=2, label=st["label"],
        )

    ax.axhline(cfg.regulatory_threshold, ls="--", color="gray", lw=1.2)
    ax.text(0.4, cfg.regulatory_threshold + 0.012,
            "DO-326A regulatory floor: 0.90", fontsize=8, color="dimgray")
    ax.axvspan(20, 40, color="#f1c40f", alpha=0.08)
    ax.text(30, 0.03, "typical EW zone\nJ/S ∈ [20,40] dB",
            fontsize=8, style="italic", ha="center",
            bbox=dict(boxstyle="round", fc="#fef9e7", ec="#d4ac0d", alpha=0.9))

    ax.set_xlabel("Jamming-to-signal ratio, J/S (dB)")
    ax.set_ylabel("Mission completion (fraction)")
    ax.set_xlim(0, 40)
    ax.set_ylim(0, 1.05)
    ax.set_xticks(range(0, 41, 5))
    ax.grid(True, ls="--", color="gray", alpha=0.3)
    ax.legend(loc="lower left", fontsize=8, framealpha=0.9)
    ax.set_title("UAV-EW-Bench-2026 — mission completion vs EW jamming",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "uav_mission_completion.pdf"))
    fig.savefig(os.path.join(out, "uav_mission_completion.png"), dpi=160)

    # -- TikZ paste-ready coordinates ------------------------------------
    lines = []
    for did in order:
        sub = rp[rp.defense == did].sort_values("js_db")
        lines.append(f"%% {_STYLE[did]['label']}")
        lines.append("  coordinates {")
        for _, r in sub.iterrows():
            half = (r.ci_high - r.ci_low) / 2.0
            lines.append(f"  ({r.js_db:g},{r.completion_mean:.2f}) "
                         f"+- ({half:.3f},{half:.3f})")
        lines.append("  };")
        lines.append("")
    with open(os.path.join(out, "tikz_coordinates.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))

    print(f"Figure + TikZ coordinates written to: {os.path.abspath(out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
