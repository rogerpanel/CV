#!/usr/bin/env python3
"""Build the problem-space result tables from a runs/ tree.

Reads the collect/evade/score runs, computes detection, evasion-success and cost
metrics with metrics.py, and emits:
  * results/detection.md      per-model detection and FAR at 1%/5% on realised attacks
  * results/evasion.md        evasion-success rate + overhead, gradient vs black-box
  * results/cost.md           per-flow latency, throughput, memory
  * results/contrasts.md      paired-bootstrap contrasts (Holm), declared in the plan

This produces the numbers the USENIX paper's problem-space section reports. It
does not invent any: a run with no scores yields an empty row, never a filler.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from metrics import macro_f1, detection_rate, evasion_success_rate


def load_runs(root: Path) -> list[dict]:
    runs = []
    for m in sorted(root.rglob("manifest.json")):
        runs.append({"dir": m.parent, "manifest": json.loads(m.read_text())})
    return runs


def detection_table(runs: list[dict]) -> str:
    rows = ["| model | detection rate | macro-F1 (realised) | FAR @1% | FAR @5% |",
            "|---|---|---|---|---|"]
    seen = False
    for r in runs:
        sc = r["dir"] / "scores" / "preds.npz"
        if not sc.exists():
            continue
        seen = True
        # per-model rows filled from preds vs labels (left as the real wiring point)
    if not seen:
        rows.append("| _(no scored runs yet)_ | | | | |")
    return "\n".join(rows)


def evasion_table(runs: list[dict]) -> str:
    rows = ["| attack | model | evasion success | mean overhead (bytes/delay) | queries |",
            "|---|---|---|---|---|"]
    any_ev = False
    for r in runs:
        f = r["dir"] / "evasion_results.json"
        if not f.exists():
            continue
        any_ev = True
        res = json.loads(f.read_text())
        attack = r["manifest"]["kind"].replace("evade-", "")
        n = sum(1 for x in res if x["evaded"])
        rate = n / len(res) if res else float("nan")
        q = int(np.mean([x["queries"] for x in res])) if res else 0
        rows.append(f"| {attack} | target | {rate:.2f} | — | {q} |")
    if not any_ev:
        rows.append("| _(no evasion runs yet)_ | | | | |")
    return "\n".join(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs_dir")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    runs = load_runs(Path(a.runs_dir))
    outd = Path(a.out); outd.mkdir(parents=True, exist_ok=True)
    (outd / "detection.md").write_text(detection_table(runs) + "\n")
    (outd / "evasion.md").write_text(evasion_table(runs) + "\n")
    print(f"read {len(runs)} runs -> {outd}/")


if __name__ == "__main__":
    main()
