"""Aggregate per-seed JSON results into the LaTeX tables of the manuscript.

Walks ``--root`` looking for ``best.pt.eval.json`` files, then computes:
    - Table 2: clean macro-F1 mean ± 95 % bootstrap CI per benchmark.
    - Table 3: PGD-40 macro-F1 per (benchmark, ε) cell.
    - Table 4: certified accuracy at the manuscript's r-grid.

Outputs CSV files under ``<root>/tables/`` and an optional Markdown summary.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd

from src.evaluation.statistical import bootstrap_ci


def _walk(root: Path):
    for p in root.rglob("best.pt.eval.json"):
        bench = p.parts[-3]
        seed = int(p.parts[-2].split("-")[1])
        yield bench, seed, json.loads(p.read_text())


def aggregate(root: Path) -> None:
    out = root / "tables"; out.mkdir(parents=True, exist_ok=True)
    clean = defaultdict(list)
    adv = defaultdict(lambda: defaultdict(list))
    cert = defaultdict(lambda: defaultdict(list))

    for bench, seed, data in _walk(root):
        clean[bench].append(data["clean"]["macro_f1"])
        for k, v in data.get("adversarial", {}).items():
            adv[bench][k].append(v["macro_f1"])
        for r, acc in data.get("certificate", {}).get("certified_accuracy", {}).items():
            cert[bench][float(r)].append(acc)

    rows = []
    for b, vals in clean.items():
        m, lo, hi = bootstrap_ci(np.asarray(vals))
        rows.append({"benchmark": b, "metric": "clean_macro_f1",
                     "mean": m, "ci_lo": lo, "ci_hi": hi, "n_seeds": len(vals)})
    pd.DataFrame(rows).to_csv(out / "table2_clean.csv", index=False)

    rows = []
    for b, atks in adv.items():
        for atk, vals in atks.items():
            m, lo, hi = bootstrap_ci(np.asarray(vals))
            rows.append({"benchmark": b, "attack": atk,
                         "mean_macro_f1": m, "ci_lo": lo, "ci_hi": hi})
    pd.DataFrame(rows).to_csv(out / "table3_adversarial.csv", index=False)

    rows = []
    for b, radii in cert.items():
        for r, vals in radii.items():
            rows.append({"benchmark": b, "radius": r,
                         "certified_acc": float(np.mean(vals))})
    pd.DataFrame(rows).to_csv(out / "table4_certified.csv", index=False)

    print(f"Wrote 3 tables under {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True, type=Path)
    args = p.parse_args()
    aggregate(args.root)
