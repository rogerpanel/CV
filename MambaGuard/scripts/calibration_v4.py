"""Calibration (ECE) of every detector, from saved per-episode scores only (no benchmark data needed).

Reads the five-seed study (reproducibility/kaggle_gpu_v4, a folder or zip) -- SSM-GAT scores (`prob`,
`val_prob`) and the lexical baseline's scores on the same episodes (`lex_test`, `lex_val`) -- and, optionally,
the independent-detector study (reproducibility/detectors_v4). For each model and benchmark it reports ECE with
M = 10 equal-width bins, raw and after Platt recalibration fitted on the validation (selection) split, as the
mean over seeds; `ECE = sum_m |B_m|/n * |acc(B_m) - conf(B_m)|`.

    python scripts/calibration_v4.py reproducibility/kaggle_gpu_v4 reproducibility/detectors_v4 \
        --out reproducibility/calibration_v4.json
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parent))
from summarise_detectors import unpack  # noqa: E402

BENCHES = ["a2a_sandbox", "mcptox", "mcp_safetybench", "agentdojo", "injecagent"]
M = 10


def ece(y, p) -> float:
    idx = np.minimum((np.asarray(p) * M).astype(int), M - 1)
    return float(sum((idx == m).sum() / len(y) * abs(y[idx == m].mean() - p[idx == m].mean())
                     for m in range(M) if (idx == m).any()))


def platt(yv, sv, st):
    if len(np.unique(yv)) < 2:
        return st
    m = LogisticRegression(C=1e4, max_iter=1000).fit(np.asarray(sv).reshape(-1, 1), yv)
    return m.predict_proba(np.asarray(st).reshape(-1, 1))[:, 1]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", type=Path, nargs="+")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    rows: dict = defaultdict(lambda: defaultdict(list))   # (model, bench) -> {"raw": [...], "platt": [...]}
    for root in unpack(args.inputs):
        for f in sorted((root / "predictions").glob("main__*.npz")):
            d = np.load(f)
            m = re.match(r"main__(\w+?)__(\w+?)__", f.name)
            bench, kind = m.group(1), m.group(2)
            pairs = []
            if kind == "flat":                      # five-seed study: SSM-GAT and the lexical baseline
                pairs.append(("ssm_gat", d["val_prob"], d["prob"]))
                if "lex_test" in d and "seed0" in f.name:
                    pairs.append(("lexical", d["lex_val"], d["lex_test"]))
            elif kind != "lexical":                 # independent detectors (lexical is already above)
                pairs.append((kind, d["val_prob"], d["prob"]))
            yv, yt = d["val_label"], d["label"]
            for name, sv, st in pairs:
                rows[(name, bench)]["raw"].append(ece(yt, st))
                rows[(name, bench)]["platt"].append(ece(yt, platt(yv, sv, st)))
    report: dict = {}
    models = sorted({k[0] for k in rows}, key=lambda n: ["ssm_gat", "lexical", "deberta_ft", "protectai", "piguard",
                                                          "promptguard2", "llmjudge"].index(n))
    print("| model | " + " | ".join(BENCHES) + " | mean |\n|---|" + "---|" * (len(BENCHES) + 1))
    for name in models:
        report[name] = {}
        for kind in ("raw", "platt"):
            vals = [float(np.mean(rows[(name, b)][kind])) if rows[(name, b)][kind] else None for b in BENCHES]
            report[name][kind] = dict(zip(BENCHES, vals))
            ok = [v for v in vals if v is not None]
            report[name][kind]["mean"] = float(np.mean(ok)) if ok else None
            print(f"| {name} {kind} | " + " | ".join(f"{v:.3f}" if v is not None else "--" for v in vals)
                  + f" | {np.mean(ok):.3f} |")
    if args.out:
        args.out.write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
