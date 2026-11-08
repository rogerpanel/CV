"""Calibration (ECE and reliability bins) of the detector and the lexical baseline.

Uses the recorded v2 GPU predictions (graph read-out MambaGuard, seeds 0-2),
which are the only run whose per-episode scores are stored in the repository,
and recomputes the lexical baseline's probabilities on exactly the same
episodes. For each model it reports

* raw ECE: the model's own probability;
* Platt ECE: the probability after a one-feature logistic regression fitted on
  the validation split (test data is never used for fitting).

ECE uses M = 10 equal-width bins: ECE = sum_m (|B_m| / n) |acc(B_m) - conf(B_m)|,
where conf is the mean predicted attack probability in bin m and acc the
fraction of attacks in it. Reliability bins are pooled over benchmarks and seeds.

    python scripts/calibration_v2.py reproducibility/kaggle_gpu_v2 --data-root data/raw \
        --out reproducibility/calibration_v2.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import expert_mixture as EM  # noqa: E402  (loads the runner as EM.R)

BENCHES = ["a2a_sandbox", "mcptox", "mcp_safetybench", "agentdojo", "injecagent"]
M = 10


def platt(yv: np.ndarray, sv: np.ndarray):
    if len(np.unique(yv)) < 2:
        return lambda s: s
    m = LogisticRegression(C=1e4, max_iter=1000).fit(sv.reshape(-1, 1), yv)
    return lambda s: m.predict_proba(np.asarray(s).reshape(-1, 1))[:, 1]


def bins(y: np.ndarray, p: np.ndarray) -> list[dict]:
    idx = np.minimum((p * M).astype(int), M - 1)
    out = []
    for m in range(M):
        sel = idx == m
        out.append({"n": int(sel.sum()), "conf": float(p[sel].mean()) if sel.any() else None,
                    "acc": float(y[sel].mean()) if sel.any() else None})
    return out


def ece(y: np.ndarray, p: np.ndarray) -> float:
    b = bins(y, p)
    return float(sum(x["n"] / len(y) * abs(x["acc"] - x["conf"]) for x in b if x["n"]))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results", type=Path)
    ap.add_argument("--data-root", default="data/raw")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    from types import SimpleNamespace

    rargs = SimpleNamespace(data_root=args.data_root, a2a_pairs=200, heldout_a2a_pairs=100, smoke=False,
                            device="cpu", agentdojo_model=None)
    data = EM.R.Data(rargs, None)
    pred = args.results / "predictions"
    report: dict = {"bins_M": M, "per_benchmark": {}, "pooled": {}}
    pooled = {k: ([], []) for k in ("det_raw", "det_platt", "lex_raw", "lex_platt")}
    for b in BENCHES:
        files = sorted(pred.glob(f"main__{b}__full__*__all__seed*.npz"))
        if not files:
            continue
        eps = {s: data.split(b, s) for s in ("train", "val", "test")}
        lv, lt = EM.lexical_probs(eps)
        yv = np.array([int(e.is_attack) for e in eps["val"]])
        yt = np.array([int(e.is_attack) for e in eps["test"]])
        row = {k: [] for k in pooled}
        for f in files:
            d = np.load(f)
            assert (d["label"] == yt).all() and (d["val_label"] == yv).all(), f.name
            scores = {"det_raw": d["prob"], "det_platt": platt(yv, d["val_prob"])(d["prob"]),
                      "lex_raw": lt, "lex_platt": platt(yv, lv)(lt)}
            for k, s in scores.items():
                row[k].append(ece(yt, s))
                pooled[k][0].append(yt)
                pooled[k][1].append(s)
        report["per_benchmark"][b] = {k: {"mean": float(np.mean(v)), "sd": float(np.std(v, ddof=1)) if len(v) > 1 else 0.0}
                                      for k, v in row.items()}
        report["per_benchmark"][b]["n_test"] = int(len(yt))
        print(b, {k: round(v["mean"], 3) for k, v in report["per_benchmark"][b].items() if isinstance(v, dict)}, flush=True)
    for k, (ys, ss) in pooled.items():
        y, s = np.concatenate(ys), np.concatenate(ss)
        report["pooled"][k] = {"ece": ece(y, s), "bins": bins(y, s), "n": int(len(y))}
        print("pooled", k, round(report["pooled"][k]["ece"], 3))
    if args.out:
        args.out.write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
