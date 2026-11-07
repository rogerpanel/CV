"""Lexical baseline: bag-of-words logistic regression on each episode's text.

Same grouped train/val/test splits as the detector. It has no graph, no
sequence model and no protocol structure, so it shows how much of each
benchmark is solvable from words alone.

    python scripts/lexical_baseline.py --data-root data/raw --out reproducibility/lexical_baseline.json
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score

from mambaguard.data.protocol_traces import EPISODE_LOADERS

FOLDERS = {"a2a_sandbox": "builtin", "mcptox": "MCPTox-Benchmark", "mcp_safetybench": "MCPSafety",
           "agentdojo": "agentdojo", "injecagent": "InjecAgent"}


def text(ep) -> str:
    return " \n ".join(json.dumps(m.payload, sort_keys=True, default=str) for m in ep.messages)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="data/raw")
    ap.add_argument("--out", default="reproducibility/lexical_baseline.json")
    args = ap.parse_args()
    res = {}
    for name, folder in FOLDERS.items():
        root = "builtin" if folder == "builtin" else os.path.join(args.data_root, folder)
        if root != "builtin" and not os.path.isdir(root):
            continue
        sp = {s: EPISODE_LOADERS[name](root, s) for s in ("train", "val", "test")}
        X = {s: [text(e) for e in v] for s, v in sp.items()}
        y = {s: np.array([int(e.is_attack) for e in v]) for s, v in sp.items()}
        vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=50000, sublinear_tf=True)
        Xtr = vec.fit_transform(X["train"])
        clf = LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000).fit(Xtr, y["train"])
        pv = clf.predict_proba(vec.transform(X["val"]))[:, 1]
        pt = clf.predict_proba(vec.transform(X["test"]))[:, 1]
        grid = np.linspace(0.01, 0.99, 99)
        thr = float(grid[np.argmax([f1_score(y["val"], pv >= t, average="macro", zero_division=0) for t in grid])]) \
            if len(set(y["val"])) == 2 else 0.5
        out = {"n_test": int(len(pt)), "benign_test": int((y["test"] == 0).sum()),
               "macro_f1_at_0.5": float(f1_score(y["test"], pt >= 0.5, average="macro", zero_division=0)),
               "macro_f1_val_threshold": float(f1_score(y["test"], pt >= thr, average="macro", zero_division=0)),
               "auroc": float(roc_auc_score(y["test"], pt)) if len(set(y["test"])) == 2 else None}
        res[name] = out
        print(name, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in out.items()}, flush=True)
    with open(args.out, "w") as fh:
        json.dump(res, fh, indent=2)


if __name__ == "__main__":
    main()
