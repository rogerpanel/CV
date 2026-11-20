"""Score a testbed flows.csv against saved v2 models.

Reuses the project's model code and the saved run layout
(OUT/meta.json, OUT/seed_<s>/{data.npz, <model>.pt|.pkl}). It aligns the testbed
flows to the run's feature schema by name, applies the run's training-set
z-scaling, and runs the deployed multi-path predictor (fixed Brownian paths) for
the SDE models. The same entry point serves Track A (models trained on the
testbed corpus) and Track B (the saved ICS3D models, for transfer).

    python scoring/score_models.py --run scoring/models/testbed --flows runs/clean/flows.csv \
        --models sode_at mlp_at ode_at lgbm sode_rs_c2 --out runs/clean/scores_models
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]       # SODE-Guard/
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import v2_run as V                                # noqa: E402


def _norm(name: str) -> str:
    return name.strip().lower().replace(" ", "_").replace("/", "_")


def align(flows: pd.DataFrame, raw_features: list[str]) -> np.ndarray:
    """Map testbed columns onto the run's raw feature order by normalised name,
    filling absent columns with NaN (imputed to the training median downstream)."""
    cols = {_norm(c): c for c in flows.columns}
    X = np.full((len(flows), len(raw_features)), np.nan, dtype=np.float64)
    missing = []
    for j, f in enumerate(raw_features):
        c = cols.get(_norm(f))
        if c is None:
            missing.append(f)
        else:
            X[:, j] = pd.to_numeric(flows[c], errors="coerce").to_numpy()
    if missing:
        print(f"[align] {len(missing)}/{len(raw_features)} features absent in testbed flows "
              f"(filled with training median): {missing[:6]}{'...' if len(missing) > 6 else ''}")
    return X


class RunArgs:
    """Minimal stand-in for the argparse namespace v2_run helpers expect."""
    def __init__(self, run: Path, split="random", clip=10.0):
        self.out = str(run); self.split = split; self.clip = clip
        self.benign_class = None; self.data_seed = 0


def scale_like(run: Path, seed: int, Xraw: np.ndarray):
    """Apply the seed's training median impute + z-score to aligned raw features."""
    names, mu, sdv, _ = V.raw_scaling(RunArgs(run), seed)   # kept-column stats
    cols = np.load(run / f"seed_{seed}" / "data.npz")["cols"]
    med = np.nanmedian(Xraw, axis=0)
    Xraw = np.where(np.isnan(Xraw), med, Xraw)
    Xk = Xraw[:, cols]
    X = np.clip((Xk - mu) / np.where(sdv > 0, sdv, 1.0), -10, 10)
    return torch.tensor(X, dtype=torch.float32)


def score(run: Path, flows: pd.DataFrame, models: list[str], seeds: list[int]):
    meta = json.loads((run / "meta.json").read_text())
    classes = meta["classes"]
    Xraw = align(flows, meta["raw_features"])
    out = {}
    for seed in seeds:
        args = RunArgs(run)
        D = V.load_split(args, seed)                 # for p, K, prior
        X = scale_like(run, seed, Xraw)
        for name in models:
            if not V.trained(name, V.seed_dir(run, seed)):
                continue
            pred = predict(args, D, run, seed, name, X)
            out.setdefault(name, {})[seed] = pred
    return classes, out


def predict(args, D, run: Path, seed: int, name: str, X: torch.Tensor) -> np.ndarray:
    sd = V.seed_dir(run, seed)
    model = V.load_any(name, D, sd)
    with torch.no_grad():
        if name in V.TREES:
            logits = model(X)
        else:
            arch = V.spec_of(name)["arch"]
            kind = {"sode": "sde", "sdetgnn": "sde", "ode": "ode", "mlp": "mlp"}[arch]
            pf = V.predictor(model, kind, mode="fixed", n_paths=8)
            logits = torch.cat([pf(X[i:i + 512]) for i in range(0, len(X), 512)])
    return logits.argmax(1).cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="saved run folder (meta.json + seed_*/)")
    ap.add_argument("--flows", required=True)
    ap.add_argument("--models", nargs="+",
                    default=["sode_at", "mlp_at", "ode_at", "sode", "lgbm", "xgb", "rf",
                             "sode_rs_c2", "mlp_rs_c2"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    flows = pd.read_csv(a.flows, low_memory=False)
    classes, preds = score(Path(a.run), flows, a.models, a.seeds)
    outd = Path(a.out); outd.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(outd / "preds.npz",
                        classes=np.array(classes),
                        **{f"{m}__seed{s}": p for m, d in preds.items() for s, p in d.items()})
    (outd / "classes.json").write_text(json.dumps(classes, indent=2))
    print(f"scored {len(flows)} flows with {len(preds)} models -> {outd/'preds.npz'}")


if __name__ == "__main__":
    main()
