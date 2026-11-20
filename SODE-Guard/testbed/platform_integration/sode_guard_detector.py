"""SODE-Guard as a detector branch for the RobustIDPS.ai platform.

The platform's own methods (CT-TGNN, MambaShield, ...) are branches of a single
surrogate network. SODE-Guard is added here as a *real* extra detector: it loads
the trained neural-SDE model from a saved v2 run and scores the platform's
uploaded flows through the deployed multi-path predictor, returning a per-flow
label, a benign-class confidence and a severity in the platform's own response
shape. It sits alongside the built-in methods in /api/model_info and answers at
its own endpoint (see platform_router.py), so the Live Monitor and the testbed
can query it exactly like the others.

Requirements inside the backend image (see INTEGRATION.md):
  * env SODE_GUARD_ROOT -> the SODE-Guard checkout (for src/ and scripts/v2_run.py)
  * env SODE_GUARD_RUN  -> a saved run folder (meta.json, full.npz, seed_*/weights)
  * env SODE_GUARD_MODEL (default "sode_at"); "sode_rs_c2" for the certified head
"""
from __future__ import annotations

import json
import os
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import torch


def _wire_paths() -> Path:
    root = Path(os.environ.get("SODE_GUARD_ROOT", "/opt/sode-guard"))
    for p in (root, root / "scripts", root / "testbed" / "scoring"):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
    return root


class SODEGuardDetector:
    """Loads once, scores many. Thread-safe for FastAPI (torch inference, no grad)."""

    def __init__(self, run: str | None = None, model: str | None = None,
                 seeds: tuple[int, ...] = (0,)):
        _wire_paths()
        import score_models as SM            # from testbed/scoring
        self._SM = SM
        self.run = Path(run or os.environ["SODE_GUARD_RUN"])
        self.model = model or os.environ.get("SODE_GUARD_MODEL", "sode_at")
        self.seeds = seeds
        self.meta = json.loads((self.run / "meta.json").read_text())
        self.classes = self.meta["classes"]
        self.benign = next((c for c in self.classes
                            if c.lower().startswith("benign") or c.lower() == "normal"),
                           self.classes[0])
        self.ready = self._probe()

    def _probe(self) -> bool:
        try:
            sd = self._SM.V.seed_dir(self.run, self.seeds[0])
            return self._SM.V.trained(self.model, sd)
        except Exception as e:                # not deployed yet; report degraded
            print(f"[sode-guard] weights not found ({e}); detector is in degraded mode")
            return False

    def info(self) -> dict:
        return {"name": "SODE-Guard", "kind": "neural-SDE (certified)",
                "model": self.model, "n_classes": len(self.classes),
                "classes": self.classes, "ready": self.ready,
                "certified": self.model.startswith("sode_rs")}

    def predict(self, df: pd.DataFrame) -> list[dict]:
        if not self.ready:
            raise RuntimeError("SODE-Guard weights are not mounted; see INTEGRATION.md")
        logits = self._logits(df)                       # [N, K] mean over seeds
        prob = torch.softmax(torch.from_numpy(logits), dim=1).numpy()
        pred = prob.argmax(1)
        bidx = self.classes.index(self.benign)
        out = []
        try:
            from models.surrogate import SurrogateIDS        # platform severity map
            sev = SurrogateIDS.severity_for
        except Exception:
            sev = lambda lbl: "benign" if lbl == self.benign else "medium"
        for i, k in enumerate(pred):
            label = self.classes[k]
            out.append({"label_pred": label,
                        "confidence": float(prob[i, k]),
                        "benign_score": float(prob[i, bidx]),
                        "severity": sev(label),
                        "detector": "SODE-Guard"})
        return out

    def _logits(self, df: pd.DataFrame) -> np.ndarray:
        SM = self._SM
        Xraw = SM.align(df, self.meta["raw_features"])
        acc = None
        for seed in self.seeds:
            args = SM.RunArgs(self.run)
            D = SM.V.load_split(args, seed)
            X = SM.scale_like(self.run, seed, Xraw)
            sd = SM.V.seed_dir(self.run, seed)
            mdl = SM.V.load_any(self.model, D, sd)
            with torch.no_grad():
                if self.model in SM.V.TREES:
                    lg = mdl(X)
                else:
                    arch = SM.V.spec_of(self.model)["arch"]
                    kind = {"sode": "sde", "sdetgnn": "sde", "ode": "ode", "mlp": "mlp"}[arch]
                    pf = SM.V.predictor(mdl, kind, mode="fixed", n_paths=8)
                    lg = torch.cat([pf(X[i:i + 512]) for i in range(0, len(X), 512)])
            lg = lg.cpu().numpy()
            acc = lg if acc is None else acc + lg
        return acc / len(self.seeds)


@lru_cache(maxsize=1)
def get_detector() -> SODEGuardDetector:
    return SODEGuardDetector()
