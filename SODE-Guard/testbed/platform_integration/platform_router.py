"""FastAPI router that exposes SODE-Guard as a platform detector branch.

Drop-in for the RobustIDPS.ai backend. It mounts three routes under the
platform's existing app and returns rows in the platform's own shape
(``label_predicted``, ``severity``, ``confidence``) plus SODE-Guard's benign
score, so the Live Monitor, the Models page and the testbed can treat it like
any built-in method.

Wiring (see INTEGRATION.md):
    # backend/main.py, after `app = FastAPI(...)`
    from platform_router import router as sode_router, branch_entry
    app.include_router(sode_router)
and add ``branch_entry()`` to the ``branch_names`` list in /api/model_info.
"""
from __future__ import annotations

import io

import pandas as pd
from fastapi import APIRouter, File, HTTPException, UploadFile

from sode_guard_detector import get_detector

router = APIRouter()


def branch_entry() -> str:
    """Name to append to /api/model_info branch_names."""
    return "SODE-Guard (Neural SDE, certified)"


@router.get("/api/sode_guard/info")
async def sode_guard_info():
    try:
        return get_detector().info()
    except Exception as e:
        # detector optional: report degraded instead of crashing the app
        return {"name": "SODE-Guard", "ready": False, "error": str(e)}


@router.post("/api/predict/sode_guard")
async def predict_sode_guard(file: UploadFile = File(...)):
    det = get_detector()
    if not det.ready:
        raise HTTPException(status_code=503,
                            detail="SODE-Guard weights not mounted; see INTEGRATION.md")
    raw = await file.read()
    try:
        df = pd.read_csv(io.BytesIO(raw), low_memory=False)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"bad CSV: {e}")
    rows = det.predict(df)
    for i, r in enumerate(rows):
        r["flow_id"] = i
        r["label_predicted"] = r.pop("label_pred")   # match platform field name
    n_threats = sum(1 for r in rows if r["severity"] != "benign")
    dist: dict = {}
    for r in rows:
        dist[r["label_predicted"]] = dist.get(r["label_predicted"], 0) + 1
    return {
        "detector": "SODE-Guard",
        "n_flows": len(rows),
        "n_threats": n_threats,
        "n_benign": len(rows) - n_threats,
        "predictions": rows,
        "attack_distribution": dist,
        "class_labels": det.classes,
        "certified": det.info()["certified"],
    }
