"""FastAPI scan endpoint smoke tests using the in-process TestClient."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Point the service at the smoke config and a fresh checkpoint dir so the
# import-time loader doesn't need a real trained model.
os.environ["FEDLORAGUARD_CONFIG"] = str(ROOT / "configs" / "smoke.yaml")
os.environ.setdefault("FEDLORAGUARD_CHECKPOINTS", "/tmp/fedloraguard_no_ckpt")
Path(os.environ["FEDLORAGUARD_CHECKPOINTS"]).mkdir(parents=True, exist_ok=True)


@pytest.fixture(scope="module")
def client():
    fastapi = pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from service.api import app

    with TestClient(app) as tc:
        yield tc


def test_healthz(client):
    r = client.get("/healthz")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_readyz(client):
    r = client.get("/readyz")
    assert r.status_code in (200, 503)


def test_metrics_renders(client):
    r = client.get("/metrics")
    assert r.status_code == 200


def test_scan_adapter_round_trip(client):
    payload = json.loads((ROOT / "docs" / "examples" / "adapter.json").read_text())
    r = client.post("/scan_adapter", json=payload)
    assert r.status_code == 200
    body = r.json()
    assert "p_malicious" in body
    assert "certificate" in body
    assert "k_star" in body["certificate"]
    assert body["adapter_id"] == payload["adapter_id"]
