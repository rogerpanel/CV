"""End-to-end smoke test: build the smoke benchmark, run 2 federated rounds,
evaluate, ensure metrics + certificate JSON are produced.  This is the test
gate the CI hook should run on every commit."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parent.parent


@pytest.mark.slow
def test_smoke_pipeline(tmp_path):
    data = tmp_path / "data"
    runs = tmp_path / "runs"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT)

    rc = subprocess.call([
        sys.executable, str(ROOT / "scripts" / "build_benchmark.py"),
        "--config", str(ROOT / "configs" / "smoke.yaml"),
        "--out", str(data),
    ], env=env)
    assert rc == 0
    assert (data / "graph.pt").exists()
    assert (data / "client_graphs.pt").exists()

    rc = subprocess.call([
        sys.executable, str(ROOT / "scripts" / "train_federated.py"),
        "--config", str(ROOT / "configs" / "smoke.yaml"),
        "--data", str(data),
        "--override", f"experiment.output_dir={runs / 'smoke'}",
    ], env=env)
    assert rc == 0
    assert (runs / "smoke" / "global.pt").exists()

    rc = subprocess.call([
        sys.executable, str(ROOT / "scripts" / "evaluate.py"),
        "--config", str(ROOT / "configs" / "smoke.yaml"),
        "--data", str(data),
        "--checkpoint", str(runs / "smoke" / "global.pt"),
        "--output", str(runs / "smoke"),
    ], env=env)
    assert rc == 0
    metrics = json.loads((runs / "smoke" / "metrics.json").read_text())
    cert = json.loads((runs / "smoke" / "certificate.json").read_text())
    assert "macro_f1" in metrics
    assert "k_star" in cert
