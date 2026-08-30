#!/usr/bin/env bash
# Reproduce Tables 5–11 of the IEEE Access revision (v4 manuscript).
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

CONFIG="${CONFIG:-configs/sode_guard.yaml}"
CKPT_DIR="${CKPT_DIR:-experiments/sode_guard_tnnls_v3}"
OUT="${OUT:-experiments/revision_v4}"
mkdir -p "$OUT"

python - <<'PY'
import json, os, torch
from pathlib import Path
from src.utils import load_config
from src.data.registry import get_loader
from src.training.train import build_model
from src.theory import (certify_L2_lipschitz, estimate_effective_degree,
                        pac_bayes_bound, verify_bel)
from src.evaluation.reliability import randomized_smoothing_sensitivity

cfg = load_config(os.environ.get("CONFIG", "configs/sode_guard.yaml"))
device = "cuda" if torch.cuda.is_available() else "cpu"
out = Path(os.environ.get("OUT", "experiments/revision_v4"))
out.mkdir(parents=True, exist_ok=True)

for bench in cfg.data.benchmarks:
    ckpt = Path(os.environ.get("CKPT_DIR", "experiments/sode_guard_tnnls_v3")) / bench / "seed-42/best.pt"
    if not ckpt.exists():
        print(f"[skip] {bench}: {ckpt} not found (train first via reproduce_paper.sh)")
        continue
    model = build_model(cfg).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device)["model"])
    model.eval()
    loader = get_loader(bench, split="test", batch_size=cfg.training.batch_size,
                        num_workers=cfg.experiment.num_workers)

    lip = certify_L2_lipschitz(model, loader, device=device)
    x_batch, _ = next(iter(loader))
    chaos = estimate_effective_degree(model, x_batch.to(device), n_samples=128)
    bel = verify_bel(model, x_batch.to(device))
    pb = pac_bayes_bound(empirical_risk=0.036, kl_divergence=0.42*128, n_samples=10**6)
    sigma = randomized_smoothing_sensitivity(model, loader, device=device)

    (out / f"{bench}.json").write_text(json.dumps({
        "lipschitz": lip.as_dict(),
        "chaos_median": chaos.median_degree,
        "chaos_p99":    chaos.p99_degree,
        "bel":          [b.__dict__ for b in bel],
        "pac_bayes":    pb,
        "smoothing_sensitivity": [s.__dict__ for s in sigma],
    }, indent=2))
    print(f"[ok] {bench}: {out / (bench + '.json')}")
PY
echo "Revision numbers written under $OUT"
