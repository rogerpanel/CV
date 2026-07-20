#!/usr/bin/env python3
"""Generate UAV-EW-Bench-2026.

Builds the flight corpus, runs the full (defence x J/S x seed) sweep through
the selected backend, and writes the benchmark artifact:

    <out>/
      per_flight.csv        one row per flight evaluation
      per_point.csv         aggregated completion + Wilson 95% CI per point
      report_points.csv     the 9 canonical J/S points (for the figure/TikZ)
      crossings.csv         J/S at which each defence crosses the 0.90 floor
      manifest.json         full provenance (config, corpus, checksums)

Usage
-----
    python scripts/generate_benchmark.py --backend sim-lite --out ./artifact
    python scripts/generate_benchmark.py --backend airsim   --out ./artifact_airsim
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone

import pandas as pd

sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(__file__), "..")))

from uavbench import __version__
from uavbench.analysis import wilson_interval
from uavbench.backends import AirSimBackend, SimLiteBackend
from uavbench.config import load_config
from uavbench.corpus import build_corpus, corpus_manifest
from uavbench.model import AdversaryContour, CompletionModel
from uavbench.runner import run_benchmark


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate UAV-EW-Bench-2026")
    ap.add_argument("--backend", choices=["sim-lite", "airsim"], default="sim-lite")
    ap.add_argument("--out", default="./artifact")
    ap.add_argument("--benchmark-config", default=None)
    ap.add_argument("--defenses-config", default=None)
    ap.add_argument("--airsim-ip", default="127.0.0.1")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.benchmark_config, args.defenses_config)
    model = CompletionModel(cfg)
    contour = AdversaryContour()

    os.makedirs(args.out, exist_ok=True)

    print(f"UAV-EW-Bench-2026 generator  (pkg v{__version__})")
    print(f"  backend            : {args.backend}")
    print(f"  defences           : {[d.id for d in cfg.defenses]}")
    print(f"  J/S sweep          : {cfg.js_levels} levels, "
          f"{cfg.js_min_db:.0f}..{cfg.js_max_db:.0f} dB")
    print(f"  repeats x seeds     : {cfg.repeats_per_point} x {cfg.seeds}")
    print(f"  corpus flights     : {cfg.total_flights}")

    # -- corpus -----------------------------------------------------------
    flights = build_corpus(cfg)

    # -- backend ----------------------------------------------------------
    if args.backend == "sim-lite":
        backend = SimLiteBackend(cfg, model)
    else:
        backend = AirSimBackend(cfg, contour, connection=args.airsim_ip)

    # -- run --------------------------------------------------------------
    # Ensure the canonical report points (0,5,...,40) are always sampled by
    # merging them into the sweep grid.
    grid = sorted(set(cfg.js_grid()) | set(cfg.js_report_points_db))
    per_flight, per_point = run_benchmark(
        cfg, backend, flights, js_grid=grid, progress=not args.quiet
    )

    # -- write per-flight & per-point -------------------------------------
    pf_path = os.path.join(args.out, "per_flight.csv")
    pp_path = os.path.join(args.out, "per_point.csv")
    per_flight.to_csv(pf_path, index=False)
    per_point.drop(columns=["seed_means"]).to_csv(pp_path, index=False)

    # -- canonical report points (for the dissertation figure) ------------
    report_rows = []
    for d in cfg.defenses:
        for js in cfg.js_report_points_db:
            sub = per_flight[(per_flight.defense == d.id) & (per_flight.js_db == js)]
            if len(sub) == 0:
                # report point not on the grid: evaluate model directly
                continue
            k = int(sub.completed.sum())
            n = int(len(sub))
            lo, hi = wilson_interval(k, n, cfg.confidence)
            report_rows.append(
                {"defense": d.id, "js_db": js, "n": n,
                 "completion_mean": k / n, "ci_low": lo, "ci_high": hi}
            )
    report_df = pd.DataFrame(report_rows)
    report_df.to_csv(os.path.join(args.out, "report_points.csv"), index=False)

    # -- 0.90 crossings ---------------------------------------------------
    cross_rows = []
    for d in cfg.defenses:
        js90 = model.crossing_js(d.id, cfg.regulatory_threshold)
        cross_rows.append(
            {"defense": d.id, "label": d.label,
             "js_at_0.90_db": round(js90, 2)}
        )
    cross_df = pd.DataFrame(cross_rows)
    cross_df.to_csv(os.path.join(args.out, "crossings.csv"), index=False)

    # -- manifest ---------------------------------------------------------
    artifacts = ["per_flight.csv", "per_point.csv",
                 "report_points.csv", "crossings.csv"]
    checksums = {a: _sha256(os.path.join(args.out, a)) for a in artifacts}
    manifest = {
        "benchmark": cfg.name,
        "version": cfg.version,
        "generator_version": __version__,
        "backend": args.backend,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "config": {
            "js_min_db": cfg.js_min_db,
            "js_max_db": cfg.js_max_db,
            "js_levels": cfg.js_levels,
            "repeats_per_point": cfg.repeats_per_point,
            "seeds": cfg.seeds,
            "confidence": cfg.confidence,
            "labelling_standard": cfg.labelling_standard,
            "regulatory_threshold": cfg.regulatory_threshold,
            "stratified_effects": cfg.stratified_enabled,
        },
        "defenses": [
            {"id": d.id, "label": d.label, "reference": d.reference}
            for d in cfg.defenses
        ],
        "corpus": corpus_manifest(cfg, flights),
        "adversary": {
            "gnss_spoofing": contour.gnss_spoofing,
            "visual_pgd": contour.visual_pgd,
            "drl_bim": contour.drl_bim,
            "pgd_epsilon": contour.pgd_epsilon,
            "pgd_steps": contour.pgd_steps,
        },
        "sha256": checksums,
        "total_flight_evaluations": int(len(per_flight)),
    }
    with open(os.path.join(args.out, "manifest.json"), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, ensure_ascii=False)

    print()
    print(f"  flight evaluations : {len(per_flight):,}")
    print("  0.90 crossings (dB):")
    for r in cross_rows:
        print(f"      {r['label']:<34} {r['js_at_0.90_db']:>5.1f}")
    print(f"\nArtifact written to: {os.path.abspath(args.out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
