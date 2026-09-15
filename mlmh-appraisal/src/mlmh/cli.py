"""Entry point: python -m mlmh <command> [args]"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import experiments as ex


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="mlmh", description="ML for mental health: leakage / external validation / calibration experiments")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("synth", help="generate the synthetic fixture under data/synthetic/")
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--days", type=int, default=8)
    s.add_argument("--scale", type=float, default=1.0, help="fraction of the real cohort sizes to simulate")

    v = sub.add_parser("verify-data", help="check the on-disk layout of each dataset and print what was recognised")
    v.add_argument("--config", default="configs/base.yaml")
    v.add_argument("--synthetic", action="store_true")

    pr = sub.add_parser("prepare", help="load, window, featurise and cache every cohort")
    pr.add_argument("--config", default="configs/base.yaml")
    pr.add_argument("--synthetic", action="store_true")
    pr.add_argument("--cohorts", nargs="*")

    r = sub.add_parser("run", help="run an experiment config (E1/E2/E3)")
    r.add_argument("config")
    r.add_argument("--synthetic", action="store_true")
    r.add_argument("--seeds", type=int, nargs="*", help="override seeds")
    r.add_argument("--models", nargs="*", help="override models")
    r.add_argument("--n-boot", type=int, help="override bootstrap replicates")

    rt = sub.add_parser("retable", help="re-emit E1/E2 LaTeX tables from stored results with seed-averaged point estimates")
    rt.add_argument("--config", default="configs/base.yaml")
    rt.add_argument("--synthetic", action="store_true")

    t = sub.add_parser("tripod", help="write the TRIPOD+AI self-audit from run manifests")
    t.add_argument("--synthetic", action="store_true")

    a = p.parse_args(argv)
    root = ex.ROOT
    if a.cmd == "synth":
        from .data.synthetic import generate

        paths = generate(root / "data" / "synthetic", seed=a.seed, n_days=a.days, scale=a.scale)
        for k, v_ in paths.items():
            print(f"[synth] {k}: {v_}")
        return 0

    cfg = ex.load_config(root / a.config) if a.cmd != "tripod" else {}
    if getattr(a, "synthetic", False):
        cfg["synthetic"] = True

    if a.cmd == "verify-data":
        from .data.loaders import load_cohort

        droot = ex.data_root(cfg)
        ok = True
        for name in cfg["cohorts"]:
            croot = ex.ROOT / cfg["cohort_roots"][name] if name in cfg.get("cohort_roots", {}) and not cfg.get("synthetic") else droot / name
            if not croot.exists():
                print(f"[verify] {name}: MISSING at {croot}")
                ok = False
                continue
            try:
                minutes, subjects = load_cohort(name, croot, **cfg.get("loader_kwargs", {}).get(name, {}))
                print(f"[verify] {name}: OK  subjects={len(subjects)} cases={int(subjects['label'].sum())} minutes={len(minutes)}  files={subjects['source_file'].nunique()}")
                print(f"          groups: {subjects['group'].value_counts().to_dict()}")
                print(f"          span: {minutes['timestamp'].min()} -> {minutes['timestamp'].max()}")
            except Exception as e:  # noqa: BLE001
                ok = False
                print(f"[verify] {name}: FAILED -- {type(e).__name__}: {e}")
        return 0 if ok else 1

    if a.cmd == "retable":
        ex.retable(cfg)
        return 0

    if a.cmd == "prepare":
        ex.prepare(cfg, cohorts=a.cohorts)
        return 0

    if a.cmd == "run":
        if a.seeds:
            cfg["seeds"] = a.seeds
        if a.models:
            cfg["models"] = a.models
        if a.n_boot:
            cfg["n_boot"] = a.n_boot
        exp = cfg.get("experiment")
        from . import extras

        runner = {"E1": ex.run_e1, "E2": ex.run_e2, "E3": ex.run_e3, "E4": extras.run_e4, "E5": extras.run_e5, "E6": extras.run_e6, "cohort_table": extras.cohort_table}.get(exp)
        if runner is None:
            print(f"config has no recognised 'experiment' key (got {exp!r})", file=sys.stderr)
            return 2
        runner(cfg)
        return 0

    if a.cmd == "tripod":
        from .reporting.tripod import self_audit

        rdir = root / "results" / ("synthetic" if a.synthetic else "real")
        out = self_audit(rdir, root / "results" / ("synthetic" if a.synthetic else "real") / "tripod_ai_self_audit.md")
        print(f"[tripod] wrote {out}")
        return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
