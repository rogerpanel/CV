"""Summarise a results folder or zip produced by kaggle/run_kaggle_study.py.

    python scripts/summarise_kaggle_results.py results.zip
    python scripts/summarise_kaggle_results.py results/ --latex --json reproducibility/kaggle_results.json

Prints Markdown tables (main results, ablations, certificates, latency) and,
with --latex, table rows that can be pasted into sp2027/MambaGuard_sp2027.tex.
All numbers are recomputed from the per-run JSON files.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import tempfile
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Any

ORDER = ["a2a_sandbox", "mcptox", "mcp_safetybench", "agentdojo", "injecagent"]
LABEL = {"a2a_sandbox": "A2A sandbox", "mcptox": "MCPTox", "mcp_safetybench": "MCP-SafetyBench",
         "agentdojo": "AgentDojo", "injecagent": "InjecAgent"}
VARIANT_LABEL = {"full": "MambaGuard (full)", "ssm_only": "SSM only (no GAT)", "gat_only": "GAT only (no SSM)",
                 "flat": "Flat readout (no graph)", "transformer": "Transformer backbone",
                 "no_lipschitz": "No Lipschitz penalty"}
BASELINES = Path(__file__).resolve().parents[1] / "reproducibility" / "benchmark_results.json"


def open_results(path: Path) -> Path:
    if path.suffix == ".zip":
        tmp = Path(tempfile.mkdtemp(prefix="mg_results_"))
        with zipfile.ZipFile(path) as zf:
            zf.extractall(tmp)
        kids = [d for d in tmp.iterdir() if d.is_dir()]
        path = kids[0] if len(kids) == 1 else tmp
    if not (path / "runs").is_dir():
        raise SystemExit(f"{path}: no runs/ folder -- is this a run_kaggle_study.py output?")
    return path


def load_json_dir(d: Path) -> dict[str, Any]:
    return {f.stem: json.loads(f.read_text()) for f in sorted(d.glob("*.json"))} if d.is_dir() else {}


def ms(vals: list[float]) -> tuple[float, float, int]:
    vals = [v for v in vals if v is not None and not (isinstance(v, float) and math.isnan(v))]
    if not vals:
        return math.nan, math.nan, 0
    return statistics.fmean(vals), (statistics.stdev(vals) if len(vals) > 1 else 0.0), len(vals)


def fmt(m: float, s: float, n: int, latex: bool = False) -> str:
    if n == 0 or math.isnan(m):
        return "--"
    pm = r"$\pm$" if latex else "±"
    return f"{m:.3f}" if n == 1 else f"{m:.3f}{pm}{s:.3f}"


def group_runs(runs: dict[str, Any]) -> dict[tuple[str, str], list[dict]]:
    g: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in runs.values():
        g[(r["benchmark"], r["variant"])].append(r)
    for v in g.values():
        v.sort(key=lambda r: r["seed"])
    return g


def row_stats(rs: list[dict]) -> dict[str, Any]:
    vt = [r["test_at_val_threshold"] for r in rs]
    at = [r["test_at_0.5"] for r in rs]
    out = {
        "seeds": [r["seed"] for r in rs], "n_test": rs[0]["n_test"], "attack_frac": rs[0]["test_attack_frac"],
        "macro_f1_val_thr": ms([m["macro_f1"] for m in vt]), "macro_f1_0.5": ms([m["macro_f1"] for m in at]),
        "auroc": ms([m.get("auroc", math.nan) for m in at]), "avg_precision": ms([m.get("avg_precision", math.nan) for m in at]),
        "recall": ms([m["attack_recall"] for m in vt]), "fpr": ms([m["fpr"] for m in vt]),
        "f1_ci_seed0": rs[0].get("macro_f1_tuned_95ci"), "auroc_ci_seed0": rs[0].get("auroc_95ci"),
        "train_time_s": ms([r["train_time_s"] for r in rs]), "params": rs[0]["params"],
        "lipschitz_bound": rs[0].get("lipschitz_bound"),
    }
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results", type=Path, help="results folder or .zip from the Kaggle run")
    ap.add_argument("--latex", action="store_true", help="also print LaTeX table rows")
    ap.add_argument("--json", type=Path, default=None, help="write the summary to this JSON file")
    args = ap.parse_args()

    root = open_results(args.results)
    runs = load_json_dir(root / "runs")
    cert = load_json_dir(root / "certify")
    lat = load_json_dir(root / "latency")
    env = json.loads((root / "environment.json").read_text()) if (root / "environment.json").exists() else {}
    base = json.loads(BASELINES.read_text())["benchmarks"] if BASELINES.exists() else {}
    errors = sorted(p.name for p in root.rglob("*.error.txt"))
    g = group_runs(runs)
    benches = [b for b in ORDER if (b, "full") in g] + sorted({b for b, _ in g} - set(ORDER))

    print(f"# MambaGuard Kaggle results ({root.name})\n")
    print(f"GPU: {env.get('gpu')}  torch {env.get('torch')}  CUDA {env.get('cuda')}  encoder: {env.get('encoder')}  "
          f"code commit: {str(env.get('code_commit'))[:10]}  started {env.get('started')}  finished {env.get('finished')}")
    print(f"{len(runs)} runs, {len(cert)} certificate reports, {len(lat)} latency reports, {len(errors)} errors\n")

    summary: dict[str, Any] = {"environment": env, "main": {}, "ablation": {}, "certify": {}, "latency": {}, "errors": errors}

    print("## Main results (test split; threshold tuned on val; mean ± std over seeds)\n")
    print("| Benchmark | n test | attack % | macro-F1 | macro-F1 @0.5 | AUROC | attack recall | FPR | length baseline | 95% CI (seed 0) |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for b in benches:
        s = row_stats(g[(b, "full")])
        summary["main"][b] = s
        lb = base.get(b, {}).get("length_baseline_bal_acc")
        ci = s["f1_ci_seed0"]
        print(f"| {LABEL.get(b, b)} | {s['n_test']} | {100 * s['attack_frac']:.0f} | {fmt(*s['macro_f1_val_thr'])} | "
              f"{fmt(*s['macro_f1_0.5'])} | {fmt(*s['auroc'])} | {fmt(*s['recall'])} | {fmt(*s['fpr'])} | "
              f"{lb if lb is not None else '--'} | {('[%.3f, %.3f]' % tuple(ci)) if ci else '--'} |")
    print(f"\nSeeds per benchmark: " + ", ".join(f"{b}={len(g[(b, 'full')])}" for b in benches))

    ab = sorted({(b, v) for b, v in g if v != "full"})
    if ab:
        print("\n## Ablations (macro-F1 / AUROC, mean ± std)\n")
        ab_benches = [b for b in benches if any(bb == b for bb, _ in ab)]
        print("| Variant | " + " | ".join(LABEL.get(b, b) for b in ab_benches) + " |")
        print("|---|" + "---|" * len(ab_benches))
        for v in VARIANT_LABEL:
            cells = []
            for b in ab_benches:
                if (b, v) not in g:
                    cells.append("--")
                    continue
                s = row_stats(g[(b, v)])
                summary["ablation"].setdefault(v, {})[b] = s
                cells.append(f"{fmt(*s['macro_f1_val_thr'])} / {fmt(*s['auroc'])}")
            if any(c != "--" for c in cells):
                print(f"| {VARIANT_LABEL[v]} | " + " | ".join(cells) + " |")

    if cert:
        print("\n## Certified accuracy (Cohen smoothing on payload embeddings, seed-0 model)\n")
        for b, rep in sorted(cert.items(), key=lambda kv: ORDER.index(kv[0]) if kv[0] in ORDER else 99):
            if "skipped" in rep:
                print(f"- {b}: skipped ({rep['skipped']})")
                continue
            summary["certify"][b] = rep
            print(f"**{LABEL.get(b, b)}** ({rep['windows']} windows, n={rep['n']}, alpha={rep['alpha']})\n")
            sig = list(rep["sigmas"])
            radii = [c["radius"] for c in rep["sigmas"][sig[0]]["cert_acc_curve"]]
            print("| sigma | abstain | " + " | ".join(f"r={r}" for r in radii) + " |")
            print("|---|---|" + "---|" * len(radii))
            for s in sig:
                d = rep["sigmas"][s]
                print(f"| {s} | {d['abstain_rate']:.2f} | " + " | ".join(f"{c['cert_acc']:.3f}" for c in d["cert_acc_curve"]) + " |")
            print()

    if lat:
        print("## Latency (untrained full model, forward pass)\n")
        print("| Benchmark | device | batch | messages | median ms/batch | ms/message | messages/s |")
        print("|---|---|---|---|---|---|---|")
        for b, rep in sorted(lat.items(), key=lambda kv: ORDER.index(kv[0]) if kv[0] in ORDER else 99):
            summary["latency"][b] = rep
            for bs, d in rep["batch"].items():
                print(f"| {LABEL.get(b, b)} | {rep['device']} | {bs} | {d['messages']} | {d['median_batch_ms']:.2f} | "
                      f"{d['per_message_ms']:.3f} | {d['messages_per_s']:.0f} |")

    if errors:
        print("\n## Errors\n")
        for e in errors:
            f = next(root.rglob(e))
            print(f"- {f.relative_to(root)}: {f.read_text()[:300]}")

    if args.latex:
        print("\n## LaTeX rows\n")
        print("% Benchmark & macro-F1 & AUROC & attack recall & FPR & length baseline \\\\")
        for b in benches:
            s = summary["main"][b]
            lb = base.get(b, {}).get("length_baseline_bal_acc")
            print(f"{LABEL.get(b, b)} & {fmt(*s['macro_f1_val_thr'], latex=True)} & {fmt(*s['auroc'], latex=True)} & "
                  f"{fmt(*s['recall'], latex=True)} & {fmt(*s['fpr'], latex=True)} & {lb if lb is not None else '--'} \\\\")
        if summary["ablation"]:
            print("\n% Ablation: variant & per-benchmark macro-F1")
            for v, per in summary["ablation"].items():
                print(f"{VARIANT_LABEL[v]} & " + " & ".join(fmt(*per[b]["macro_f1_val_thr"], latex=True) if b in per else "--"
                                                           for b in ab_benches) + " \\\\")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(summary, indent=2, default=str))
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
