"""Summarise results folders or zips produced by kaggle/run_kaggle_study.py.

    python scripts/summarise_kaggle_results.py results_main.zip results_heldout.zip --latex
    python scripts/summarise_kaggle_results.py reproducibility/kaggle_gpu_v2 --json reproducibility/kaggle_results_v2.json

Several inputs are merged (e.g. the main and held-out sessions). Prints
Markdown tables -- main results with the lexical and length controls, encoder
ablation, architecture ablation, held-out-template results, certificates,
latency, errors -- and, with --latex, table rows for the manuscript. All
numbers are recomputed from the per-run JSON files. Version-1 results (first
Kaggle run) are read too.
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
ENC_LABEL = {"minilm": "one MiniLM vector / message", "chunk": "chunked MiniLM (mean+max)",
             "chunk_lex": "chunked MiniLM + lexical"}
LENGTH_BASELINE = {"a2a_sandbox": 0.53, "mcptox": 0.50, "mcp_safetybench": 0.50, "agentdojo": 0.50, "injecagent": 0.50}


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


def load(paths: list[Path]) -> tuple[list[dict], dict, dict, list[dict], list[str]]:
    runs, cert, lat, envs, errors = [], {}, {}, [], []
    for p in paths:
        root = open_results(p)
        env = json.loads((root / "environment.json").read_text()) if (root / "environment.json").exists() else {}
        envs.append(env)
        for f in sorted((root / "runs").glob("*.json")):
            r = json.loads(f.read_text())
            if "study" not in r:  # version-1 record
                r["study"] = "main" if r.get("variant") == "full" else "arch"
                r["encoder"] = env.get("encoder", "minilm")
                r["fold"] = "all"
                r["n_test_benign"] = round(r["n_test"] * (1 - r["test_attack_frac"]))
            runs.append(r)
        for kind, store in (("certify", cert), ("latency", lat)):
            for f in sorted((root / kind).glob("*.json")) if (root / kind).is_dir() else []:
                store[f.stem] = json.loads(f.read_text())
        errors += [f"{e.relative_to(root)}: {e.read_text().strip().splitlines()[-1][:200]}" for e in root.rglob("*.error.txt")]
    return runs, cert, lat, envs, errors


def ms(vals) -> tuple[float, float, int]:
    vals = [v for v in vals if v is not None and not (isinstance(v, float) and math.isnan(v))]
    if not vals:
        return math.nan, math.nan, 0
    return statistics.fmean(vals), (statistics.stdev(vals) if len(vals) > 1 else 0.0), len(vals)


def fmt(t, latex: bool = False) -> str:
    m, s, n = t
    if n == 0 or math.isnan(m):
        return "--"
    pm = r"$\pm$" if latex else "±"
    return f"{m:.3f}" if n == 1 else f"{m:.3f}{pm}{s:.3f}"


def stats(rs: list[dict]) -> dict[str, Any]:
    vt = [r["test_at_val_threshold"] for r in rs]
    at = [r["test_at_0.5"] for r in rs]
    return {"n": len(rs), "n_test": rs[0].get("n_test"), "n_test_benign": rs[0].get("n_test_benign"),
            "f1": ms([m["macro_f1"] for m in vt]), "f1_05": ms([m["macro_f1"] for m in at]),
            "auroc": ms([m.get("auroc") for m in at]), "recall": ms([m["attack_recall"] for m in vt]),
            "fpr": ms([m["fpr"] for m in vt]), "ci": rs[0].get("macro_f1_tuned_95ci"),
            "lip": rs[0].get("lipschitz_bound"), "params": rs[0].get("params"), "time": rs[0].get("train_time_s")}


def pick(runs, **kw) -> list[dict]:
    return sorted([r for r in runs if all(r.get(k) == v for k, v in kw.items())], key=lambda r: (r.get("seed") is None, r.get("seed")))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results", type=Path, nargs="+", help="results folders or .zip files")
    ap.add_argument("--latex", action="store_true")
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args()
    runs, cert, lat, envs, errors = load(args.results)
    main_enc = next((e.get("main_encoder") for e in envs if e.get("main_encoder")), None) or \
        next((e.get("encoder") for e in envs if e.get("encoder")), "minilm")
    main_var = next((e.get("main_variant") for e in envs if e.get("main_variant")), "full")
    out: dict[str, Any] = {"environment": envs, "main_encoder": main_enc, "main": {}, "encoder_ablation": {},
                           "arch_ablation": {}, "heldout": {}, "certify": cert, "latency": lat, "errors": errors}
    benches = [b for b in ORDER if any(r["benchmark"] == b for r in runs)]

    print("# MambaGuard Kaggle results\n")
    for e in envs:
        print(f"- session: study={e.get('study', 'main')} GPU={e.get('gpu')} torch={e.get('torch')} "
              f"encoder={e.get('main_encoder', e.get('encoder'))} started={e.get('started')} finished={e.get('finished')}")
    print(f"- {len(runs)} run files, {len(cert)} certificates, {len(lat)} latency reports, {len(errors)} errors\n")

    # ---- main --------------------------------------------------------------
    print(f"## Main results (encoder: {ENC_LABEL.get(main_enc, main_enc)}; threshold tuned on val; mean ± sd over seeds)\n")
    print("| Benchmark | n test (benign) | MambaGuard F1 | AUROC | recall | FPR | Lexical F1 | Lexical AUROC | Length |")
    print("|---|---|---|---|---|---|---|---|---|")
    for b in benches:
        rs = pick(runs, study="main", benchmark=b, variant=main_var, encoder=main_enc, fold="all")
        lx = pick(runs, study="main", benchmark=b, variant="lexical", fold="all")
        if not rs:
            continue
        s = stats(rs)
        l = stats(lx) if lx else None
        out["main"][b] = {"mambaguard": s, "lexical": l}
        print(f"| {LABEL[b]} | {s['n_test']} ({s['n_test_benign']}) | {fmt(s['f1'])} | {fmt(s['auroc'])} | {fmt(s['recall'])} | "
              f"{fmt(s['fpr'])} | {fmt(l['f1']) if l else '--'} | {fmt(l['auroc']) if l else '--'} | {LENGTH_BASELINE[b]} |")

    # ---- encoder ablation ---------------------------------------------------
    encs = sorted({r["encoder"] for r in runs if r["study"] == "enc"})
    if encs:
        print("\n## Encoder ablation (macro-F1 / AUROC; seeds 0-1 for the ablated encoders)\n")
        cols = [e for e in ("minilm", "chunk", "chunk_lex", "chunk_mil") if e in encs and e != main_enc] + [main_enc]
        eb = [b for b in benches if any(r["study"] == "enc" and r["benchmark"] == b for r in runs)]
        print("| Encoder | " + " | ".join(LABEL[b] for b in eb) + " |")
        print("|---|" + "---|" * len(eb))
        for e in cols:
            cells = []
            for b in eb:
                rs = pick(runs, study="enc", benchmark=b, encoder=e) if e != main_enc else \
                    [r for r in pick(runs, study="main", benchmark=b, variant=main_var, encoder=e) if r["seed"] in (0, 1)]
                if rs:
                    s = stats(rs)
                    out["encoder_ablation"].setdefault(e, {})[b] = s
                    cells.append(f"{fmt(s['f1'])} / {fmt(s['auroc'])}")
                else:
                    cells.append("--")
            print(f"| {ENC_LABEL.get(e, e)} | " + " | ".join(cells) + " |")

    # ---- architecture ablation ----------------------------------------------
    ab = sorted({r["benchmark"] for r in runs if r["study"] == "arch"}, key=ORDER.index)
    if ab:
        print("\n## Architecture ablation (macro-F1 / AUROC)\n")
        print("| Variant | " + " | ".join(LABEL[b] for b in ab) + " | params | Lf (first bench) |")
        print("|---|" + "---|" * (len(ab) + 2))
        for v in VARIANT_LABEL:
            cells, first = [], None
            for b in ab:
                rs = pick(runs, study="main" if v == main_var else "arch", benchmark=b, variant=v)
                rs = [r for r in rs if r["encoder"] == main_enc or r["study"] == "arch"]
                if rs:
                    s = stats(rs)
                    first = first or s
                    out["arch_ablation"].setdefault(v, {})[b] = s
                    cells.append(f"{fmt(s['f1'])} / {fmt(s['auroc'])}")
                else:
                    cells.append("--")
            if first:
                print(f"| {VARIANT_LABEL[v]} | " + " | ".join(cells) + f" | {first['params']} | {first['lip']:.3g} |")

    # ---- held-out templates ---------------------------------------------------
    ho = [b for b in ORDER if any(r["study"] == "heldout" and r["benchmark"] == b for r in runs)]
    if ho:
        print("\n## Held-out templates (test attacks use a wording/family absent from training)\n")
        print("| Benchmark | fold | test att/ben | MambaGuard F1 | AUROC | recall | FPR | Lexical F1 | Lexical AUROC | recall |")
        print("|---|---|---|---|---|---|---|---|---|---|")
        for b in ho:
            folds = sorted({r["fold"] for r in runs if r["study"] == "heldout" and r["benchmark"] == b})
            per = {}
            for f in folds:
                d = pick(runs, study="heldout", benchmark=b, fold=f, variant=main_var)
                l = pick(runs, study="heldout", benchmark=b, fold=f, variant="lexical")
                sd, sl = (stats(d) if d else None), (stats(l) if l else None)
                per[f] = {"mambaguard": sd, "lexical": sl}
                ref = sd or sl
                att = round(ref["n_test"] - ref["n_test_benign"]) if ref else "?"
                print(f"| {LABEL[b]} | {f} | {att}/{ref['n_test_benign'] if ref else '?'} | "
                      f"{fmt(sd['f1']) if sd else '--'} | {fmt(sd['auroc']) if sd else '--'} | {fmt(sd['recall']) if sd else '--'} | "
                      f"{fmt(sd['fpr']) if sd else '--'} | {fmt(sl['f1']) if sl else '--'} | {fmt(sl['auroc']) if sl else '--'} | "
                      f"{fmt(sl['recall']) if sl else '--'} |")
            agg = {}
            for who in ("mambaguard", "lexical"):
                vals = [v[who] for v in per.values() if v[who]]
                agg[who] = {k: ms([v[k][0] for v in vals]) for k in ("f1", "auroc", "recall", "fpr")} if vals else None
            out["heldout"][b] = {"folds": per, "mean_over_folds": agg}
            m, l = agg["mambaguard"], agg["lexical"]
            print(f"| **{LABEL[b]}** | **mean over folds** | | {fmt(m['f1']) if m else '--'} | {fmt(m['auroc']) if m else '--'} | "
                  f"{fmt(m['recall']) if m else '--'} | {fmt(m['fpr']) if m else '--'} | {fmt(l['f1']) if l else '--'} | "
                  f"{fmt(l['auroc']) if l else '--'} | {fmt(l['recall']) if l else '--'} |")

    # ---- certificates / latency ----------------------------------------------
    if cert:
        print("\n## Certified accuracy (seed-0 main model)\n")
        for b in [x for x in ORDER if x in cert]:
            rep = cert[b]
            if "skipped" in rep:
                print(f"- {b}: skipped ({rep['skipped']})")
                continue
            sig = list(rep["sigmas"])
            radii = [c["radius"] for c in rep["sigmas"][sig[0]]["cert_acc_curve"]][:5]
            print(f"**{LABEL[b]}** ({rep['windows']} windows, n={rep['n']})\n")
            print("| sigma | abstain | " + " | ".join(f"r={r}" for r in radii) + " |")
            print("|---|---|" + "---|" * len(radii))
            for s in sig:
                d = rep["sigmas"][s]
                print(f"| {s} | {d['abstain_rate']:.2f} | " + " | ".join(f"{c['cert_acc']:.3f}" for c in d["cert_acc_curve"][:5]) + " |")
            print()
    if lat:
        print("## Latency (forward pass on the GPU; payload encoding listed separately)\n")
        print("| Benchmark | batch-1 ms/window | best msg/s | encoder ms/msg (batch 1) | encoder ms/msg (batched) |")
        print("|---|---|---|---|---|")
        for b in [x for x in ORDER if x in lat]:
            rep = lat[b]
            if "batch" not in rep:
                continue
            best = max(rep["batch"].values(), key=lambda x: x["messages_per_s"])
            enc = rep.get("encoder") or {}
            print(f"| {LABEL[b]} | {rep['batch']['1']['median_batch_ms']:.2f} | {best['messages_per_s']:.0f} | "
                  f"{enc.get('per_message_ms_batch1', float('nan')):.2f} | {enc.get('per_message_ms_batched', float('nan')):.2f} |")

    if errors:
        print("\n## Errors\n")
        for e in errors:
            print(f"- {e}")

    if args.latex:
        print("\n## LaTeX rows\n")
        print("% main: benchmark & n (benign) & MG F1 & MG AUROC & Lex F1 & Lex AUROC & Len")
        for b, d in out["main"].items():
            s, l = d["mambaguard"], d["lexical"]
            print(f"{LABEL[b]} & {s['n_test']} ({s['n_test_benign']}) & {fmt(s['f1'], True)} & {fmt(s['auroc'], True)} & "
                  f"{fmt(l['f1'], True) if l else '--'} & {fmt(l['auroc'], True) if l else '--'} & {LENGTH_BASELINE[b]} \\\\")
        if out["heldout"]:
            print("\n% held-out mean over folds: benchmark & MG F1 & MG AUROC & Lex F1 & Lex AUROC")
            for b, d in out["heldout"].items():
                m, l = d["mean_over_folds"]["mambaguard"], d["mean_over_folds"]["lexical"]
                print(f"{LABEL[b]} & {fmt(m['f1'], True) if m else '--'} & {fmt(m['auroc'], True) if m else '--'} & "
                      f"{fmt(l['f1'], True) if l else '--'} & {fmt(l['auroc'], True) if l else '--'} \\\\")
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(out, indent=2, default=str))
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
