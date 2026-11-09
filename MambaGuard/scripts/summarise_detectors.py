"""One comparison table for every detector: the paper's neural detector and the independent detectors.

Reads the five-seed study (results_v4_all[.zip], from kaggle/run_kaggle_study.py) and the independent-detector
study (results_baselines[.zip], from kaggle/run_detector_baselines.py). AUROC is computed from the saved
per-episode scores on the same test episodes; for the neural detector it is the mean over seeds (main study) and
the mean over folds of the per-fold seed means (held-out study), as in the paper.

    python scripts/summarise_detectors.py results_v4_bundle.zip [--latex] [--json out.json]
    python scripts/summarise_detectors.py results_v4_all.zip results_baselines.zip
"""
from __future__ import annotations

import argparse
import json
import re
import tempfile
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

BENCH = ["a2a_sandbox", "mcptox", "mcp_safetybench", "agentdojo", "injecagent"]
SHORT = {"a2a_sandbox": "A2A", "mcptox": "MCPTox", "mcp_safetybench": "MCP-SB", "agentdojo": "AgentDojo", "injecagent": "InjecAgent"}
ORDER = ["neural", "lexical", "deberta_ft", "protectai", "piguard", "promptguard2", "llmjudge"]
NAMES = {"neural": "Neural detector (ours)", "lexical": "Lexical (TF-IDF)", "deberta_ft": "DeBERTa-v3 fine-tuned",
         "protectai": "ProtectAI DeBERTa", "piguard": "PIGuard", "promptguard2": "Prompt-Guard-2", "llmjudge": "LLM judge (Qwen2.5-1.5B)"}
NEURAL_RE = re.compile(r"(main|heldout)__(\w+?)__flat__chunk_mil__(.+)__seed(\d+)\.npz$")
BASE_RE = re.compile(r"(main|heldout|transfer)__(\w+?)__(\w+?)__(.+)__seed(\d+)\.npz$")


def unpack(paths: list[Path]) -> list[Path]:
    """Expand zips (including a bundle of zips) into folders that contain predictions/."""
    roots, queue = [], list(paths)
    while queue:
        p = queue.pop()
        if p.suffix == ".zip":
            tmp = Path(tempfile.mkdtemp(prefix="mg_sum_"))
            with zipfile.ZipFile(p) as z:
                z.extractall(tmp)
            queue += sorted(tmp.rglob("*.zip"))
            roots += [d.parent for d in tmp.rglob("predictions") if d.is_dir()]
        elif (p / "predictions").is_dir():
            roots.append(p)
    return roots


def auroc(f: Path) -> float | None:
    d = np.load(f)
    y, s = d["label"], d["prob"]
    return float(roc_auc_score(y, s)) if len(np.unique(y)) == 2 else None


def collect(roots: list[Path]) -> dict:
    res: dict = defaultdict(lambda: defaultdict(list))   # (study, det, bench) -> fold -> [auroc per seed]
    for r in roots:
        for f in sorted((r / "predictions").glob("*.npz")):
            m = NEURAL_RE.match(f.name)
            if m:
                study, bench, fold, _ = m.groups()
                det = "neural"
            else:
                m = BASE_RE.match(f.name)
                if not m or m.group(3) in ("flat", "full", "ssm_only", "gat_only", "transformer", "no_lipschitz"):
                    continue
                study, bench, det, fold, _ = m.groups()
            a = auroc(f)
            if a is not None:
                res[(study, det, bench)][fold].append(a)
    return res


def cell(res, study, det, bench, fold=None):
    folds = res.get((study, det, bench))
    if not folds:
        return None
    if fold is not None:
        return float(np.mean(folds[fold])) if fold in folds else None
    return float(np.mean([np.mean(v) for v in folds.values()]))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", type=Path, nargs="+")
    ap.add_argument("--latex", action="store_true")
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args()
    res = collect(unpack(args.inputs))
    dets = [d for d in ORDER if any(k[1] == d for k in res)]
    out: dict = {"main": {}, "heldout": {}, "transfer": {}}
    fmt = lambda v: f"{v:.3f}" if v is not None else "--"  # noqa: E731
    for study, title in (("main", "In distribution: test AUROC"), ("heldout", "Held-out templates: mean test AUROC over folds")):
        print(f"\n## {title}\n\n| Detector | " + " | ".join(SHORT[b] for b in BENCH) + " |\n|---|" + "---|" * len(BENCH))
        for d in dets:
            vals = [cell(res, study, d, b) for b in BENCH]
            out[study][d] = dict(zip(BENCH, vals))
            print(f"| {NAMES[d]} | " + " | ".join(fmt(v) for v in vals) + " |")
    tr = sorted({(k[2], f) for k in res if k[0] == "transfer" for f in res[k]})
    if tr:
        print("\n## Cross-benchmark transfer: test AUROC (trained detectors learn on the source; zero-shot ones do not train)\n")
        print("| Target (source) | " + " | ".join(NAMES[d] for d in dets if d != "neural") + " |\n|---|" + "---|" * (len(dets) - 1))
        for bench, fold in tr:
            vals = [cell(res, "transfer", d, bench, fold) for d in dets if d != "neural"]
            out["transfer"][f"{bench} {fold}"] = dict(zip([d for d in dets if d != "neural"], vals))
            print(f"| {SHORT[bench]} ({fold.replace('from_', '')}) | " + " | ".join(fmt(v) for v in vals) + " |")
    if args.latex:
        print("\n% LaTeX rows: detector & in-distribution AUROC x5 & held-out AUROC x5")
        for d in dets:
            row = [out["main"][d].get(b) for b in BENCH] + [out["heldout"][d].get(b) for b in BENCH]
            print(NAMES[d] + " & " + " & ".join(f"${v:.2f}$" if v is not None else "--" for v in row) + r" \\")
    if args.json:
        args.json.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
