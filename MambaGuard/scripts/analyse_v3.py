"""Hybrid detector, conformal thresholds and paired tests from a v3 results folder/zip.

Works from the saved prediction files alone (no data, no GPU). Each detector
run of the v3 study stores, for the same validation / calibration / test
episodes, three expert scores:

    det  MambaGuard P(attack)
    lex  TF-IDF logistic-regression P(attack)        (the lexical baseline)
    nov  Mahalanobis novelty to benign training episodes (label-free)

Pre-registered analysis (fixed before any v3 result existed):

1. Each expert is Platt-calibrated on the validation (selection) split.
2. HYBRID = convex combination w_d*det + w_l*lex + w_n*nov, weights on a
   0.1-grid of the simplex chosen by validation macro-F1 (ties: lower
   validation Brier score, then the most balanced det/lex weights -- no
   preference for our own expert); threshold tuned on validation.
3. Split conformal: tau = ceil((n+1)(1-alpha))-th smallest calibration-benign
   score, alpha in {0.05, 0.10}; plus adaptive conformal inference (Gibbs &
   Candes 2021, gamma = 0.05) over a random test stream, updated on benign
   windows only, for long-run false-alarm control under benign shift.
4. Paired bootstrap (2,000 resamples of test episodes, shared across seeds) of
   HYBRID minus LEXICAL macro-F1 and AUROC per benchmark; two-sided p-values,
   Holm-Bonferroni across the five main benchmarks. Held-out folds: sign test
   over (fold, seed) pairs.

    python scripts/analyse_v3.py results_v3_all.zip --json reproducibility/analysis_v3.json
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import re
import tempfile
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score

EXPERTS = ("det", "lex", "nov")
ORDER = ["a2a_sandbox", "mcptox", "mcp_safetybench", "agentdojo", "injecagent"]


def open_results(path: Path) -> Path:
    if path.suffix == ".zip":
        tmp = Path(tempfile.mkdtemp(prefix="mg_v3_"))
        with zipfile.ZipFile(path) as zf:
            zf.extractall(tmp)
        kids = [d for d in tmp.iterdir() if d.is_dir()]
        path = kids[0] if len(kids) == 1 else tmp
    return path


def macro_f1(y, s, thr):
    """Binary macro-F1 (same value as sklearn's, vectorised for the bootstrap)."""
    p = s >= thr
    y = np.asarray(y).astype(bool)
    tp, fp, fn = np.sum(p & y), np.sum(p & ~y), np.sum(~p & y)
    tn = len(y) - tp - fp - fn
    f_pos = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
    f_neg = 2 * tn / (2 * tn + fn + fp) if (2 * tn + fn + fp) else 0.0
    return float((f_pos + f_neg) / 2)


def auroc(y, s):
    """Mann-Whitney AUROC with tie correction (equals sklearn's roc_auc_score)."""
    from scipy.stats import rankdata

    y = np.asarray(y).astype(bool)
    n1, n0 = int(y.sum()), int((~y).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = rankdata(s)
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def tune_thr(y, s):
    if len(np.unique(y)) < 2:
        return 0.5
    grid = np.unique(np.concatenate([np.linspace(0.01, 0.99, 99), s]))
    sc = np.array([macro_f1(y, s, t) for t in grid])
    return float(np.median(grid[sc >= sc.max() - 1e-12]))


def platt(yv, sv):
    if len(np.unique(yv)) < 2 or np.std(sv) < 1e-12:
        lo, hi = float(np.min(sv)) if len(sv) else 0.0, float(np.max(sv)) if len(sv) else 1.0
        return lambda s: np.clip((s - lo) / (hi - lo + 1e-12), 0, 1)
    m = LogisticRegression(C=1e4, max_iter=1000).fit(sv.reshape(-1, 1), yv)
    return lambda s: m.predict_proba(np.asarray(s).reshape(-1, 1))[:, 1] if len(s) else np.zeros(0)


SIMPLEX = [w for w in itertools.product(np.round(np.arange(0, 1.01, 0.1), 1), repeat=3) if abs(sum(w) - 1) < 1e-9]


def hybrid_weights(yv, cv: dict) -> tuple:
    best, key = None, None
    for w in SIMPLEX:
        s = sum(wi * cv[e] for wi, e in zip(w, EXPERTS))
        k = (round(macro_f1(yv, s, tune_thr(yv, s)), 6), -round(float(np.mean((s - yv) ** 2)), 6), -abs(w[0] - w[1]))
        if key is None or k > key:
            best, key = w, k
    return best


def conformal(ycal, scal, alpha):
    ben = np.sort(scal[ycal == 0])
    k = math.ceil((len(ben) + 1) * (1 - alpha))
    return None if k > len(ben) or len(ben) == 0 else float(ben[k - 1])


def aci(ycal, scal, yt, st, alpha=0.05, gamma=0.05, seed=0):
    ben = np.sort(scal[ycal == 0])
    if len(ben) < 5:
        return None
    a_t, flags = alpha, np.zeros(len(yt), dtype=int)
    for i in np.random.default_rng(seed).permutation(len(yt)):
        q = 1.0 - min(max(a_t, 0.0), 1.0)
        tau = np.inf if q >= 1 else float(np.quantile(ben, q))
        flags[i] = int(st[i] > tau)
        if yt[i] == 0:  # benign label arrives: update the working level
            a_t = a_t + gamma * (alpha - flags[i])
    fpr = float(flags[yt == 0].mean()) if (yt == 0).any() else float("nan")
    rec = float(flags[yt == 1].mean()) if (yt == 1).any() else float("nan")
    return {"fpr": fpr, "recall": rec}


def analyse_file(f: Path) -> dict:
    d = dict(np.load(f))
    if "lex_test" not in d:
        return {}
    yv, yc, yt = d["val_label"], d.get("cal_label", np.zeros(0)), d["label"]
    raw = {e: {"val": d[f"{e}_val" if e != "det" else "val_prob"],
               "cal": d[f"{e}_cal" if e != "det" else "cal_prob"],
               "test": d[f"{e}_test" if e != "det" else "prob"]} for e in EXPERTS}
    cal = {}
    for e in EXPERTS:
        g = platt(yv, raw[e]["val"])
        cal[e] = {s: g(raw[e][s]) for s in ("val", "cal", "test")}
    w = hybrid_weights(yv, {e: cal[e]["val"] for e in EXPERTS})
    models = {e: cal[e] for e in EXPERTS}
    models["hybrid"] = {s: sum(wi * cal[e][s] for wi, e in zip(w, EXPERTS)) for s in ("val", "cal", "test")}
    out = {"weights": dict(zip(EXPERTS, map(float, w))), "n_test": int(len(yt)), "n_test_benign": int((yt == 0).sum()),
           "n_cal_benign": int((yc == 0).sum()), "models": {}, "_y": yt, "_scores": {}}
    for name, m in models.items():
        thr = tune_thr(yv, m["val"])
        pred = (m["test"] >= thr).astype(int)
        r = {"macro_f1": macro_f1(yt, m["test"], thr), "auroc": auroc(yt, m["test"]),
             "recall": float(pred[yt == 1].mean()) if (yt == 1).any() else float("nan"),
             "fpr": float(pred[yt == 0].mean()) if (yt == 0).any() else float("nan")}
        for a in (0.05, 0.10):
            tau = conformal(yc, m["cal"], a) if len(yc) else None
            if tau is None:
                r[f"conformal_{a}"] = None
            else:
                p = (m["test"] > tau).astype(int)
                r[f"conformal_{a}"] = {"fpr": float(p[yt == 0].mean()) if (yt == 0).any() else float("nan"),
                                       "recall": float(p[yt == 1].mean()) if (yt == 1).any() else float("nan")}
        r["aci_0.05"] = aci(yc, m["cal"], yt, m["test"]) if len(yc) else None
        out["models"][name] = r
        out["_scores"][name] = (m["test"], thr)
    return out


def parse(name: str) -> dict | None:
    m = re.match(r"(main|heldout)__(\w+?)__(\w+?)__(\w+?)__(.+)__seed(\d+)\.npz$", name)
    return None if not m else dict(study=m[1], bench=m[2], variant=m[3], enc=m[4], fold=m[5], seed=int(m[6]))


def paired_bootstrap(cases: list[dict], a: str, b: str, B: int = 2000, seed: int = 0) -> dict:
    y = cases[0]["_y"]
    rng = np.random.default_rng(seed)
    dF, dA = [], []
    for _ in range(B):
        idx = rng.integers(0, len(y), len(y))
        yy = y[idx]
        if len(np.unique(yy)) < 2:
            continue
        fa = np.mean([macro_f1(yy, c["_scores"][a][0][idx], c["_scores"][a][1]) for c in cases])
        fb = np.mean([macro_f1(yy, c["_scores"][b][0][idx], c["_scores"][b][1]) for c in cases])
        aa = np.mean([auroc(yy, c["_scores"][a][0][idx]) for c in cases])
        ab = np.mean([auroc(yy, c["_scores"][b][0][idx]) for c in cases])
        dF.append(fa - fb)
        dA.append(aa - ab)
    def summ(v):
        v = np.array(v)
        p = 2 * min((v <= 0).mean(), (v >= 0).mean())
        return {"mean": float(v.mean()), "ci95": [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))],
                "p": float(min(1.0, p))}
    return {"macro_f1": summ(dF), "auroc": summ(dA)}


def holm(ps: dict) -> dict:
    items = sorted(ps.items(), key=lambda kv: kv[1])
    m, out, run = len(items), {}, 0.0
    for i, (k, p) in enumerate(items):
        run = max(run, min(1.0, (m - i) * p))
        out[k] = run
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results", type=Path, nargs="+")
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args()
    groups: dict = defaultdict(list)
    for r in args.results:
        root = open_results(r)
        for f in sorted((root / "predictions").glob("*.npz")):
            meta = parse(f.name)
            if not meta:
                continue
            res = analyse_file(f)
            if res:
                groups[(meta["study"], meta["bench"], meta["fold"])].append({**meta, **res})
    if not groups:
        raise SystemExit("no v3 prediction files (with expert scores) found")
    names = ["det", "lex", "nov", "hybrid"]
    report: dict = {"main": {}, "heldout": {}}

    def agg(cases, model, key, sub=None):
        v = []
        for c in cases:
            x = c["models"][model].get(key) if sub is None else (c["models"][model].get(key) or {}).get(sub)
            if x is not None and not (isinstance(x, float) and math.isnan(x)):
                v.append(x)
        return (float(np.mean(v)), float(np.std(v, ddof=1)) if len(v) > 1 else 0.0) if v else (float("nan"), 0.0)

    print("# v3 analysis: experts, hybrid, conformal\n")
    print("## In-distribution (mean ± sd over seeds; validation-tuned threshold)\n")
    print("| Benchmark | weights d/l/n | " + " | ".join(f"{n} F1 / AUROC" for n in names) + " |")
    print("|---|---|" + "---|" * len(names))
    pvals = {}
    for b in ORDER:
        cases = groups.get(("main", b, "all"))
        if not cases:
            continue
        ws = np.mean([[c["weights"][e] for e in EXPERTS] for c in cases], axis=0)
        cells = [f"{agg(cases, n, 'macro_f1')[0]:.3f}±{agg(cases, n, 'macro_f1')[1]:.3f} / {agg(cases, n, 'auroc')[0]:.3f}"
                 for n in names]
        test = paired_bootstrap(cases, "hybrid", "lex")
        pvals[b] = test["macro_f1"]["p"]
        report["main"][b] = {"weights_mean": ws.tolist(), "n_seeds": len(cases),
                             "models": {n: {k: agg(cases, n, k) for k in ("macro_f1", "auroc", "recall", "fpr")} for n in names},
                             "hybrid_minus_lexical": test,
                             "conformal": {n: {a: (agg(cases, n, f"conformal_{a}", "fpr"), agg(cases, n, f"conformal_{a}", "recall"))
                                               for a in (0.05, 0.10)} for n in names},
                             "aci_0.05": {n: (agg(cases, n, "aci_0.05", "fpr"), agg(cases, n, "aci_0.05", "recall")) for n in names}}
        print(f"| {b} | {ws[0]:.2f}/{ws[1]:.2f}/{ws[2]:.2f} | " + " | ".join(cells) + " |")
    adj = holm(pvals)
    print("\n## Hybrid minus lexical (paired bootstrap over test episodes; Holm across benchmarks)\n")
    print("| Benchmark | ΔF1 [95% CI] | p (Holm) | ΔAUROC [95% CI] | p |")
    print("|---|---|---|---|---|")
    for b, v in report["main"].items():
        t = v["hybrid_minus_lexical"]
        v["hybrid_minus_lexical"]["macro_f1"]["p_holm"] = adj[b]
        f, a = t["macro_f1"], t["auroc"]
        print(f"| {b} | {f['mean']:+.3f} [{f['ci95'][0]:+.3f}, {f['ci95'][1]:+.3f}] | {adj[b]:.3f} | "
              f"{a['mean']:+.3f} [{a['ci95'][0]:+.3f}, {a['ci95'][1]:+.3f}] | {a['p']:.3f} |")
    print("\n## Conformal false-alarm control (test FPR / recall; alpha = 0.05 split, and ACI)\n")
    print("| Benchmark | " + " | ".join(f"{n} split" for n in names) + " | " + " | ".join(f"{n} ACI" for n in names) + " |")
    print("|---|" + "---|" * (2 * len(names)))
    for b, v in report["main"].items():
        sp = [v["conformal"][n][0.05] for n in names]
        ac = [v["aci_0.05"][n] for n in names]
        fmt = lambda t: "n/a" if math.isnan(t[0][0]) else f"{t[0][0]:.3f} / {t[1][0]:.3f}"
        print(f"| {b} | " + " | ".join(fmt(x) for x in sp) + " | " + " | ".join(fmt(x) for x in ac) + " |")

    print("\n## Held-out templates (mean over folds of the per-fold seed means)\n")
    print("| Benchmark | folds | " + " | ".join(f"{n} F1 / AUROC" for n in names) + " | hybrid>lex (F1) | hybrid>lex (AUROC) |")
    print("|---|---|" + "---|" * (len(names) + 2))
    wins_f = wins_a = total = 0
    for b in ORDER:
        folds = sorted(k[2] for k in groups if k[0] == "heldout" and k[1] == b)
        if not folds:
            continue
        per = {n: {"f1": [], "au": []} for n in names}
        wf = wa = 0
        for fo in folds:
            cases = groups[("heldout", b, fo)]
            for n in names:
                per[n]["f1"].append(agg(cases, n, "macro_f1")[0])
                per[n]["au"].append(agg(cases, n, "auroc")[0])
            for c in cases:
                total += 1
                gf = c["models"]["hybrid"]["macro_f1"] > c["models"]["lex"]["macro_f1"]
                ga = c["models"]["hybrid"]["auroc"] > c["models"]["lex"]["auroc"]
                wf += gf
                wa += ga
        wins_f += wf
        wins_a += wa
        n_cases = sum(len(groups[("heldout", b, fo)]) for fo in folds)
        report["heldout"][b] = {n: {"macro_f1": float(np.nanmean(per[n]["f1"])), "auroc": float(np.nanmean(per[n]["au"]))}
                                for n in names}
        report["heldout"][b]["wins_f1"], report["heldout"][b]["wins_auroc"], report["heldout"][b]["cases"] = wf, wa, n_cases
        cells = [f"{np.nanmean(per[n]['f1']):.3f} / {np.nanmean(per[n]['au']):.3f}" for n in names]
        print(f"| {b} | {len(folds)} | " + " | ".join(cells) + f" | {wf}/{n_cases} | {wa}/{n_cases} |")
    if total:
        from scipy.stats import binomtest

        pf = binomtest(wins_f, total, 0.5).pvalue
        pa = binomtest(wins_a, total, 0.5).pvalue
        report["heldout_sign_test"] = {"wins_f1": wins_f, "wins_auroc": wins_a, "n": total, "p_f1": pf, "p_auroc": pa}
        print(f"\nSign test over all held-out (fold, seed) pairs: hybrid beats lexical in F1 {wins_f}/{total} "
              f"(p={pf:.3g}), in AUROC {wins_a}/{total} (p={pa:.3g}).")
    if args.json:
        args.json.write_text(json.dumps(report, indent=1, default=str))
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
