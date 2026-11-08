"""Offline expert mixture and conformal thresholds on recorded GPU predictions.

Reads the detector's saved validation/test probabilities from a
run_kaggle_study.py output (no GPU needed), recomputes the lexical baseline's
probabilities on exactly the same episodes, and evaluates

* the two experts alone (validation-tuned threshold, as in the paper);
* three mixtures p = w * p_detector + (1 - w) * p_lexical:
    equal   w = 0.5 (no tuning);
    hedge   w from Hedge / multiplicative weights run over the validation
            stream with absolute loss and eta = sqrt(8 ln N / T)
            (the same update as mambaguard.certification.hedge);
    tuned   w chosen on validation macro-F1 from {0, 0.1, ..., 1};
* split-conformal thresholds: tau is the ceil((n+1)(1-alpha))-th smallest
  validation *benign* score, so a new benign window is flagged with
  probability <= alpha when benign traffic is exchangeable -- whatever the
  attacks look like. Reported for alpha in {0.05, 0.10}; undefined when the
  validation split has too few benign windows.

Everything tuned uses validation data only; test data is used once.

    python scripts/expert_mixture.py reproducibility/kaggle_gpu_v2 --data-root data/raw \
        --out reproducibility/expert_mixture_v2.json
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _runner():
    spec = importlib.util.spec_from_file_location("run_kaggle_study", ROOT / "kaggle" / "run_kaggle_study.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["run_kaggle_study"] = mod
    spec.loader.exec_module(mod)
    return mod


R = _runner()


def open_results(path: Path) -> Path:
    """Accept a results folder or a zip of one (as written by run_kaggle_study.py)."""
    if path.suffix == ".zip":
        import tempfile
        import zipfile

        tmp = Path(tempfile.mkdtemp(prefix="mg_results_"))
        with zipfile.ZipFile(path) as zf:
            zf.extractall(tmp)
        kids = [d for d in tmp.iterdir() if d.is_dir()]
        path = kids[0] if len(kids) == 1 and not (tmp / "predictions").exists() else tmp
    return path


def lexical_probs(eps: dict) -> tuple[np.ndarray, np.ndarray]:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression

    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=50000, sublinear_tf=True)
    xtr = vec.fit_transform([R.episode_text(e) for e in eps["train"]])
    clf = LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000).fit(
        xtr, [int(e.is_attack) for e in eps["train"]])
    pv = clf.predict_proba(vec.transform([R.episode_text(e) for e in eps["val"]]))[:, 1]
    pt = clf.predict_proba(vec.transform([R.episode_text(e) for e in eps["test"]]))[:, 1]
    return pv, pt


def hedge_weight(yv: np.ndarray, experts_v: list[np.ndarray], seed: int = 0) -> np.ndarray:
    order = np.random.default_rng(seed).permutation(len(yv))
    n, T = len(experts_v), len(yv)
    eta = math.sqrt(8.0 * math.log(n) / T)   # losses in [0, 1]
    logw = np.zeros(n)
    for i in order:
        loss = np.array([abs(e[i] - yv[i]) for e in experts_v])
        logw -= eta * loss
    w = np.exp(logw - logw.max())
    return w / w.sum()


def conformal_threshold(yv: np.ndarray, pv: np.ndarray, alpha: float) -> float | None:
    ben = np.sort(pv[yv == 0])
    k = math.ceil((len(ben) + 1) * (1 - alpha))
    return None if k > len(ben) else float(ben[k - 1])


def evaluate(yv, pv, yt, pt) -> dict:
    thr = R.tune_threshold(yv, pv)
    out = {"val_tuned": R.metrics(yt, pt, thr), "threshold": thr}
    for a in (0.05, 0.10):
        tau = conformal_threshold(yv, pv, a)
        if tau is None:
            out[f"conformal_{a}"] = None
        else:
            m = R.metrics(yt, (pt > tau).astype(float), 0.5)
            m["tau"] = tau
            out[f"conformal_{a}"] = m
    return out


def run_case(det_files: list[Path], eps: dict) -> dict:
    lv, lt = lexical_probs(eps)
    ids = [e.episode_id for e in eps["test"]]
    yt_ref = np.array([int(e.is_attack) for e in eps["test"]])
    yv_ref = np.array([int(e.is_attack) for e in eps["val"]])
    per_seed = []
    for f in det_files:
        d = np.load(f)
        assert list(d["episode_id"]) == ids, f"{f.name}: test episodes do not align"
        assert (d["label"] == yt_ref).all() and (d["val_label"] == yv_ref).all(), f"{f.name}: labels do not align"
        dv, dt = d["val_prob"], d["prob"]
        w_h = hedge_weight(yv_ref, [dv, lv])[0]
        grid = np.round(np.linspace(0, 1, 11), 2)
        f1 = [R.metrics(yv_ref, w * dv + (1 - w) * lv, R.tune_threshold(yv_ref, w * dv + (1 - w) * lv))["macro_f1"]
              for w in grid]
        w_t = float(grid[int(np.argmax(f1))])
        res = {"seed_file": f.name, "w_hedge": float(w_h), "w_tuned": w_t,
               "detector": evaluate(yv_ref, dv, yt_ref, dt), "lexical": evaluate(yv_ref, lv, yt_ref, lt)}
        for name, w in (("equal", 0.5), ("hedge", w_h), ("tuned", w_t)):
            res[f"mix_{name}"] = evaluate(yv_ref, w * dv + (1 - w) * lv, yt_ref, w * dt + (1 - w) * lt)
        per_seed.append(res)
    return {"n_test": len(ids), "n_test_benign": int((yt_ref == 0).sum()), "n_val_benign": int((yv_ref == 0).sum()),
            "seeds": per_seed}


def mean_of(rows: list[dict], who: str, mode: str, key: str) -> float | None:
    vals = [r[who][mode][key] for r in rows if r[who].get(mode) and key in r[who][mode]]
    vals = [v for v in vals if v is not None and not (isinstance(v, float) and math.isnan(v))]
    return float(np.mean(vals)) if vals else None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results", type=Path)
    ap.add_argument("--data-root", default="data/raw")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--studies", nargs="+", default=["main", "heldout"])
    args = ap.parse_args()
    pred = open_results(args.results) / "predictions"
    rargs = SimpleNamespace(data_root=args.data_root, a2a_pairs=200, heldout_a2a_pairs=100, smoke=False,
                            device="cpu", agentdojo_model=None)
    data = R.Data(rargs, None)
    out: dict = {"main": {}, "heldout": {}}
    benches = ["a2a_sandbox", "mcptox", "mcp_safetybench", "agentdojo", "injecagent"]
    if "main" in args.studies:
        for b in benches:
            files = sorted(pred.glob(f"main__{b}__full__*__all__seed*.npz"))
            if not files:
                continue
            eps = {s: data.split(b, s) for s in ("train", "val", "test")}
            out["main"][b] = run_case(files, eps)
            print("main", b, "done", flush=True)
    if "heldout" in args.studies:
        for b in benches:
            for fold, key, held in R.heldout_folds(b):
                tag = fold.replace("/", "_").replace(" ", "_")
                files = sorted(pred.glob(f"heldout__{b}__full__*__{tag}__seed*.npz"))
                if not files:
                    continue
                eps = R.heldout_split(data, b, key, held, rargs)
                out["heldout"].setdefault(b, {})[fold] = run_case(files, eps)
            print("heldout", b, "done", flush=True)

    whos = ["detector", "lexical", "mix_equal", "mix_hedge", "mix_tuned"]
    print("\n## Validation-tuned threshold: macro-F1 / AUROC (mean over seeds; held-out: mean over folds)\n")
    print("| study | benchmark | " + " | ".join(whos) + " |")
    print("|---|---|" + "---|" * len(whos))
    summary: dict = {}
    for study in ("main", "heldout"):
        for b, v in out[study].items():
            cases = [v] if study == "main" else list(v.values())
            cells = []
            for who in whos:
                f1 = np.mean([mean_of(c["seeds"], who, "val_tuned", "macro_f1") for c in cases])
                au = np.mean([mean_of(c["seeds"], who, "val_tuned", "auroc") or np.nan for c in cases])
                summary.setdefault(study, {}).setdefault(b, {})[who] = {"macro_f1": float(f1), "auroc": float(au)}
                cells.append(f"{f1:.3f} / {au:.3f}")
            print(f"| {study} | {b} | " + " | ".join(cells) + " |")
    print("\n## Conformal threshold (alpha = 0.05): test FPR / attack recall\n")
    print("| study | benchmark | " + " | ".join(whos) + " |")
    print("|---|---|" + "---|" * len(whos))
    for study in ("main", "heldout"):
        for b, v in out[study].items():
            cases = [v] if study == "main" else list(v.values())
            cells = []
            for who in whos:
                fp = [mean_of(c["seeds"], who, "conformal_0.05", "fpr") for c in cases]
                rc = [mean_of(c["seeds"], who, "conformal_0.05", "attack_recall") for c in cases]
                fp, rc = [x for x in fp if x is not None], [x for x in rc if x is not None]
                summary[study][b][who]["conformal_0.05"] = {"fpr": float(np.mean(fp)) if fp else None,
                                                            "recall": float(np.mean(rc)) if rc else None}
                cells.append(f"{np.mean(fp):.3f} / {np.mean(rc):.3f}" if fp else "n/a")
            print(f"| {study} | {b} | " + " | ".join(cells) + " |")
    if args.out:
        args.out.write_text(json.dumps({"summary": summary, "detail": out}, indent=1, default=str))
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
