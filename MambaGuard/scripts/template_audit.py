"""Template audit: how much of detection is explained by similarity to training attacks?

CPU only; needs the benchmark clones (data/raw) but no GPU and no trained model.
For every benchmark's in-distribution split and every held-out-template fold:

* template similarity: the *injected content* of an attack episode is the set
  of chunks of its attacker-controlled messages (chunked as by the payload
  encoder) that occur verbatim in no benign episode of the benchmark -- the
  poisoned tool, the injected sentence run, the altered A2A field. Each test
  attack's injected content is embedded with TF-IDF (fitted on the training
  attacks' injected content, same settings as the lexical baseline) and
  compared by cosine similarity with the nearest training attack.
  1 = a training attack with the same wording exists.
* the lexical baseline's test AUROC on that split (as in the paper);
* the neural detector's AUROC on that fold, read from the five-seed run's
  printed results (reproducibility/kaggle_gpu_v3/RESULTS_TRANSCRIBED.md);
* Spearman rank correlation, across all splits, between median template
  similarity and AUROC.

In-distribution splits also get
* the 10 heaviest attack and 5 heaviest benign features of the lexical model;
* a feature-removal curve: delete the k most attack-indicative features
  (k = 0, 10, 50, 200, 1000, ranked by the full model), refit, test AUROC.

    python scripts/template_audit.py --data-root data/raw --out reproducibility/template_audit.json
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy.stats import spearmanr
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import expert_mixture as EM  # noqa: E402  (loads kaggle/run_kaggle_study.py as EM.R)
from mambaguard.data.chunked_encoder import chunk_payload  # noqa: E402

R = EM.R
BENCHES = ["a2a_sandbox", "mcptox", "mcp_safetybench", "agentdojo", "injecagent"]
NAMES = {"A2A sandbox": "a2a_sandbox", "MCPTox": "mcptox", "MCP-SafetyBench": "mcp_safetybench",
         "AgentDojo": "agentdojo", "InjecAgent": "injecagent"}
KS = [0, 10, 50, 200, 1000]
VEC = dict(ngram_range=(1, 2), min_df=2, max_features=50000, sublinear_tf=True)


def _payload(m) -> str:
    return m.payload if isinstance(m.payload, str) else json.dumps(m.payload)


def benign_chunks(episodes) -> set:
    return {c for e in episodes if not e.is_attack for m in e.messages for c in chunk_payload(_payload(m))}


def injected_text(ep, benign: set) -> str:
    msgs = [m for m in ep.messages if getattr(m, "label", 0)] or ep.messages
    chunks = [c for m in msgs for c in chunk_payload(_payload(m))]
    new = [c for c in chunks if c not in benign]
    return " \n ".join(new or chunks)


def fit_lexical(train, test):
    vec = TfidfVectorizer(**VEC)
    xtr = vec.fit_transform([R.episode_text(e) for e in train])
    ytr = np.array([int(e.is_attack) for e in train])
    clf = LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000).fit(xtr, ytr)
    xte = vec.transform([R.episode_text(e) for e in test])
    return vec, clf, xtr, ytr, xte


def auroc(y, s) -> float | None:
    return float(roc_auc_score(y, s)) if len(set(y)) == 2 else None


def similarity(train, test, benign: set) -> dict:
    tr_att = [e for e in train if e.is_attack]
    te_att = [e for e in test if e.is_attack]
    vec = TfidfVectorizer(**VEC).fit([injected_text(e, benign) for e in tr_att])
    a = vec.transform([injected_text(e, benign) for e in tr_att])
    b = vec.transform([injected_text(e, benign) for e in te_att])
    nn = np.asarray((b @ a.T).max(axis=1).todense()).ravel()
    return {"median": float(np.median(nn)), "q25": float(np.percentile(nn, 25)), "q75": float(np.percentile(nn, 75)),
            "frac_ge_0.9": float(np.mean(nn >= 0.9)), "n_test_attacks": int(len(nn))}


def removal_curve(xtr, ytr, xte, yte, coef) -> dict:
    order = np.argsort(-coef)  # most attack-indicative first
    out = {}
    for k in KS:
        keep = np.ones(xtr.shape[1], dtype=bool)
        keep[order[:k]] = False
        clf = LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000).fit(xtr[:, keep], ytr)
        out[str(k)] = auroc(yte, clf.predict_proba(xte[:, keep])[:, 1])
    return out


def detector_fold_auroc(path: Path) -> dict:
    """Detector AUROC per held-out fold from the transcribed runner table."""
    out: dict = {}
    if not path.exists():
        return out
    text = path.read_text()
    sec = text.split("## Held-out templates (test attacks", 1)[-1].split("## Certified", 1)[0]
    for line in sec.splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) < 6 or cells[0] not in NAMES or "mean" in cells[1]:
            continue
        au = re.match(r"([0-9.]+)", cells[4])
        if au:
            out[(NAMES[cells[0]], cells[1])] = float(au.group(1))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", default="data/raw")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    rargs = SimpleNamespace(data_root=args.data_root, a2a_pairs=200, heldout_a2a_pairs=100, smoke=False,
                            device="cpu", agentdojo_model=None, calibration_split=False)
    data = R.Data(rargs, None)
    det_fold = detector_fold_auroc(ROOT / "reproducibility" / "kaggle_gpu_v3" / "RESULTS_TRANSCRIBED.md")
    det_main = {"a2a_sandbox": 0.997, "mcptox": 0.781, "mcp_safetybench": 0.761, "agentdojo": 0.987, "injecagent": 1.0}
    report: dict = {"main": {}, "heldout": {}}
    for b in BENCHES:
        tr, te = data.split(b, "train"), data.split(b, "test")
        ben = benign_chunks(tr + data.split(b, "val") + te)
        vec, clf, xtr, ytr, xte = fit_lexical(tr, te)
        yte = np.array([int(e.is_attack) for e in te])
        names = np.array(vec.get_feature_names_out())
        coef = clf.coef_.ravel()
        report["main"][b] = {
            "similarity": similarity(tr, te, ben),
            "lexical_auroc": auroc(yte, clf.predict_proba(xte)[:, 1]),
            "detector_auroc": det_main[b],
            "top_attack_features": names[np.argsort(-coef)[:10]].tolist(),
            "top_benign_features": names[np.argsort(coef)[:5]].tolist(),
            "removal_auroc": removal_curve(xtr, ytr, xte, yte, coef),
        }
        print("main", b, report["main"][b]["similarity"]["median"], report["main"][b]["lexical_auroc"],
              report["main"][b]["removal_auroc"], flush=True)
        for fold, key, held in R.heldout_folds(b):
            eps = R.heldout_split(data, b, key, held, rargs)
            fben = ben if b != "agentdojo" else benign_chunks(eps["train"] + eps["val"] + eps["test"])
            _, clf_f, _, _, xte_f = fit_lexical(eps["train"], eps["test"])
            y = np.array([int(e.is_attack) for e in eps["test"]])
            report["heldout"].setdefault(b, {})[fold] = {
                "similarity": similarity(eps["train"], eps["test"], fben),
                "lexical_auroc": auroc(y, clf_f.predict_proba(xte_f)[:, 1]),
                "detector_auroc": det_fold.get((b, fold)),
            }
            r = report["heldout"][b][fold]
            print("  ", fold, round(r["similarity"]["median"], 3), r["lexical_auroc"], r["detector_auroc"], flush=True)
    pts = [(v["similarity"]["median"], v["lexical_auroc"], v["detector_auroc"]) for v in report["main"].values()]
    pts += [(v["similarity"]["median"], v["lexical_auroc"], v["detector_auroc"])
            for f in report["heldout"].values() for v in f.values()]
    for who, i in (("lexical", 1), ("detector", 2)):
        xs = [p[0] for p in pts if p[i] is not None]
        ys = [p[i] for p in pts if p[i] is not None]
        rho, p = spearmanr(xs, ys)
        report[f"spearman_{who}"] = {"rho": float(rho), "p": float(p), "n": len(xs)}
        print(f"Spearman(similarity, {who} AUROC) = {rho:.3f} (p={p:.2g}, n={len(xs)})")
    for b, folds in report["heldout"].items():
        med = [v["similarity"]["median"] for v in folds.values()]
        report["main"][b]["heldout_similarity_mean"] = float(np.mean(med))
    if args.out:
        args.out.write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
