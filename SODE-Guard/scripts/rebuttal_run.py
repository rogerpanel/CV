"""Rebuttal experiments on real flow CSVs (designed for a single Colab GPU).

Stages (run in order, or ``all``):

    prepare     read the flow CSVs in chunks, keep numeric features, cap each
                class, stratified 70/15/15 split, z-score with train statistics
    train       SODE-Guard (paper setting), SODE-Guard without L_AC,
                certifiable SODE-Guard, SDE-TGNN (re-implementation), an
                E-GraphSAGE MLP, the same MLP with PGD-AT, and a Gaussian-noise
                MLP used as the randomised-smoothing base classifier
    attack      clean metrics and the attack suite at eps (l_inf, z-units):
                PGD-40, PGD-200 with restarts, EOT-PGD-100, APGD-CE/DLR with
                EOT, expected-margin PGD, Square (black box), transfer;
                plus an eps sweep and a step sweep for the stochastic models
    certify     Theorem A radii for the certifiable model and randomised
                smoothing radii for the Gaussian MLP, at matched l2
    regulariser near-boundary mass P[G(x) <= beta] with and without L_AC
    report      print a summary of results.json

Every stochastic model is scored two ways: with the fixed seeds 0..7 used by the
deployed predictor ("fixed") and with fresh paths on every call ("fresh").

Example (Colab):
    python scripts/rebuttal_run.py prepare --csv "/content/data/*.csv" --out /content/run
    python scripts/rebuttal_run.py all --out /content/run
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.attacks.eot_autoattack import (apgd, eot_pgd, expected_margin_pgd_chunked,  # noqa: E402
                                        square_attack)
from src.baselines.egraphsage_baseline import EGraphSAGEBaseline                    # noqa: E402
from src.baselines.lipschitz_mlp import LipschitzMLP                                # noqa: E402
from src.baselines.sde_tgnn import SDE_TGNN                                          # noqa: E402
from src.certify import certify_mean, model_lipschitz                               # noqa: E402
from src.evaluation.reliability import certify_smoothing                            # noqa: E402
from src.models.sode_guard import SODEGuard, SODEGuardConfig, fresh_seeds           # noqa: E402
from src.training.adversarial import adversarial_step                              # noqa: E402
from src.training.loss import CrossEntropyWithAC                                    # noqa: E402
from src.utils.metrics import aggregate_metrics                                    # noqa: E402

LABEL_CANDIDATES = ["label", "Label", "LABEL", "attack_cat", "Attack_type", "attack_type",
                    "Category", "category", "class", "Class", "Attack", "attack", "type"]
ID_HINTS = ("flow id", "flow_id", "timestamp", "src ip", "dst ip", "source ip",
            "destination ip", "src_ip", "dst_ip", "ip.src", "ip.dst", "unnamed")
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def log(*a):
    print(time.strftime("[%H:%M:%S]"), *a, flush=True)


# ------------------------------------------------------------------ prepare
def find_label(cols: list[str], override: str | None) -> str:
    if override:
        return override
    for c in LABEL_CANDIDATES:
        if c in cols:
            return c
    raise SystemExit(f"No label column found among {cols[:30]}...; pass --label-col")


def cmd_prepare(a):
    files = sorted(glob.glob(a.csv))
    if not files:
        raise SystemExit(f"No files match {a.csv}")
    rng = np.random.default_rng(a.seed)
    header = pd.read_csv(files[0], nrows=5)
    label = find_label(list(header.columns), a.label_col)
    log("files:", len(files), "| label column:", label)
    kept: dict[str, list[pd.DataFrame]] = {}
    counts: dict[str, int] = {}
    for f in files:
        for chunk in pd.read_csv(f, chunksize=a.chunksize, low_memory=False):
            if label not in chunk.columns:
                log(f"skip {f}: no column {label}")
                break
            chunk = chunk.sample(frac=1.0, random_state=int(rng.integers(1 << 30)))
            for cls, part in chunk.groupby(chunk[label].astype(str)):
                have = counts.get(cls, 0)
                if have >= a.cap_per_class:
                    continue
                take = part.iloc[: a.cap_per_class - have]
                kept.setdefault(cls, []).append(take)
                counts[cls] = have + len(take)
        log(f"read {f}; classes so far: {len(counts)}")
    df = pd.concat([pd.concat(v) for k, v in kept.items() if counts[k] >= a.min_per_class], ignore_index=True)
    dropped = {k: n for k, n in counts.items() if n < a.min_per_class}
    y_str = df[label].astype(str)
    feat = df.drop(columns=[label])
    feat = feat[[c for c in feat.columns if not any(h in c.strip().lower() for h in ID_HINTS)]]
    feat = feat.apply(pd.to_numeric, errors="coerce")
    feat = feat.loc[:, feat.notna().mean() > 0.5]
    feat = feat.replace([np.inf, -np.inf], np.nan)
    classes = sorted(y_str.unique())
    y = y_str.map({c: i for i, c in enumerate(classes)}).to_numpy()

    from sklearn.model_selection import train_test_split
    idx = np.arange(len(y))
    split_note = "stratified random 70/15/15"
    if a.split == "temporal":
        tcol = a.time_col or next((c for c in df.columns if "time" in c.lower()), None)
        if tcol is None:
            raise SystemExit("no timestamp column found; pass --time-col")
        order = np.argsort(pd.to_datetime(df[tcol], errors="coerce").fillna(pd.Timestamp(0)).to_numpy(), kind="stable")
        n = len(order)
        tr, va, te = order[: int(.7 * n)], order[int(.7 * n): int(.85 * n)], order[int(.85 * n):]
        split_note = f"temporal 70/15/15 by {tcol}"
    else:
        tr, rest = train_test_split(idx, train_size=0.70, stratify=y, random_state=a.seed)
        va, te = train_test_split(rest, train_size=0.5, stratify=y[rest], random_state=a.seed)
    Xtr = feat.iloc[tr]
    med = Xtr.median()
    feat = feat.fillna(med)
    std = feat.iloc[tr].std().replace(0, np.nan)
    keep = std.dropna().index
    feat = feat[keep]
    mean, std = feat.iloc[tr].mean(), feat.iloc[tr].std()
    X = ((feat - mean) / std).clip(-a.clip, a.clip).to_numpy(np.float32)
    raw = feat.to_numpy(np.float64)
    row_hash = pd.util.hash_pandas_object(pd.DataFrame(raw), index=False).to_numpy()
    train_set = set(row_hash[tr].tolist())
    leakage = {"test_rows_identical_to_a_train_row": float(np.mean([h in train_set for h in row_hash[te]])),
               "duplicate_rate_overall": float(1 - len(set(row_hash.tolist())) / len(row_hash)),
               "same_features_different_label_rate": float(
                   pd.DataFrame({"h": row_hash, "y": y}).groupby("h")["y"].nunique().gt(1).mean())}
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "data.npz", Xtr=X[tr], ytr=y[tr], Xva=X[va], yva=y[va], Xte=X[te], yte=y[te])
    meta = {"files": files, "label_column": label, "classes": classes,
            "class_counts": {c: int((y_str == c).sum()) for c in classes},
            "dropped_small_classes": dropped, "features": list(keep), "n_features": len(keep),
            "cap_per_class": a.cap_per_class, "clip": a.clip, "seed": a.seed,
            "sizes": {"train": len(tr), "val": len(va), "test": len(te)},
            "split": split_note, "leakage": leakage,
            "classes_in_test": int(len(np.unique(y[te]))), "classes_in_train": int(len(np.unique(y[tr])))}
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    log(f"saved {len(y)} flows, {len(keep)} features, {len(classes)} classes ({split_note}) -> {out}")
    log("leakage check:", leakage)


# ------------------------------------------------------------------ models
def load_data(out: Path):
    d = np.load(out / "data.npz")
    t = lambda k: torch.from_numpy(d[k])
    return t("Xtr").float(), t("ytr").long(), t("Xva").float(), t("yva").long(), t("Xte").float(), t("yte").long()


def build(name: str, p: int, K: int):
    if name in {"sode", "sode_noac"}:
        return SODEGuard(SODEGuardConfig(feature_dim=p, num_classes=K))
    if name == "sode_cert":
        return SODEGuard(SODEGuardConfig(feature_dim=p, num_classes=K, certifiable=True))
    if name == "sdetgnn":
        return SDE_TGNN(num_classes=K, feature_dim=p)
    if name == "ode":
        return SODEGuard(SODEGuardConfig(feature_dim=p, num_classes=K, use_diffusion=False))
    if name == "ode_cert":
        return SODEGuard(SODEGuardConfig(feature_dim=p, num_classes=K, certifiable=True, use_diffusion=False))
    if name == "lip_mlp":
        return LipschitzMLP(feature_dim=p, num_classes=K)
    if name in {"mlp", "mlp_at", "mlp_gauss"}:
        return EGraphSAGEBaseline(feature_dim=p, num_classes=K)
    raise KeyError(name)


STOCHASTIC = {"sode", "sode_noac", "sode_cert", "sdetgnn"}
ODE = {"ode", "ode_cert"}
MODELS = ["sode", "sode_noac", "sode_cert", "sdetgnn", "ode", "ode_cert", "mlp", "mlp_at", "mlp_gauss", "lip_mlp"]
ATTACK_MODELS = ["sode", "sode_cert", "sdetgnn", "ode", "mlp", "mlp_at"]


def predictor(model, name: str, mode: str, n_paths: int = 8):
    if name in ODE:
        return lambda x: model.forward_mean(x, seeds=[0])
    if name in STOCHASTIC:
        if mode == "fixed":
            seeds = list(range(n_paths))
            return lambda x: model.forward_mean(x, seeds=seeds)
        return lambda x: model.forward_mean(x, n_paths=n_paths)
    return model


def batched_logits(f, X, bs=1024):
    with torch.no_grad():
        return torch.cat([f(X[i:i + bs].to(DEV)).float().cpu() for i in range(0, len(X), bs)])


def train_one(name, Xtr, ytr, Xva, yva, K, a, out: Path):
    torch.manual_seed(a.seed)
    model = build(name, Xtr.shape[1], K).to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr, weight_decay=1e-5)
    steps = a.epochs * math.ceil(len(Xtr) / a.batch)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    ac = CrossEntropyWithAC(ac_weight=0.1 if name in {"sode", "sode_cert", "ode", "ode_cert"} else 0.0,
                            n_ac_paths=1 if name in ODE else a.train_paths)
    g = torch.Generator().manual_seed(a.seed)
    t0 = time.time()
    for ep in range(a.epochs):
        model.train()
        perm = torch.randperm(len(Xtr), generator=g)
        tot = 0.0
        for i in range(0, len(Xtr), a.batch):
            idx = perm[i:i + a.batch]
            x, y = Xtr[idx].to(DEV), ytr[idx].to(DEV)
            if name in STOCHASTIC or name in ODE:
                loss, _ = ac(model, x, y)
            elif name == "mlp_at":
                loss = adversarial_step(model, x, y, method="pgd_at", eps=a.eps, steps=7)
            elif name == "mlp_gauss":
                loss = F.cross_entropy(model(x + a.sigma_train * torch.randn_like(x)), y)
            else:
                loss = F.cross_entropy(model(x), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            tot += float(loss.detach()) * len(idx)
        model.eval()
        va_pred = batched_logits(predictor(model, name, "fixed"), Xva).argmax(1)
        f1 = aggregate_metrics(yva.numpy(), va_pred.numpy())["macro_f1"]
        log(f"[{name}] epoch {ep + 1}/{a.epochs} loss {tot / len(Xtr):.4f} val macro-F1 {f1:.4f}")
    torch.save(model.state_dict(), out / f"{name}.pt")
    return {"train_seconds": time.time() - t0, "val_macro_f1": f1}


def load_model(name, out: Path, p, K):
    m = build(name, p, K).to(DEV)
    m.load_state_dict(torch.load(out / f"{name}.pt", map_location=DEV))
    return m.eval()


def cmd_train(a):
    out = Path(a.out)
    Xtr, ytr, Xva, yva, _, _ = load_data(out)
    K = int(max(ytr.max(), yva.max()) + 1)
    res = load_results(out)
    for name in a.models:
        res.setdefault("train", {})[name] = train_one(name, Xtr, ytr, Xva, yva, K, a, out)
        save_results(out, res)


# ------------------------------------------------------------------ attacks
def load_results(out: Path) -> dict:
    p = out / "results.json"
    return json.loads(p.read_text()) if p.exists() else {}


def save_results(out: Path, res: dict):
    (out / "results.json").write_text(json.dumps(res, indent=2))


def subset(X, y, n, seed):
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(len(X), generator=g)[:n]
    return X[idx], y[idx]


def run_batched(attack, X, y, bs):
    return torch.cat([attack(X[i:i + bs].to(DEV), y[i:i + bs].to(DEV)).detach().cpu()
                      for i in range(0, len(X), bs)])


def pgd_restarts(f, x, y, eps, steps, restarts):
    """Per-example worst case over random restarts (keeps any misclassifying start)."""
    best = x.clone()
    fooled = torch.zeros(len(x), dtype=torch.bool, device=x.device)
    for _ in range(restarts):
        xa = eot_pgd(f, x, y, eps=eps, steps=steps, eot=1)
        with torch.no_grad():
            wrong = f(xa).argmax(1) != y
        new = wrong & ~fooled
        best[new] = xa[new]
        fooled |= wrong
    return best


def scores(scorers, X_adv, y):
    out = {}
    for mode, f in scorers.items():
        pred = batched_logits(f, X_adv).argmax(1)
        out[mode] = pred
    return out


def cmd_attack(a):
    out = Path(a.out)
    Xtr, ytr, Xva, yva, Xte, yte = load_data(out)
    p, K = Xte.shape[1], int(max(ytr.max(), yte.max()) + 1)
    res = load_results(out)
    X, y = subset(Xte, yte, a.n_attack, a.seed)
    surrogate = load_model("mlp", out, p, K)
    transfer_x = run_batched(lambda xb, yb: eot_pgd(surrogate, xb, yb, eps=a.eps, steps=a.eot_steps, eot=1), X, y, a.attack_batch)
    for name in a.attack_models:
        m = load_model(name, out, p, K)
        stoch = name in STOCHASTIC
        scorers = {"fixed": predictor(m, name, "fixed")}
        if stoch:
            scorers["fresh"] = predictor(m, name, "fresh")
        f_fixed, f_fresh = scorers["fixed"], scorers.get("fresh", scorers["fixed"])
        r = {"clean": {}}
        # clean metrics on the full test split
        for mode, f in scorers.items():
            logits = batched_logits(f, Xte)
            probs = F.softmax(logits, 1).numpy()
            met = aggregate_metrics(yte.numpy(), logits.argmax(1).numpy(), probs)
            r["clean"][mode] = met
        atks = {
            "PGD-40 (fixed paths)": lambda xb, yb: eot_pgd(f_fixed, xb, yb, eps=a.eps, steps=40, eot=1),
            f"PGD-{a.long_steps} x{a.restarts} restarts (fixed)": lambda xb, yb: pgd_restarts(f_fixed, xb, yb, a.eps, a.long_steps, a.restarts),
            "APGD-CE": lambda xb, yb: apgd(f_fresh, xb, yb, eps=a.eps, n_iter=a.apgd_iters, loss="ce", eot=a.eot_apgd if stoch else 1),
            "APGD-DLR": lambda xb, yb: apgd(f_fresh, xb, yb, eps=a.eps, n_iter=a.apgd_iters, loss="dlr", eot=a.eot_apgd if stoch else 1),
            f"Square ({a.square_queries} queries)": lambda xb, yb: square_attack(f_fresh, xb, yb, eps=a.eps, n_queries=a.square_queries),
        }
        if stoch:
            atks[f"EOT-PGD-{a.eot_steps} (EOT {a.eot})"] = lambda xb, yb: eot_pgd(f_fresh, xb, yb, eps=a.eps, steps=a.eot_steps, eot=a.eot)
            atks["E-margin PGD (256 paths)"] = lambda xb, yb: expected_margin_pgd_chunked(
                m, xb, yb, eps=a.eps, steps=a.eot_steps, n_paths=256, chunk=a.em_chunk)
        yn = y.numpy()
        worst = {mode: None for mode in scorers}
        r["attacks"] = {}
        adv_sets = {"Transfer from MLP": transfer_x}
        for an, fn in atks.items():
            t0 = time.time()
            adv_sets[an] = run_batched(fn, X, y, a.em_batch if an.startswith("E-margin") else a.attack_batch)
            log(f"[{name}] {an} done in {time.time() - t0:.0f}s")
        for an, xa in adv_sets.items():
            preds = scores(scorers, xa, y)
            r["attacks"][an] = {}
            for mode, pr in preds.items():
                r["attacks"][an][mode] = aggregate_metrics(yn, pr.numpy())["macro_f1"]
                w = worst[mode]
                if w is None:
                    worst[mode] = pr.clone()
                else:
                    flip = (pr != y) & (w == y)
                    w[flip] = pr[flip]
        r["attacks"]["Worst case (all attacks)"] = {mode: aggregate_metrics(yn, w.numpy())["macro_f1"]
                                                   for mode, w in worst.items()}
        clean_sub = scores(scorers, X, y)
        r["clean_subset_macro_f1"] = {mode: aggregate_metrics(yn, pr.numpy())["macro_f1"] for mode, pr in clean_sub.items()}
        if stoch and name in {"sode", "sdetgnn"}:
            sweep = {}
            for e in a.eps_sweep:
                xa = run_batched(lambda xb, yb: eot_pgd(f_fresh, xb, yb, eps=e, steps=a.eot_steps, eot=max(a.eot // 4, 1)),
                                 X[: a.n_sweep], y[: a.n_sweep], a.attack_batch)
                sweep[str(e)] = aggregate_metrics(y[: a.n_sweep].numpy(), batched_logits(f_fresh, xa).argmax(1).numpy())["macro_f1"]
            r["eps_sweep_eot"] = sweep
            steps_r = {}
            for s in a.step_sweep:
                xa = run_batched(lambda xb, yb: eot_pgd(f_fresh, xb, yb, eps=a.eps, steps=s, eot=max(a.eot // 4, 1)),
                                 X[: a.n_sweep], y[: a.n_sweep], a.attack_batch)
                steps_r[str(s)] = aggregate_metrics(y[: a.n_sweep].numpy(), batched_logits(f_fresh, xa).argmax(1).numpy())["macro_f1"]
            r["step_sweep_eot"] = steps_r
        res.setdefault("attack", {})[name] = r
        res["attack_settings"] = {"eps_linf": a.eps, "n_attack": a.n_attack, "restarts": a.restarts,
                                  "eot": a.eot, "eot_apgd": a.eot_apgd, "long_steps": a.long_steps, "eot_steps": a.eot_steps, "apgd_iters": a.apgd_iters, "square_queries": a.square_queries,
                                  "n_sweep": a.n_sweep, "seed": a.seed}
        save_results(out, res)
        log(f"[{name}] worst-case macro-F1: {r['attacks']['Worst case (all attacks)']}")


# ------------------------------------------------------------------ certify
def lipschitz_margin_report(logits, y, L, grid):
    """Deterministic certificate r = M / (sqrt(2) L) for an L-Lipschitz logit map."""
    top2 = logits.topk(2, dim=1)
    pred = top2.indices[:, 0]
    margin = top2.values[:, 0] - top2.values[:, 1]
    r = torch.where(pred == y, margin / (math.sqrt(2) * L), torch.zeros_like(margin)).numpy()
    return {"L": float(L), "clean_accuracy": float((pred == y).float().mean()),
            "certified_accuracy_l2": {str(g): float((r > 0).mean() if g == 0 else (r >= g).mean()) for g in grid},
            "median_radius_l2_certified": float(np.median(r[r > 0])) if (r > 0).any() else 0.0}


def cmd_certify(a):
    out = Path(a.out)
    Xtr, ytr, _, _, Xte, yte = load_data(out)
    p, K = Xte.shape[1], int(max(ytr.max(), yte.max()) + 1)
    res = load_results(out)
    m = load_model("sode_cert", out, p, K)
    lip = model_lipschitz(m)
    with torch.no_grad():
        norms = m.sample_logits(Xtr[:2048].to(DEV), list(range(64))).norm(dim=-1)
    B = float(1.1 * norms.max())
    X, y = subset(Xte, yte, a.n_cert, a.seed + 1)
    radii, preds, proj = [], [], []
    for i in range(0, len(X), a.cert_batch):
        c = certify_mean(m, X[i:i + a.cert_batch].to(DEV), L=lip.L, B=B, n=a.cert_paths)
        radii.append(c.radius_l2.cpu()); preds.append(c.prediction.cpu()); proj.append(c.frac_projected)
    r2 = torch.cat(radii); pr = torch.cat(preds)
    r2c = torch.where(pr == y, r2, torch.zeros_like(r2)).numpy()
    deployed = batched_logits(predictor(m, "sode_cert", "fixed"), X).argmax(1)
    grid = [0.0, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.273, 0.5, 1.0]
    cert = {"lipschitz": lip.as_dict(), "B": B, "n_paths": a.cert_paths, "alpha": 1e-3,
            "n_flows": len(X), "frac_projected": float(np.mean(proj)),
            "abstain_rate": float((pr < 0).float().mean()),
            "certified_accuracy_l2": {str(g): float((r2c > g).mean() if g == 0 else (r2c >= g).mean()) for g in grid},
            "median_radius_l2_certified": float(np.median(r2c[r2c > 0])) if (r2c > 0).any() else 0.0,
            "agreement_deployed_vs_certified": float((deployed == pr)[pr >= 0].float().mean()) if (pr >= 0).any() else None}
    # same flows, other certificates: certifiable neural ODE (Theorem A with K_g = 0)
    # and the Lipschitz-bounded MLP (Lipschitz-margin certificate, no Monte-Carlo step)
    mo = load_model("ode_cert", out, p, K)
    lo = model_lipschitz(mo)
    with torch.no_grad():
        Bo = float(1.1 * mo.forward_mean(Xtr[:2048].to(DEV), seeds=[0]).norm(dim=-1).max())
        z = torch.cat([mo.forward_mean(X[i:i + 256].to(DEV), seeds=[0]).cpu() for i in range(0, len(X), 256)])
    cert["ode_cert"] = lipschitz_margin_report(z, y, lo.L, grid) | {"lipschitz": lo.as_dict(), "B": Bo}
    ml = load_model("lip_mlp", out, p, K)
    with torch.no_grad():
        zl = torch.cat([ml(X[i:i + 256].to(DEV)).cpu() for i in range(0, len(X), 256)])
    cert["lip_mlp"] = lipschitz_margin_report(zl, y, ml.lipschitz_constant(), grid)
    ms = load_model("mlp_gauss", out, p, K)
    rs = {}
    for s in a.sigmas:
        r = np.concatenate([certify_smoothing(ms, X[i:i + 64].to(DEV), y[i:i + 64].to(DEV), sigma=s, K=K,
                                              n0=100, n=a.smooth_n) for i in range(0, len(X), 64)])
        rs[str(s)] = {str(g): float((r > 0).mean() if g == 0 else (r >= g).mean()) for g in grid}
    cert["randomized_smoothing_certified_accuracy_l2"] = rs
    res["certify"] = cert
    save_results(out, res)
    log("certify:", json.dumps({k: v for k, v in cert.items() if k != "lipschitz"})[:800])


# ------------------------------------------------------------------ regulariser link
def near_boundary(m, X, betas, n_paths=256):
    with torch.no_grad():
        k = m.sample_logits(X, fresh_seeds(64)).mean(1).argmax(1)
        L = m.sample_logits(X, fresh_seeds(n_paths))
        top = L.gather(-1, k.view(-1, 1, 1).expand(-1, n_paths, 1)).squeeze(-1)
        oth = L.scatter(-1, k.view(-1, 1, 1).expand(-1, n_paths, 1), float("-inf")).max(-1).values
        G = top - oth
    return {str(b): float((G <= b).float().mean()) for b in betas}


def cmd_regulariser(a):
    out = Path(a.out)
    Xtr, ytr, _, _, Xte, yte = load_data(out)
    p, K = Xte.shape[1], int(max(ytr.max(), yte.max()) + 1)
    res = load_results(out)
    X, _ = subset(Xte, yte, a.n_reg, a.seed + 2)
    betas = [0.0, 0.1, 0.25, 0.5, 1.0, 2.0]
    reg = {}
    for name in ["sode", "sode_noac"]:
        m = load_model(name, out, p, K)
        vals = [near_boundary(m, X[i:i + 256].to(DEV), betas) for i in range(0, len(X), 256)]
        reg[name] = {b: float(np.mean([v[b] for v in vals])) for b in vals[0]}
    res["regulariser_near_boundary_mass"] = reg
    save_results(out, res)
    log("near-boundary mass P[G<=beta]:", json.dumps(reg))


# ------------------------------------------------------------------ report
def cmd_report(a):
    res = load_results(Path(a.out))
    meta = json.loads((Path(a.out) / "meta.json").read_text())
    print(f"# data: {meta['sizes']} flows, {meta['n_features']} features, {len(meta['classes'])} classes")
    for name, r in res.get("attack", {}).items():
        print(f"\n## {name}")
        print("clean macro-F1:", {k: round(v['macro_f1'], 4) for k, v in r["clean"].items()})
        for an, v in r["attacks"].items():
            print(f"  {an:34s}", {k: round(x, 4) for k, x in v.items()})
        if "eps_sweep_eot" in r:
            print("  eps sweep (EOT-PGD, fresh):", r["eps_sweep_eot"])
            print("  step sweep (EOT-PGD, fresh):", r["step_sweep_eot"])
    if "certify" in res:
        c = res["certify"]
        print("\n## certify  L =", round(c["lipschitz"]["L"], 3), "| abstain", round(c["abstain_rate"], 3))
        print("  SODE-Guard (Theorem A):", c["certified_accuracy_l2"])
        for key in ("ode_cert", "lip_mlp"):
            if key in c:
                print(f"  {key} (L={c[key]['L']:.3f}):", c[key]["certified_accuracy_l2"])
        for s, v in c["randomized_smoothing_certified_accuracy_l2"].items():
            print(f"  smoothing sigma={s}:", v)
    if "regulariser_near_boundary_mass" in res:
        print("\n## near-boundary mass P[G<=beta]:", res["regulariser_near_boundary_mass"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["prepare", "train", "attack", "certify", "regulariser", "report", "all"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--csv", help="glob of flow CSV files (prepare)")
    ap.add_argument("--label-col")
    ap.add_argument("--cap-per-class", type=int, default=20000)
    ap.add_argument("--min-per-class", type=int, default=200)
    ap.add_argument("--chunksize", type=int, default=500_000)
    ap.add_argument("--clip", type=float, default=10.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--models", nargs="+", default=MODELS, help="models to train")
    ap.add_argument("--attack-models", nargs="+", default=ATTACK_MODELS, help="models to attack")
    ap.add_argument("--split", choices=["random", "temporal"], default="random")
    ap.add_argument("--time-col", help="timestamp column for --split temporal (auto-detected)")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--train-paths", type=int, default=4)
    ap.add_argument("--sigma-train", type=float, default=0.25)
    ap.add_argument("--eps", type=float, default=0.03)
    ap.add_argument("--n-attack", type=int, default=1000)
    ap.add_argument("--attack-batch", type=int, default=500)
    ap.add_argument("--em-batch", type=int, default=250, help="batch for the 256-path attack")
    ap.add_argument("--em-chunk", type=int, default=32, help="paths per backward pass in that attack")
    ap.add_argument("--restarts", type=int, default=5)
    ap.add_argument("--long-steps", type=int, default=200, help="PGD steps per restart")
    ap.add_argument("--eot-steps", type=int, default=100, help="steps of EOT-PGD, E-margin PGD, transfer, eps sweep")
    ap.add_argument("--apgd-iters", type=int, default=100)
    ap.add_argument("--eot", type=int, default=32)
    ap.add_argument("--eot-apgd", type=int, default=8)
    ap.add_argument("--square-queries", type=int, default=5000)
    ap.add_argument("--eps-sweep", type=float, nargs="+", default=[0.01, 0.03, 0.1, 0.3, 1.0])
    ap.add_argument("--step-sweep", type=int, nargs="+", default=[10, 40, 100, 200])
    ap.add_argument("--n-sweep", type=int, default=1000)
    ap.add_argument("--n-cert", type=int, default=500)
    ap.add_argument("--cert-batch", type=int, default=50)
    ap.add_argument("--cert-paths", type=int, default=4096)
    ap.add_argument("--sigmas", type=float, nargs="+", default=[0.12, 0.25, 0.5])
    ap.add_argument("--smooth-n", type=int, default=1000)
    ap.add_argument("--n-reg", type=int, default=2000)
    a = ap.parse_args()
    log("device:", DEV)
    if a.stage == "prepare":
        cmd_prepare(a)
    elif a.stage == "all":
        cmd_train(a); cmd_attack(a); cmd_certify(a); cmd_regulariser(a); cmd_report(a)
    else:
        {"train": cmd_train, "attack": cmd_attack, "certify": cmd_certify,
         "regulariser": cmd_regulariser, "report": cmd_report}[a.stage](a)


if __name__ == "__main__":
    main()
