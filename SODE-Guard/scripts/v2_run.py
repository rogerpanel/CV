"""SODE-Guard v2 re-run: modules 1-3 of the research report, over several seeds.

Modules
    1  clean macro-F1: logit-adjusted (or focal / class-balanced) loss, optional
       class-balanced sampling, AdamW + cosine LR, a full training budget with
       early stopping on validation macro-F1, an optional differentiable quantile
       input encoding, and a post-hoc logit adjustment tuned on validation data
    2  EOT-matched adversarial training: the adversary differentiates through
       the trained predictor itself (for SODE-Guard, the M-path mean logits with
       fresh Brownian paths), N-FGSM / PGD-K / TRADES, an eps ramp and early
       stopping on robust validation macro-F1 (PGD-10)
    3  smoothing certificate: Gaussian input noise (a pure-diffusion prefix of
       the SDE) with consistency regularisation in training, and the Cohen et
       al. certificate over the joint input noise and Brownian paths

Stages
    prepare    read the CSVs, cap each class, drop identifier / label-like columns,
               remove exact duplicates and feature vectors with conflicting labels
    train      for every seed: a new stratified 60/20/20 split (validation split
               into a selection half and a calibration half), z-scoring with the
               training statistics, then every requested model
    attack     clean metrics on the full test split and the adaptive attack suite
               on a class-stratified test subset (profile lite or full)
    certify    randomised-smoothing certificates for the *_rs models
    aggregate  mean and 95% CI over seeds; paired bootstrap contrasts with Holm
               correction over a fixed family; writes summary.json and summary.md
    report     print summary.md

Layout: OUT/full.npz, OUT/meta.json, OUT/seed_<s>/{data.npz, results.json, <model>.pt, preds_*.npz}

Example (Colab):
    python scripts/v2_run.py prepare --csv /content/ics3d/Containers_Dataset.csv --label-col Label \
        --drop-cols "Src Port" --out /content/v2
    python scripts/v2_run.py train --out /content/v2 --seeds 0 1 2 3 4
    python scripts/v2_run.py attack --out /content/v2 --seeds 0 1 2 3 4 --profile lite
    python scripts/v2_run.py certify --out /content/v2 --seeds 0 1 2 3 4 --profile lite
    python scripts/v2_run.py aggregate --out /content/v2
"""
from __future__ import annotations

import argparse
import copy
import glob
import json
import math
import pickle
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sklearn.metrics import f1_score                                                  # noqa: E402
import rebuttal_run as rr                                                              # noqa: E402
from src.attacks.eot_autoattack import apgd, eot_pgd, expected_margin_pgd_chunked, square_attack  # noqa: E402
from src.baselines.egraphsage_baseline import EGraphSAGEBaseline                       # noqa: E402
from src.baselines.sde_tgnn import SDE_TGNN                                             # noqa: E402
from src.certify.smoothing_sde import certify_randomized, radius_ceiling              # noqa: E402
from src.evaluation.stats import holm, mean_ci, paired_bootstrap                       # noqa: E402
from src.models.pipeline import Pipeline, QuantileEncoder                              # noqa: E402
from src.models.sode_guard import SODEGuard, SODEGuardConfig                           # noqa: E402
from src.training.imbalance import (ImbalanceLoss, _predict as _vote_or_mean,  # noqa: E402
                                    class_balanced_sampler, fast_macro_f1, tune_per_class_bias, tune_tau)
from src.training.loss import CrossEntropyWithAC                                        # noqa: E402
from src.training.robust import adversarial_loss, eps_schedule, gaussian_consistency_loss  # noqa: E402
from src.utils.metrics import aggregate_metrics                                        # noqa: E402

DEV, log = rr.DEV, rr.log
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*eval_set.*")
RADII = [0.0, 0.05, 0.1, 0.2, 0.273, 0.5]

# ------------------------------------------------------------------ model zoo
BASE_SPEC = dict(arch="sode", train="clean", loss="la", sampler=0.0, encoding="none", epochs=None,
                 posthoc="tau", at_method=None, at_steps=5, recipe="v2", source=None)
SPECS = {
    # module 1 applied to every neural model
    "sode": {}, "ode": {"arch": "ode"}, "mlp": {"arch": "mlp"}, "sdetgnn": {"arch": "sdetgnn"},
    # module 2: the same adversarial-training recipe for every architecture
    "sode_at": {"train": "at"}, "ode_at": {"arch": "ode", "train": "at"}, "mlp_at": {"arch": "mlp", "train": "at"},
    # module 3: Gaussian input noise + consistency, certified by smoothing
    "sode_rs": {"train": "rs"}, "ode_rs": {"arch": "ode", "train": "rs"}, "mlp_rs": {"arch": "mlp", "train": "rs"},
    # the ICLR-rebuttal recipe (CE + L_AC, 10 epochs, no early stopping, no post-hoc) on the v2 split
    "sode_base": {"loss": "ce", "epochs": 10, "posthoc": "none", "recipe": "rebuttal"},
    # ablations of module 1
    "sode_noposthoc": {"source": "sode", "posthoc": "none"},
    "sode_ce": {"loss": "ce"}, "sode_focal": {"loss": "focal"}, "sode_cb": {"loss": "cb"},
    "sode_sampler": {"loss": "ce", "sampler": 0.5}, "sode_quant": {"encoding": "quantile"},
    "sode_perclass": {"source": "sode", "posthoc": "per_class"},
    # ablations of module 2
    "sode_at_pgd": {"train": "at", "at_method": "pgd"}, "sode_at_trades": {"train": "at", "at_method": "trades"},
    "mlp_at_pgd": {"arch": "mlp", "train": "at", "at_method": "pgd"},
    # gradient-boosted trees and a random forest
    "lgbm": {"arch": "lgbm"}, "xgb": {"arch": "xgb"}, "rf": {"arch": "rf"},
}
CORE = ["sode_base", "sode", "ode", "mlp", "lgbm", "xgb", "rf", "sode_at", "ode_at", "mlp_at", "sode_rs", "mlp_rs"]
ABLATIONS = ["sode_noposthoc", "sode_perclass", "sode_ce", "sode_focal", "sode_cb", "sode_sampler", "sode_quant",
             "sode_at_pgd", "sode_at_trades", "mlp_at_pgd", "ode_rs", "sdetgnn"]
TREES = {"lgbm", "xgb", "rf"}

# contrasts declared before the run: (key, model A, model B, metric); Holm over the ones present
CONTRASTS = [
    ("module 1: sode vs sode_base (clean)", "sode", "sode_base", "clean"),
    ("sode vs mlp (clean)", "sode", "mlp", "clean"),
    ("sode vs ode (clean)", "sode", "ode", "clean"),
    ("sode vs lgbm (clean)", "sode", "lgbm", "clean"),
    ("module 2: sode_at vs sode (worst case)", "sode_at", "sode", "worst"),
    ("sode_at vs mlp_at (worst case)", "sode_at", "mlp_at", "worst"),
    ("sode_at vs ode_at (worst case)", "sode_at", "ode_at", "worst"),
    ("module 3: sode_rs vs mlp_rs (certified @0.273)", "sode_rs", "mlp_rs", "cert@0.273"),
]

PROFILES = {
    "lite": dict(n_attack=500, restarts=2, long_steps=100, eot=8, eot_apgd=4, apgd_iters=50, eot_steps=50,
                 square_queries=1000, em_paths=64, n_cert=500, smooth_n=2000),
    "full": dict(n_attack=1000, restarts=5, long_steps=200, eot=32, eot_apgd=8, apgd_iters=100, eot_steps=100,
                 square_queries=5000, em_paths=256, n_cert=500, smooth_n=10000),
}


def spec_of(name):
    return {**BASE_SPEC, **SPECS[name]}


def kind_of(spec):
    return {"sode": "sde", "sdetgnn": "sde", "ode": "ode", "mlp": "mlp"}.get(spec["arch"], "tree")


# ------------------------------------------------------------------ prepare
def cmd_prepare(a):
    files = sorted(glob.glob(a.csv, recursive=True))
    if not files:
        raise SystemExit(f"No files match {a.csv}")
    rng = np.random.default_rng(a.data_seed)
    header = pd.read_csv(files[0], nrows=5)
    label = rr.find_label(list(header.columns), a.label_col)
    log("files:", len(files), "| label column:", label)
    kept, counts = {}, {}
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
    df = pd.concat([pd.concat(v) for v in kept.values()], ignore_index=True)
    tcol = a.time_col or next((c for c in df.columns if c.strip().lower() in ("timestamp", "time", "ts")), None)
    t = (pd.to_datetime(df[tcol], errors="coerce").astype("int64").to_numpy() if tcol
         else np.full(len(df), -1, dtype=np.int64))
    y_str = df[label].astype(str)
    feat = df.drop(columns=[label])
    feat = feat[[c for c in feat.columns if not any(h in c.strip().lower() for h in rr.ID_HINTS)]]
    leaky = [c for c in feat.columns if "label" in c.lower() or c.lower().startswith("attack") or c in (a.drop_cols or [])]
    feat = feat.drop(columns=leaky).apply(pd.to_numeric, errors="coerce")
    feat = feat.loc[:, feat.notna().mean() > 0.5].replace([np.inf, -np.inf], np.nan)
    # exact duplicates and conflicting labels (Arp et al. 2022; Engelen et al. 2021)
    h = pd.util.hash_pandas_object(feat, index=False).to_numpy()
    conflict = pd.DataFrame({"h": h, "y": y_str.to_numpy()}).groupby("h")["y"].transform("nunique").to_numpy() > 1
    first = ~pd.Series(h).duplicated().to_numpy()
    keep = first & ~conflict
    dedup = {"rows_read": int(len(h)), "rows_with_conflicting_labels": int(conflict.sum()),
             "exact_duplicates_removed": int((~first & ~conflict).sum()), "rows_kept": int(keep.sum())}
    feat, y_str, t = feat[keep].reset_index(drop=True), y_str[keep].reset_index(drop=True), t[keep]
    vc = y_str.value_counts()
    small = {k: int(v) for k, v in vc.items() if v < a.min_per_class}
    ok = ~y_str.isin(list(small)).to_numpy()
    feat, y_str, t = feat[ok], y_str[ok], t[ok]
    classes = sorted(y_str.unique())
    y = y_str.map({c: i for i, c in enumerate(classes)}).to_numpy()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "full.npz", X=feat.to_numpy(np.float32), y=y, t=t)
    meta = {"files": files, "label_column": label, "time_column": tcol, "classes": classes,
            "class_counts_after_dedup": {c: int((y_str == c).sum()) for c in classes},
            "dropped_small_classes": small, "dropped_label_like_columns": leaky,
            "raw_features": list(feat.columns), "cap_per_class": a.cap_per_class, "dedup": dedup}
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    log(f"saved {len(y)} flows, {feat.shape[1]} raw features, {len(classes)} classes -> {out}")
    log("dedup:", dedup)


# ------------------------------------------------------------------ split per seed
class Data:
    pass


def seed_dir(out, seed):
    d = Path(out) / f"seed_{seed}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def make_split(a, seed):
    from sklearn.model_selection import train_test_split
    out, sd = Path(a.out), seed_dir(a.out, seed)
    if (sd / "data.npz").exists():
        return
    d = np.load(out / "full.npz")
    Xr, y, t = d["X"].astype(np.float64), d["y"], d["t"]
    idx = np.arange(len(y))
    if a.split == "class_time":
        if (t < 0).all():
            raise SystemExit("no timestamp column was found at prepare time")
        tr, va, te = [], [], []
        for c in np.unique(y):
            ic = idx[y == c][np.argsort(t[y == c], kind="stable")]
            n = len(ic)
            tr += list(ic[: int(.6 * n)]); va += list(ic[int(.6 * n): int(.8 * n)]); te += list(ic[int(.8 * n):])
        tr, va, te = map(np.array, (tr, va, te))
    else:
        tr, rest = train_test_split(idx, train_size=0.6, stratify=y, random_state=seed)
        va, te = train_test_split(rest, train_size=0.5, stratify=y[rest], random_state=seed)
    sel, cal = train_test_split(va, train_size=0.5, stratify=y[va], random_state=seed)
    med = np.nanmedian(Xr[tr], axis=0)
    Xr = np.where(np.isnan(Xr), med, Xr)
    mu, sdv = Xr[tr].mean(0), Xr[tr].std(0)
    cols = np.where(sdv > 0)[0]
    X = np.clip((Xr[:, cols] - mu[cols]) / sdv[cols], -a.clip, a.clip).astype(np.float32)
    np.savez_compressed(sd / "data.npz", Xtr=X[tr], ytr=y[tr], Xsel=X[sel], ysel=y[sel], Xcal=X[cal], ycal=y[cal],
                        Xte=X[te], yte=y[te], cols=cols)
    (sd / "split.json").write_text(json.dumps({"seed": seed, "split": a.split, "n_features": int(len(cols)),
                                               "sizes": {"train": len(tr), "selection": len(sel), "calibration": len(cal), "test": len(te)}}))
    log(f"[seed {seed}] split {a.split}: train {len(tr)}, sel {len(sel)}, cal {len(cal)}, test {len(te)}, {len(cols)} features")


def load_split(a, seed) -> Data:
    d = np.load(seed_dir(a.out, seed) / "data.npz")
    D = Data()
    for k in ("tr", "sel", "cal", "te"):
        setattr(D, "X" + k, torch.from_numpy(d["X" + k]).float())
        setattr(D, "y" + k, torch.from_numpy(d["y" + k]).long())
    D.p = D.Xtr.shape[1]
    D.K = int(max(D.ytr.max(), D.yte.max()) + 1)
    D.counts = torch.bincount(D.ytr, minlength=D.K)
    D.prior = D.counts.float() / D.counts.sum()
    return D


def strat_idx(y: torch.Tensor, n: int, seed: int) -> torch.Tensor:
    """Class-stratified subset (up to ceil(n/K) per class, topped up at random)."""
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(y), generator=g)
    classes = y.unique()
    quota = math.ceil(n / len(classes))
    picked = torch.cat([perm[y[perm] == c][:quota] for c in classes])
    if len(picked) < n:
        rest = perm[~torch.isin(perm, picked)]
        picked = torch.cat([picked, rest[: n - len(picked)]])
    return picked[:n]


def test_weights(y_sub: torch.Tensor, y_test: torch.Tensor, K: int) -> np.ndarray:
    """Importance weights that reweight a stratified subset to the test class mix."""
    pt = torch.bincount(y_test, minlength=K).float() / len(y_test)
    ps = torch.bincount(y_sub, minlength=K).float().clamp_min(1) / len(y_sub)
    return (pt / ps)[y_sub].numpy()


# ------------------------------------------------------------------ models
def build_model(name, D):
    spec = spec_of(name)
    p, K = D.p, D.K
    if spec["arch"] == "sode":
        base = SODEGuard(SODEGuardConfig(feature_dim=p, num_classes=K))
    elif spec["arch"] == "ode":
        base = SODEGuard(SODEGuardConfig(feature_dim=p, num_classes=K, use_diffusion=False))
    elif spec["arch"] == "sdetgnn":
        base = SDE_TGNN(num_classes=K, feature_dim=p)
    elif spec["arch"] == "mlp":
        base = EGraphSAGEBaseline(feature_dim=p, num_classes=K)
    else:
        raise KeyError(name)
    prep = QuantileEncoder(D.Xtr.numpy()) if spec["encoding"] == "quantile" else None
    return Pipeline(base, prep, K)


def predictor(model, kind, mode="fixed", n_paths=8):
    if kind == "sde":
        if mode == "fixed":
            seeds = list(range(n_paths))
            return lambda x: model.forward_mean(x, seeds=seeds)
        return lambda x: model.forward_mean(x, n_paths=n_paths)
    if kind == "ode":
        return lambda x: model.forward_mean(x, seeds=[0])
    return model


def logits_of(f, X, bs=1024):
    return rr.batched_logits(f, X, bs)


def noisy_logit_samples(f, X, sigma, S, bs=512):
    with torch.no_grad():
        return torch.stack([torch.cat([f(X[i:i + bs].to(DEV) + sigma * torch.randn_like(X[i:i + bs].to(DEV))).float().cpu()
                                       for i in range(0, len(X), bs)]) for _ in range(S)])


def val_score(model, spec, kind, D, a, sub_idx):
    model.eval()
    if spec["train"] == "rs":
        s = noisy_logit_samples(predictor(model, kind, "fresh"), D.Xsel, a.sigma_rs, 8)
        votes = F.one_hot(s.argmax(-1), D.K).sum(0).argmax(-1).numpy()
        return fast_macro_f1(D.ysel.numpy(), votes, D.K)
    f = predictor(model, kind, "fixed")
    if spec["train"] == "at":
        X, y = D.Xsel[sub_idx], D.ysel[sub_idx]
        xa = rr.run_batched(lambda xb, yb: eot_pgd(f, xb, yb, eps=a.eps, steps=10, eot=1), X, y, a.attack_batch)
        return fast_macro_f1(y.numpy(), logits_of(f, xa).argmax(1).numpy(), D.K)
    return fast_macro_f1(D.ysel.numpy(), logits_of(f, D.Xsel).argmax(1).numpy(), D.K)


def posthoc(model, spec, kind, D, a):
    """Module 1, post-hoc logit adjustment tuned on the selection half."""
    model.logit_bias.zero_()
    if spec["posthoc"] == "none":
        return {"posthoc": "none"}
    model.eval()
    vote = spec["train"] == "rs"
    if vote:
        samples = noisy_logit_samples(predictor(model, kind, "fresh"), D.Xsel, a.sigma_rs, a.rs_val_draws)
    else:
        samples = logits_of(predictor(model, kind, "fixed"), D.Xsel).unsqueeze(0)
    ysel = D.ysel.numpy()
    tau, grid = tune_tau(samples, ysel, D.prior, vote=vote)
    bias = -tau * D.prior.log()
    info = {"posthoc": spec["posthoc"], "tau": tau, "val_macro_f1_by_tau": grid,
            "val_macro_f1_after_posthoc": grid[tau]}
    if spec["posthoc"] == "per_class":
        bias = tune_per_class_bias(samples, ysel, vote=vote, init=bias)
        info["per_class_bias"] = bias.tolist()
        info["val_macro_f1_after_posthoc"] = fast_macro_f1(ysel, _vote_or_mean(samples, bias, vote), D.K)
    model.logit_bias.copy_(bias.to(model.logit_bias.device))
    return info


def train_neural(name, D, a, sd):
    spec, t0 = spec_of(name), time.time()
    kind = kind_of(spec)
    if spec["source"]:                      # same weights as another model, different post-hoc rule
        model = build_model(name, D).to(DEV)
        model.load_state_dict(torch.load(sd / f"{spec['source']}.pt", map_location=DEV))
        info = posthoc(model, spec, kind, D, a)
        torch.save(model.state_dict(), sd / f"{name}.pt")
        return {"source": spec["source"], "val_score": info.get("val_macro_f1_after_posthoc"), **info}
    torch.manual_seed(a.seed_offset + D.seed)
    model = build_model(name, D).to(DEV)
    epochs = spec["epochs"] or a.epochs
    rebuttal = spec["recipe"] == "rebuttal"
    if rebuttal:
        opt = torch.optim.Adam(model.parameters(), lr=a.lr, weight_decay=1e-5)
        ac = CrossEntropyWithAC(ac_weight=0.1, n_ac_paths=a.train_paths if kind == "sde" else 1)
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)
    loss_fn = ImbalanceLoss(spec["loss"], D.counts, tau=a.la_tau).to(DEV)
    steps = epochs * math.ceil(len(D.Xtr) / a.batch)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    g = torch.Generator().manual_seed(D.seed)
    sampler = class_balanced_sampler(D.ytr, spec["sampler"], generator=g) if spec["sampler"] > 0 else None
    f = (lambda x: model.forward_mean(x, n_paths=a.train_paths)) if kind == "sde" else predictor(model, kind)
    ramp = max(1, round(a.eps_ramp * epochs))
    method = spec["at_method"] or a.at_method
    sub_idx = strat_idx(D.ysel, a.n_val_robust, D.seed)
    best, best_state, best_ep, bad, hist = -1.0, None, -1, 0, []
    for ep in range(epochs):
        model.train()
        perm = sampler() if sampler else torch.randperm(len(D.Xtr), generator=g)
        eps_e = eps_schedule(ep, a.eps, ramp)
        tot = 0.0
        for i in range(0, len(perm), a.batch):
            idx = perm[i:i + a.batch]
            x, y = D.Xtr[idx].to(DEV), D.ytr[idx].to(DEV)
            if rebuttal:
                loss, _ = ac(model.base, x, y)
            elif spec["train"] == "at":
                loss = adversarial_loss(model, f, x, y, loss_fn, method=method, eps=eps_e,
                                        steps=spec["at_steps"], trades_beta=a.trades_beta)
            elif spec["train"] == "rs":
                loss = gaussian_consistency_loss(f, x, y, loss_fn, sigma=a.sigma_rs, m=a.rs_m,
                                                 lam=a.cons_lambda, eta=a.cons_eta)
            else:
                loss = loss_fn(f(x), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            tot += float(loss.detach()) * len(idx)
        last = ep == epochs - 1
        if rebuttal:
            if last:
                best, best_ep = val_score(model, spec, kind, D, a, sub_idx), ep
            log(f"[seed {D.seed}][{name}] epoch {ep + 1}/{epochs} loss {tot / len(perm):.4f}")
            continue
        if (ep + 1) % a.eval_every == 0 or last:
            s = val_score(model, spec, kind, D, a, sub_idx)
            hist.append((ep + 1, s))
            if s > best:
                best, best_ep, bad = s, ep, 0
                best_state = copy.deepcopy(model.state_dict())
            else:
                bad += a.eval_every
            log(f"[seed {D.seed}][{name}] epoch {ep + 1}/{epochs} loss {tot / len(perm):.4f} "
                f"val {'robust ' if spec['train'] == 'at' else ''}macro-F1 {s:.4f} (best {best:.4f} @ {best_ep + 1})")
            if bad >= a.patience:
                log(f"[seed {D.seed}][{name}] early stop")
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    info = posthoc(model, spec, kind, D, a)
    torch.save(model.state_dict(), sd / f"{name}.pt")
    out = {"train_seconds": time.time() - t0, "epochs_run": ep + 1, "best_epoch": best_ep + 1,
           "val_score": best, "val_history": hist, **info}
    if spec["encoding"] == "quantile":
        out["quantile_max_slope"] = model.prep.max_slope()
    return out


class TreePredictor:
    def __init__(self, clf, bias: np.ndarray, K: int):
        self.clf, self.bias, self.K = clf, torch.tensor(bias, dtype=torch.float32), K

    def __call__(self, x):
        p = np.zeros((len(x), self.K))
        p[:, self.clf.classes_] = self.clf.predict_proba(x.detach().cpu().float().numpy())
        return (torch.log(torch.tensor(p, dtype=torch.float32) + 1e-9) + self.bias).to(x.device)


def train_tree(name, D, a, sd):
    t0 = time.time()
    Xtr, ytr, Xsel, ysel = D.Xtr.numpy(), D.ytr.numpy(), D.Xsel.numpy(), D.ysel.numpy()
    w = (len(ytr) / (D.K * D.counts.float().clamp_min(1)))[D.ytr].numpy()
    seed = a.seed_offset + D.seed
    if name == "lgbm":
        import lightgbm as lgb
        clf = lgb.LGBMClassifier(n_estimators=3000, learning_rate=0.05, num_leaves=63, subsample=0.8, subsample_freq=1,
                                 colsample_bytree=0.8, random_state=seed, verbose=-1)
        clf.fit(Xtr, ytr, sample_weight=w, eval_set=[(Xsel, ysel)],
                callbacks=[lgb.early_stopping(100, verbose=False)])
    elif name == "xgb":
        import xgboost as xgb
        clf = xgb.XGBClassifier(n_estimators=3000, learning_rate=0.05, max_depth=8, subsample=0.8, colsample_bytree=0.8,
                                tree_method="hist", device="cuda" if DEV == "cuda" else "cpu",
                                early_stopping_rounds=100, eval_metric="mlogloss", random_state=seed)
        clf.fit(Xtr, ytr, sample_weight=w, eval_set=[(Xsel, ysel)], verbose=False)
    else:
        from sklearn.ensemble import RandomForestClassifier
        clf = RandomForestClassifier(n_estimators=500, class_weight="balanced_subsample", n_jobs=-1, random_state=seed)
        clf.fit(Xtr, ytr)
    pr = TreePredictor(clf, np.zeros(D.K), D.K)
    samples = pr(D.Xsel).unsqueeze(0)
    tau, grid = tune_tau(samples, ysel, D.prior)
    bias = (-tau * D.prior.log()).numpy()
    with open(sd / f"{name}.pkl", "wb") as fh:
        pickle.dump({"clf": clf, "bias": bias}, fh)
    val = fast_macro_f1(ysel, (samples[0] + torch.tensor(bias)).argmax(1).numpy(), D.K)
    return {"train_seconds": time.time() - t0, "val_score": val, "posthoc": "tau", "tau": tau, "val_macro_f1_by_tau": grid}


def load_any(name, D, sd):
    if name in TREES:
        with open(sd / f"{name}.pkl", "rb") as fh:
            obj = pickle.load(fh)
        return TreePredictor(obj["clf"], obj["bias"], D.K)
    m = build_model(name, D).to(DEV)
    m.load_state_dict(torch.load(sd / f"{name}.pt", map_location=DEV))
    return m.eval()


def trained(name, sd):
    return (sd / f"{name}.{'pkl' if name in TREES else 'pt'}").exists()


def load_results(sd):
    p = sd / "results.json"
    return json.loads(p.read_text()) if p.exists() else {}


def save_results(sd, res):
    (sd / "results.json").write_text(json.dumps(res, indent=2))


def ordered(models):
    """Models with a ``source`` after their source."""
    return [m for m in models if not spec_of(m)["source"]] + [m for m in models if spec_of(m)["source"]]


def cmd_train(a):
    for seed in a.seeds:
        make_split(a, seed)
        sd, D = seed_dir(a.out, seed), load_split(a, seed)
        D.seed = seed
        res = load_results(sd)
        for name in ordered(a.models):
            if trained(name, sd) and not a.force:
                log(f"[seed {seed}][{name}] already trained, skipping (use --force to retrain)")
                continue
            src = spec_of(name)["source"]
            if src and not trained(src, sd):
                log(f"[seed {seed}][{name}] needs {src}; skipped")
                continue
            info = train_tree(name, D, a, sd) if name in TREES else train_neural(name, D, a, sd)
            res.setdefault("train", {})[name] = info
            save_results(sd, res)
            log(f"[seed {seed}][{name}] done: val {info.get('val_score') or float('nan'):.4f}, tau {info.get('tau')}")


# ------------------------------------------------------------------ attack
def macro(y, p, K, w=None):
    return fast_macro_f1(np.asarray(y), np.asarray(p), K, w)


def cmd_attack(a):
    for seed in a.seeds:
        sd, D = seed_dir(a.out, seed), load_split(a, seed)
        res = load_results(sd)
        idx = strat_idx(D.yte, a.n_attack, seed)
        X, y = D.Xte[idx], D.yte[idx]
        w = test_weights(y, D.yte, D.K)
        yn = y.numpy()
        transfer = {}
        for src in ("mlp", "mlp_at"):
            if trained(src, sd):
                s = load_any(src, D, sd)
                transfer[f"Transfer from {src}"] = rr.run_batched(
                    lambda xb, yb: eot_pgd(s, xb, yb, eps=a.eps, steps=a.eot_steps, eot=1), X, y, a.attack_batch)
        for name in a.attack_models:
            if not trained(name, sd) or spec_of(name)["train"] == "rs":
                continue
            if name in res.get("attack", {}) and not a.force:
                log(f"[seed {seed}][{name}] already attacked, skipping")
                continue
            spec = spec_of(name)
            kind = kind_of(spec)
            m = load_any(name, D, sd)
            scorers = {"fixed": predictor(m, kind, "fixed")}
            if kind == "sde":
                scorers["fresh"] = predictor(m, kind, "fresh")
            f_fixed, f_fresh = scorers["fixed"], scorers.get("fresh", scorers["fixed"])
            r = {"clean": {}}
            clean_pred = None
            for mode, f in scorers.items():
                logits = logits_of(f, D.Xte)
                pred = logits.argmax(1).numpy()
                met = aggregate_metrics(D.yte.numpy(), pred, F.softmax(logits, 1).numpy())
                met["per_class_f1"] = f1_score(D.yte.numpy(), pred, average=None, labels=list(range(D.K)), zero_division=0).tolist()
                r["clean"][mode] = met
                if mode == "fixed":
                    clean_pred = pred
            stoch = kind == "sde"
            atks = {f"Square ({a.square_queries} queries)": lambda xb, yb: square_attack(f_fresh, xb, yb, eps=a.eps, n_queries=a.square_queries)}
            if kind != "tree":
                atks.update({
                    "PGD-40 (fixed paths)": lambda xb, yb: eot_pgd(f_fixed, xb, yb, eps=a.eps, steps=40, eot=1),
                    f"PGD-{a.long_steps} x{a.restarts} restarts (fixed)": lambda xb, yb: rr.pgd_restarts(f_fixed, xb, yb, a.eps, a.long_steps, a.restarts),
                    "APGD-CE": lambda xb, yb: apgd(f_fresh, xb, yb, eps=a.eps, n_iter=a.apgd_iters, loss="ce", eot=a.eot_apgd if stoch else 1),
                    "APGD-DLR": lambda xb, yb: apgd(f_fresh, xb, yb, eps=a.eps, n_iter=a.apgd_iters, loss="dlr", eot=a.eot_apgd if stoch else 1),
                })
            if stoch:
                atks[f"EOT-PGD-{a.eot_steps} (EOT {a.eot})"] = lambda xb, yb: eot_pgd(f_fresh, xb, yb, eps=a.eps, steps=a.eot_steps, eot=a.eot)
                atks[f"E-margin PGD ({a.em_paths} paths)"] = lambda xb, yb: expected_margin_pgd_chunked(
                    m, xb, yb, eps=a.eps, steps=a.eot_steps, n_paths=a.em_paths, chunk=a.em_chunk)
            adv = dict(transfer)
            for an, fn in atks.items():
                t0 = time.time()
                adv[an] = rr.run_batched(fn, X, y, a.em_batch if an.startswith("E-margin") else a.attack_batch)
                log(f"[seed {seed}][{name}] {an}: {time.time() - t0:.0f}s")
            worst = {mode: None for mode in scorers}
            r["attacks"] = {}
            for an, xa in adv.items():
                r["attacks"][an] = {}
                for mode, f in scorers.items():
                    pr = logits_of(f, xa).argmax(1)
                    r["attacks"][an][mode] = macro(yn, pr.numpy(), D.K)
                    if worst[mode] is None:
                        worst[mode] = pr.clone()
                    else:
                        flip = (pr != y) & (worst[mode] == y)
                        worst[mode][flip] = pr[flip]
            clean_sub = {mode: logits_of(f, X).argmax(1).numpy() for mode, f in scorers.items()}
            r["clean_subset_macro_f1"] = {mode: macro(yn, p, D.K) for mode, p in clean_sub.items()}
            r["worst_case_macro_f1"] = {mode: macro(yn, wp.numpy(), D.K) for mode, wp in worst.items()}
            r["worst_case_macro_f1_test_reweighted"] = {mode: macro(yn, wp.numpy(), D.K, w) for mode, wp in worst.items()}
            if kind == "tree":
                r["note"] = "not differentiable: gradient attacks do not apply; Square and transfer only"
            if seed in a.sanity_seeds and name in a.sanity_models and kind != "tree":
                n_s = min(a.n_sweep, len(X))
                fs = f_fresh
                eot_s = max(a.eot // 4, 1) if stoch else 1
                r["eps_sweep"] = {str(e): macro(yn[:n_s], logits_of(fs, rr.run_batched(
                    lambda xb, yb: eot_pgd(fs, xb, yb, eps=e, steps=a.eot_steps, eot=eot_s), X[:n_s], y[:n_s], a.attack_batch)).argmax(1).numpy(), D.K)
                    for e in a.eps_sweep}
                r["step_sweep"] = {str(s): macro(yn[:n_s], logits_of(fs, rr.run_batched(
                    lambda xb, yb: eot_pgd(fs, xb, yb, eps=a.eps, steps=s, eot=eot_s), X[:n_s], y[:n_s], a.attack_batch)).argmax(1).numpy(), D.K)
                    for s in a.step_sweep}
            np.savez_compressed(sd / f"preds_attack_{name}.npz", clean_test=clean_pred, idx=idx.numpy(),
                                **{f"worst_{k}": v.numpy() for k, v in worst.items()})
            res.setdefault("attack", {})[name] = r
            res["attack_settings"] = {k: getattr(a, k) for k in ("eps", "profile", "n_attack", "restarts", "long_steps", "eot",
                                                                 "eot_apgd", "apgd_iters", "eot_steps", "square_queries", "em_paths")}
            save_results(sd, res)
            log(f"[seed {seed}][{name}] clean {r['clean']['fixed']['macro_f1']:.4f} | worst case {r['worst_case_macro_f1']}")


# ------------------------------------------------------------------ certify
def cmd_certify(a):
    for seed in a.seeds:
        sd, D = seed_dir(a.out, seed), load_split(a, seed)
        res = load_results(sd)
        idx = strat_idx(D.yte, a.n_cert, seed + 1)
        X, y = D.Xte[idx], D.yte[idx]
        w = test_weights(y, D.yte, D.K)
        for name in a.models:
            if not trained(name, sd) or spec_of(name)["train"] != "rs":
                continue
            if name in res.get("certify", {}) and not a.force:
                log(f"[seed {seed}][{name}] already certified, skipping")
                continue
            kind = kind_of(spec_of(name))
            m = load_any(name, D, sd)
            f = predictor(m, kind, "fresh", n_paths=a.rs_paths)
            t0 = time.time()
            preds, radii = [], []
            for i in range(0, len(X), a.cert_batch):
                p_, r_ = certify_randomized(f, X[i:i + a.cert_batch].to(DEV), sigma=a.sigma_rs, K=D.K,
                                            n0=100, n=a.smooth_n, alpha=1e-3, batch=a.smooth_batch)
                preds.append(p_); radii.append(r_)
            pred, rad = np.concatenate(preds), np.concatenate(radii)
            correct = pred == y.numpy()
            cert = {"sigma": a.sigma_rs, "n": a.smooth_n, "alpha": 1e-3, "n_flows": len(X), "paths_per_sample": a.rs_paths if kind == "sde" else None,
                    "radius_ceiling": radius_ceiling(a.sigma_rs, a.smooth_n), "abstain_rate": float((pred < 0).mean()),
                    "certified_accuracy_l2": {str(r): float((correct & (rad >= r)).mean()) for r in RADII},
                    "certified_accuracy_l2_test_reweighted": {str(r): float(np.average(correct & (rad >= r), weights=w)) for r in RADII},
                    "acr": float((rad * correct).mean()), "seconds": time.time() - t0,
                    "per_class_certified_recall@0.273": [float((correct & (rad >= 0.273))[y.numpy() == c].mean()) for c in range(D.K)]}
            np.savez_compressed(sd / f"preds_cert_{name}.npz", pred=pred, radius=rad, idx=idx.numpy())
            res.setdefault("certify", {})[name] = cert
            save_results(sd, res)
            log(f"[seed {seed}][{name}] certified accuracy {cert['certified_accuracy_l2']} ({cert['seconds']:.0f}s)")


# ------------------------------------------------------------------ aggregate
def cmd_aggregate(a):
    out = Path(a.out)
    seeds = sorted(int(p.name.split("_")[1]) for p in out.glob("seed_*") if (p / "results.json").exists())
    meta = json.loads((out / "meta.json").read_text())
    K = len(meta["classes"])
    per = {s: load_results(out / f"seed_{s}") for s in seeds}
    models = sorted({m for r in per.values() for sec in ("train", "attack", "certify") for m in r.get(sec, {})})
    summ = {}
    for m in models:
        rows = {}
        for s in seeds:
            r = per[s]
            if m in r.get("attack", {}):
                at = r["attack"][m]
                rows.setdefault("clean macro-F1", []).append(at["clean"]["fixed"]["macro_f1"])
                rows.setdefault("clean accuracy", []).append(at["clean"]["fixed"]["accuracy"])
                rows.setdefault("per-class F1", []).append(at["clean"]["fixed"]["per_class_f1"])
                for mode, v in at["worst_case_macro_f1"].items():
                    rows.setdefault(f"worst-case macro-F1 ({mode})", []).append(v)
                for mode, v in at["worst_case_macro_f1_test_reweighted"].items():
                    rows.setdefault(f"worst-case macro-F1 ({mode}, test mix)", []).append(v)
            if m in r.get("certify", {}):
                c = r["certify"][m]
                for rad, v in c["certified_accuracy_l2"].items():
                    rows.setdefault(f"certified acc @{rad}", []).append(v)
                rows.setdefault("ACR", []).append(c["acr"])
            if m in r.get("train", {}) and "val_score" in r["train"][m]:
                rows.setdefault("validation score", []).append(r["train"][m]["val_score"])
        summ[m] = {k: (np.mean(np.array(v), 0).round(4).tolist() if k == "per-class F1" else mean_ci(v)) for k, v in rows.items()}

    def metric_for(kind):
        if kind == "clean" or kind == "worst":
            return lambda yy, pp, ww: fast_macro_f1(yy, pp, K, ww)
        return lambda yy, pp, ww: float(np.average(pp, weights=ww))

    contrasts, pvals = {}, {}
    for key, ma, mb, kind in CONTRASTS:
        pairs = []
        for s in seeds:
            sd = out / f"seed_{s}"
            if kind in ("clean", "worst"):
                fa, fb = sd / f"preds_attack_{ma}.npz", sd / f"preds_attack_{mb}.npz"
                if not (fa.exists() and fb.exists()):
                    continue
                A, B = np.load(fa), np.load(fb)
                d = np.load(sd / "data.npz")
                if kind == "clean":
                    pairs.append((d["yte"], A["clean_test"], B["clean_test"], None))
                else:
                    yy = d["yte"][A["idx"]]
                    wa = test_weights(torch.from_numpy(yy), torch.from_numpy(d["yte"]), K)
                    pairs.append((yy, A["worst_fixed"], B["worst_fixed"], wa))
            else:
                r = float(kind.split("@")[1])
                fa, fb = sd / f"preds_cert_{ma}.npz", sd / f"preds_cert_{mb}.npz"
                if not (fa.exists() and fb.exists()):
                    continue
                A, B = np.load(fa), np.load(fb)
                d = np.load(sd / "data.npz")
                yy = d["yte"][A["idx"]]
                ind = lambda Z: ((Z["pred"] == yy) & (Z["radius"] >= r)).astype(float)
                pairs.append((yy, ind(A), ind(B), None))
        if pairs:
            contrasts[key] = paired_bootstrap(pairs, metric_for(kind), n_boot=a.n_boot)
            pvals[key] = contrasts[key]["p"]
    for k, p in holm(pvals).items():
        contrasts[k]["p_holm"] = p
    (out / "summary.json").write_text(json.dumps({"seeds": seeds, "models": summ, "contrasts": contrasts}, indent=2))

    fmt = lambda c: "–" if not c or c.get("mean") is None else f"{c['mean']:.3f} ± {(c['hi'] - c['mean']):.3f}"
    lines = [f"# SODE-Guard v2 re-run: {len(seeds)} seed(s) {seeds}", "",
             f"Data: {sum(meta['class_counts_after_dedup'].values())} flows after deduplication, {K} classes. "
             "Mean ± half-width of the 95% t-interval over seeds.", "",
             "## Clean and adversarial (l-inf eps = %.3g, z-units)" % a.eps, "",
             "| model | clean macro-F1 | clean accuracy | worst-case macro-F1 (fixed) | worst-case (fresh) | worst-case (fixed, test mix) |",
             "|---|---|---|---|---|---|"]
    for m in models:
        s = summ[m]
        if "clean macro-F1" in s:
            lines.append(f"| {m} | {fmt(s['clean macro-F1'])} | {fmt(s['clean accuracy'])} | {fmt(s.get('worst-case macro-F1 (fixed)'))} | "
                         f"{fmt(s.get('worst-case macro-F1 (fresh)'))} | {fmt(s.get('worst-case macro-F1 (fixed, test mix)'))} |")
    cert_models = [m for m in models if "certified acc @0.0" in summ[m]]
    if cert_models:
        lines += ["", "## Certified accuracy (randomised smoothing, l2)", "",
                  "| model | " + " | ".join(f"r = {r}" for r in RADII) + " | ACR |", "|---|" + "---|" * (len(RADII) + 1)]
        for m in cert_models:
            lines.append(f"| {m} | " + " | ".join(fmt(summ[m].get(f"certified acc @{r}")) for r in RADII) + f" | {fmt(summ[m]['ACR'])} |")
    lines += ["", "## Per-class clean F1 (mean over seeds)", "", "| model | " + " | ".join(meta["classes"]) + " |",
              "|---|" + "---|" * K]
    for m in models:
        if "per-class F1" in summ[m]:
            lines.append(f"| {m} | " + " | ".join(f"{v:.2f}" for v in summ[m]["per-class F1"]) + " |")
    if contrasts:
        lines += ["", "## Declared contrasts (paired bootstrap over flows within seeds, Holm-adjusted)", "",
                  "| contrast | difference A − B | 95% CI | p | p (Holm) |", "|---|---|---|---|---|"]
        for k, c in contrasts.items():
            lines.append(f"| {k} | {c['diff']:+.3f} | [{c['lo']:+.3f}, {c['hi']:+.3f}] | {c['p']:.3f} | {c['p_holm']:.3f} |")
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def cmd_report(a):
    print((Path(a.out) / "summary.md").read_text())


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["prepare", "train", "attack", "certify", "aggregate", "report"])
    ap.add_argument("--out", required=True)
    # prepare
    ap.add_argument("--csv"); ap.add_argument("--label-col"); ap.add_argument("--time-col")
    ap.add_argument("--drop-cols", nargs="*")
    ap.add_argument("--cap-per-class", type=int, default=20000)
    ap.add_argument("--min-per-class", type=int, default=100)
    ap.add_argument("--chunksize", type=int, default=500_000)
    ap.add_argument("--data-seed", type=int, default=42)
    # split / seeds
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--seed-offset", type=int, default=0)
    ap.add_argument("--split", choices=["random", "class_time"], default="random")
    ap.add_argument("--clip", type=float, default=10.0)
    # models and training (module 1)
    ap.add_argument("--models", nargs="+", default=CORE, help=f"core: {CORE}; ablations: {ABLATIONS}")
    ap.add_argument("--attack-models", nargs="+", default=None, help="default: --models")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--patience", type=int, default=12)
    ap.add_argument("--eval-every", type=int, default=1)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--la-tau", type=float, default=1.0)
    ap.add_argument("--train-paths", type=int, default=8, help="Brownian paths per training forward (matches the 8-path predictor)")
    # module 2
    ap.add_argument("--eps", type=float, default=0.03)
    ap.add_argument("--at-method", choices=["nfgsm", "pgd", "trades"], default="nfgsm")
    ap.add_argument("--eps-ramp", type=float, default=0.15, help="fraction of epochs over which eps ramps up")
    ap.add_argument("--trades-beta", type=float, default=6.0)
    ap.add_argument("--n-val-robust", type=int, default=1000)
    # module 3
    ap.add_argument("--sigma-rs", type=float, default=0.25)
    ap.add_argument("--rs-m", type=int, default=2)
    ap.add_argument("--cons-lambda", type=float, default=10.0)
    ap.add_argument("--cons-eta", type=float, default=0.5)
    ap.add_argument("--rs-paths", type=int, default=8)
    ap.add_argument("--rs-val-draws", type=int, default=16)
    # attacks / certification
    ap.add_argument("--profile", choices=list(PROFILES), default="lite")
    for k in PROFILES["lite"]:
        ap.add_argument("--" + k.replace("_", "-"), type=int, default=None)
    ap.add_argument("--attack-batch", type=int, default=500)
    ap.add_argument("--em-batch", type=int, default=250)
    ap.add_argument("--em-chunk", type=int, default=32)
    ap.add_argument("--cert-batch", type=int, default=50)
    ap.add_argument("--smooth-batch", type=int, default=64)
    ap.add_argument("--sanity-seeds", type=int, nargs="*", default=[0])
    ap.add_argument("--sanity-models", nargs="*", default=["sode_at", "mlp_at", "sode"])
    ap.add_argument("--eps-sweep", type=float, nargs="+", default=[0.01, 0.03, 0.1, 0.3, 1.0])
    ap.add_argument("--step-sweep", type=int, nargs="+", default=[10, 40, 100])
    ap.add_argument("--n-sweep", type=int, default=500)
    ap.add_argument("--n-boot", type=int, default=1000)
    a = ap.parse_args()
    for k, v in PROFILES[a.profile].items():
        if getattr(a, k) is None:
            setattr(a, k, v)
    a.attack_models = a.attack_models or a.models
    unknown = [m for m in a.models + a.attack_models if m not in SPECS]
    if unknown:
        raise SystemExit(f"unknown models {unknown}; choose from {sorted(SPECS)}")
    log("device:", DEV, "| stage:", a.stage, "| profile:", a.profile)
    {"prepare": cmd_prepare, "train": cmd_train, "attack": cmd_attack, "certify": cmd_certify,
     "aggregate": cmd_aggregate, "report": cmd_report}[a.stage](a)


if __name__ == "__main__":
    main()
