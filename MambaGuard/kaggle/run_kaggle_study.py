"""MambaGuard GPU study for Kaggle (or any CUDA machine).

Runs, in priority order, and saves each result as soon as it finishes:

1. main      -- full model x every benchmark x every seed
2. ablation  -- architecture switches on the ablation benchmarks
3. certify   -- randomised-smoothing certificate of trained full models
4. latency   -- streaming and best-batch GPU latency

Everything goes to ``--out`` (default /kaggle/working/results) and is zipped to
``<out>.zip`` at the end, or when the time budget runs out. Re-running the
script skips runs whose result file already exists, so an interrupted session
can be resumed: in the same session just run it again; in a new session pass
``--resume-from`` the results folder of the earlier one (attached as input).

    python kaggle/run_kaggle_study.py --data-root /tmp/mambaguard_data --seeds 0 1 2
    python kaggle/run_kaggle_study.py --smoke          # ~5-minute end-to-end check
"""
from __future__ import annotations

import argparse
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mambaguard.certification.certificate import composed_certificate_v2  # noqa: E402
from mambaguard.certification.smoothing import _clopper_pearson_lower, _phi_inv  # noqa: E402
from mambaguard.data.canonicalize import (  # noqa: E402
    DEFAULT_METADATA_SCHEMA,
    HashingCanonicaliser,
    MessageCanonicaliser,
)
from mambaguard.data.datasets import ProtocolMessageDataset, protocol_collate  # noqa: E402
from mambaguard.data.protocol_traces import EPISODE_LOADERS, flatten  # noqa: E402
from mambaguard.models import MambaGuard, MambaGuardConfig  # noqa: E402
from mambaguard.training import MambaGuardTrainer  # noqa: E402
from mambaguard.training.callbacks import EarlyStopping  # noqa: E402
from mambaguard.utils.seed import set_seed  # noqa: E402

BENCHMARKS = {  # name: (folder under data root, or "builtin"; default max epochs)
    "a2a_sandbox": ("builtin", 15),
    "mcptox": ("MCPTox-Benchmark", 20),
    "mcp_safetybench": ("MCPSafety", 30),
    "agentdojo": ("agentdojo", 30),
    "injecagent": ("InjecAgent", 20),
}
VARIANTS = {  # name: (model overrides, trainer overrides)
    "full": ({}, {}),
    "ssm_only": ({"n_gat_layers": 0}, {}),
    "gat_only": ({"n_blocks": 0}, {}),
    "flat": ({"readout": "flat"}, {}),
    "transformer": ({"backbone": "transformer"}, {}),
    "no_lipschitz": ({}, {"lipschitz_lambda": 0.0}),
}


# ---------------------------------------------------------------------------
def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def env_info(args: argparse.Namespace) -> dict[str, Any]:
    info: dict[str, Any] = {
        "python": platform.python_version(), "torch": torch.__version__,
        "cuda": torch.version.cuda, "device": args.device,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "bf16": bool(torch.cuda.is_available() and torch.cuda.is_bf16_supported()),
        "encoder": args.encoder, "seeds": args.seeds, "balance": args.balance,
        "started": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    try:
        info["code_commit"] = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
    except Exception:  # noqa: BLE001
        info["code_commit"] = None
    try:
        import sentence_transformers  # type: ignore

        info["sentence_transformers"] = sentence_transformers.__version__
    except Exception:  # noqa: BLE001
        pass
    return info


def make_encoder(args: argparse.Namespace):
    if args.encoder == "hash":
        return HashingCanonicaliser(d_p=384)
    return MessageCanonicaliser(model_name=args.encoder_model, device=args.device, batch_size=256)


def benchmark_root(name: str, data_root: str) -> str:
    folder = BENCHMARKS[name][0]
    return "builtin" if folder == "builtin" else os.path.join(data_root, folder)


def load_split(name: str, root: str, split: str, args: argparse.Namespace):
    kw: dict[str, Any] = {}
    if name == "agentdojo" and args.agentdojo_model:
        kw["model"] = args.agentdojo_model
    if name == "a2a_sandbox":
        kw["n_pairs_per_finding"] = args.a2a_pairs
    eps = EPISODE_LOADERS[name](root, split, **kw)
    if args.smoke:  # keep both classes so every metric is defined
        k = (48 if split == "train" else 24) // 2
        att = [e for e in eps if e.is_attack]
        ben = [e for e in eps if not e.is_attack]
        eps = att[:k] + ben[:k]
    return eps


def build_datasets(name: str, args: argparse.Namespace, encoder) -> dict[str, ProtocolMessageDataset]:
    root = benchmark_root(name, args.data_root)
    out = {}
    for split in ("train", "val", "test"):
        msgs = list(flatten(load_split(name, root, split, args)))
        out[split] = ProtocolMessageDataset(msgs, canonicaliser=encoder, compute_embeddings=True)
    return out


def window_labels(ds: ProtocolMessageDataset) -> np.ndarray:
    return np.array([int(ds[i][0].extras.get("is_attack_episode", 0)) for i in range(len(ds))])


def window_ids(ds: ProtocolMessageDataset) -> list[str]:
    return [ds[i][0].extras.get("episode_id", str(i)) for i in range(len(ds))]


def loader(ds, shuffle: bool, args: argparse.Namespace, noise_sigma: float = 0.0):
    from torch.utils.data import DataLoader, WeightedRandomSampler

    def collate(batch):
        b = protocol_collate(batch)
        if noise_sigma > 0:
            noise = torch.randn_like(b["payload_emb"]) * noise_sigma
            b["payload_emb"] = b["payload_emb"] + noise * b["attention_mask"].unsqueeze(-1)
        return b

    kw: dict[str, Any] = {"batch_size": args.batch_size, "collate_fn": collate,
                          "num_workers": args.workers, "pin_memory": args.device == "cuda"}
    if shuffle and args.balance:
        y = window_labels(ds)
        counts = np.bincount(y, minlength=2).astype(float)
        w = 1.0 / counts[y].clip(min=1)
        kw["sampler"] = WeightedRandomSampler(torch.as_tensor(w, dtype=torch.double), num_samples=len(ds), replacement=True)
    else:
        kw["shuffle"] = shuffle
    return DataLoader(ds, **kw)


def model_config(overrides: dict[str, Any]) -> MambaGuardConfig:
    d_mu = len(DEFAULT_METADATA_SCHEMA)
    return MambaGuardConfig(d_p=384, d_mu=d_mu, edge_dim=5 + d_mu, num_classes=2, **overrides)


@torch.no_grad()
def predict(model, ds, args: argparse.Namespace) -> np.ndarray:
    model.eval()
    probs = []
    for batch in loader(ds, False, args):
        batch = {k: (v.to(args.device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        logits = model(batch)["logits"].float()
        probs.append(torch.softmax(logits, dim=-1)[:, 1].cpu())
    return torch.cat(probs).numpy()


def metrics(y: np.ndarray, p: np.ndarray, thr: float = 0.5) -> dict[str, float]:
    from sklearn.metrics import accuracy_score, average_precision_score, f1_score, roc_auc_score

    pred = (p >= thr).astype(int)
    out = {"macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
           "accuracy": float(accuracy_score(y, pred)),
           "attack_recall": float(pred[y == 1].mean()) if (y == 1).any() else float("nan"),
           "fpr": float(pred[y == 0].mean()) if (y == 0).any() else float("nan")}
    if len(np.unique(y)) == 2:
        out["auroc"] = float(roc_auc_score(y, p))
        out["avg_precision"] = float(average_precision_score(y, p))
    return out


def tune_threshold(y: np.ndarray, p: np.ndarray) -> float:
    from sklearn.metrics import f1_score

    if len(np.unique(y)) < 2:
        return 0.5
    grid = np.unique(np.concatenate([np.linspace(0.01, 0.99, 99), p]))
    scores = [f1_score(y, (p >= t).astype(int), average="macro", zero_division=0) for t in grid]
    return float(grid[int(np.argmax(scores))])


def bootstrap_ci(y: np.ndarray, p: np.ndarray, thr: float, n: int = 1000, seed: int = 0) -> dict[str, list[float]]:
    from sklearn.metrics import f1_score, roc_auc_score

    rng = np.random.default_rng(seed)
    f1s, aucs = [], []
    for _ in range(n):
        idx = rng.integers(0, len(y), len(y))
        yy, pp = y[idx], p[idx]
        f1s.append(f1_score(yy, (pp >= thr).astype(int), average="macro", zero_division=0))
        if len(np.unique(yy)) == 2:
            aucs.append(roc_auc_score(yy, pp))
    q = lambda a: [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))] if a else [math.nan, math.nan]
    return {"macro_f1_tuned_95ci": q(f1s), "auroc_95ci": q(aucs)}


# ---------------------------------------------------------------------------
def run_one(name: str, variant: str, seed: int, data: dict, args: argparse.Namespace, out: Path) -> dict[str, Any]:
    set_seed(seed)
    mo, to = VARIANTS[variant]
    model = MambaGuard(model_config(mo))
    ckpt_dir = out / "checkpoints" / f"{name}__{variant}__seed{seed}"
    epochs = 2 if args.smoke else max(1, int(round(BENCHMARKS[name][1] * args.epoch_scale)))
    tcfg = {"epochs": epochs, "num_classes": 2, "lr": args.lr, "out_dir": str(ckpt_dir), "log_every": 0,
            "monitor": "macro_f1", "amp_dtype": args.amp, "warmup_steps": 200, **to}
    stopper = EarlyStopping(patience=args.patience, monitor="val_macro_f1")
    trainer = MambaGuardTrainer(model, loader(data["train"], True, args, args.noise_aug),
                                loader(data["val"], False, args), cfg=tcfg, device=args.device, callbacks=[stopper])
    t0 = time.time()
    state = trainer.train()
    train_time = time.time() - t0
    trainer.load(str(ckpt_dir / "best.pt"))
    yv, pv = window_labels(data["val"]), predict(trainer.model, data["val"], args)
    yt, pt = window_labels(data["test"]), predict(trainer.model, data["test"], args)
    thr = tune_threshold(yv, pv)
    res = {
        "benchmark": name, "variant": variant, "seed": seed, "epochs_max": epochs,
        "epochs_run": state.epoch + 1, "best_epoch": state.best_epoch, "train_time_s": round(train_time, 1),
        "n_train": len(data["train"]), "n_val": len(data["val"]), "n_test": len(data["test"]),
        "test_attack_frac": float(yt.mean()), "params": int(sum(p.numel() for p in trainer.model.parameters())),
        "threshold_val_tuned": thr,
        "test_at_0.5": metrics(yt, pt, 0.5), "test_at_val_threshold": metrics(yt, pt, thr),
        **bootstrap_ci(yt, pt, thr),
        "lipschitz_bound": float(trainer.model.lipschitz_bound()),
    }
    np.savez_compressed(out / "predictions" / f"{name}__{variant}__seed{seed}.npz",
                        episode_id=np.array(window_ids(data["test"])), label=yt, prob=pt,
                        val_label=yv, val_prob=pv)
    keep = args.keep_checkpoints and variant == "full" and seed == args.seeds[0]
    if keep:  # weights + config only (~15 MB); drop optimizer state and last.pt
        ck = torch.load(ckpt_dir / "best.pt", map_location="cpu", weights_only=False)
        torch.save({"model": ck["model"], "config": ck.get("config"), "epoch": ck.get("epoch")}, ckpt_dir / "best.pt")
        (ckpt_dir / "last.pt").unlink(missing_ok=True)
    else:
        shutil.rmtree(ckpt_dir, ignore_errors=True)
    return res


@torch.no_grad()
def certify(name: str, data: dict, args: argparse.Namespace, out: Path) -> dict[str, Any]:
    """Cohen-style certificate on the payload embeddings of whole test windows."""
    ck = out / "checkpoints" / f"{name}__full__seed{args.seeds[0]}" / "best.pt"
    if not ck.exists():
        return {"skipped": f"{ck} missing (run main first with --keep-checkpoints)"}
    model = MambaGuard(model_config({})).to(args.device)
    model.load_state_dict(torch.load(ck, map_location=args.device, weights_only=False)["model"])
    model.eval()
    ds = data["test"]
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(ds))[: (8 if args.smoke else args.cert_windows)]
    y = window_labels(ds)
    n0, n = (8, 32) if args.smoke else (args.cert_n0, args.cert_n)
    report: dict[str, Any] = {"benchmark": name, "n0": n0, "n": n, "alpha": args.cert_alpha, "windows": len(idx), "sigmas": {}}
    for sigma in args.cert_sigmas:
        radii, correct = [], []
        for i in idx:
            base = protocol_collate([ds[int(i)]])

            def votes(m: int) -> np.ndarray:
                c = np.zeros(2, dtype=int)
                left = m
                while left > 0:
                    b = min(args.cert_batch, left)
                    left -= b
                    batch = protocol_collate([ds[int(i)]] * b)
                    batch = {k: (v.to(args.device) if torch.is_tensor(v) else v) for k, v in batch.items()}
                    noise = torch.randn_like(batch["payload_emb"]) * sigma
                    batch["payload_emb"] = batch["payload_emb"] + noise * batch["attention_mask"].unsqueeze(-1)
                    pred = model(batch)["logits"].argmax(dim=-1).cpu().numpy()
                    c += np.bincount(pred, minlength=2)
                return c

            top = int(np.argmax(votes(n0)))
            k = int(votes(n)[top])
            p_low = _clopper_pearson_lower(k, n, args.cert_alpha)
            r = sigma * _phi_inv(p_low) if p_low > 0.5 else -1.0
            radii.append(r)
            correct.append(int(top == y[int(i)]))
            del base
        radii_a, correct_a = np.array(radii), np.array(correct)
        grid = [0.0, 0.02, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0]
        curve = [{"radius": r, "cert_acc": float(((radii_a >= r) & (radii_a >= 0) & (correct_a == 1)).mean())} for r in grid]
        bound = [{"radius": c["radius"], "composed_lower_bound": float(composed_certificate_v2(1.0, c["cert_acc"], 1.0, 10_000, 5, args.cert_alpha))}
                 for c in curve]
        report["sigmas"][str(sigma)] = {"cert_acc_curve": curve, "composed_bound_V1_B1_T1e4_N5": bound,
                                        "abstain_rate": float((radii_a < 0).mean()),
                                        "median_radius_certified": float(np.median(radii_a[radii_a >= 0])) if (radii_a >= 0).any() else None}
    return report


@torch.no_grad()
def latency(name: str, data: dict, args: argparse.Namespace) -> dict[str, Any]:
    model = MambaGuard(model_config({})).to(args.device).eval()
    ds = data["test"]
    sync = torch.cuda.synchronize if args.device == "cuda" else (lambda: None)
    res: dict[str, Any] = {"benchmark": name, "device": args.device, "batch": {}}
    for bs in (1, 8, 32, 128):
        if bs > len(ds):
            break
        windows = [ds[i % len(ds)] for i in range(bs)]
        batch = {k: (v.to(args.device) if torch.is_tensor(v) else v) for k, v in protocol_collate(windows).items()}
        n_msgs = int(batch["attention_mask"].sum().item())
        for _ in range(10):
            model(batch)
        sync()
        times = []
        for _ in range(5 if args.smoke else 100):
            t = time.perf_counter()
            model(batch)
            sync()
            times.append(time.perf_counter() - t)
        med = float(np.median(times))
        res["batch"][str(bs)] = {"windows": bs, "messages": n_msgs, "median_batch_ms": med * 1e3,
                                 "p99_batch_ms": float(np.percentile(times, 99)) * 1e3,
                                 "per_message_ms": med * 1e3 / max(n_msgs, 1),
                                 "messages_per_s": n_msgs / med}
    return res


# ---------------------------------------------------------------------------
def summarise(out: Path) -> dict[str, Any]:
    runs = [json.loads(p.read_text()) for p in sorted((out / "runs").glob("*.json"))]
    groups: dict[str, list[dict]] = {}
    for r in runs:
        groups.setdefault(f"{r['benchmark']}__{r['variant']}", []).append(r)
    table = {}
    for key, rs in groups.items():
        def ms(path):
            vals = [r[path[0]][path[1]] for r in rs if path[1] in r[path[0]]]
            return {"mean": float(np.mean(vals)), "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0, "n": len(vals)} if vals else None
        table[key] = {"seeds": [r["seed"] for r in rs],
                      "macro_f1_at_0.5": ms(("test_at_0.5", "macro_f1")),
                      "macro_f1_val_threshold": ms(("test_at_val_threshold", "macro_f1")),
                      "auroc": ms(("test_at_0.5", "auroc")),
                      "attack_recall_val_threshold": ms(("test_at_val_threshold", "attack_recall")),
                      "fpr_val_threshold": ms(("test_at_val_threshold", "fpr")),
                      "test_attack_frac": rs[0]["test_attack_frac"], "n_test": rs[0]["n_test"], "params": rs[0]["params"]}
    return table


def resume(src: Path, out: Path) -> None:
    """Copy finished runs from an earlier session's results (folder or zip)."""
    if src.suffix == ".zip":
        tmp = Path(tempfile.mkdtemp(prefix="mg_resume_"))
        with zipfile.ZipFile(src) as zf:
            zf.extractall(tmp)
        kids = [d for d in tmp.iterdir() if d.is_dir()]
        src = kids[0] if len(kids) == 1 else tmp
    if not src.is_dir():
        log(f"--resume-from {src}: not found, starting fresh")
        return
    n = 0
    for f in src.rglob("*"):
        rel = f.relative_to(src)
        if f.is_file() and not f.name.endswith(".error.txt") and rel.parts[0] in ("runs", "predictions", "checkpoints", "certify", "latency"):
            dst = out / rel
            if not dst.exists():
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(f, dst)
                n += 1
    log(f"resumed {n} files from {src}")


def finish(out: Path, info: dict[str, Any], zip_checkpoints: bool = True) -> None:
    info["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    (out / "environment.json").write_text(json.dumps(info, indent=2))
    (out / "summary.json").write_text(json.dumps(summarise(out), indent=2))
    # results.zip: everything except model weights (small, the file to bring back)
    # <out>_checkpoints.zip: the kept seed-0 weights, written separately
    files = [f for f in sorted(out.rglob("*")) if f.is_file()]
    archives = [(str(out) + ".zip", [f for f in files if "checkpoints" not in f.relative_to(out).parts])]
    ckpts = [f for f in files if "checkpoints" in f.relative_to(out).parts]
    if zip_checkpoints and ckpts:
        archives.append((str(out) + "_checkpoints.zip", ckpts))
    for archive, members in archives:
        with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as zf:
            for f in members:
                zf.write(f, f.relative_to(out.parent))
        log(f"wrote {archive} ({os.path.getsize(archive) / 1e6:.1f} MB) -- download it from the notebook's Output tab")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", default="/tmp/mambaguard_data",
                    help="benchmark clones (kept out of /kaggle/working so they are not saved as output)")
    ap.add_argument("--resume-from", default=None,
                    help="results folder (or .zip) of an earlier session; finished runs are copied and skipped")
    ap.add_argument("--out", default="/kaggle/working/results")
    ap.add_argument("--benchmarks", nargs="+", default=list(BENCHMARKS))
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--ablations", nargs="*", default=None,
                    help="variants to run (default: all of %s)" % ", ".join(v for v in VARIANTS if v != "full"))
    ap.add_argument("--skip-ablations", action="store_true")
    ap.add_argument("--ablation-benchmarks", nargs="*", default=["a2a_sandbox", "mcptox"])
    ap.add_argument("--ablation-seeds", nargs="*", type=int, default=None, help="default: first two seeds")
    ap.add_argument("--encoder", choices=["minilm", "hash"], default="minilm")
    ap.add_argument("--encoder-model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--amp", default="auto", help="auto | fp32 | bf16 | fp16")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--epoch-scale", type=float, default=1.0)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--balance", action=argparse.BooleanOptionalAction, default=True,
                    help="class-balanced sampling of training windows")
    ap.add_argument("--noise-aug", type=float, default=0.0, help="Gaussian noise on payload embeddings during training")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--a2a-pairs", type=int, default=200)
    ap.add_argument("--agentdojo-model", default=None)
    ap.add_argument("--certify", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--cert-sigmas", nargs="+", type=float, default=[0.05, 0.1, 0.25])
    ap.add_argument("--cert-windows", type=int, default=100)
    ap.add_argument("--cert-n0", type=int, default=32)
    ap.add_argument("--cert-n", type=int, default=500)
    ap.add_argument("--cert-alpha", type=float, default=1e-3)
    ap.add_argument("--cert-batch", type=int, default=50)
    ap.add_argument("--latency", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--keep-checkpoints", action=argparse.BooleanOptionalAction, default=True,
                    help="keep the first-seed full-model checkpoint per benchmark (needed for --certify)")
    ap.add_argument("--zip-checkpoints", action=argparse.BooleanOptionalAction, default=True,
                    help="also write <out>_checkpoints.zip with the kept model weights")
    ap.add_argument("--time-budget-h", type=float, default=11.0)
    ap.add_argument("--smoke", action="store_true", help="tiny end-to-end check: few windows, 2 epochs")
    args = ap.parse_args()
    if args.smoke:
        args.seeds = args.seeds[:1]
        args.out = args.out.rstrip("/") + "_smoke"
    if not args.ablations:
        args.ablations = [v for v in VARIANTS if v != "full"]
    if args.skip_ablations:
        args.ablations = []
    unknown = set(args.ablations) - set(VARIANTS)
    if unknown:
        ap.error(f"unknown ablation(s): {sorted(unknown)}")
    ab_seeds = args.ablation_seeds if args.ablation_seeds is not None else args.seeds[:2]

    out = Path(args.out)
    if args.resume_from:
        resume(Path(args.resume_from), out)
    for d in ("runs", "predictions", "checkpoints", "certify", "latency"):
        (out / d).mkdir(parents=True, exist_ok=True)
    info = env_info(args)
    log(f"device={args.device} gpu={info['gpu']} encoder={args.encoder} amp={args.amp}")
    deadline = time.time() + args.time_budget_h * 3600
    encoder = make_encoder(args)

    plan: list[tuple[str, str, int]] = [(b, "full", s) for b in args.benchmarks for s in args.seeds]
    plan += [(b, v, s) for b in args.ablation_benchmarks if b in args.benchmarks
             for v in args.ablations for s in ab_seeds]
    cache: dict[str, dict] = {}
    try:
        for name, variant, seed in plan:
            dest = out / "runs" / f"{name}__{variant}__seed{seed}.json"
            if dest.exists():
                log(f"skip {dest.name} (done)")
                continue
            if time.time() > deadline:
                log("time budget reached; stopping training")
                break
            root = benchmark_root(name, args.data_root)
            if root != "builtin" and not os.path.isdir(root):
                log(f"skip {name}: {root} not found")
                continue
            if name not in cache:
                log(f"embedding {name} ...")
                cache = {name: build_datasets(name, args, encoder)}  # keep one benchmark in memory
            log(f"train {name} / {variant} / seed {seed}")
            try:
                res = run_one(name, variant, seed, cache[name], args, out)
                dest.write_text(json.dumps(res, indent=2))
                t = res["test_at_val_threshold"]
                log(f"  macro-F1 {t['macro_f1']:.3f}  AUROC {t.get('auroc', float('nan')):.3f}  ({res['train_time_s']}s)")
            except Exception as exc:  # noqa: BLE001
                (out / "runs" / f"{name}__{variant}__seed{seed}.error.txt").write_text(repr(exc))
                log(f"  ERROR {exc!r}")
                if args.device == "cuda":
                    torch.cuda.empty_cache()

        for kind, fn in (("certify", certify), ("latency", latency)):
            if not getattr(args, kind):
                continue
            for name in args.benchmarks:
                dest = out / kind / f"{name}.json"
                if dest.exists() or time.time() > deadline:
                    continue
                root = benchmark_root(name, args.data_root)
                if root != "builtin" and not os.path.isdir(root):
                    continue
                if name not in cache:
                    cache = {name: build_datasets(name, args, encoder)}
                log(f"{kind} {name}")
                try:
                    rep = fn(name, cache[name], args, out) if kind == "certify" else fn(name, cache[name], args)
                    dest.write_text(json.dumps(rep, indent=2))
                except Exception as exc:  # noqa: BLE001
                    (out / kind / f"{name}.error.txt").write_text(repr(exc))
                    log(f"  ERROR {exc!r}")
    finally:
        finish(out, info, args.zip_checkpoints)


if __name__ == "__main__":
    main()
