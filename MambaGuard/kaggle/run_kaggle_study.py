"""MambaGuard GPU study for Kaggle (or any CUDA machine), version 2.

Two studies, run as two Kaggle sessions (``--study main`` then ``--study
heldout``), or together with ``--study all``. Every result is saved as soon as
it finishes; re-running skips finished results, and ``--resume-from`` copies
them from an earlier session's output.

main
  1. main   -- full model, chunked+lexical encoder, every benchmark x seeds
  2. lex    -- TF-IDF logistic-regression baseline on the same splits
  3. enc    -- encoder ablation on the payload benchmarks (minilm, chunk)
  4. arch   -- architecture ablation on the A2A sandbox and MCPTox
  5. certify, latency (detector forward pass and payload encoding)

heldout
  Leave-template-out: the attack wordings / families of one fold are removed
  from training and validation, and the test split keeps only those attacks
  (plus benign episodes from the test groups). Detector and lexical baseline
  run on the same folds.
    mcptox          wrapper style (3 folds) and paradigm (3 folds)
    agentdojo       attack family, GPT-4o recorded runs (6 folds)
    injecagent      attacker case (4 hashed folds)
    mcp_safetybench attack family (4 folds)
    a2a_sandbox     vulnerability class, leave-one-finding-out (11 folds)

    python kaggle/run_kaggle_study.py --study main     --data-root /tmp/mambaguard_data
    python kaggle/run_kaggle_study.py --study heldout  --data-root /tmp/mambaguard_data
    python kaggle/run_kaggle_study.py --smoke          # ~10-minute end-to-end check
"""
from __future__ import annotations

import argparse
import hashlib
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
from typing import Any, Callable

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
from mambaguard.data.chunked_encoder import ChunkedEncoder  # noqa: E402
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
PAYLOAD_BENCHMARKS = ["mcptox", "mcp_safetybench", "agentdojo"]
VARIANTS = {  # name: (model overrides, trainer overrides)
    "full": ({}, {}),
    "ssm_only": ({"n_gat_layers": 0}, {}),
    "gat_only": ({"n_blocks": 0}, {}),
    "flat": ({"readout": "flat"}, {}),
    "transformer": ({"backbone": "transformer"}, {}),
    "no_lipschitz": ({}, {"lipschitz_lambda": 0.0}),
}
MAIN_ENCODER = "chunk_lex"
HELDOUT_AGENTDOJO_MODEL = "gpt-4o-2024-05-13"   # the only recorded model attacked with many templates
AGENTDOJO_CAP = {"train": 100, "val": 25, "test": 40}  # attack episodes per family per split (held-out study)


# ---------------------------------------------------------------------------
def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def h64(s: str) -> int:
    return int.from_bytes(hashlib.blake2b(s.encode(), digest_size=8).digest(), "little")


def env_info(args: argparse.Namespace) -> dict[str, Any]:
    info: dict[str, Any] = {
        "runner_version": 2, "python": platform.python_version(), "torch": torch.__version__,
        "cuda": torch.version.cuda, "device": args.device,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "bf16_native": bool(torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8),
        "study": args.study, "main_encoder": MAIN_ENCODER, "sentence_backend": args.sentence_backend,
        "seeds": args.seeds, "heldout_seeds": args.heldout_seeds, "balance": args.balance,
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


# ---------------------------------------------------------------------------
# encoders and data
# ---------------------------------------------------------------------------
class Encoders:
    """Builds each payload encoder once; the sentence model is shared."""

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self._sent = None
        self._made: dict[str, Any] = {}

    def sentence(self):
        if self._sent is None:
            if self.args.sentence_backend == "hash":   # offline smoke tests only
                self._sent = HashingCanonicaliser(d_p=384)
            else:
                self._sent = MessageCanonicaliser(model_name=self.args.encoder_model, device=self.args.device,
                                                  batch_size=256)
        return self._sent

    def get(self, kind: str):
        if kind not in self._made:
            if kind == "minilm":
                self._made[kind] = self.sentence()
            elif kind in ("chunk", "chunk_lex"):
                self._made[kind] = ChunkedEncoder(self.sentence(), mode=kind, d_lex=self.args.d_lex)
            else:
                raise ValueError(kind)
        return self._made[kind]

    def dim(self, kind: str) -> int:
        e = self.get(kind)
        return int(getattr(e, "d_p", 384))


def benchmark_root(name: str, data_root: str) -> str:
    folder = BENCHMARKS[name][0]
    return "builtin" if folder == "builtin" else os.path.join(data_root, folder)


class Data:
    """Episode and embedding cache for one benchmark configuration at a time."""

    def __init__(self, args: argparse.Namespace, encoders: Encoders) -> None:
        self.args = args
        self.encoders = encoders
        self.key: tuple | None = None
        self.episodes: dict[str, list] = {}
        self.msgs: dict[tuple[str, str], list] = {}

    def _reset(self, key: tuple) -> None:
        if key != self.key:
            self.key, self.episodes, self.msgs = key, {}, {}
            if self.args.device == "cuda":
                torch.cuda.empty_cache()

    def split(self, name: str, split: str, agentdojo_model: str | None = None, a2a_pairs: int | None = None) -> list:
        self._reset((name, agentdojo_model, a2a_pairs))
        if split not in self.episodes:
            kw: dict[str, Any] = {}
            if name == "agentdojo" and agentdojo_model:
                kw["model"] = agentdojo_model
            if name == "a2a_sandbox":
                kw["n_pairs_per_finding"] = a2a_pairs or self.args.a2a_pairs
            self.episodes[split] = EPISODE_LOADERS[name](benchmark_root(name, self.args.data_root), split, **kw)
        return self.episodes[split]

    def dataset(self, eps: list, enc: str) -> ProtocolMessageDataset:
        missing = [e for e in eps if (enc, e.episode_id) not in self.msgs]
        if missing:
            msgs = list(flatten(missing))
            emb = self.encoders.get(enc).encode_batch([m.payload for m in msgs])
            emb = emb.detach().float().cpu() if hasattr(emb, "detach") else torch.as_tensor(emb)
            by_ep: dict[str, list] = {}
            for i, m in enumerate(msgs):
                mm = _copy_msg(m)
                mm.payload_emb = emb[i]
                by_ep.setdefault(m.extras["episode_id"], []).append(mm)
            for e in missing:
                self.msgs[(enc, e.episode_id)] = by_ep.get(e.episode_id, [])
        return ProtocolMessageDataset([m for e in eps for m in self.msgs[(enc, e.episode_id)]])


def _copy_msg(m):
    import copy

    mm = copy.copy(m)
    mm.extras = dict(m.extras)
    return mm


def smoke_cut(eps: list, split: str, args: argparse.Namespace) -> list:
    if not args.smoke:
        return eps
    k = (48 if split == "train" else 24) // 2
    att = [e for e in eps if e.is_attack]
    ben = [e for e in eps if not e.is_attack]
    return att[:k] + ben[:k]


def window_labels(ds: ProtocolMessageDataset) -> np.ndarray:
    return np.array([int(ds[i][0].extras.get("is_attack_episode", 0)) for i in range(len(ds))])


def window_ids(ds: ProtocolMessageDataset) -> list[str]:
    return [ds[i][0].extras.get("episode_id", str(i)) for i in range(len(ds))]


def loader(ds, shuffle: bool, args: argparse.Namespace):
    from torch.utils.data import DataLoader, WeightedRandomSampler

    kw: dict[str, Any] = {"batch_size": args.batch_size, "collate_fn": protocol_collate,
                          "num_workers": args.workers, "pin_memory": args.device == "cuda"}
    if shuffle and args.balance:
        y = window_labels(ds)
        counts = np.bincount(y, minlength=2).astype(float)
        w = 1.0 / counts[y].clip(min=1)
        kw["sampler"] = WeightedRandomSampler(torch.as_tensor(w, dtype=torch.double), num_samples=len(ds), replacement=True)
    else:
        kw["shuffle"] = shuffle
    return DataLoader(ds, **kw)


def model_config(d_p: int, overrides: dict[str, Any]) -> MambaGuardConfig:
    d_mu = len(DEFAULT_METADATA_SCHEMA)
    return MambaGuardConfig(d_p=d_p, d_mu=d_mu, edge_dim=5 + d_mu, num_classes=2, **overrides)


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------
@torch.no_grad()
def predict(model, ds, args: argparse.Namespace) -> np.ndarray:
    model.eval()
    probs = []
    for batch in loader(ds, False, args):
        batch = {k: (v.to(args.device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        probs.append(torch.softmax(model(batch)["logits"].float(), dim=-1)[:, 1].cpu())
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
    scores = np.array([f1_score(y, (p >= t).astype(int), average="macro", zero_division=0) for t in grid])
    best = grid[scores >= scores.max() - 1e-12]  # ties: middle of the best range
    return float(np.median(best))


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


def summary_block(yv, pv, yt, pt) -> dict[str, Any]:
    thr = tune_threshold(yv, pv)
    return {"threshold_val_tuned": thr, "test_attack_frac": float(yt.mean()),
            "n_test": int(len(yt)), "n_test_benign": int((yt == 0).sum()),
            "test_at_0.5": metrics(yt, pt, 0.5), "test_at_val_threshold": metrics(yt, pt, thr),
            **bootstrap_ci(yt, pt, thr)}


# ---------------------------------------------------------------------------
# detector and lexical runs
# ---------------------------------------------------------------------------
def run_detector(tag: str, name: str, variant: str, seed: int, sets: dict, d_p: int, max_epochs: int,
                 args: argparse.Namespace, out: Path, keep: bool) -> dict[str, Any]:
    set_seed(seed)
    mo, to = VARIANTS[variant]
    model = MambaGuard(model_config(d_p, mo))
    ckpt_dir = out / "checkpoints" / tag
    epochs = 2 if args.smoke else max(1, int(round(max_epochs * args.epoch_scale)))
    tcfg = {"epochs": epochs, "num_classes": 2, "lr": args.lr, "out_dir": str(ckpt_dir), "log_every": 0,
            "monitor": "macro_f1", "amp_dtype": args.amp, "warmup_steps": 200, **to}
    stopper = EarlyStopping(patience=args.patience, monitor="val_macro_f1")
    trainer = MambaGuardTrainer(model, loader(sets["train"], True, args), loader(sets["val"], False, args),
                                cfg=tcfg, device=args.device, callbacks=[stopper])
    t0 = time.time()
    state = trainer.train()
    train_time = time.time() - t0
    trainer.load(str(ckpt_dir / "best.pt"))
    yv, pv = window_labels(sets["val"]), predict(trainer.model, sets["val"], args)
    yt, pt = window_labels(sets["test"]), predict(trainer.model, sets["test"], args)
    res = {"model": "mambaguard", "epochs_max": epochs, "epochs_run": state.epoch + 1, "best_epoch": state.best_epoch,
           "train_time_s": round(train_time, 1), "n_train": len(sets["train"]), "n_val": len(sets["val"]),
           "params": int(sum(p.numel() for p in trainer.model.parameters())), "d_p": d_p,
           **summary_block(yv, pv, yt, pt), "lipschitz_bound": float(trainer.model.lipschitz_bound())}
    np.savez_compressed(out / "predictions" / f"{tag}.npz", episode_id=np.array(window_ids(sets["test"])),
                        label=yt, prob=pt, val_label=yv, val_prob=pv)
    if keep:  # weights + config only; drop optimiser state and last.pt
        ck = torch.load(ckpt_dir / "best.pt", map_location="cpu", weights_only=False)
        torch.save({"model": ck["model"], "config": ck.get("config"), "d_p": d_p}, ckpt_dir / "best.pt")
        (ckpt_dir / "last.pt").unlink(missing_ok=True)
    else:
        shutil.rmtree(ckpt_dir, ignore_errors=True)
    return res


def episode_text(ep) -> str:
    return " \n ".join(m.payload if isinstance(m.payload, str) else json.dumps(m.payload) for m in ep.messages)


def run_lexical(eps: dict[str, list]) -> dict[str, Any]:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression

    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=50000, sublinear_tf=True)
    xtr = vec.fit_transform([episode_text(e) for e in eps["train"]])
    ytr = np.array([int(e.is_attack) for e in eps["train"]])
    clf = LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000).fit(xtr, ytr)
    yv = np.array([int(e.is_attack) for e in eps["val"]])
    yt = np.array([int(e.is_attack) for e in eps["test"]])
    pv = clf.predict_proba(vec.transform([episode_text(e) for e in eps["val"]]))[:, 1]
    pt = clf.predict_proba(vec.transform([episode_text(e) for e in eps["test"]]))[:, 1]
    return {"model": "lexical_tfidf_lr", "n_train": len(ytr), "n_val": len(yv), **summary_block(yv, pv, yt, pt)}


# ---------------------------------------------------------------------------
# held-out-template folds
# ---------------------------------------------------------------------------
def _agentdojo_family(t: str) -> str:
    if t.startswith("important_instructions"):
        return "important_instructions"
    if t.endswith("dos") or t == "dos":
        return "dos"
    return t  # tool_knowledge, ignore_previous, injecagent, direct


_SAFETY_FAMILY = {
    "poisoning": lambda t: t.startswith("Tool Poisoning"),
    "name_competition": lambda t: t in ("Function Overlapping", "Tool Shadowing", "Preference Manipulation"),
    "result_injection": lambda t: t in ("Function Return Injection", "Retrieval-Agent Deception", "Intent Injection",
                                        "Replay Injection", "Identity Spoofing"),
    "host_user": lambda t: t in ("Remote Access Control", "Malicious Code Execution", "Credential Theft",
                                 "Data Tampering", "Excessive Privileges Misuse"),
}


def heldout_folds(name: str) -> list[tuple[str, Callable[[Any], str | None], set]]:
    """(fold name, template key of an episode -- None for benign --, held-out keys)."""
    def attack_only(f):
        return lambda e: f(e) if e.is_attack else None

    if name == "mcptox":
        w = attack_only(lambda e: str(e.extras.get("wrapper")))
        p = attack_only(lambda e: str(e.extras.get("paradigm")))
        return ([(f"wrapper={v}", w, {v}) for v in ("important_tag", "plain", "ignore_previous")]
                + [(f"paradigm={v}", p, {v}) for v in ("Template-1", "Template-2", "Template-3")])
    if name == "agentdojo":
        f = attack_only(lambda e: _agentdojo_family(e.attack_type))
        return [(f"family={v}", f, {v}) for v in
                ("important_instructions", "tool_knowledge", "ignore_previous", "injecagent", "direct", "dos")]
    if name == "injecagent":
        f = attack_only(lambda e: f"case{h64(str(e.extras.get('attacker_tools'))) % 4}")
        return [(f"attacker_cases={k}", f, {f"case{k}"}) for k in range(4)]
    if name == "mcp_safetybench":
        def fam(e):
            return next((k for k, test in _SAFETY_FAMILY.items() if test(e.attack_type)), "other")
        f = attack_only(fam)
        return [(f"family={k}", f, {k}) for k in _SAFETY_FAMILY]
    if name == "a2a_sandbox":  # benign twins carry their finding: test keeps the held-out finding's pairs
        f = lambda e: str(e.extras.get("finding"))
        findings = ["scope_amplification", "multi_hop_identity_loss", "cross_client_context_injection",
                    "unverified_webhook_url", "unbounded_delegation_depth", "post_revocation_stream",
                    "unattested_skill_claims", "artifact_chunk_corruption", "agent_card_jws_trust",
                    "indefinite_interrupted_tasks", "stale_task_state"]
        return [(f"finding={v}", f, {v}) for v in findings]
    raise ValueError(name)


def heldout_split(data: Data, name: str, key, held: set, args: argparse.Namespace) -> dict[str, list]:
    model = HELDOUT_AGENTDOJO_MODEL if name == "agentdojo" else None
    pairs = args.heldout_a2a_pairs if name == "a2a_sandbox" else None
    out = {}
    for split in ("train", "val", "test"):
        eps = data.split(name, split, agentdojo_model=model, a2a_pairs=pairs)
        if name == "agentdojo":  # cap attacks per family so folds stay small
            cap = AGENTDOJO_CAP[split]
            kept, count = [], {}
            for e in sorted(eps, key=lambda e: h64(e.episode_id)):
                if e.is_attack:
                    fam = _agentdojo_family(e.attack_type)
                    if count.get(fam, 0) >= cap:
                        continue
                    count[fam] = count.get(fam, 0) + 1
                kept.append(e)
            eps = kept
        if split == "test":
            sel = [e for e in eps if (key(e) in held) or (key(e) is None and not e.is_attack)]
        else:
            sel = [e for e in eps if key(e) not in held]
        out[split] = smoke_cut(sel, split, args)
    return out


# ---------------------------------------------------------------------------
# certificate and latency
# ---------------------------------------------------------------------------
@torch.no_grad()
def certify(model, ds, args: argparse.Namespace) -> dict[str, Any]:
    """Cohen certificate on the payload embeddings of whole test windows."""
    model.eval()
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(ds))[: (8 if args.smoke else args.cert_windows)]
    y = window_labels(ds)
    n0, n = (8, 32) if args.smoke else (args.cert_n0, args.cert_n)
    report: dict[str, Any] = {"n0": n0, "n": n, "alpha": args.cert_alpha, "windows": len(idx), "sigmas": {}}
    for sigma in args.cert_sigmas:
        radii, correct = [], []
        for i in idx:
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
                    c += np.bincount(model(batch)["logits"].argmax(dim=-1).cpu().numpy(), minlength=2)
                return c

            top = int(np.argmax(votes(n0)))
            k = int(votes(n)[top])
            p_low = _clopper_pearson_lower(k, n, args.cert_alpha)
            radii.append(sigma * _phi_inv(p_low) if p_low > 0.5 else -1.0)
            correct.append(int(top == y[int(i)]))
        ra, ca = np.array(radii), np.array(correct)
        grid = [0.0, 0.02, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0]
        curve = [{"radius": r, "cert_acc": float(((ra >= r) & (ra >= 0) & (ca == 1)).mean())} for r in grid]
        bound = [{"radius": c["radius"],
                  "composed_lower_bound": float(composed_certificate_v2(1.0, c["cert_acc"], 1.0, 10_000, 5, args.cert_alpha))}
                 for c in curve]
        report["sigmas"][str(sigma)] = {"cert_acc_curve": curve, "composed_bound_V1_B1_T1e4_N5": bound,
                                        "abstain_rate": float((ra < 0).mean()),
                                        "median_radius_certified": float(np.median(ra[ra >= 0])) if (ra >= 0).any() else None}
    return report


@torch.no_grad()
def latency(model, ds, eps: list, encoder, args: argparse.Namespace) -> dict[str, Any]:
    model.eval()
    sync = torch.cuda.synchronize if args.device == "cuda" else (lambda: None)
    res: dict[str, Any] = {"device": args.device, "batch": {}, "encoder": {}}
    reps = 5 if args.smoke else 100
    for bs in (1, 8, 32, 128):
        if bs > len(ds):
            break
        batch = {k: (v.to(args.device) if torch.is_tensor(v) else v)
                 for k, v in protocol_collate([ds[i % len(ds)] for i in range(bs)]).items()}
        n_msgs = int(batch["attention_mask"].sum().item())
        for _ in range(10):
            model(batch)
        sync()
        times = []
        for _ in range(reps):
            t = time.perf_counter()
            model(batch)
            sync()
            times.append(time.perf_counter() - t)
        med = float(np.median(times))
        res["batch"][str(bs)] = {"windows": bs, "messages": n_msgs, "median_batch_ms": med * 1e3,
                                 "p99_batch_ms": float(np.percentile(times, 99)) * 1e3,
                                 "per_message_ms": med * 1e3 / max(n_msgs, 1), "messages_per_s": n_msgs / med}
    payloads = [m.payload for e in eps[:60] for m in e.messages][:200]
    if payloads:  # payload encoding cost (fresh encoder calls, no cache)
        encoder.encode_batch(payloads[:4])
        t = time.perf_counter()
        for p in payloads[:50]:
            encoder.encode_batch([p])
        one = (time.perf_counter() - t) / min(50, len(payloads))
        t = time.perf_counter()
        encoder.encode_batch(payloads)
        many = (time.perf_counter() - t) / len(payloads)
        res["encoder"] = {"per_message_ms_batch1": one * 1e3, "per_message_ms_batched": many * 1e3,
                          "messages": len(payloads)}
    return res


# ---------------------------------------------------------------------------
# plan
# ---------------------------------------------------------------------------
def build_plan(args: argparse.Namespace) -> list[dict[str, Any]]:
    """Ordered list of jobs; each has a unique ``tag`` (= result file name)."""
    jobs: list[dict[str, Any]] = []
    b_ok = [b for b in args.benchmarks]
    if args.study in ("main", "all"):
        for b in b_ok:
            for s in args.seeds:
                jobs.append(dict(kind="det", study="main", bench=b, variant="full", enc=MAIN_ENCODER, fold="all", seed=s))
        for b in b_ok:
            jobs.append(dict(kind="lex", study="main", bench=b, fold="all"))
        enc_seeds = args.seeds[:2]
        for b in [x for x in PAYLOAD_BENCHMARKS if x in b_ok]:
            for e in args.enc_ablation:
                for s in enc_seeds:
                    jobs.append(dict(kind="det", study="enc", bench=b, variant="full", enc=e, fold="all", seed=s))
        for b in [x for x in args.arch_benchmarks if x in b_ok]:
            for v in args.ablations:
                for s in enc_seeds:
                    jobs.append(dict(kind="det", study="arch", bench=b, variant=v, enc=MAIN_ENCODER, fold="all", seed=s))
        for b in b_ok:
            jobs.append(dict(kind="certify", study="main", bench=b))
            jobs.append(dict(kind="latency", study="main", bench=b))
    if args.study in ("heldout", "all"):
        for b in [x for x in args.heldout_benchmarks if x in b_ok]:
            folds = heldout_folds(b)
            if args.smoke:
                folds = folds[:2]
            seeds = args.heldout_seeds[:1] if (b == "a2a_sandbox" or args.smoke) else args.heldout_seeds
            for fold, _, _ in folds:
                jobs.append(dict(kind="lex", study="heldout", bench=b, fold=fold))
                for s in seeds:
                    jobs.append(dict(kind="det", study="heldout", bench=b, variant="full", enc=MAIN_ENCODER, fold=fold, seed=s))
    for j in jobs:
        if j["kind"] == "det":
            j["tag"] = f"{j['study']}__{j['bench']}__{j['variant']}__{j['enc']}__{j['fold']}__seed{j['seed']}"
        elif j["kind"] == "lex":
            j["tag"] = f"{j['study']}__{j['bench']}__lexical__{j['fold']}"
        else:
            j["tag"] = j["bench"]
        j["tag"] = j["tag"].replace("/", "_").replace(" ", "_")
    return jobs


def job_dir(kind: str) -> str:
    return {"det": "runs", "lex": "runs", "certify": "certify", "latency": "latency"}[kind]


def main_sets(data: Data, name: str, args: argparse.Namespace) -> dict[str, list]:
    return {s: smoke_cut(data.split(name, s, agentdojo_model=args.agentdojo_model), s, args)
            for s in ("train", "val", "test")}


def run_job(j: dict, data: Data, encoders: Encoders, args: argparse.Namespace, out: Path) -> dict[str, Any]:
    name = j["bench"]
    if j["study"] == "heldout":
        fold = next(f for f in heldout_folds(name) if f[0] == j["fold"])
        eps = heldout_split(data, name, fold[1], fold[2], args)
    else:
        eps = main_sets(data, name, args)
    meta = {k: j[k] for k in ("study", "bench", "fold") if k in j}
    meta["benchmark"] = name
    if name == "agentdojo":
        meta["agentdojo_model"] = HELDOUT_AGENTDOJO_MODEL if j["study"] == "heldout" else (args.agentdojo_model or "default")
    if j["kind"] == "lex":
        return {**meta, "variant": "lexical", "encoder": "tfidf", "seed": None, **run_lexical(eps)}
    if j["kind"] == "det":
        sets = {s: data.dataset(eps[s], j["enc"]) for s in eps}
        max_ep = BENCHMARKS[name][1]
        if j["study"] == "heldout" and name == "a2a_sandbox":
            max_ep = args.heldout_a2a_epochs
        keep = args.keep_checkpoints and j["study"] == "main" and j["seed"] == args.seeds[0]
        res = run_detector(j["tag"], name, j["variant"], j["seed"], sets, encoders.dim(j["enc"]), max_ep, args, out, keep)
        stats = getattr(encoders.get(j["enc"]), "stats", {})
        return {**meta, "variant": j["variant"], "encoder": j["enc"], "seed": j["seed"], **res, "encoder_stats": stats}
    # certify / latency use the kept seed-0 main checkpoint
    tag = f"main__{name}__full__{MAIN_ENCODER}__all__seed{args.seeds[0]}"
    ck = out / "checkpoints" / tag / "best.pt"
    if not ck.exists():
        return {**meta, "skipped": f"{ck} missing"}
    payload = torch.load(ck, map_location=args.device, weights_only=False)
    d_p = int(payload.get("d_p", encoders.dim(MAIN_ENCODER)))
    model = MambaGuard(model_config(d_p, {})).to(args.device)
    model.load_state_dict(payload["model"])
    ds = data.dataset(eps["test"], MAIN_ENCODER)
    if j["kind"] == "certify":
        return {**meta, "encoder": MAIN_ENCODER, "d_p": d_p, **certify(model, ds, args)}
    return {**meta, "encoder": MAIN_ENCODER, "d_p": d_p, **latency(model, ds, eps["test"], encoders.get(MAIN_ENCODER), args)}


# ---------------------------------------------------------------------------
def summarise(out: Path) -> dict[str, Any]:
    groups: dict[str, list] = {}
    for f in sorted((out / "runs").glob("*.json")):
        r = json.loads(f.read_text())
        key = f"{r.get('study')}__{r.get('benchmark')}__{r.get('variant')}__{r.get('encoder')}__{r.get('fold')}"
        groups.setdefault(key, []).append(r)
    table = {}
    for key, rs in groups.items():
        def ms(path):
            vals = [r[path[0]].get(path[1]) for r in rs if path[0] in r]
            vals = [v for v in vals if v is not None and not (isinstance(v, float) and math.isnan(v))]
            if not vals:
                return None
            return {"mean": float(np.mean(vals)), "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0, "n": len(vals)}
        table[key] = {"n_runs": len(rs), "macro_f1_val_threshold": ms(("test_at_val_threshold", "macro_f1")),
                      "auroc": ms(("test_at_0.5", "auroc")), "attack_recall": ms(("test_at_val_threshold", "attack_recall")),
                      "fpr": ms(("test_at_val_threshold", "fpr")), "n_test": rs[0].get("n_test")}
    return table


def resume(src: Path, out: Path) -> None:
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
    ap.add_argument("--study", choices=["main", "heldout", "all"], default="main")
    ap.add_argument("--data-root", default="/tmp/mambaguard_data")
    ap.add_argument("--out", default=None, help="default: /kaggle/working/results_<study>")
    ap.add_argument("--resume-from", default=None, help="results folder or .zip of an earlier session")
    ap.add_argument("--benchmarks", nargs="+", default=list(BENCHMARKS))
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--enc-ablation", nargs="*", default=["minilm", "chunk"],
                    help="encoders compared with chunk_lex on the payload benchmarks (seeds 0, 1)")
    ap.add_argument("--ablations", nargs="*", default=[v for v in VARIANTS if v != "full"])
    ap.add_argument("--arch-benchmarks", nargs="*", default=["a2a_sandbox", "mcptox"])
    ap.add_argument("--heldout-benchmarks", nargs="*", default=["mcptox", "agentdojo", "injecagent", "mcp_safetybench", "a2a_sandbox"])
    ap.add_argument("--heldout-seeds", nargs="+", type=int, default=[0, 1], help="A2A held-out folds use the first seed only")
    ap.add_argument("--heldout-a2a-pairs", type=int, default=100)
    ap.add_argument("--heldout-a2a-epochs", type=int, default=10)
    ap.add_argument("--encoder-model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--sentence-backend", choices=["minilm", "hash"], default="minilm",
                    help="hash = offline smoke test without MiniLM; never report its numbers")
    ap.add_argument("--d-lex", type=int, default=512)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--amp", default="auto", help="auto | fp32 | bf16 | fp16")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--epoch-scale", type=float, default=1.0)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--balance", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--a2a-pairs", type=int, default=200)
    ap.add_argument("--agentdojo-model", default=None, help="main study; default: the loader's choice (claude-3-5-sonnet)")
    ap.add_argument("--cert-sigmas", nargs="+", type=float, default=[0.05, 0.1, 0.25])
    ap.add_argument("--cert-windows", type=int, default=100)
    ap.add_argument("--cert-n0", type=int, default=32)
    ap.add_argument("--cert-n", type=int, default=500)
    ap.add_argument("--cert-alpha", type=float, default=1e-3)
    ap.add_argument("--cert-batch", type=int, default=50)
    ap.add_argument("--skip", nargs="*", default=[], choices=["enc", "arch", "certify", "latency", "lex"],
                    help="parts of the main study to skip")
    ap.add_argument("--keep-checkpoints", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--zip-checkpoints", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--time-budget-h", type=float, default=11.0)
    ap.add_argument("--smoke", action="store_true", help="tiny end-to-end check: few windows, 2 epochs, 2 folds")
    args = ap.parse_args()
    if args.smoke:
        args.seeds, args.heldout_seeds = args.seeds[:1], args.heldout_seeds[:1]
        args.enc_ablation = args.enc_ablation[:1]
        args.ablations = args.ablations[:2]
    out = Path(args.out or f"/kaggle/working/results_{args.study}")
    if args.smoke:
        out = out.with_name(out.name + "_smoke")
    if args.resume_from:
        resume(Path(args.resume_from), out)
    for d in ("runs", "predictions", "checkpoints", "certify", "latency"):
        (out / d).mkdir(parents=True, exist_ok=True)
    info = env_info(args)
    log(f"study={args.study} device={args.device} gpu={info['gpu']} encoder={MAIN_ENCODER} amp={args.amp}")
    deadline = time.time() + args.time_budget_h * 3600
    encoders = Encoders(args)
    data = Data(args, encoders)
    jobs = build_plan(args)
    skip_map = {"enc": lambda j: j["study"] == "enc", "arch": lambda j: j["study"] == "arch",
                "certify": lambda j: j["kind"] == "certify", "latency": lambda j: j["kind"] == "latency",
                "lex": lambda j: j["kind"] == "lex" and j["study"] == "main"}
    jobs = [j for j in jobs if not any(skip_map[s](j) for s in args.skip)]
    log(f"{len(jobs)} jobs planned")
    try:
        for i, j in enumerate(jobs, 1):
            dest = out / job_dir(j["kind"]) / f"{j['tag']}.json"
            if dest.exists():
                continue
            if time.time() > deadline:
                log("time budget reached; stopping")
                break
            root = benchmark_root(j["bench"], args.data_root)
            if root != "builtin" and not os.path.isdir(root):
                log(f"skip {j['tag']}: {root} not found")
                continue
            log(f"[{i}/{len(jobs)}] {j['kind']} {j['tag']}")
            t0 = time.time()
            try:
                res = run_job(j, data, encoders, args, out)
                dest.write_text(json.dumps(res, indent=2, default=str))
                if "test_at_val_threshold" in res:
                    t = res["test_at_val_threshold"]
                    log(f"   macro-F1 {t['macro_f1']:.3f}  AUROC {t.get('auroc', float('nan')):.3f}  "
                        f"recall {t['attack_recall']:.3f}  FPR {t['fpr']:.3f}  ({time.time() - t0:.0f}s)")
            except Exception as exc:  # noqa: BLE001
                import traceback

                (out / job_dir(j["kind"]) / f"{j['tag']}.error.txt").write_text(traceback.format_exc())
                log(f"   ERROR {exc!r}")
                if args.device == "cuda":
                    torch.cuda.empty_cache()
    finally:
        finish(out, info, args.zip_checkpoints)


if __name__ == "__main__":
    main()
