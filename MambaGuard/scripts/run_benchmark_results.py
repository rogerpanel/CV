"""Train and evaluate MambaGuard on every protocol benchmark, on CPU.

Writes reproducibility/benchmark_results.json. Uses the hashing payload encoder
(no download, no GPU), so the numbers are fully reproducible on a laptop. Also
computes the episode-length shortcut baseline for each benchmark, so the paper
can report the detector against it.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mambaguard.data.canonicalize import DEFAULT_METADATA_SCHEMA, HashingCanonicaliser  # noqa: E402
from mambaguard.data.datasets import ProtocolMessageDataset, protocol_collate  # noqa: E402
from mambaguard.data.protocol_traces import EPISODE_LOADERS, flatten  # noqa: E402
from mambaguard.evaluation.metrics import compute_metrics  # noqa: E402
from mambaguard.models import MambaGuard, MambaGuardConfig  # noqa: E402
from mambaguard.training import MambaGuardTrainer  # noqa: E402
from mambaguard.utils.seed import set_seed  # noqa: E402

DATA = os.environ.get("DATA_ROOT", str(ROOT / "data" / "raw"))
ROOTS = {
    "mcptox": f"{DATA}/MCPTox-Benchmark",
    "mcp_safetybench": f"{DATA}/MCPSafety",
    "agentdojo": f"{DATA}/agentdojo",
    "injecagent": f"{DATA}/InjecAgent",
    "a2a_sandbox": "builtin",
}
EPOCHS = {"mcptox": 4, "mcp_safetybench": 6, "agentdojo": 6, "injecagent": 5, "a2a_sandbox": 4}
SEEDS = [int(s) for s in os.environ.get("SEEDS", "0").split()]


def _loader(name, root, split):
    return list(flatten(EPISODE_LOADERS[name](root, split)))


def _episodes(name, root, split):
    return EPISODE_LOADERS[name](root, split)


def length_baseline(name, root) -> float:
    tr, te = _episodes(name, root, "train"), _episodes(name, root, "test")
    if not tr or not te:
        return float("nan")
    xs = sorted({len(e.messages) for e in tr})
    best = (0.0, xs[0], 1)
    for t in xs:
        for d in (1, -1):
            acc = sum(((d * (len(e.messages) - t) >= 0) == e.is_attack) for e in tr) / len(tr)
            if acc > best[0]:
                best = (acc, t, d)
    _, t, d = best
    pos = [e for e in te if e.is_attack]
    neg = [e for e in te if not e.is_attack]
    tpr = np.mean([d * (len(e.messages) - t) >= 0 for e in pos]) if pos else 0.0
    tnr = np.mean([d * (len(e.messages) - t) < 0 for e in neg]) if neg else 0.0
    return float((tpr + tnr) / 2)


def dataset(name, root, split, enc):
    return ProtocolMessageDataset(_loader(name, root, split), canonicaliser=enc, compute_embeddings=True)


def run_one(name: str, root: str, seed: int) -> dict:
    set_seed(seed)
    enc = HashingCanonicaliser(d_p=384)
    from torch.utils.data import DataLoader

    tr = dataset(name, root, "train", enc)
    va = dataset(name, root, "val", enc)
    te = dataset(name, root, "test", enc)
    dl = lambda ds, sh: DataLoader(ds, batch_size=32, shuffle=sh, collate_fn=protocol_collate)
    cfg = MambaGuardConfig(d_mu=len(DEFAULT_METADATA_SCHEMA), edge_dim=5 + len(DEFAULT_METADATA_SCHEMA), num_classes=2)
    model = MambaGuard(cfg)
    tcfg = {"epochs": EPOCHS[name], "num_classes": 2, "lr": 3e-4, "out_dir": f"/tmp/mg_{name}_{seed}",
            "log_every": 0, "monitor": "macro_f1", "amp_dtype": "fp32"}
    trainer = MambaGuardTrainer(model, dl(tr, True), dl(va, False), cfg=tcfg, device="cpu")
    t0 = time.time()
    trainer.train()
    trainer.load(os.path.join(tcfg["out_dir"], "best.pt"))
    metrics = trainer.evaluate(dl(te, False))
    return {
        "seed": seed, "epochs": EPOCHS[name], "train_time_s": round(time.time() - t0, 1),
        "n_train": len(tr), "n_test": len(te),
        "test_macro_f1": round(metrics.get("macro_f1", float("nan")), 4),
        "test_accuracy": round(metrics.get("accuracy", float("nan")), 4),
        "test_recall_attack": round(metrics.get("per_class_recall", [float("nan"), float("nan")])[1], 4),
        "test_fpr": round(1 - metrics.get("per_class_recall", [float("nan")])[0], 4),
        "latency_ms_per_msg": round(metrics.get("mean_latency_ms", float("nan")), 3),
    }


def main() -> None:
    out: dict = {"encoder": "hashing-384 (feature hashing over token n-grams; no external model)",
                 "device": "cpu", "readout": "graph", "params": None, "benchmarks": {}}
    out["params"] = sum(p.numel() for p in MambaGuard(MambaGuardConfig(num_classes=2)).parameters())
    for name, root in ROOTS.items():
        if root != "builtin" and not os.path.isdir(root):
            out["benchmarks"][name] = {"skipped": f"{root} not found"}
            print(f"[skip] {name}: {root} not found", flush=True)
            continue
        print(f"[run] {name}", flush=True)
        entry: dict = {"length_baseline_bal_acc": round(length_baseline(name, root), 4), "runs": []}
        for seed in SEEDS:
            try:
                entry["runs"].append(run_one(name, root, seed))
                print(f"   seed {seed}: {entry['runs'][-1]}", flush=True)
            except Exception as exc:  # noqa: BLE001
                entry["runs"].append({"seed": seed, "error": repr(exc)})
                print(f"   seed {seed} ERROR: {exc!r}", flush=True)
        f1s = [r["test_macro_f1"] for r in entry["runs"] if "test_macro_f1" in r]
        if f1s:
            entry["macro_f1_mean"] = round(float(np.mean(f1s)), 4)
            entry["macro_f1_std"] = round(float(np.std(f1s)), 4)
        out["benchmarks"][name] = entry

    # MCPSecBench: evaluation only, with the MCPTox-trained model if available.
    outpath = ROOT / "reproducibility" / "benchmark_results.json"
    outpath.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
