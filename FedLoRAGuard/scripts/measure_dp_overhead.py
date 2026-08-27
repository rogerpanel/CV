#!/usr/bin/env python
"""Measure the wall-clock overhead of the DP + SecAgg stack against a
non-private FedAvg baseline on identical hardware and identical local work.

This is the instrumentation referenced in Appendix "Reproducibility ->
Training-time overhead of the privacy stack" of the manuscript.

Usage::

    python scripts/measure_dp_overhead.py \
        --config configs/smoke.yaml --data /tmp/fedloraguard_bench \
        --warmup 3 --rounds 30 --out runs/overhead.json
"""
from __future__ import annotations

import argparse
import copy
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from fedloraguard.federated.client import ClientConfig, FederatedClient
from fedloraguard.federated.sampling import build_query_batch
from fedloraguard.federated.secure_agg import SecureAggregator
from fedloraguard.federated.fltrust import fltrust_score, fltrust_normalize
from fedloraguard.models.verifier import build_verifier
from fedloraguard.privacy.dp_sgd import ClipNoiseConfig
from fedloraguard.utils import load_config, set_seed


def _one_round(
    verifier: nn.Module,
    clients: Dict[int, FederatedClient],
    sampled_ids: List[int],
    root_batch,
    criterion: nn.Module,
    lr: float,
    dp_enabled: bool,
    secagg_enabled: bool,
    fltrust_enabled: bool,
    clip_norm: float,
    noise_multiplier: float,
    aggregator: SecureAggregator,
) -> float:
    """Run one full federated round; return wall-clock elapsed seconds."""
    t0 = time.perf_counter()
    global_state = {k: v.detach().clone() for k, v in verifier.state_dict().items()}

    # FLTrust reference gradient (server side).
    if fltrust_enabled and root_batch:
        ref_client = next(iter(clients.values()))
        ref_client.set_state(global_state)
        ref_grads = ref_client.reference_gradient(root_batch, criterion)
    else:
        ref_grads = None

    # Local updates on each sampled client.
    clip_noise = ClipNoiseConfig(
        clip_norm=clip_norm,
        noise_multiplier=noise_multiplier if dp_enabled else 0.0,
        enabled=dp_enabled,
    )
    client_grads: Dict[int, List[torch.Tensor]] = {}
    for cid in sampled_ids:
        clients[cid].set_state(global_state)
        client_grads[cid] = clients[cid].local_update(criterion, lr, clip_noise)

    # FLTrust weighting.
    if fltrust_enabled and ref_grads is not None:
        weights = {cid: fltrust_score(g, ref_grads) for cid, g in client_grads.items()}
        normalized = {cid: fltrust_normalize(g, ref_grads) for cid, g in client_grads.items()}
    else:
        weights = {cid: 1.0 for cid in client_grads}
        normalized = client_grads

    contributions = {cid: [w * g for g in normalized[cid]]
                     for cid, w in weights.items()}
    # Secure aggregation (or plain sum-of-tensors fallback if disabled).
    if secagg_enabled:
        summed = aggregator.aggregate(contributions)
    else:
        ids = list(contributions.keys())
        summed = [torch.stack([contributions[c][p] for c in ids], dim=0).sum(dim=0)
                  for p in range(len(contributions[ids[0]]))]
    total_w = max(sum(weights.values()), 1e-9)
    avg = [g / total_w for g in summed]

    # Apply.
    with torch.no_grad():
        for p, g in zip(verifier.parameters(), avg):
            p.add_(g, alpha=-lr)

    return time.perf_counter() - t0


def _make_clients(cfg, client_graphs, device):
    clients: Dict[int, FederatedClient] = {}
    for cid, graph in client_graphs.items():
        v = build_verifier(cfg)
        clients[int(cid)] = FederatedClient(
            ClientConfig(
                client_id=int(cid),
                local_epochs=cfg["federated"]["local_epochs"],
                batch_size=cfg["federated"]["batch_size"],
            ),
            v, graph, device=device,
        )
    return clients


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--rounds", type=int, default=30)
    ap.add_argument("--out", default="runs/overhead.json")
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["experiment"]["seed"])
    device = "cpu"

    client_graphs = torch.load(Path(args.data) / "client_graphs.pt", weights_only=False)
    root_set = torch.load(Path(args.data) / "root_set.pt", weights_only=False)
    root_batch = build_query_batch(root_set, batch_size=cfg["federated"]["fltrust"]["root_set_size"])

    m = min(cfg["federated"]["clients_per_round"], len(client_graphs))
    rng = np.random.default_rng(cfg["experiment"]["seed"])

    # Run each configuration on a fresh model + fresh clients, identical seed.
    def _driver(dp: bool, secagg: bool) -> Dict[str, float]:
        set_seed(cfg["experiment"]["seed"])
        verifier = build_verifier(cfg).to(device)
        clients = _make_clients(cfg, client_graphs, device)
        aggregator = SecureAggregator(num_clients=len(client_graphs), enabled=secagg)
        criterion = nn.CrossEntropyLoss()
        lr = cfg["optim"]["lr"]
        cn = cfg["privacy"]["clip_norm"]
        nm = cfg["privacy"]["noise_multiplier"]

        sampled_ids = list(client_graphs.keys())[:m]
        # Warm up.
        for _ in range(args.warmup):
            _one_round(verifier, clients, sampled_ids, root_batch, criterion,
                       lr, dp, secagg, cfg["federated"]["fltrust"]["enabled"],
                       cn, nm, aggregator)
        # Measure.
        times = []
        for _ in range(args.rounds):
            elapsed = _one_round(verifier, clients, sampled_ids, root_batch, criterion,
                                 lr, dp, secagg, cfg["federated"]["fltrust"]["enabled"],
                                 cn, nm, aggregator)
            times.append(elapsed)
        return {
            "mean": float(statistics.mean(times)),
            "median": float(statistics.median(times)),
            "stdev": float(statistics.pstdev(times)) if len(times) > 1 else 0.0,
            "min": float(min(times)),
            "max": float(max(times)),
            "samples": times,
        }

    print(f"Warm-up rounds: {args.warmup}  |  measurement rounds: {args.rounds}")
    print(f"Sampled clients per round: m={m}  |  device: {device}")
    print()
    print("Measuring: FedAvg (DP off, SecAgg off) ...")
    baseline = _driver(dp=False, secagg=False)
    print(f"  median = {baseline['median']*1e3:.1f} ms   mean = {baseline['mean']*1e3:.1f} ms   stdev = {baseline['stdev']*1e3:.1f} ms")
    print("Measuring: FedAvg + DP (SecAgg off) ...")
    dp_only = _driver(dp=True, secagg=False)
    print(f"  median = {dp_only['median']*1e3:.1f} ms   mean = {dp_only['mean']*1e3:.1f} ms")
    print("Measuring: FedAvg + DP + SecAgg (full stack) ...")
    full = _driver(dp=True, secagg=True)
    print(f"  median = {full['median']*1e3:.1f} ms   mean = {full['mean']*1e3:.1f} ms")

    overhead_dp_pct = 100.0 * (dp_only["median"] - baseline["median"]) / baseline["median"]
    overhead_full_pct = 100.0 * (full["median"] - baseline["median"]) / baseline["median"]

    result = {
        "config": args.config,
        "warmup_rounds": args.warmup,
        "measurement_rounds": args.rounds,
        "sampled_clients_per_round": m,
        "device": device,
        "baseline_fedavg": baseline,
        "dp_only": dp_only,
        "dp_plus_secagg": full,
        "overhead_dp_only_pct": overhead_dp_pct,
        "overhead_full_stack_pct": overhead_full_pct,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)

    print()
    print(f"=== Overhead (median) ===")
    print(f"  DP alone      : {overhead_dp_pct:+.2f} %")
    print(f"  DP + SecAgg   : {overhead_full_pct:+.2f} %  <== value for the manuscript XX%")


if __name__ == "__main__":
    main()
