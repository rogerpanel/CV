"""P50/P95 latency and throughput benchmark — Table 5 of the manuscript."""
from __future__ import annotations
import time
import numpy as np
import torch


@torch.no_grad()
def benchmark_latency(model, loader, *, device, warmup_batches: int = 5,
                      profile_batches: int = 50) -> dict:
    model.eval()
    samples = []
    n_seen = 0
    t0_all = time.time()
    for i, (x, _) in enumerate(loader):
        x = x.to(device)
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        _ = model.forward_mc(x)
        if device == "cuda":
            torch.cuda.synchronize()
        dt = (time.time() - t0) * 1000.0
        if i >= warmup_batches:
            samples.append(dt / x.shape[0])           # ms / sample
            n_seen += x.shape[0]
        if i >= warmup_batches + profile_batches:
            break
    arr = np.asarray(samples)
    elapsed = time.time() - t0_all
    return {
        "median_ms": float(np.median(arr)),
        "p95_ms": float(np.percentile(arr, 95)),
        "throughput_flows_per_s": float(n_seen / max(elapsed, 1e-9)),
        "samples": int(n_seen),
    }


def run_eval_cli():                                     # pragma: no cover
    import argparse, json
    from pathlib import Path
    from ..utils import load_config
    from ..training.train import build_model, evaluate
    from ..data.registry import get_loader
    from .robustness import evaluate_attacks
    from .certificate import certify_dataset

    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--benchmark", required=True)
    p.add_argument("--attacks", nargs="*", default=["pgd40"])
    p.add_argument("--epsilons", nargs="*", type=float,
                   default=[0.005, 0.01, 0.02, 0.03, 0.05, 0.10])
    p.add_argument("--certify", action="store_true")
    p.add_argument("--chaos-degree", type=int, default=4)
    args = p.parse_args()

    cfg = load_config(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model(cfg).to(device)
    sd = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(sd["model"]); model.eval()

    test_loader = get_loader(args.benchmark, split="test",
                             batch_size=cfg.training.batch_size,
                             num_workers=cfg.experiment.num_workers)
    out = {"clean": evaluate(model, test_loader, device)}
    atk_cfg = {a: {"type": "pgd" if a.startswith("pgd") else a,
                   "epsilons": args.epsilons, "steps": 40}
               for a in args.attacks}
    out["adversarial"] = evaluate_attacks(model, test_loader, device=device, attack_cfg=atk_cfg)
    if args.certify:
        out["certificate"] = certify_dataset(
            model, test_loader, device=device,
            chaos_degree=args.chaos_degree,
            smoothing_paths=cfg.evaluation.certified_radius.smoothing_paths,
            beta=cfg.evaluation.certified_radius.margin_threshold,
        )
    out["latency"] = benchmark_latency(model, test_loader, device=device)
    print(json.dumps(out, indent=2))
    Path(args.checkpoint).with_suffix(".eval.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":                              # pragma: no cover
    run_eval_cli()
