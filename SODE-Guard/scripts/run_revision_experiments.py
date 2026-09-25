"""Experiments E1–E4 for the revised paper; writes JSON and LaTeX table rows.

    E1  adaptive attacks on SODE-Guard at eps (l∞): PGD-40 on the scored
        predictor, EOT-PGD-100 (64 draws), APGD-CE / APGD-DLR with EOT,
        expected-margin PGD on F(x); plus the per-example worst case.
    E2  gradient-masking checks: ε sweep to chance, transfer from an
        undefended surrogate, Square (black box), PGD-step monotonicity.
    E3  PGD-AT / TRADES on a baseline, evaluated like E1.
    E4  Theorem A radii (N=512 paths, empirical-Bernstein correction) vs
        randomised smoothing at matched l2, and the Proposition B flip bound.

Example (GPU, trained checkpoints from scripts/reproduce_paper.sh):
    python scripts/run_revision_experiments.py e1 --config configs/sode_guard.yaml \
        --checkpoint experiments/sode_guard_tnnls_v3/ics3d/seed-42/best.pt \
        --benchmark ics3d --n-test 2000 --out experiments/revision
"""
from __future__ import annotations
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.attacks import PGD                                                     # noqa: E402
from src.attacks.eot_autoattack import (apgd, eot_pgd, expected_margin_pgd,     # noqa: E402
                                        fixed_predictor, stochastic_predictor)
from src.baselines import (CNN_LSTM, EGraphSAGEBaseline, IDSGraphMamba,          # noqa: E402
                           RTIDSTransformer, SurrogateIDS7B)
from src.certify import certify_mean, model_lipschitz, path_flip_bound          # noqa: E402
from src.data.registry import get_loader                                        # noqa: E402
from src.evaluation import gradient_masking as gm                               # noqa: E402
from src.evaluation.reliability import certify_smoothing                        # noqa: E402
from src.training.adversarial import train_adversarial                          # noqa: E402
from src.training.train import build_model                                      # noqa: E402
from src.utils import load_config, set_global_seed                              # noqa: E402
from src.utils.metrics import macro_f1                                          # noqa: E402

BASELINES = {"egraphsage": EGraphSAGEBaseline, "rtids": RTIDSTransformer, "cnn_lstm": CNN_LSTM,
             "ids_graphmamba": IDSGraphMamba, "ens7b": SurrogateIDS7B}


def device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_sode(cfg, ckpt: str):
    model = build_model(cfg).to(device())
    model.load_state_dict(torch.load(ckpt, map_location=device())["model"])
    return model.eval()


def load_baseline(name: str, ckpt: str | None, num_classes: int):
    cls = BASELINES[name]
    model = cls(num_classes=num_classes).to(device())
    if ckpt:
        model.load_state_dict(torch.load(ckpt, map_location=device()))
    return model.eval()


def test_tensors(bench: str, n: int, seed: int):
    loader = get_loader(bench, split="test", batch_size=4096, num_workers=2, seed=seed)
    xs, ys = [], []
    for x, y in loader:
        xs.append(x)
        ys.append(y)
        if sum(len(t) for t in ys) >= n:
            break
    return torch.cat(xs)[:n].to(device()), torch.cat(ys)[:n].to(device())


def batched(fn, x, y, bs: int):
    return torch.cat([fn(x[i:i + bs], y[i:i + bs]) for i in range(0, len(x), bs)])


def predictions(predict, x, bs: int = 512) -> torch.Tensor:
    with torch.no_grad():
        return torch.cat([predict(x[i:i + bs]).argmax(-1) for i in range(0, len(x), bs)])


def attack_suite(attack_predict, score_predict, x, y, eps, *, bs, eot, paths_em) -> dict:
    """Run every E1 attack; score all adversarial points with ``score_predict``."""
    atks = {
        "PGD-40": lambda a, b: PGD(attack_predict, eps=eps, steps=40)(a, b),
        "EOT-PGD-100": lambda a, b: eot_pgd(attack_predict, a, b, eps=eps, steps=100, eot=eot),
        "APGD-CE": lambda a, b: apgd(attack_predict, a, b, eps=eps, loss="ce", eot=max(eot // 4, 1)),
        "APGD-DLR": lambda a, b: apgd(attack_predict, a, b, eps=eps, loss="dlr", eot=max(eot // 4, 1)),
    }
    if paths_em:
        atks["E-margin PGD"] = lambda a, b: expected_margin_pgd(paths_em, a, b, eps=eps, steps=100)
    y_np = y.cpu().numpy()
    clean_pred = predictions(score_predict, x)
    out = {"clean": macro_f1(y_np, clean_pred.cpu().numpy())}
    worst = clean_pred.clone()
    for name, fn in atks.items():
        pred = predictions(score_predict, batched(fn, x, y, bs))
        out[name] = macro_f1(y_np, pred.cpu().numpy())
        wrong = (pred != y) & (worst == y)
        worst[wrong] = pred[wrong]
    out["worst"] = macro_f1(y_np, worst.cpu().numpy())
    return out


def tex_row(name: str, res: dict, cols: list[str]) -> str:
    return name + " & " + " & ".join(f"{res[c]:.3f}" for c in cols) + r" \\"


def cmd_e1(args, cfg):
    x, y = test_tensors(args.benchmark, args.n_test, args.seed)
    rows = {}
    sode = load_sode(cfg, args.checkpoint)
    score = fixed_predictor(sode, cfg.sde.monte_carlo.eval_paths) if args.deployment == "fixed" \
        else stochastic_predictor(sode, cfg.sde.monte_carlo.eval_paths)
    rows["SODE-Guard"] = attack_suite(stochastic_predictor(sode, cfg.sde.monte_carlo.eval_paths), score,
                                      x, y, args.eps, bs=args.batch, eot=args.eot,
                                      paths_em=stochastic_predictor(sode, args.em_paths))
    for spec in args.baseline or []:
        name, ckpt = spec.split("=", 1)
        model = load_baseline(name, ckpt, cfg.data.label_space)
        rows[name] = attack_suite(model, model, x, y, args.eps, bs=args.batch, eot=1, paths_em=None)
    return rows


def cmd_e2(args, cfg):
    x, y = test_tensors(args.benchmark, args.n_test, args.seed)
    sode = load_sode(cfg, args.checkpoint)
    predict = stochastic_predictor(sode, cfg.sde.monte_carlo.eval_paths)
    out = {"eps_sweep": gm.epsilon_sweep(predict, x, y),
           "steps": gm.step_monotonicity(predict, x, y, eps=args.eps),
           "square": gm.black_box(predict, x, y, eps=args.eps)}
    if args.surrogate:
        name, ckpt = args.surrogate.split("=", 1)
        out["transfer"] = gm.transfer_attack(load_baseline(name, ckpt, cfg.data.label_space),
                                             predict, x, y, eps=args.eps)
    out["chance_macro_f1"] = 1.0 / cfg.data.label_space
    return out


def cmd_e3(args, cfg):
    set_global_seed(args.seed)
    loader = get_loader(args.benchmark, split="train", batch_size=cfg.training.batch_size,
                        num_workers=cfg.experiment.num_workers, seed=args.seed)
    out = {}
    for method in args.methods:
        model = load_baseline(args.at_baseline, None, cfg.data.label_space).train()
        train_adversarial(model, loader, method=method, eps=args.eps, epochs=args.epochs, device=device())
        model.eval()
        ckpt = Path(args.out) / f"{args.at_baseline}_{method}_seed{args.seed}.pt"
        torch.save(model.state_dict(), ckpt)
        x, y = test_tensors(args.benchmark, args.n_test, args.seed)
        out[f"{args.at_baseline}+{method}"] = attack_suite(model, model, x, y, args.eps,
                                                           bs=args.batch, eot=1, paths_em=None)
    return out


def calibrate_B(model, bench: str, seed: int, n: int = 2048, paths: int = 64) -> float:
    """Ball radius B for P_B, fixed on training data before certification."""
    loader = get_loader(bench, split="train", batch_size=n, num_workers=2, seed=seed)
    x, _ = next(iter(loader))
    with torch.no_grad():
        norms = model.sample_logits(x.to(device()), list(range(paths))).norm(dim=-1)
    return float(1.1 * norms.max())


def cmd_e4(args, cfg):
    x, y = test_tensors(args.benchmark, args.n_test, args.seed)
    sode = load_sode(cfg, args.checkpoint)
    lip = model_lipschitz(sode)
    B = args.B or calibrate_B(sode, args.benchmark, args.seed)
    certs = [certify_mean(sode, x[i:i + args.batch], L=lip.L, B=B, n0=64, n=args.cert_paths,
                          alpha=args.alpha) for i in range(0, len(x), args.batch)]
    pred = torch.cat([c.prediction for c in certs])
    r2 = torch.cat([c.radius_l2 for c in certs])
    r2 = torch.where(pred == y, r2, torch.zeros_like(r2)).cpu().numpy()
    deployed = predictions(fixed_predictor(sode, cfg.sde.monte_carlo.eval_paths), x)
    out = {"lipschitz": lip.as_dict(), "B": B, "alpha": args.alpha, "n_paths": args.cert_paths,
           "frac_projected": float(np.mean([c.frac_projected for c in certs])),
           "certified_correct_frac": float((r2 > 0).mean()),
           "median_radius_l2_certified": float(np.median(r2[r2 > 0])) if (r2 > 0).any() else 0.0,
           "agreement_deployed_vs_certified": float((deployed == pred)[pred >= 0].float().mean()),
           "radii_l2": r2.tolist()}
    flips = {}
    for e in args.flip_eps:
        fb = torch.cat([path_flip_bound(sode, x[i:i + args.batch], L=lip.L, eps_l2=e)["flip_bound"]
                        for i in range(0, len(x), args.batch)])
        flips[str(e)] = float(fb.mean())
    out["flip_bound_mean"] = flips
    if args.smoothing_base:
        name, ckpt = args.smoothing_base.split("=", 1)
        base = load_baseline(name, ckpt, cfg.data.label_space)
        out["smoothing"] = {}
        for s in args.sigmas:
            r = np.concatenate([certify_smoothing(base, x[i:i + 64], y[i:i + 64], sigma=s,
                                                  K=cfg.data.label_space, n=args.smoothing_n)
                                for i in range(0, len(x), 64)])
            out["smoothing"][str(s)] = r.tolist()
    plot_e4(out, Path(args.out) / f"fig_e4_{args.benchmark}.pdf", feature_dim=cfg.data.feature_dim)
    return out


def plot_e4(res: dict, path: Path, feature_dim: int) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    radii = np.asarray(res["radii_l2"])
    grid = np.linspace(0, max(radii.max(), 1e-3) * 1.05, 200)
    ax.plot(grid, [(radii >= g).mean() for g in grid], label="SODE-Guard (Thm. A)", lw=2)
    for s, r in res.get("smoothing", {}).items():
        r = np.asarray(r)
        ax.plot(grid, [(r >= g).mean() for g in grid], "--", label=f"Rand. smoothing σ={s}")
    ax.set_xlabel(r"$\ell_2$ radius $r$ (z-scored features)")
    ax.set_ylabel("certified accuracy")
    top = ax.secondary_xaxis("top", functions=(lambda r: r / math.sqrt(feature_dim),
                                               lambda r: r * math.sqrt(feature_dim)))
    top.set_xlabel(r"implied $\ell_\infty$ radius $r/\sqrt{83}$")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("experiment", choices=["e1", "e2", "e3", "e4"])
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint")
    p.add_argument("--benchmark", default="ics3d")
    p.add_argument("--eps", type=float, default=0.03)
    p.add_argument("--n-test", type=int, default=2000)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="experiments/revision")
    p.add_argument("--deployment", choices=["fixed", "fresh"], default="fixed")
    p.add_argument("--eot", type=int, default=64)
    p.add_argument("--em-paths", type=int, default=256)
    p.add_argument("--baseline", action="append", help="name=checkpoint, e.g. ens7b=path.pt")
    p.add_argument("--surrogate", help="undefended surrogate for the transfer check, name=ckpt")
    p.add_argument("--at-baseline", default="ens7b", choices=list(BASELINES))
    p.add_argument("--methods", nargs="+", default=["pgd_at", "trades"])
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--cert-paths", type=int, default=512)
    p.add_argument("--alpha", type=float, default=1e-3)
    p.add_argument("--B", type=float, default=None)
    p.add_argument("--flip-eps", type=float, nargs="+", default=[0.01, 0.03, 0.1, 0.3])
    p.add_argument("--smoothing-base", help="name=ckpt of a Gaussian-noise-trained base classifier")
    p.add_argument("--sigmas", type=float, nargs="+", default=[0.12, 0.25, 0.5])
    p.add_argument("--smoothing-n", type=int, default=10_000)
    args = p.parse_args()

    cfg = load_config(args.config)
    Path(args.out).mkdir(parents=True, exist_ok=True)
    set_global_seed(args.seed)
    res = {"e1": cmd_e1, "e2": cmd_e2, "e3": cmd_e3, "e4": cmd_e4}[args.experiment](args, cfg)
    out = Path(args.out) / f"{args.experiment}_{args.benchmark}_seed{args.seed}.json"
    out.write_text(json.dumps(res, indent=2))
    if args.experiment in {"e1", "e3"}:
        cols = ["clean", "PGD-40", "EOT-PGD-100", "APGD-CE", "APGD-DLR", "E-margin PGD", "worst"]
        lines = [tex_row(k, {c: v.get(c, float("nan")) for c in cols}, cols) for k, v in res.items()]
        (Path(args.out) / f"{args.experiment}_rows.tex").write_text("\n".join(lines) + "\n")
    print(json.dumps({k: v for k, v in res.items() if k != "radii_l2"}, indent=2, default=str)[:4000])


if __name__ == "__main__":
    main()
