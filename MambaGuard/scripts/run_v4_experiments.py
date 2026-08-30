"""Substantiating numerical experiments for the MambaGuard v4 rewrite.

Runs everything the reviewers asked for that we can compute locally without
GPU / real datasets, and writes results to reproducibility/experiments.json.
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


OUT = Path("/home/user/CV/MambaGuard/reproducibility")
OUT.mkdir(parents=True, exist_ok=True)
RESULT: dict = {}


# -----------------------------------------------------------------------
# (a) Exact SiLU Lipschitz constant
# -----------------------------------------------------------------------
def silu_lipschitz() -> dict:
    xs = torch.linspace(-8.0, 8.0, 20001, dtype=torch.float64, requires_grad=True)
    y = F.silu(xs)
    g = torch.autograd.grad(y.sum(), xs)[0]
    L = float(g.abs().max().item())
    x_star = float(xs[g.abs().argmax()].item())
    return {"L_silu": L, "argmax_x": x_star}


RESULT["silu"] = silu_lipschitz()


# -----------------------------------------------------------------------
# (b) L_f for the default MambaGuard config
# (c) L_f sweep vs Lipschitz penalty lambda_L (via spectral-norm rescale)
# -----------------------------------------------------------------------
def _spec_norm(w: torch.Tensor) -> float:
    W = w.detach()
    if W.ndim == 1:
        return float(W.norm(p=2).item())
    W = W.reshape(W.shape[0], -1).float()
    try:
        return float(torch.linalg.matrix_norm(W, ord=2).item())
    except Exception:
        return float(torch.linalg.svdvals(W)[0].item())


def _apply_spectral_cap(model: torch.nn.Module, max_norm: float) -> None:
    """Rescale linear-layer weights so their spectral norm <= max_norm.

    This is the deployment-time projection version of the Lipschitz penalty
    used at training time; it lets us sweep the effective L_f as if the
    training-time lambda_L had driven every W to have spec-norm max_norm.
    """
    for m in model.modules():
        if isinstance(m, torch.nn.Linear):
            w = m.weight.data
            s = _spec_norm(w)
            if s > max_norm:
                w.mul_(max_norm / s)


def lipschitz_sweep() -> dict:
    from mambaguard.certification import compute_lipschitz_bound
    from mambaguard.models import MambaGuard, MambaGuardConfig

    torch.manual_seed(0)
    cfg = MambaGuardConfig()  # paper defaults: 4 SSM blocks, 4 GAT heads
    model = MambaGuard(cfg).eval()
    n_params = sum(p.numel() for p in model.parameters())
    baseline = compute_lipschitz_bound(model)

    caps = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
    sweep = []
    for cap in caps:
        m2 = MambaGuard(cfg).eval()
        m2.load_state_dict(model.state_dict())
        _apply_spectral_cap(m2, cap)
        rep = compute_lipschitz_bound(m2)
        scalar = {k: float(v) for k, v in rep.items() if isinstance(v, (int, float))}
        sweep.append({"spec_cap": cap, **scalar})
    baseline_scalar = {k: float(v) for k, v in baseline.items() if isinstance(v, (int, float))}
    return {
        "num_params": int(n_params),
        "baseline": baseline_scalar,
        "baseline_ssm_per_layer": [float(x) for x in baseline["L_ssm_components"]],
        "baseline_gat_per_layer": [float(x) for x in baseline["L_gat_components"]],
        "spec_cap_sweep": sweep,
    }


RESULT["lipschitz"] = lipschitz_sweep()


# -----------------------------------------------------------------------
# (d) Bochner encoding Lipschitz in Delta t (learnable omega)
# -----------------------------------------------------------------------
def bochner_lipschitz() -> dict:
    from mambaguard.models.bochner_encoding import BochnerTimeEncoding

    torch.manual_seed(0)
    enc = BochnerTimeEncoding(d_T=64).eval()
    with torch.no_grad():
        omega = enc.omega.detach()
    omega_max = float(omega.abs().max().item())
    lipschitz_delta_t = float(math.sqrt(2.0 / enc.omega.numel() * (omega**2).sum().item()))
    dt = torch.linspace(0.0, 100.0, 501, dtype=torch.float32)
    phi = enc(dt.unsqueeze(-1)).squeeze(-2) if enc(dt.unsqueeze(-1)).dim() == 3 else enc(dt)
    diffs = (phi[1:] - phi[:-1]).norm(dim=-1) / (dt[1:] - dt[:-1])
    return {
        "omega_max": omega_max,
        "L_bochner_dt_theory": lipschitz_delta_t,
        "L_bochner_dt_empirical_max": float(diffs.max().item()),
        "d_T": int(enc.omega.numel() * 2),
    }


RESULT["bochner"] = bochner_lipschitz()


# -----------------------------------------------------------------------
# (e) Hedge regret vs Cesa-Bianchi-Lugosi theoretical bound
# -----------------------------------------------------------------------
def hedge_experiment() -> dict:
    from mambaguard.certification.hedge import HedgeDefender

    rng = np.random.default_rng(42)
    n_actions = 5
    horizons = [1000, 5000, 10_000, 50_000]
    results = []
    for T in horizons:
        actions = [f"a{i}" for i in range(n_actions)]
        hd = HedgeDefender(actions=actions, horizon=T, B=1.0)
        cum = np.zeros(n_actions, dtype=float)
        played = 0.0
        for _ in range(T):
            losses = rng.uniform(0.0, 1.0, size=n_actions)
            idx = hd.sample(rng=rng)
            played += float(losses[idx])
            cum += losses
            hd.update(losses)
        regret = played - float(cum.min())
        results.append({
            "T": T,
            "empirical_regret": regret,
            "theoretical_bound": float(hd.regret_bound()),
            "average_empirical_regret": regret / T,
            "average_theoretical_bound": float(hd.average_regret_bound()),
        })
    return {"num_actions": n_actions, "runs": results}


RESULT["hedge"] = hedge_experiment()


# -----------------------------------------------------------------------
# (f) Stackelberg LP solution on the default 5x7 utility matrix
# -----------------------------------------------------------------------
def stackelberg_experiment() -> dict:
    try:
        from mambaguard.certification import (
            ATTACKER_ACTIONS,
            DEFENDER_ACTIONS,
            StackelbergSolver,
            default_utility_matrix,
        )
        U = default_utility_matrix(B=1.0)
        solver = StackelbergSolver(
            defender_actions=list(DEFENDER_ACTIONS),
            attacker_actions=list(ATTACKER_ACTIONS),
            utility_matrix=U,
        )
        sol = solver.solve()
        return {
            "V_star": float(sol.value),
            "pi_D": [float(x) for x in sol.pi_D],
            "attacker_best_response_idx": int(sol.attacker_best_response),
            "attacker_best_response": ATTACKER_ACTIONS[sol.attacker_best_response],
            "defender_actions": list(DEFENDER_ACTIONS),
            "attacker_actions": list(ATTACKER_ACTIONS),
        }
    except ImportError as exc:
        return {"error": str(exc)}


RESULT["stackelberg"] = stackelberg_experiment()


# -----------------------------------------------------------------------
# (g) Composed three-layer certificate across epsilon sweep
# -----------------------------------------------------------------------
def certificate_sweep() -> dict:
    from mambaguard.certification import composed_certificate

    L_f = RESULT["lipschitz"]["baseline"].get("L_f", 1.0)
    caps = RESULT["lipschitz"]["spec_cap_sweep"]
    capped = next((c["L_f"] for c in caps if abs(c["spec_cap"] - 1.0) < 1e-6), L_f)
    V_star = RESULT["stackelberg"].get("V_star", 1.0)
    B = 1.0
    T = 10_000
    n_actions = len(RESULT["stackelberg"].get("defender_actions", ["a"] * 5))
    epsilons = [0.005, 0.01, 0.02, 0.05, 0.10]
    rows = []
    for eps in epsilons:
        rows.append({
            "epsilon": eps,
            "cert_lower_bound_capped": float(
                composed_certificate(V_star, capped, eps, B, T, n_actions)
            ),
        })
    return {
        "V_star": V_star,
        "L_f_capped_at_specnorm_1": float(capped),
        "L_f_raw": float(L_f),
        "T": T,
        "B": B,
        "num_actions": n_actions,
        "sweep": rows,
    }


RESULT["certificate"] = certificate_sweep()


# -----------------------------------------------------------------------
# (h) Latency decomposition on CPU (batch-1, batch-32, batch-256)
# -----------------------------------------------------------------------
def latency_experiment() -> dict:
    from mambaguard.models import MambaGuard, MambaGuardConfig

    torch.manual_seed(0)
    cfg = MambaGuardConfig()
    model = MambaGuard(cfg).eval()
    d_p = cfg.d_p
    d_mu = cfg.d_mu
    device = "cpu"
    L = 16
    results = {}
    for bs in (1, 32, 128):
        p = torch.randn(bs, L, d_p, device=device)
        mu = torch.randn(bs, L, d_mu, device=device)
        batch = {"p": p, "mu": mu}
        for _ in range(3):
            with torch.no_grad():
                _ = model(batch)
        times = []
        for _ in range(15):
            t0 = time.perf_counter()
            with torch.no_grad():
                _ = model(batch)
            times.append(time.perf_counter() - t0)
        arr = np.asarray(times) * 1000.0
        results[f"batch_{bs}"] = {
            "batch_latency_ms_mean": float(arr.mean()),
            "batch_latency_ms_p50": float(np.percentile(arr, 50)),
            "batch_latency_ms_p95": float(np.percentile(arr, 95)),
            "per_message_latency_ms_mean": float(arr.mean() / bs),
            "throughput_msg_s": float(1000.0 * bs / arr.mean()),
            "batch_size": bs,
            "seq_len": L,
        }
    return {"device": device, "note": "CPU proxy; A100 numbers in paper are ~10x faster.", "runs": results}


RESULT["latency"] = latency_experiment()


# -----------------------------------------------------------------------
# (i) Ablation: SSM-only, GAT-only, Transformer-instead-of-SSM
# -----------------------------------------------------------------------
def ablation_variants() -> dict:
    from mambaguard.models import MambaGuard, MambaGuardConfig
    import torch.nn as nn

    torch.manual_seed(0)
    cfg = MambaGuardConfig()
    L = 16
    bs = 8

    def _fwd_ms(model, batch, iters=10):
        with torch.no_grad():
            for _ in range(3):
                model(batch)
        ts = []
        for _ in range(iters):
            t0 = time.perf_counter()
            with torch.no_grad():
                model(batch)
            ts.append(time.perf_counter() - t0)
        return float(np.mean(ts) * 1000.0)

    batch = {"p": torch.randn(bs, L, cfg.d_p), "mu": torch.randn(bs, L, cfg.d_mu)}
    variants = {}

    full = MambaGuard(cfg).eval()
    variants["full"] = {
        "params": int(sum(p.numel() for p in full.parameters())),
        "fwd_batch_latency_ms": _fwd_ms(full, batch),
    }

    cfg_no_gat = MambaGuardConfig(n_gat_layers=0)
    m_no_gat = MambaGuard(cfg_no_gat).eval()
    variants["mamba_only"] = {
        "params": int(sum(p.numel() for p in m_no_gat.parameters())),
        "fwd_batch_latency_ms": _fwd_ms(m_no_gat, batch),
    }

    try:
        cfg_no_ssm = MambaGuardConfig(n_blocks=0)
        m_no_ssm = MambaGuard(cfg_no_ssm).eval()
        variants["gat_only"] = {
            "params": int(sum(p.numel() for p in m_no_ssm.parameters())),
            "fwd_batch_latency_ms": _fwd_ms(m_no_ssm, batch),
        }
    except Exception as exc:
        variants["gat_only"] = {"error": str(exc)}

    class TxProxy(nn.Module):
        def __init__(self, cfg):
            super().__init__()
            self.in_proj = nn.Linear(cfg.d_p + cfg.d_mu, cfg.d_model)
            enc_layer = nn.TransformerEncoderLayer(
                d_model=cfg.d_model, nhead=4, dim_feedforward=cfg.d_model * 2, batch_first=True
            )
            self.enc = nn.TransformerEncoder(enc_layer, num_layers=cfg.n_blocks)
            self.head = nn.Linear(cfg.d_model, cfg.num_classes)

        def forward(self, messages, graph=None):
            x = torch.cat([messages["p"], messages["mu"]], dim=-1)
            x = self.in_proj(x)
            x = self.enc(x)
            return {"logits": self.head(x.mean(dim=1))}

    tx = TxProxy(cfg).eval()
    variants["transformer_instead_of_ssm"] = {
        "params": int(sum(p.numel() for p in tx.parameters())),
        "fwd_batch_latency_ms": _fwd_ms(tx, batch),
    }

    return {"batch_size": bs, "seq_len": L, "device": "cpu", "variants": variants}


try:
    RESULT["ablation"] = ablation_variants()
except Exception as exc:
    RESULT["ablation"] = {"error": str(exc)}


# -----------------------------------------------------------------------
# (j) Data-dependent Tsuzuku Lipschitz-margin certified radius
# -----------------------------------------------------------------------
def tsuzuku_certified_radius() -> dict:
    from mambaguard.models import MambaGuard, MambaGuardConfig

    torch.manual_seed(0)
    cfg = MambaGuardConfig()
    model = MambaGuard(cfg).eval()
    _apply_spectral_cap(model, 1.0)
    from mambaguard.certification import compute_lipschitz_bound
    rep = compute_lipschitz_bound(model)
    L_f = float(rep["L_f"])

    bs = 128
    L = 16
    batch = {"p": torch.randn(bs, L, cfg.d_p), "mu": torch.randn(bs, L, cfg.d_mu)}
    with torch.no_grad():
        out = model(batch)
    logits = out["logits"]
    top2 = logits.topk(2, dim=-1).values
    margins = (top2[:, 0] - top2[:, 1]).cpu().numpy()
    radii = margins / (math.sqrt(2.0) * L_f)
    return {
        "L_f_capped": L_f,
        "num_samples": int(bs),
        "margin_mean": float(np.mean(margins)),
        "margin_std": float(np.std(margins)),
        "certified_radius_mean": float(np.mean(radii)),
        "certified_radius_p50": float(np.percentile(radii, 50)),
        "certified_radius_p95": float(np.percentile(radii, 95)),
        "certified_frac_at_eps_005": float(np.mean(radii >= 0.005)),
        "certified_frac_at_eps_01": float(np.mean(radii >= 0.01)),
        "certified_frac_at_eps_05": float(np.mean(radii >= 0.05)),
    }


try:
    RESULT["tsuzuku"] = tsuzuku_certified_radius()
except Exception as exc:
    RESULT["tsuzuku"] = {"error": str(exc)}


# -----------------------------------------------------------------------
# Save
# -----------------------------------------------------------------------
out_path = OUT / "experiments.json"
with open(out_path, "w", encoding="utf-8") as fh:
    json.dump(RESULT, fh, indent=2, default=float)

print(json.dumps(RESULT, indent=2, default=float))
