"""Certified-radius evaluation using the anti-concentration bound.

For each test example we:
    1. Estimate the smoothed margin m_i via N MC paths through the SDE.
    2. Estimate the local Lipschitz constant L_g of the smoothed score by
       finite differences (cheap because the SDE smooths g).
    3. Invert the Carbery–Wright bound to obtain r*_i.

We report Cohen-et-al-style *certified accuracy* curves: for each r the
fraction of test points whose r*_i ≥ r and whose top-1 class is correct.
"""
from __future__ import annotations
import numpy as np
import torch

from ..regularizers.anti_concentration import (
    anti_concentration_certificate, certified_radius,
)


@torch.no_grad()
def _estimate_lipschitz(model, x: torch.Tensor, eps_probe: float = 1e-3,
                        num_probes: int = 4) -> float:
    """Crude local Lipschitz estimate via random directions."""
    L = 0.0
    base = model.forward_mc(x).clamp_min(1e-12).log()
    for _ in range(num_probes):
        d = torch.randn_like(x); d = d / (d.flatten(1).norm(dim=1, keepdim=True) + 1e-9).view(-1, *([1] * (x.ndim - 1)))
        perturb = model.forward_mc(x + eps_probe * d).clamp_min(1e-12).log()
        diff = (perturb - base).flatten(1).norm(dim=1) / eps_probe
        L = max(L, float(diff.max().item()))
    return L


def certify_dataset(model, loader, *, device, chaos_degree: int = 4,
                    beta: float = 0.05, confidence: float = 0.95,
                    smoothing_paths: int = 256,
                    radii: tuple[float, ...] = (0.0, 0.005, 0.01, 0.02, 0.05, 0.10)
                    ) -> dict:
    model.eval()
    all_margins, all_radii, all_correct = [], [], []
    L_est = None
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        if L_est is None:
            L_est = _estimate_lipschitz(model, x[:64])
        probs = model.forward_mc(x, n_paths=smoothing_paths)
        pred = probs.argmax(-1)
        top2, _ = torch.topk(probs, k=2, dim=-1)
        margin = (top2[:, 0] - top2[:, 1])
        r_star = certified_radius(margin, lipschitz=L_est,
                                  chaos_degree=chaos_degree,
                                  beta=beta, confidence=confidence)
        all_margins.append(margin.cpu().numpy())
        all_radii.append(r_star.cpu().numpy())
        all_correct.append((pred == y).cpu().numpy())
    margins = np.concatenate(all_margins)
    r = np.concatenate(all_radii)
    correct = np.concatenate(all_correct)
    curve = {float(rr): float(((r >= rr) & correct).mean()) for rr in radii}
    return {
        "lipschitz_estimate": float(L_est or 0.0),
        "median_margin": float(np.median(margins)),
        "median_radius": float(np.median(r)),
        "certified_accuracy": curve,
    }
