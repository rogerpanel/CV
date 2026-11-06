"""Empirical lower bound on the Lipschitz constant via gradient ascent.

The analytic composed Lipschitz bound ``L_f`` from
:mod:`lipschitz_bounds` is only claimed to be an upper bound. To
quantify how loose the bound is we also need an empirical *lower*
bound, obtained by attacking the Lipschitz ratio directly.

Given a clean input ``x_ref`` we maximise

    phi(delta) = || f(x_ref + delta) - f(x_ref) ||_2 / || delta ||_2

over ``delta`` in the L2 ball of radius ``bound`` via projected
gradient ascent. Any ratio observed during the search is a valid lower
bound on the local Lipschitz constant of ``f``; we report the best.
"""
from __future__ import annotations

from typing import Any, Callable, Mapping

import torch


def _extract_output(out: Any) -> torch.Tensor:
    if isinstance(out, dict):
        for k in ("logits", "agent_features", "token_features"):
            if k in out:
                return out[k]
        # fall back to first tensor value
        for v in out.values():
            if isinstance(v, torch.Tensor):
                return v
        raise ValueError("model output has no tensor")
    return out


def _project_l2(delta: torch.Tensor, radius: float) -> torch.Tensor:
    flat = delta.reshape(-1)
    n = flat.norm(p=2).clamp(min=1e-12)
    factor = torch.clamp(torch.as_tensor(radius / n), max=1.0)
    return (flat * factor).reshape_as(delta)


def empirical_lipschitz(
    model: Callable[..., Any],
    x_ref: Mapping[str, torch.Tensor] | torch.Tensor,
    n_steps: int = 200,
    lr: float = 0.01,
    bound: float = 1.0,
    input_key: str = "p",
    seed: int | None = 0,
    n_restarts: int = 4,
) -> dict:
    """Empirical lower bound on the local Lipschitz constant of ``model``.

    Parameters
    ----------
    model : callable
        ``model(x)`` or ``model(dict)``. ``model.eval()`` is called.
    x_ref : tensor or mapping
        Clean reference input. If a mapping, only ``input_key`` is
        perturbed.
    n_steps : int
        Gradient ascent steps.
    lr : float
        Step size.
    bound : float
        L2 ball radius for the perturbation.
    seed : int
        Seeds the initial perturbations for reproducibility.
    n_restarts : int
        Independent random restarts; the best ratio is returned.
    """
    if hasattr(model, "eval"):
        model.eval()
    if seed is not None:
        torch.manual_seed(seed)

    if isinstance(x_ref, torch.Tensor):
        is_dict = False
        base = x_ref.detach()
    else:
        is_dict = True
        base = x_ref[input_key].detach()

    with torch.no_grad():
        if is_dict:
            y_ref = _extract_output(model(dict(x_ref)))
        else:
            y_ref = _extract_output(model(x_ref))
        y_ref = y_ref.detach()

    best_ratio = 0.0
    best_delta_norm = 0.0
    best_output_shift = 0.0
    history = []
    for r in range(n_restarts):
        delta = torch.randn_like(base) * 1e-2
        delta = _project_l2(delta, bound)
        for step in range(n_steps):
            delta = delta.detach().requires_grad_(True)
            if is_dict:
                batch = dict(x_ref)
                batch[input_key] = base + delta
                out = _extract_output(model(batch))
            else:
                out = _extract_output(model(base + delta))
            diff = out - y_ref
            num = diff.reshape(-1).norm(p=2)
            den = delta.reshape(-1).norm(p=2).clamp(min=1e-8)
            ratio = num / den
            grad = torch.autograd.grad(ratio, delta, retain_graph=False)[0]
            # Gradient ASCENT on ratio.
            with torch.no_grad():
                g = grad.reshape(-1)
                gn = g.norm(p=2).clamp(min=1e-12)
                step_vec = (lr * (g / gn)).reshape_as(delta)
                delta = (delta.detach() + step_vec)
                delta = _project_l2(delta, bound)
            r_val = float(ratio.detach().item())
            if r_val > best_ratio:
                best_ratio = r_val
                best_delta_norm = float(den.detach().item())
                best_output_shift = float(num.detach().item())
        history.append(best_ratio)
    return {
        "empirical_L_lower": float(best_ratio),
        "best_delta_norm": float(best_delta_norm),
        "best_output_shift": float(best_output_shift),
        "n_steps": int(n_steps),
        "n_restarts": int(n_restarts),
        "lr": float(lr),
        "bound": float(bound),
        "per_restart_best": [float(x) for x in history],
    }
