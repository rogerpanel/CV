"""Adaptive attacks for the stochastic SODE-Guard predictor.

All attacks take a ``predict`` callable mapping inputs to logits. For
SODE-Guard use ``stochastic_predictor`` (fresh Brownian paths on every call,
the deployed randomised model) or ``fixed_predictor`` (the same seeds on every
call, a deterministic network). Features are z-scored, so no box constraint is
applied unless ``box`` is given.

    eot_pgd              PGD with the gradient averaged over ``eot`` draws.
    expected_margin_pgd  PGD on the CW margin of the mean predictor F(x),
                         estimated with many fresh paths per step.
    apgd                 APGD-CE / APGD-DLR (Croce & Hein, 2020) with EOT.
    square_attack        Score-based black-box Square attack (Andriushchenko
                         et al., 2020) adapted to feature vectors.
"""
from __future__ import annotations
import math
from typing import Callable, Optional

import torch
import torch.nn.functional as F

Predict = Callable[[torch.Tensor], torch.Tensor]


def stochastic_predictor(model, n_paths: int = 8) -> Predict:
    return lambda x: model.forward_mean(x, n_paths=n_paths)


def fixed_predictor(model, n_paths: int = 8) -> Predict:
    seeds = list(range(n_paths))
    return lambda x: model.forward_mean(x, seeds=seeds)


def cw_margin(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """z_y − max_{k≠y} z_k per example (negative ⇒ misclassified)."""
    true = logits.gather(1, y[:, None]).squeeze(1)
    other = logits.scatter(1, y[:, None], float("-inf")).max(dim=1).values
    return true - other


def dlr_loss(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Difference-of-logits-ratio loss (Croce & Hein, 2020); larger is worse for the model."""
    srt = logits.sort(dim=1, descending=True).values
    return -cw_margin(logits, y) / (srt[:, 0] - srt[:, 2] + 1e-12)


def _project(delta: torch.Tensor, eps: float, norm: str) -> torch.Tensor:
    if norm == "linf":
        return delta.clamp(-eps, eps)
    n = delta.flatten(1).norm(dim=1).clamp_min(1e-12).view(-1, *[1] * (delta.ndim - 1))
    return delta * torch.clamp(eps / n, max=1.0)


def _step(grad: torch.Tensor, norm: str) -> torch.Tensor:
    if norm == "linf":
        return grad.sign()
    n = grad.flatten(1).norm(dim=1).clamp_min(1e-12).view(-1, *[1] * (grad.ndim - 1))
    return grad / n


def _clip_box(x: torch.Tensor, box: Optional[tuple[torch.Tensor, torch.Tensor]]) -> torch.Tensor:
    return x if box is None else torch.max(torch.min(x, box[1]), box[0])


def _loss_and_grad(predict: Predict, x: torch.Tensor, y: torch.Tensor,
                   loss_fn, eot: int) -> tuple[torch.Tensor, torch.Tensor]:
    grad = torch.zeros_like(x)
    loss_sum = torch.zeros(x.shape[0], device=x.device)
    for _ in range(eot):
        xr = x.detach().requires_grad_(True)
        loss = loss_fn(predict(xr), y)
        grad += torch.autograd.grad(loss.sum(), xr)[0]
        loss_sum += loss.detach()
    return loss_sum / eot, grad / eot


def _random_start(x: torch.Tensor, eps: float, norm: str) -> torch.Tensor:
    if norm == "linf":
        return torch.empty_like(x).uniform_(-eps, eps)
    d = torch.randn_like(x)
    return _project(d * eps, eps, norm)


def eot_pgd(predict: Predict, x: torch.Tensor, y: torch.Tensor, *, eps: float,
            steps: int = 100, eot: int = 64, alpha: Optional[float] = None,
            norm: str = "linf", box=None) -> torch.Tensor:
    alpha = alpha if alpha is not None else 2.5 * eps / steps
    ce = lambda z, t: F.cross_entropy(z, t, reduction="none")
    delta = _random_start(x, eps, norm)
    for _ in range(steps):
        _, g = _loss_and_grad(predict, x + delta, y, ce, eot)
        delta = _project(delta + alpha * _step(g, norm), eps, norm)
        delta = _clip_box(x + delta, box) - x
    return (x + delta).detach()


def expected_margin_pgd(predict_mean: Predict, x: torch.Tensor, y: torch.Tensor, *,
                        eps: float, steps: int = 100, alpha: Optional[float] = None,
                        norm: str = "linf", box=None) -> torch.Tensor:
    """Minimise the CW margin of the mean predictor; ``predict_mean`` should
    average many fresh paths (e.g. ``stochastic_predictor(model, 256)``)."""
    alpha = alpha if alpha is not None else 2.5 * eps / steps
    neg_margin = lambda z, t: -cw_margin(z, t)
    delta = _random_start(x, eps, norm)
    best = x.clone()
    best_m = torch.full((x.shape[0],), float("inf"), device=x.device)
    for _ in range(steps):
        loss, g = _loss_and_grad(predict_mean, x + delta, y, neg_margin, 1)
        better = -loss < best_m
        best[better] = (x + delta)[better].detach()
        best_m = torch.where(better, -loss, best_m)
        delta = _project(delta + alpha * _step(g, norm), eps, norm)
        delta = _clip_box(x + delta, box) - x
    return best.detach()


def _apgd_checkpoints(n_iter: int) -> list[int]:
    p = [0.0, 0.22]
    while p[-1] < 1.0:
        p.append(p[-1] + max(p[-1] - p[-2] - 0.03, 0.06))
    return sorted({int(math.ceil(q * n_iter)) for q in p if q <= 1.0})


def apgd(predict: Predict, x: torch.Tensor, y: torch.Tensor, *, eps: float,
         n_iter: int = 100, loss: str = "ce", eot: int = 16, norm: str = "linf",
         rho: float = 0.75, box=None) -> torch.Tensor:
    """Auto-PGD with step-size halving and momentum 0.75; returns the best point."""
    if loss == "ce":
        loss_fn = lambda z, t: F.cross_entropy(z, t, reduction="none")
    elif loss == "dlr":
        loss_fn = dlr_loss
    else:
        raise ValueError("loss must be 'ce' or 'dlr'")
    bs = x.shape[0]
    view = (-1, *[1] * (x.ndim - 1))
    eta = torch.full((bs,), 2.0 * eps, device=x.device)
    checkpoints = _apgd_checkpoints(n_iter)

    x_k = x + _random_start(x, eps, norm)
    l_k, g = _loss_and_grad(predict, x_k, y, loss_fn, eot)
    x_best, l_best, g_best = x_k.clone(), l_k.clone(), g.clone()
    x_prev = x_k.clone()
    successes = torch.zeros(bs, device=x.device)
    l_best_at_ckpt = l_best.clone()
    eta_reduced_at_ckpt = torch.ones(bs, dtype=torch.bool, device=x.device)
    last_ckpt = 0

    for k in range(n_iter):
        z = x + _project(x_k + eta.view(view) * _step(g, norm) - x, eps, norm)
        a = 0.75 if k > 0 else 1.0
        x_new = x_k + a * (z - x_k) + (1 - a) * (x_k - x_prev)
        x_new = _clip_box(x + _project(x_new - x, eps, norm), box)
        x_prev = x_k
        x_k = x_new
        l_k, g = _loss_and_grad(predict, x_k, y, loss_fn, eot)
        improved = l_k > l_best
        successes += improved.float()
        x_best[improved] = x_k[improved]
        g_best[improved] = g[improved]
        l_best = torch.where(improved, l_k, l_best)

        if k + 1 in checkpoints and k + 1 > last_ckpt:
            span = k + 1 - last_ckpt
            cond1 = successes < rho * span
            cond2 = (~eta_reduced_at_ckpt) & (l_best_at_ckpt >= l_best)
            reduce = cond1 | cond2
            eta = torch.where(reduce, eta / 2.0, eta)
            x_k[reduce] = x_best[reduce]
            g[reduce] = g_best[reduce]
            eta_reduced_at_ckpt = reduce
            l_best_at_ckpt = l_best.clone()
            successes.zero_()
            last_ckpt = k + 1
    return x_best.detach()


@torch.no_grad()
def square_attack(predict: Predict, x: torch.Tensor, y: torch.Tensor, *, eps: float,
                  n_queries: int = 5000, p_init: float = 0.3, box=None) -> torch.Tensor:
    """l∞ Square attack on feature vectors: resample ±eps on a random coordinate
    subset whose size follows the original schedule; keep a proposal only if it
    lowers the CW margin."""
    d = x[0].numel()
    flat = lambda t: t.view(t.shape[0], -1)
    delta = eps * torch.sign(torch.randn_like(x))
    x_adv = _clip_box(x + delta, box)
    margin = cw_margin(predict(x_adv), y)
    for i in range(n_queries):
        active = margin > 0
        if not active.any():
            break
        frac = p_init * 0.5 ** sum(i > t * n_queries / 10000 for t in (10, 50, 200, 500, 1000, 2000, 4000, 6000, 8000))
        s = max(1, int(round(frac * d)))
        idx = torch.stack([torch.randperm(d, device=x.device)[:s] for _ in range(int(active.sum()))])
        cand = flat(x_adv[active].clone())
        base = flat(x[active])
        signs = eps * torch.sign(torch.randn(idx.shape, device=x.device))
        cand.scatter_(1, idx, base.gather(1, idx) + signs)
        cand = _clip_box(cand.view_as(x_adv[active]), None if box is None else (box[0], box[1]))
        new_margin = cw_margin(predict(cand), y[active])
        accept = new_margin < margin[active]
        sub = x_adv[active]
        sub[accept] = cand[accept]
        x_adv[active] = sub
        m = margin[active]
        m[accept] = new_margin[accept]
        margin[active] = m
    return x_adv


def bel_pgd(model, x: torch.Tensor, y: torch.Tensor, *, eps: float, steps: int = 20,
            n_paths: int = 256, alpha: Optional[float] = None) -> torch.Tensor:
    """l∞ PGD that ascends the path-error probability P[ψ_y(X_T) ≤ max_{k≠y} ψ_k(X_T)],
    a discontinuous functional, using the regularised BEL input gradient."""
    from ..theory.bel import bel_input_gradient
    alpha = alpha if alpha is not None else 2.5 * eps / steps
    head = model.head
    delta = _random_start(x, eps, "linf")
    for _ in range(steps):
        xr = x + delta
        rows = []
        for i in range(xr.shape[0]):
            label = y[i]
            phi = lambda z, label=label: (cw_margin(head(z), label.expand(z.shape[0])) <= 0).to(z.dtype)
            rows.append(bel_input_gradient(model, xr[i:i + 1], phi, n_paths=n_paths))
        g = torch.cat(rows)
        delta = _project(delta + alpha * g.sign(), eps, "linf")
    return (x + delta).detach()
