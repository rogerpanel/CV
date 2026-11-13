"""Robust-training modules for any predictor ``f: x -> logits`` (modules 2 and 3).

Module 2, EOT-matched adversarial training
    The adversary differentiates through exactly the predictor that is trained:
    for SODE-Guard, the M-path mean logits with fresh Brownian paths on every
    call, so each attack step sees new paths (expectation over transformation).
      nfgsm   N-FGSM (de Jorge et al., NeurIPS 2022): noise U[-2eps, 2eps],
              one signed-gradient step of size eps, no projection. About 2x
              the cost of a clean step and resistant to catastrophic overfitting.
      pgd     PGD-K with a random start and step 2.5 eps / K (Madry et al., 2018).
      trades  TRADES (Zhang et al., 2019): CE(f(x)) + beta KL(f(x) || f(x_adv)),
              x_adv from PGD-K on the KL term.
    ``eps_schedule`` ramps eps linearly over the first epochs.

Module 3, smoothing-ready training
    ``gaussian_consistency_loss``: classification loss on m Gaussian-noised
    copies x + sigma * eps_i plus the consistency regulariser of Jeong & Shin
    (NeurIPS 2020), lambda * mean_i KL(p_bar || p_i) + eta * H(p_bar). The noise is
    the input-space "diffusion prefix" of the SDE: integrating dX = sigma dW for
    unit time from X_0 = x gives exactly x + sigma * N(0, I).
"""
from __future__ import annotations

from typing import Callable

import torch
import torch.nn.functional as F

from ..attacks.eot_autoattack import _project, _random_start

Predict = Callable[[torch.Tensor], torch.Tensor]


def eps_schedule(epoch: int, eps: float, ramp_epochs: int) -> float:
    if ramp_epochs <= 0:
        return eps
    return eps * min(1.0, (epoch + 1) / ramp_epochs)


def _grad(f: Predict, x: torch.Tensor, y: torch.Tensor, loss_fn) -> torch.Tensor:
    x = x.detach().requires_grad_(True)
    loss = loss_fn(f(x), y)
    return torch.autograd.grad(loss, x)[0]


def make_adversarial(f: Predict, x: torch.Tensor, y: torch.Tensor, *, method: str,
                     eps: float, steps: int = 5) -> torch.Tensor:
    ce = lambda z, t: F.cross_entropy(z, t)
    if eps <= 0:
        return x.detach()
    if method == "nfgsm":
        eta = (torch.rand_like(x) * 2 - 1) * 2 * eps
        g = _grad(f, x + eta, y, ce)
        return (x + eta + eps * g.sign()).detach()
    if method == "pgd":
        alpha = 2.5 * eps / steps
        delta = _random_start(x, eps, "linf")
        for _ in range(steps):
            g = _grad(f, x + delta, y, ce)
            delta = _project(delta.detach() + alpha * g.sign(), eps, "linf")
        return (x + delta).detach()
    raise ValueError(method)


def trades_adversarial(f: Predict, x: torch.Tensor, *, eps: float, steps: int = 5) -> torch.Tensor:
    alpha = 2.5 * eps / steps
    with torch.no_grad():
        p_clean = F.softmax(f(x), dim=-1)
    delta = 0.001 * torch.randn_like(x)
    for _ in range(steps):
        delta = delta.detach().requires_grad_(True)
        loss = F.kl_div(F.log_softmax(f(x + delta), dim=-1), p_clean, reduction="sum")
        g = torch.autograd.grad(loss, delta)[0]
        delta = _project(delta.detach() + alpha * g.sign(), eps, "linf")
    return (x + delta).detach()


def adversarial_loss(model, f: Predict, x: torch.Tensor, y: torch.Tensor, loss_fn, *,
                     method: str, eps: float, steps: int = 5, trades_beta: float = 6.0) -> torch.Tensor:
    """One adversarial-training loss. The adversary runs in eval mode (no dropout);
    the loss is taken in train mode."""
    was = model.training
    model.eval()
    if method == "trades":
        x_adv = trades_adversarial(f, x, eps=eps, steps=steps)
        model.train(was)
        z = f(x)
        robust = F.kl_div(F.log_softmax(f(x_adv), dim=-1), F.softmax(z, dim=-1), reduction="batchmean")
        return loss_fn(z, y) + trades_beta * robust
    x_adv = make_adversarial(f, x, y, method=method, eps=eps, steps=steps)
    model.train(was)
    return loss_fn(f(x_adv), y)


def gaussian_consistency_loss(f: Predict, x: torch.Tensor, y: torch.Tensor, loss_fn, *,
                              sigma: float, m: int = 2, lam: float = 10.0, eta: float = 0.5) -> torch.Tensor:
    B = x.shape[0]
    xn = x.repeat(m, 1) + sigma * torch.randn(m * B, *x.shape[1:], device=x.device)
    z = f(xn)                                                      # (m*B, K)
    cls = loss_fn(z, y.repeat(m))
    if m < 2 or lam == 0:
        return cls
    logp = F.log_softmax(z, dim=-1).view(m, B, -1)
    p = logp.exp()
    p_bar = p.mean(0)
    log_p_bar = p_bar.clamp_min(1e-12).log()
    kl = (p_bar.unsqueeze(0) * (log_p_bar.unsqueeze(0) - logp)).sum(-1).mean()
    ent = -(p_bar * log_p_bar).sum(-1).mean()
    return cls + lam * kl + eta * ent
