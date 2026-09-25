"""Adversarially trained baselines: PGD-AT (Madry et al., 2018) and TRADES
(Zhang et al., 2019), for any model mapping features to logits."""
from __future__ import annotations

import torch
import torch.nn.functional as F

from ..attacks.eot_autoattack import _project, _random_start


def _pgd_ce(model, x, y, eps, steps, alpha):
    delta = _random_start(x, eps, "linf")
    for _ in range(steps):
        delta.requires_grad_(True)
        loss = F.cross_entropy(model(x + delta), y)
        g = torch.autograd.grad(loss, delta)[0]
        delta = _project(delta.detach() + alpha * g.sign(), eps, "linf")
    return (x + delta).detach()


def _pgd_kl(model, x, eps, steps, alpha):
    with torch.no_grad():
        p_clean = F.softmax(model(x), dim=-1)
    delta = 0.001 * torch.randn_like(x)
    for _ in range(steps):
        delta.requires_grad_(True)
        loss = F.kl_div(F.log_softmax(model(x + delta), dim=-1), p_clean, reduction="sum")
        g = torch.autograd.grad(loss, delta)[0]
        delta = _project(delta.detach() + alpha * g.sign(), eps, "linf")
    return (x + delta).detach()


def adversarial_step(model, x: torch.Tensor, y: torch.Tensor, *, method: str,
                     eps: float, steps: int = 10, alpha: float | None = None,
                     trades_beta: float = 6.0) -> torch.Tensor:
    """Return the training loss for one mini-batch."""
    alpha = alpha if alpha is not None else 2.5 * eps / steps
    was_training = model.training
    model.eval()
    if method == "pgd_at":
        x_adv = _pgd_ce(model, x, y, eps, steps, alpha)
        model.train(was_training)
        return F.cross_entropy(model(x_adv), y)
    if method == "trades":
        x_adv = _pgd_kl(model, x, eps, steps, alpha)
        model.train(was_training)
        logits = model(x)
        robust = F.kl_div(F.log_softmax(model(x_adv), dim=-1), F.softmax(logits, dim=-1),
                          reduction="batchmean")
        return F.cross_entropy(logits, y) + trades_beta * robust
    raise ValueError("method must be 'pgd_at' or 'trades'")


def train_adversarial(model, loader, *, method: str, eps: float, epochs: int = 40,
                      lr: float = 5e-4, weight_decay: float = 1e-5, steps: int = 10,
                      device: str = "cpu", log=print) -> None:
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs * len(loader))
    for epoch in range(epochs):
        model.train()
        total = 0.0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            loss = adversarial_step(model, x, y, method=method, eps=eps, steps=steps)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            total += float(loss)
        log(f"[{method}] epoch {epoch} loss {total / max(len(loader), 1):.4f}")
