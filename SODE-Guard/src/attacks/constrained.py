"""Feature-constrained attacks on flow statistics (future-work item: realisable perturbations).

The unconstrained threat model lets the attacker move every z-scored statistic
by eps. A traffic-level attacker who controls only its own packets is weaker.
``FlowConstraints`` encodes three restrictions in the raw (unscaled) feature
space and projects every attack iterate onto them:

* mutable features only: statistics of the server-to-client direction
  (``Bwd``/``Backward``), ports, protocol, TCP flag counts and initial window
  sizes are fixed, because the attacker's own packets do not set them;
* domain: a feature never leaves the range seen in training at its lower end
  (byte and packet counts, durations and rates stay non-negative);
* integrality: features whose training values are all integers (packet counts,
  header lengths, ...) change by whole units only, rounded towards the clean
  value so the eps budget still holds.

The constrained attacks are PGD (fixed paths), EOT-PGD (fresh paths), a masked
Square random search, and transfer of constrained adversarial examples.
"""
from __future__ import annotations

from typing import Callable, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from .eot_autoattack import cw_margin

IMMUTABLE_PATTERNS = ("port", "protocol", "bwd", "backward", "flag", "init_win", "init win", "win bytes")
Predict = Callable[[torch.Tensor], torch.Tensor]


class FlowConstraints:
    def __init__(self, names: Sequence[str], mu: np.ndarray, sd: np.ndarray, raw_train: np.ndarray,
                 eps: float, device: str = "cpu"):
        lower = np.char.lower(np.asarray(names, dtype=str))
        immutable = np.array([any(p in n for p in IMMUTABLE_PATTERNS) for n in lower])
        integer = np.all(np.isclose(raw_train, np.round(raw_train)), axis=0)
        t = lambda v: torch.tensor(np.asarray(v, dtype=np.float32), device=device)
        self.names = list(names)
        self.eps = eps
        self.mu, self.sd = t(mu), t(sd)
        self.lo = t(raw_train.min(0))
        self.mutable = t(~immutable)
        self.integer = torch.tensor(integer, device=device)
        self.summary = {"n_features": len(names), "n_mutable": int((~immutable).sum()),
                        "n_integer": int(integer.sum()), "immutable": [n for n, i in zip(names, immutable) if i]}

    def project(self, xa: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        d = (xa - x).clamp(-self.eps, self.eps) * self.mutable
        raw = (x + d) * self.sd + self.mu
        raw = torch.maximum(raw, self.lo)
        out = (raw - self.mu) / self.sd
        return torch.where(self.mutable > 0, out, x)       # immutable features exactly unchanged

    def finalize(self, xa: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Integer features move by whole units, truncated towards the clean value."""
        xa = self.project(xa, x)
        raw_a, raw_c = xa * self.sd + self.mu, x * self.sd + self.mu
        step = torch.trunc(raw_a - raw_c)
        raw_a = torch.where(self.integer, raw_c + step, raw_a)
        return self.project((raw_a - self.mu) / self.sd, x).detach()


def constrained_pgd(f: Predict, x: torch.Tensor, y: torch.Tensor, C: FlowConstraints, *,
                    steps: int = 100, eot: int = 1, restarts: int = 1) -> torch.Tensor:
    alpha = 2.5 * C.eps / steps
    best, fooled = x.clone(), torch.zeros(len(x), dtype=torch.bool, device=x.device)
    for _ in range(restarts):
        xa = C.project(x + (torch.rand_like(x) * 2 - 1) * C.eps, x)
        for _ in range(steps):
            g = torch.zeros_like(x)
            for _ in range(eot):
                xr = xa.detach().requires_grad_(True)
                g += torch.autograd.grad(F.cross_entropy(f(xr), y), xr)[0]
            xa = C.project(xa.detach() + alpha * g.sign(), x)
        xa = C.finalize(xa, x)
        with torch.no_grad():
            wrong = f(xa).argmax(1) != y
        new = wrong & ~fooled
        best[new] = xa[new]
        fooled |= wrong
        best[~fooled] = xa[~fooled]
    return best.detach()


@torch.no_grad()
def constrained_square(f: Predict, x: torch.Tensor, y: torch.Tensor, C: FlowConstraints, *,
                       n_queries: int = 1000, p_init: float = 0.3) -> torch.Tensor:
    """Random search over the mutable coordinates (masked Square attack)."""
    idx_mut = torch.nonzero(C.mutable > 0).squeeze(1)
    dm = len(idx_mut)
    if dm == 0:
        return x.clone()
    xa = x.clone()
    xa[:, idx_mut] += C.eps * torch.sign(torch.randn(len(x), dm, device=x.device))
    xa = C.finalize(xa, x)
    margin = cw_margin(f(xa), y)
    for i in range(n_queries):
        active = margin > 0
        if not active.any():
            break
        frac = p_init * 0.5 ** sum(i > t * n_queries / 10000 for t in (10, 50, 200, 500, 1000, 2000, 4000, 6000, 8000))
        s = max(1, int(round(frac * dm)))
        cand = xa[active].clone()
        pick = idx_mut[torch.rand(int(active.sum()), dm, device=x.device).argsort(1)[:, :s]]
        signs = C.eps * torch.sign(torch.randn(pick.shape, device=x.device))
        cand.scatter_(1, pick, x[active].gather(1, pick) + signs)
        cand = C.finalize(cand, x[active])
        new_margin = cw_margin(f(cand), y[active])
        accept = new_margin < margin[active]
        sub = xa[active]
        sub[accept] = cand[accept]
        xa[active] = sub
        m = margin[active]
        m[accept] = new_margin[accept]
        margin[active] = m
    return xa
