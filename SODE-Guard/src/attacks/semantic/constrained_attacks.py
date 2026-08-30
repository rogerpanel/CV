"""PGD-40 and Carlini–Wagner adversaries restricted to feasible feature space.

Both attacks call the ``FeasibilityProjector`` after every gradient step
so the reported robust accuracy uses only perturbations that correspond
to *realisable* packet flows (Reviewer 2's problem-space concern).
"""
from __future__ import annotations
import torch
import torch.nn.functional as F

from ..pgd import PGD
from ..cw import CarliniWagnerL2
from .feasibility import FeasibilityProjector


class ConstrainedPGD(PGD):
    def __init__(self, *args, projector: FeasibilityProjector,
                 clip_min: float = -8.0, clip_max: float = 8.0, **kw):
        super().__init__(*args, clip_min=clip_min, clip_max=clip_max, **kw)
        self.projector = projector

    def __call__(self, x, y):
        x_adv = x.clone().detach()
        if self.random_start:
            x_adv = x_adv + torch.empty_like(x_adv).uniform_(-self.eps, self.eps)
            x_adv = self.projector(x_adv).clamp(*self.clip).detach()

        for _ in range(self.steps):
            x_adv.requires_grad_(True)
            logits = self.model(x_adv)
            loss = F.cross_entropy(logits, y)
            grad = torch.autograd.grad(loss, x_adv, retain_graph=False)[0]
            x_adv = x_adv.detach() + self.alpha * grad.sign()
            delta = (x_adv - x).clamp(-self.eps, self.eps)
            x_adv = self.projector((x + delta).clamp(*self.clip)).detach()
        return x_adv


class ConstrainedCW(CarliniWagnerL2):
    def __init__(self, *args, projector: FeasibilityProjector, **kw):
        super().__init__(*args, **kw)
        self.projector = projector

    def __call__(self, x, y):
        x_adv = super().__call__(x, y)
        return self.projector(x_adv)
