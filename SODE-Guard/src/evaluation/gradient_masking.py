"""Gradient-masking diagnostics (Carlini et al., 2019, "On evaluating
adversarial robustness"). A defence whose robustness comes from masked
gradients typically fails at least one of these checks:

    (i)   unbounded ε: accuracy must reach chance as ε grows;
    (ii)  transfer: examples crafted on an undefended surrogate should not be
          markedly stronger than white-box ones;
    (iii) black-box: a score-based attack (Square) should not beat white-box PGD;
    (iv)  iterations: more PGD steps must never increase accuracy.
"""
from __future__ import annotations
import numpy as np
import torch

from ..attacks.eot_autoattack import eot_pgd, square_attack
from ..utils.metrics import macro_f1


def _f1(predict, x, y) -> float:
    with torch.no_grad():
        pred = predict(x).argmax(-1)
    return macro_f1(y.cpu().numpy(), pred.cpu().numpy())


def epsilon_sweep(predict, x, y, *, epsilons=(0.03, 0.1, 0.3, 1.0, 3.0), steps=100, eot=8) -> dict:
    return {float(e): _f1(predict, eot_pgd(predict, x, y, eps=e, steps=steps, eot=eot), y)
            for e in epsilons}


def transfer_attack(surrogate, target_predict, x, y, *, eps: float, steps: int = 100) -> float:
    x_adv = eot_pgd(surrogate, x, y, eps=eps, steps=steps, eot=1)
    return _f1(target_predict, x_adv, y)


def black_box(predict, x, y, *, eps: float, n_queries: int = 5000) -> float:
    return _f1(predict, square_attack(predict, x, y, eps=eps, n_queries=n_queries), y)


def step_monotonicity(predict, x, y, *, eps: float, steps=(10, 20, 40, 100, 200), eot=8) -> dict:
    out = {int(s): _f1(predict, eot_pgd(predict, x, y, eps=eps, steps=s, eot=eot), y) for s in steps}
    vals = np.array(list(out.values()))
    out["monotone_non_increasing"] = bool(np.all(np.diff(vals) <= 0.01))
    return out
