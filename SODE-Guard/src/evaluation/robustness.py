"""Adversarial evaluation harness.

Produces the Table 3 numbers of the manuscript: macro-F1 under PGD-40 over the
ε-grid {0.005, 0.01, 0.02, 0.03, 0.05, 0.10}, plus FGSM, C&W, DeepFool and
Gaussian sanity baselines.
"""
from __future__ import annotations
import numpy as np
import torch

from ..attacks import PGD, FGSM, CarliniWagnerL2, DeepFool, GaussianNoise
from ..utils.metrics import aggregate_metrics


def _run(model, x_adv, y) -> tuple[np.ndarray, np.ndarray]:
    with torch.no_grad():
        probs = model.forward_mc(x_adv)
    return probs.argmax(-1).cpu().numpy(), y.cpu().numpy()


def evaluate_attacks(model, loader, *, device, attack_cfg) -> dict[str, dict]:
    model.eval()
    results: dict[str, dict] = {}
    for atk_name, params in attack_cfg.items():
        atk_type = params.get("type", atk_name)
        if atk_type in {"pgd", "pgd40"}:
            for eps in params["epsilons"]:
                atk = PGD(model, eps=eps, steps=params.get("steps", 40),
                          alpha=params.get("step_size", 2.5 * eps / params.get("steps", 40)))
                results[f"{atk_name}@{eps}"] = _sweep(model, loader, atk, device)
        elif atk_type == "fgsm":
            for eps in params["epsilons"]:
                results[f"{atk_name}@{eps}"] = _sweep(model, loader, FGSM(model, eps), device)
        elif atk_type == "cw_l2":
            atk = CarliniWagnerL2(model, c=params.get("c", 1.0),
                                  iterations=params.get("iterations", 100))
            results[atk_name] = _sweep(model, loader, atk, device)
        elif atk_type == "deepfool":
            atk = DeepFool(model, num_classes=model.cfg.num_classes,
                           iterations=params.get("iterations", 50),
                           overshoot=params.get("overshoot", 0.02))
            results[atk_name] = _sweep(model, loader, atk, device)
        elif atk_type == "gaussian_noise":
            for s in params["sigmas"]:
                results[f"{atk_name}@{s}"] = _sweep(model, loader, GaussianNoise(s), device)
    return results


def _sweep(model, loader, attack, device) -> dict:
    yt, yp = [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        x_adv = attack(x, y)
        p, t = _run(model, x_adv, y)
        yp.append(p); yt.append(t)
    yt = np.concatenate(yt); yp = np.concatenate(yp)
    return aggregate_metrics(yt, yp)
