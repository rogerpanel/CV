"""Adversarial evaluation harness (PGD, FGSM, C&W, DeepFool, Gaussian noise).

For SODE-Guard the attack differentiates through exactly the predictor that is
scored: the deployed mean over ``mc_paths_eval`` fixed-seed paths. Adaptive
attacks against the randomised predictor live in ``attacks/eot_autoattack.py``
and ``scripts/run_revision_experiments.py``.
"""
from __future__ import annotations
import numpy as np
import torch

from ..attacks import PGD, FGSM, CarliniWagnerL2, DeepFool, GaussianNoise
from ..attacks.eot_autoattack import fixed_predictor
from ..utils.metrics import aggregate_metrics


def scored_predictor(model):
    if hasattr(model, "forward_mean"):
        return fixed_predictor(model, model.cfg.mc_paths_eval)
    return model


def evaluate_attacks(model, loader, *, device, attack_cfg) -> dict[str, dict]:
    model.eval()
    predict = scored_predictor(model)
    results: dict[str, dict] = {}
    for atk_name, params in attack_cfg.items():
        atk_type = params.get("type", atk_name)
        if atk_type in {"pgd", "pgd40"}:
            steps = params.get("steps", 40)
            for eps in params["epsilons"]:
                atk = PGD(predict, eps=eps, steps=steps, norm=params.get("norm", "linf"),
                          alpha=params.get("step_size", 2.5 * eps / steps))
                results[f"{atk_name}@{eps}"] = _sweep(predict, loader, atk, device)
        elif atk_type == "fgsm":
            for eps in params["epsilons"]:
                results[f"{atk_name}@{eps}"] = _sweep(predict, loader, FGSM(predict, eps), device)
        elif atk_type == "cw_l2":
            atk = CarliniWagnerL2(predict, c=params.get("c", 1.0),
                                  iterations=params.get("iterations", 100))
            results[atk_name] = _sweep(predict, loader, atk, device)
        elif atk_type == "deepfool":
            atk = DeepFool(predict, num_classes=model.cfg.num_classes,
                           iterations=params.get("iterations", 50),
                           overshoot=params.get("overshoot", 0.02))
            results[atk_name] = _sweep(predict, loader, atk, device)
        elif atk_type == "gaussian_noise":
            for s in params["sigmas"]:
                results[f"{atk_name}@{s}"] = _sweep(predict, loader, GaussianNoise(s), device)
    return results


def _sweep(predict, loader, attack, device) -> dict:
    yt, yp = [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        x_adv = attack(x, y)
        with torch.no_grad():
            yp.append(predict(x_adv).argmax(-1).cpu().numpy())
        yt.append(y.cpu().numpy())
    return aggregate_metrics(np.concatenate(yt), np.concatenate(yp))
