"""DeepFool attack (Moosavi-Dezfooli et al., CVPR 2016) — multi-class variant."""
from __future__ import annotations
import torch
import torch.nn as nn


class DeepFool:
    def __init__(self, model: nn.Module, num_classes: int = 34,
                 iterations: int = 50, overshoot: float = 0.02):
        self.model = model
        self.num_classes = num_classes
        self.iterations = int(iterations)
        self.overshoot = float(overshoot)

    def __call__(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x_adv = x.clone().detach()
        for b in range(x.shape[0]):
            xb = x_adv[b:b + 1].clone().detach().requires_grad_(True)
            for _ in range(self.iterations):
                logits = self.model(xb).squeeze(0)
                pred = logits.argmax().item()
                if pred != y[b].item():
                    break
                grads = []
                for k in range(self.num_classes):
                    g = torch.autograd.grad(logits[k], xb, retain_graph=True)[0]
                    grads.append(g.detach())
                grads = torch.cat(grads, dim=0)
                f_k = logits.detach() - logits[y[b]].detach()
                w_k = grads - grads[y[b]]
                w_norm = w_k.flatten(1).norm(dim=-1) + 1e-12
                ratios = f_k.abs() / w_norm
                ratios[y[b]] = float("inf")
                l_hat = ratios.argmin().item()
                r = (f_k[l_hat].abs() / (w_norm[l_hat] ** 2 + 1e-12)) * w_k[l_hat]
                xb = (xb + (1 + self.overshoot) * r).detach().requires_grad_(True)
            x_adv[b] = xb.detach()
        return x_adv.clamp(0.0, 1.0)
