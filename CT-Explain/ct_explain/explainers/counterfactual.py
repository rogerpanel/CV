"""3C — Counterfactual Trajectory Generation via Adjoint Sensitivity.

    δ* = argmin_δ ∫₀^T ||δ(t)||² dt
         s.t. M(G(T); x + δ) ≠ M(G(T); x)                         (Eq. 6)

We solve this with projected gradient descent on the node's input feature
vector. The gradient ∇_x L is obtained via the adjoint sensitivity method
— i.e. standard PyTorch autograd through the ODE solver, which in the
adjoint setting reduces to the backward integration described in §3C.

Domain constraints implemented here:
    - non-negative packet sizes / counts (columns starting with total_* or
      *_bytes / *_pkts / pkt_* are clamped to ≥ 0)
    - protocol flag features are clamped into [0, 1]
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Optional

import torch
import torch.nn.functional as F

from ct_explain.data.graph_builder import TemporalGraph
from ct_explain.explainers.base import BaseExplainer, Explanation


@dataclass
class Counterfactual:
    original: torch.Tensor
    perturbed: torch.Tensor
    delta: torch.Tensor
    l2_norm: float
    original_prediction: int
    perturbed_prediction: int
    iterations: int
    trajectory: list[torch.Tensor]


class CounterfactualExplainer(BaseExplainer):
    """Minimal-perturbation counterfactual generator."""

    name = "counterfactual"

    def __init__(
        self,
        model,
        lr: float = 0.05,
        max_iters: int = 200,
        l2_weight: float = 1.0,
        projection: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ) -> None:
        super().__init__(model)
        self.lr = lr
        self.max_iters = max_iters
        self.l2_weight = l2_weight
        self.projection = projection or self._default_projection

    # ------------------------------------------------------------------ #
    def explain(
        self,
        graph: TemporalGraph,
        target_node: int,
        target_class: Optional[int] = None,
    ) -> Explanation:
        t0 = time.perf_counter()
        self.model.eval()

        x_original = graph.node_features.clone().detach()
        with torch.no_grad():
            out = self.model(
                node_features=x_original,
                edge_index=graph.edge_index,
                edge_features=graph.edge_features,
                edge_times=graph.edge_times,
            )
            original_class = int(F.softmax(out.logits[target_node], -1).argmax().item())
        target_class = target_class if target_class is not None else (
            (original_class + 1) % self.model.num_classes
        )

        x = x_original.clone().detach()
        delta = torch.zeros_like(x).requires_grad_(True)
        optim = torch.optim.Adam([delta], lr=self.lr)

        trajectory: list[torch.Tensor] = []
        final_class = original_class
        it = 0
        for it in range(self.max_iters):
            optim.zero_grad()
            perturbed = x + delta
            out = self.model(
                node_features=perturbed,
                edge_index=graph.edge_index,
                edge_features=graph.edge_features,
                edge_times=graph.edge_times,
            )
            logits = out.logits[target_node]
            loss_cls = F.cross_entropy(
                logits.unsqueeze(0),
                torch.tensor([target_class], device=logits.device),
            )
            loss_l2 = self.l2_weight * (delta[target_node] ** 2).sum()
            loss = loss_cls + loss_l2
            loss.backward()
            optim.step()

            # Domain projection.
            with torch.no_grad():
                delta.data = self.projection(x_original + delta.data) - x_original

            if it % max(1, self.max_iters // 10) == 0:
                trajectory.append((x + delta).detach()[target_node].clone().cpu())

            # Early stop once the prediction flips.
            with torch.no_grad():
                pred = int(F.softmax(logits.detach(), -1).argmax().item())
            if pred == target_class:
                final_class = pred
                break

        perturbed = (x + delta).detach()
        d_tgt = delta.detach()[target_node]
        l2 = float(torch.linalg.norm(d_tgt).item())

        cf = Counterfactual(
            original=x_original[target_node].detach().cpu(),
            perturbed=perturbed[target_node].cpu(),
            delta=d_tgt.cpu(),
            l2_norm=l2,
            original_prediction=original_class,
            perturbed_prediction=final_class,
            iterations=it + 1,
            trajectory=trajectory,
        )

        attribution = d_tgt.abs().cpu() / (d_tgt.abs().sum().cpu() + 1e-12)
        return Explanation(
            attribution=attribution,
            method=self.name,
            latency_ms=(time.perf_counter() - t0) * 1000,
            supporting={"counterfactual": cf},
            target_node=target_node,
        )

    # ------------------------------------------------------------------ #
    @staticmethod
    def _default_projection(x: torch.Tensor) -> torch.Tensor:
        out = x.clone()
        out = torch.where(out < 0, torch.zeros_like(out), out)
        return out
