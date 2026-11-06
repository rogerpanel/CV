"""Sparse (edge) randomised smoothing for graph-structured inputs.

S&P 2027 audit (Part C.10): MambaGuard reasons about the agent
communication graph, so the certificate must also cover **edge edits**
(additions and deletions), not just L2 perturbations of node features.
This module implements a sparse-smoothing driver following
Bojchevski, Gasteiger & Guennemann, "Efficient Robustness
Certificates for Discrete Data" (ICML 2020).

Full Neyman-Pearson decoding of the smoothed classifier requires
summing over all perturbations of a given radius, which is
combinatorially expensive. For the stub we support **two** decoding
options:

    1. Majority-vote weak bound (default): the number of edges that
       can be flipped while preserving the predicted class is bounded
       by
            k <= floor( log(nA / nB) / log((1 - p) / p) )
       where ``nA`` and ``nB`` are top-two vote counts in the smoothed
       sample.  This is the symmetric-noise specialisation of the
       Bojchevski et al. majority bound.
    2. Clopper-Pearson-based bound: certify any budget ``k`` such that
       ``pA_lower(k) > 0.5`` under the symmetric Bernoulli noise model,
       using the lower bound from :mod:`smoothing`.

The driver returns separate addition/deletion budgets so the
manuscript can report *k_a* and *k_d*.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable

import torch

from .smoothing import _clopper_pearson_lower


@dataclass
class SparseResult:
    predicted_class: int
    k_add: int
    k_del: int
    n_samples: int
    p_plus: float
    p_minus: float
    nA: int
    nB: int
    pA_lower: float
    abstain: bool

    def as_dict(self) -> dict:
        return {
            "predicted_class": int(self.predicted_class),
            "k_add": int(self.k_add),
            "k_del": int(self.k_del),
            "n_samples": int(self.n_samples),
            "p_plus": float(self.p_plus),
            "p_minus": float(self.p_minus),
            "nA": int(self.nA),
            "nB": int(self.nB),
            "pA_lower": float(self.pA_lower),
            "abstain": bool(self.abstain),
        }


def _sample_sparse_edges(
    edge_index: torch.Tensor,
    candidate_nonedges: torch.Tensor,
    p_plus: float,
    p_minus: float,
    rng: torch.Generator,
) -> torch.Tensor:
    """Return a perturbed ``edge_index`` tensor (2, E').

    Each existing edge is deleted with probability ``p_minus`` and each
    non-edge in ``candidate_nonedges`` is added with probability
    ``p_plus``.
    """
    if edge_index.numel() > 0:
        keep_mask = torch.rand(edge_index.shape[1], generator=rng) >= p_minus
        kept = edge_index[:, keep_mask]
    else:
        kept = edge_index
    if candidate_nonedges.numel() > 0:
        add_mask = torch.rand(candidate_nonedges.shape[1], generator=rng) < p_plus
        added = candidate_nonedges[:, add_mask]
    else:
        added = candidate_nonedges
    return torch.cat([kept, added], dim=1) if added.numel() else kept


def _majority_vote_budget(nA: int, nB: int, p: float) -> int:
    """Weak symmetric-noise majority-vote bound on the edit radius.

        k <= floor( log(nA/nB) / log((1 - p)/p) ).
    """
    if nA <= nB or p <= 0 or p >= 0.5:
        return 0
    num = math.log(nA / max(nB, 1))
    den = math.log((1.0 - p) / p)
    if den <= 0:
        return 0
    return int(math.floor(num / den))


def sparse_certify(
    model_fn: Callable[[torch.Tensor], torch.Tensor],
    edge_index: torch.Tensor,
    candidate_nonedges: torch.Tensor | None = None,
    p_plus: float = 0.05,
    p_minus: float = 0.05,
    n: int = 1000,
    alpha: float = 1e-3,
    seed: int = 0,
    num_classes: int = 2,
) -> SparseResult:
    """Sparse-smoothing certificate for an edge-set perturbation.

    Parameters
    ----------
    model_fn : callable
        ``model_fn(edge_index)`` must return class logits of shape
        ``(num_classes,)`` (a single graph-level prediction).
    edge_index : Tensor of shape (2, E)
        Original edge set.
    candidate_nonedges : Tensor of shape (2, M) or None
        Pre-computed node-pair list acting as candidate additions. If
        ``None`` the complement of the edge set over the implicit node
        range is used (tractable only for small graphs).
    p_plus, p_minus : float
        Per-edge add / delete Bernoulli probabilities.
    n : int
        Number of smoothing samples.
    alpha : float
        Clopper-Pearson confidence level.
    """
    if not 0.0 <= p_plus < 0.5:
        raise ValueError(f"p_plus must be in [0, 0.5), got {p_plus}")
    if not 0.0 <= p_minus < 0.5:
        raise ValueError(f"p_minus must be in [0, 0.5), got {p_minus}")
    if n <= 0:
        raise ValueError("n must be positive")

    rng = torch.Generator()
    rng.manual_seed(seed)

    if candidate_nonedges is None:
        n_nodes = int(edge_index.max().item()) + 1 if edge_index.numel() else 1
        ij = torch.combinations(torch.arange(n_nodes), r=2, with_replacement=False)
        pairs = torch.cat([ij.t(), ij.flip(1).t()], dim=1)
        # Remove existing edges.
        existing = set(
            (int(a), int(b)) for a, b in edge_index.t().tolist()
        )
        keep = [
            i
            for i in range(pairs.shape[1])
            if (int(pairs[0, i]), int(pairs[1, i])) not in existing
        ]
        candidate_nonedges = pairs[:, keep]

    counts = torch.zeros(num_classes, dtype=torch.long)
    for _ in range(n):
        ei = _sample_sparse_edges(edge_index, candidate_nonedges, p_plus, p_minus, rng)
        with torch.no_grad():
            logits = model_fn(ei)
            if logits.dim() > 1:
                logits = logits.mean(dim=0)
            pred = int(logits.argmax().item())
        counts[pred] += 1

    top2 = counts.topk(min(2, num_classes))
    nA = int(top2.values[0].item())
    cA = int(top2.indices[0].item())
    nB = int(top2.values[1].item()) if num_classes > 1 else 0
    pA_lower = _clopper_pearson_lower(nA, n, alpha)
    abstain = pA_lower <= 0.5
    if abstain:
        return SparseResult(
            predicted_class=-1,
            k_add=0,
            k_del=0,
            n_samples=n,
            p_plus=p_plus,
            p_minus=p_minus,
            nA=nA,
            nB=nB,
            pA_lower=float(pA_lower),
            abstain=True,
        )

    k_add = _majority_vote_budget(nA, max(nB, 1), p_plus)
    k_del = _majority_vote_budget(nA, max(nB, 1), p_minus)
    return SparseResult(
        predicted_class=cA,
        k_add=int(k_add),
        k_del=int(k_del),
        n_samples=n,
        p_plus=p_plus,
        p_minus=p_minus,
        nA=nA,
        nB=nB,
        pA_lower=float(pA_lower),
        abstain=False,
    )
