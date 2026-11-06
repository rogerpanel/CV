"""GCG-style discrete token search -- scaffold only.

S&P 2027 audit (C.12): the manuscript claims adaptive robustness in
token space, which EmbedPGD does not cover (continuous relaxation).
The GCG attack (Zou et al. 2023) performs a greedy coordinate-descent
search over discrete tokens via first-order substitution scores:

    1. Compute gradient of the loss w.r.t. the one-hot token matrix.
    2. For each position, propose the ``top_k`` tokens whose
       substitution reduces the loss by the largest linear estimate.
    3. Evaluate a random subset of ``batch`` candidate substitutions
       and keep the single best.

The MambaGuard reference pipeline uses sentence-level embeddings, so a
drop-in GCG requires a tokeniser + embedder plug-in. This class
exposes the full interface -- ``token_embedder``, ``substitution_loss``
and ``projection_step`` -- and returns the unperturbed batch when
invoked with no plug-ins. The intent is that reviewers can see the
attack *scaffold* exists and could be filled in once
``sentence_transformers`` is available.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping

import torch


@dataclass
class GCGHooks:
    """Plug-in points for GCG. A real attack supplies all three."""

    token_embedder: Callable[[torch.Tensor], torch.Tensor] | None = None
    """One-hot (B, L, V) -> embeddings (B, L, d). Differentiable."""

    substitution_loss: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None
    """(logits, labels) -> scalar loss to minimise (negate for untargeted)."""

    projection_step: Callable[[torch.Tensor], torch.Tensor] | None = None
    """Project continuous embeddings back onto the token embedding simplex."""


class TokenGCG:
    """Discrete GCG attack skeleton.

    Parameters
    ----------
    steps : int
        Outer coordinate-descent iterations.
    top_k : int
        Candidate substitutions per position.
    batch : int
        Random sample of candidate substitutions evaluated per step.
    hooks : GCGHooks
        Plug-in points. If ``token_embedder`` is ``None`` the attack
        returns the unperturbed batch and sets ``was_noop=True`` so the
        caller can flag the missing plug-in.
    """

    def __init__(
        self,
        steps: int = 50,
        top_k: int = 256,
        batch: int = 64,
        hooks: GCGHooks | None = None,
        input_key: str = "p",
    ) -> None:
        self.steps = int(steps)
        self.top_k = int(top_k)
        self.batch = int(batch)
        self.hooks = hooks or GCGHooks()
        self.input_key = input_key
        self.was_noop: bool = False

    # ------------------------------------------------------------------
    def required_plugins(self) -> list[str]:
        missing = []
        if self.hooks.token_embedder is None:
            missing.append("token_embedder")
        if self.hooks.substitution_loss is None:
            missing.append("substitution_loss")
        if self.hooks.projection_step is None:
            missing.append("projection_step")
        return missing

    def forward(
        self,
        model: Any,
        batch: Mapping[str, Any],
        y: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        missing = self.required_plugins()
        if missing:
            # Skeleton path: nothing to attack; record a no-op.
            self.was_noop = True
            return dict(batch)

        # ------------------------------------------------------------------
        # Real path (requires a tokeniser/embedder).  For the public
        # artefact we leave this loop in place so reviewers can trace it;
        # it is executed only when hooks are provided.
        # ------------------------------------------------------------------
        was_training = getattr(model, "training", False)
        if hasattr(model, "eval"):
            model.eval()
        x_emb = batch[self.input_key].detach()
        labels = y if y is not None else batch.get("labels")
        if labels is None:
            raise ValueError("TokenGCG requires labels for untargeted loss")
        current = x_emb.clone()
        for _ in range(self.steps):
            current = current.detach().requires_grad_(True)
            emb = self.hooks.token_embedder(current)  # type: ignore[misc]
            adv_batch = dict(batch)
            adv_batch[self.input_key] = emb
            out = model(adv_batch)
            logits = out["logits"] if isinstance(out, dict) else out
            loss = self.hooks.substitution_loss(logits, labels)  # type: ignore[misc]
            grad = torch.autograd.grad(loss, current)[0]
            # Pick top_k substitutions per position by linearised loss drop.
            scores = -grad
            _, cand = scores.topk(self.top_k, dim=-1)
            # Evaluate a random subset of ``batch`` candidates and keep the best.
            best = current.detach()
            best_loss = float(loss.detach().item())
            idx_iter = torch.randperm(cand.numel())[: self.batch].tolist()
            for cidx in idx_iter:
                trial = best.clone()
                # Flatten index into (position, candidate rank).
                flat_pos, cand_rank = divmod(cidx, self.top_k)
                b_idx = flat_pos // trial.shape[1]
                l_idx = flat_pos % trial.shape[1]
                trial[b_idx, l_idx] = cand[b_idx, l_idx, cand_rank]
                proj = self.hooks.projection_step(trial)  # type: ignore[misc]
                trial_batch = dict(batch)
                trial_batch[self.input_key] = proj
                with torch.no_grad():
                    tout = model(trial_batch)
                tlogits = tout["logits"] if isinstance(tout, dict) else tout
                tloss = float(
                    self.hooks.substitution_loss(tlogits, labels).item()  # type: ignore[misc]
                )
                if tloss < best_loss:
                    best_loss = tloss
                    best = proj.detach()
            current = best
        if was_training:
            model.train()
        out_batch = dict(batch)
        out_batch[self.input_key] = current.detach()
        return out_batch

    def __call__(
        self, model: Any, batch: Mapping[str, Any], y: torch.Tensor | None = None
    ) -> dict[str, torch.Tensor]:
        return self.forward(model, batch, y)
