"""Class-imbalance modules (module 1 of the v2 plan).

Losses
    ce      plain cross-entropy
    la      logit-adjusted cross-entropy (Menon et al., ICLR 2021): CE(z + tau log pi, y);
            predictions use the raw logits z, which corrects the prior shift
    focal   focal loss (Lin et al., 2017), gamma = 2
    cb      class-balanced CE with effective-number weights (Cui et al., CVPR 2019)

Sampling
    class_balanced_sampler(y, power): sampling weight of a flow of class k is
    n_k^{-power} (0 = uniform, 0.5 = square-root, 1 = fully balanced), with the
    largest-to-smallest weight ratio clipped.

Post-hoc adjustment (no retraining)
    tune_tau: subtract tau * log(pi) from the logits, tau chosen on a validation
    split to maximise macro-F1 (Menon et al., post-hoc logit adjustment).
    tune_per_class_bias: coordinate ascent over per-class biases.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def class_prior(y: torch.Tensor, K: int) -> torch.Tensor:
    counts = torch.bincount(y, minlength=K).float().clamp_min(1.0)
    return counts / counts.sum()


class ImbalanceLoss:
    def __init__(self, kind: str, counts: torch.Tensor, tau: float = 1.0,
                 gamma: float = 2.0, cb_beta: float = 0.999, label_smoothing: float = 0.0):
        if kind not in {"ce", "la", "focal", "cb"}:
            raise ValueError(f"unknown loss {kind}")
        self.kind, self.tau, self.gamma, self.ls = kind, tau, gamma, label_smoothing
        counts = counts.float().clamp_min(1.0)
        self.log_prior = (counts / counts.sum()).log()
        eff = (1.0 - cb_beta ** counts) / (1.0 - cb_beta)          # effective number of samples
        w = 1.0 / eff
        self.cb_weight = w / w.mean()

    def to(self, device):
        self.log_prior = self.log_prior.to(device)
        self.cb_weight = self.cb_weight.to(device)
        return self

    def __call__(self, logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if self.kind == "la":
            return F.cross_entropy(logits + self.tau * self.log_prior, y, label_smoothing=self.ls)
        if self.kind == "cb":
            return F.cross_entropy(logits, y, weight=self.cb_weight, label_smoothing=self.ls)
        if self.kind == "focal":
            logp = F.log_softmax(logits, dim=-1).gather(1, y[:, None]).squeeze(1)
            return (-(1.0 - logp.exp()) ** self.gamma * logp).mean()
        return F.cross_entropy(logits, y, label_smoothing=self.ls)


def class_balanced_sampler(y: torch.Tensor, power: float, max_ratio: float = 50.0,
                           generator: torch.Generator | None = None):
    """Returns a function giving a fresh epoch of indices (with replacement)."""
    K = int(y.max()) + 1
    counts = torch.bincount(y, minlength=K).float().clamp_min(1.0)
    w_class = counts ** (-power)
    w_class = w_class.clamp(max=w_class.min() * max_ratio)
    w = w_class[y]

    def epoch():
        return torch.multinomial(w, len(y), replacement=True, generator=generator)
    return epoch


# ------------------------------------------------------------------ post-hoc
def fast_macro_f1(y: np.ndarray, p: np.ndarray, K: int, w: np.ndarray | None = None) -> float:
    """Macro-F1 over the labels present in y or p (as sklearn with zero_division=0),
    with optional sample weights."""
    w = np.ones(len(y)) if w is None else w
    cm = np.bincount(y * K + p, weights=w, minlength=K * K).reshape(K, K)
    tp = np.diag(cm)
    fp, fn = cm.sum(0) - tp, cm.sum(1) - tp
    denom = 2 * tp + fp + fn
    f1 = np.where(denom > 0, 2 * tp / np.maximum(denom, 1e-12), 0.0)
    present = (cm.sum(0) + cm.sum(1)) > 0
    return float(f1[present].mean())


def _predict(samples: torch.Tensor, bias: torch.Tensor, vote: bool) -> np.ndarray:
    """samples: (S, N, K) logits. Mean-logit argmax, or a majority vote over S."""
    if vote:
        votes = (samples + bias).argmax(-1)                       # (S, N)
        K = samples.shape[-1]
        counts = F.one_hot(votes, K).sum(0)                      # (N, K)
        return counts.argmax(-1).numpy()
    return (samples.mean(0) + bias).argmax(-1).numpy()


def tune_tau(samples: torch.Tensor, y: np.ndarray, prior: torch.Tensor, *, vote: bool = False,
             grid=np.round(np.arange(-1.0, 2.51, 0.25), 2)) -> tuple[float, dict]:
    K = samples.shape[-1]
    lp = prior.log().cpu()
    scores = {float(t): fast_macro_f1(y, _predict(samples, -t * lp, vote), K) for t in grid}
    best = max(scores, key=lambda t: (scores[t], -abs(t)))      # ties -> smallest |tau|
    return best, scores


def tune_per_class_bias(samples: torch.Tensor, y: np.ndarray, *, vote: bool = False,
                        init: torch.Tensor | None = None, passes: int = 3,
                        grid=np.arange(-3.0, 3.01, 0.25)) -> torch.Tensor:
    K = samples.shape[-1]
    b = torch.zeros(K) if init is None else init.clone()
    best = fast_macro_f1(y, _predict(samples, b, vote), K)
    for _ in range(passes):
        for k in range(K):
            for v in grid:
                c = b.clone()
                c[k] = float(v)
                s = fast_macro_f1(y, _predict(samples, c, vote), K)
                if s > best + 1e-9:
                    best, b = s, c
    return b
