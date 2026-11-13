"""Production runtime — the object instantiated by the RobustIDPS plug-in.

Brings together:
  * `DyGFormerHGT`            — encoder
  * `SafetyHead`              — vocabulary classifier
  * `ConformalCalibrator`     — split-conformal threshold
  * `AdaptiveConformal`       — online adaptive coverage (ACI, Algorithm 1)
  * `DriftMonitor`            — KS-based drift early warning
  * `AnalystRouter`           — routes flagged steps to the SOC review queue
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from conformalguard.adaptive.drift_monitor import DriftMonitor
from conformalguard.adaptive.gibbs_candes import AdaptiveConformal
from conformalguard.conformal.calibration import ConformalCalibrator
from conformalguard.conformal.nonconformity import NegLogProbScore, SafetyHead
from conformalguard.conformal.prediction_set import PredictionSet, Verdict
from conformalguard.encoder.dygformer_hgt import DyGFormerHGT, EncoderConfig
from conformalguard.graph.execution_graph import ExecutionGraph
from conformalguard.violations.classes import VIOLATION_CLASSES


@dataclass
class GuardDecision:
    verdict: Verdict
    prediction_set: PredictionSet
    proposed_action: int
    latency_ms: float
    drift: float
    alpha_eff: float


class ConformalGuardRuntime:
    """End-to-end inference pipeline."""

    def __init__(
        self,
        encoder: DyGFormerHGT,
        head: SafetyHead,
        calibrator: ConformalCalibrator,
        adaptive: AdaptiveConformal | None = None,
        drift_monitor: DriftMonitor | None = None,
        device: str | None = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.encoder = encoder.to(self.device).eval()
        self.head = head.to(self.device).eval()
        self.calibrator = calibrator
        # ACI needs the calibration scores to re-read the quantile at the
        # working level; without them the guard runs the static threshold.
        self.adaptive = adaptive
        self.drift_monitor = drift_monitor
        self.score = NegLogProbScore()

    # --------------------------------------------------------------------
    @torch.no_grad()
    def evaluate_step(
        self,
        graph: ExecutionGraph,
        step_t: float,
        proposed_action: int,
    ) -> GuardDecision:
        t0 = time.perf_counter()
        sub = graph.subgraph_at(step_t)
        h = self.encoder.encode_graph(sub, step_t=step_t).unsqueeze(0)
        logits = self.head(h)
        all_scores = self.score.all_candidates(logits).squeeze(0)

        # Threshold: static split-conformal quantile, or the quantile re-read
        # at the ACI working level alpha_t when online adaptation is enabled.
        tau_eff = (self.adaptive.threshold() if self.adaptive is not None
                   else self.calibrator.threshold())
        pset = PredictionSet.from_logits(
            logits.squeeze(0),
            threshold=tau_eff,
            nonconformity_all_candidates=all_scores,
            proposed_action=proposed_action,
        )
        latency = (time.perf_counter() - t0) * 1000.0

        # NOTE: the ACI update is *not* applied here.  err_t is defined only on
        # steps whose proposed action is later labelled safe, so the caller
        # feeds the label back through `feedback()` when it arrives.
        drift = 0.0
        if self.drift_monitor is not None:
            drift = self.drift_monitor.observe(float(all_scores[proposed_action])).ks_statistic

        return GuardDecision(
            verdict=pset.verdict,
            prediction_set=pset,
            proposed_action=proposed_action,
            latency_ms=latency,
            drift=drift,
            alpha_eff=(self.adaptive.alpha_t if self.adaptive is not None
                       else self.calibrator.alpha),
        )

    # --------------------------------------------------------------------
    def feedback(self, decision: GuardDecision, proposed_action_was_safe: bool) -> None:
        """Delayed label for a past decision (Algorithm 1, line 5).

        Only labelled-safe steps update the working level: err = 1 iff a safe
        proposed action was blocked.  Steps whose proposed action is labelled
        violating carry no coverage error and are skipped.
        """
        if self.adaptive is None or not proposed_action_was_safe:
            return
        self.adaptive.update(int(decision.verdict != Verdict.ALLOW))

    # --------------------------------------------------------------------
    @classmethod
    def load(cls, ckpt_path: str | Path) -> "ConformalGuardRuntime":
        ckpt = torch.load(ckpt_path, map_location="cpu")
        cfg = EncoderConfig(**ckpt["cfg"]["encoder"])
        encoder = DyGFormerHGT(cfg); encoder.load_state_dict(ckpt["encoder"])
        head = SafetyHead(d_model=cfg.d_model, vocab_size=len(VIOLATION_CLASSES))
        head.load_state_dict(ckpt["head"])
        calibrator = ConformalCalibrator()
        cal_state = ckpt.get("calibrator")
        if cal_state:
            calibrator.load_state_dict(cal_state)
        adaptive = None
        if ckpt.get("cal_scores") is not None:
            adaptive = AdaptiveConformal(np.asarray(ckpt["cal_scores"]),
                                         alpha=calibrator.alpha)
        return cls(encoder=encoder, head=head, calibrator=calibrator, adaptive=adaptive)
