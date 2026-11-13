"""Tests for analyst routing decisions."""

from __future__ import annotations

from conformalguard.conformal.prediction_set import PredictionSet, Verdict
from conformalguard.deployment.routing import AnalystRouter, RoutingDecision


def _pset(verdict: Verdict, width: int = 3) -> PredictionSet:
    return PredictionSet(
        candidates=list(range(width)),
        scores=[0.1] * width,
        threshold=1.0,
        verdict=verdict,
    )


def test_critical_block_with_drift_is_auto_blocked():
    r = AnalystRouter()
    item = r.route("trace1", 0.0, _pset(Verdict.BLOCK, width=11),
                   top_violation_id=1,    # pii_exfiltration -> critical
                   drift=1.0)             # priority = 0.6 + 0.2 + 0.2 = 1.0
    assert item.decision == RoutingDecision.AUTO_BLOCK


def test_critical_block_without_drift_goes_on_call():
    r = AnalystRouter()
    item = r.route("trace1", 0.0, _pset(Verdict.BLOCK, width=8),
                   top_violation_id=1, drift=0.5)   # priority = 0.84
    assert item.decision == RoutingDecision.ESCALATE_ONCALL


def test_blocked_action_is_never_routed_as_allow():
    r = AnalystRouter()
    item = r.route("t2", 1.0, _pset(Verdict.BLOCK, width=1),
                   top_violation_id=0, drift=0.0)   # lowest possible priority
    assert item.decision == RoutingDecision.REVIEW_QUEUE


def test_allowed_action_is_never_queued():
    r = AnalystRouter()
    item = r.route("t3", 0.0, _pset(Verdict.ALLOW, width=1),
                   top_violation_id=1, drift=1.0)   # high priority, but allowed
    assert item.decision == RoutingDecision.ALLOW
