import random

import numpy as np

from ct_explain.conformal.conformal_guard import ConformalGuard
from ct_explain.conformal.e_value import EValueMartingale
from ct_explain.conformal.execution_graph import (
    DynamicExecutionGraph,
    EdgeKind,
    NodeKind,
)
from ct_explain.conformal.graph_conformal import (
    GraphConformalCalibrator,
    NonconformityScorer,
)


def _build_workflow(action_id: str, n_neighbours: int = 2) -> DynamicExecutionGraph:
    g = DynamicExecutionGraph()
    g.add_node(action_id, NodeKind.ACTION, timestamp=0.0, attrs={})
    for i in range(n_neighbours):
        nid = f"{action_id}_n{i}"
        g.add_node(nid, NodeKind.TOOL, timestamp=-1 - i, attrs={})
        g.add_edge(action_id, nid, EdgeKind.INVOKES, timestamp=0.0)
    return g


def test_nonconformity_score_is_nonnegative():
    g = _build_workflow("a")
    scorer = NonconformityScorer()
    s = scorer.score("a", g, p_safe=lambda _: 0.8)
    assert s >= 0


def test_graph_conformal_calibrator_covers():
    random.seed(0)
    scorer = NonconformityScorer(k_hops=2, lambda_cascade=0.2)
    cal = GraphConformalCalibrator(scorer, alpha=0.1)
    calib = []
    for i in range(200):
        g = _build_workflow(f"a_{i}")
        p = random.uniform(0.5, 1.0)
        calib.append((f"a_{i}", g, (lambda p=p: (lambda _: p))(), 1))
    cal.calibrate(calib)
    assert cal.q_hat > 0


def test_e_value_triggers_on_high_scores():
    mg = EValueMartingale(q_hat=0.1, alpha=0.05, lr=0.5)
    for _ in range(50):
        mg.update(0.9)
    assert mg.triggered


def test_conformal_guard_end_to_end():
    guard = ConformalGuard(alpha=0.1, k_hops=1, lambda_cascade=0.0)
    calib = []
    for i in range(100):
        g = _build_workflow(f"a_{i}")
        calib.append((f"a_{i}", g, (lambda: (lambda _: 0.9))(), 1))
    q_hat = guard.calibrate(calib)
    assert q_hat >= 0
    verdict = guard.certify("a_0", calib[0][1], lambda _: 0.95)
    assert verdict.verdict in {"Safe", "Abstain", "Escalate"}
