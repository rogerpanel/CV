"""Human-AI collaboration controller.

One object per SOC session. Coordinates:

    (1) detection + uncertainty from CT-TGNN / SDE-TGNN,
    (2) the four explanations (on-demand or eagerly),
    (3) ConformalGuard safety verdict,
    (4) calibrated attribution intervals,
    (5) active-learning feedback capture.

Emits events consumable by either the Flask REST API or a WebSocket
listener — the paper's field-deployment section mentions sub-second push
notifications, which the WebSocket side of `robustidps_web_app/backend`
already supports.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import torch

from ct_explain.conformal.conformal_guard import ConformalGuard, ConformalVerdict
from ct_explain.conformal.execution_graph import DynamicExecutionGraph
from ct_explain.data.graph_builder import TemporalGraph
from ct_explain.explainers import (
    AttentionFlowExplainer,
    CalibratedExplanations,
    CounterfactualExplainer,
    GameTheoreticExplainer,
    UncertaintyAttributionExplainer,
)
from ct_explain.soc.active_learning import ActiveLearningBuffer, Feedback
from ct_explain.soc.dashboard import Alert, AlertTriageDashboard, InvestigationPanel


@dataclass
class SessionConfig:
    eager_explanations: bool = True
    alpha: float = 0.05
    explanation_budget_ms: float = 100.0


class HumanAICollaboration:
    def __init__(
        self,
        model,
        class_names: list[str],
        conformal_guard: Optional[ConformalGuard] = None,
        config: Optional[SessionConfig] = None,
    ) -> None:
        self.model = model
        self.class_names = class_names
        self.config = config or SessionConfig()
        self.guard = conformal_guard or ConformalGuard(alpha=self.config.alpha)

        self.att_exp = AttentionFlowExplainer(model)
        self.unc_exp = UncertaintyAttributionExplainer(model)
        self.cf_exp = CounterfactualExplainer(model)
        self.gt_exp = GameTheoreticExplainer(model)
        self.calibrated = CalibratedExplanations(self.unc_exp, alpha=self.config.alpha)

        self.dashboard = AlertTriageDashboard()
        self.panel = InvestigationPanel()
        self.feedback_buffer = ActiveLearningBuffer()

    # ------------------------------------------------------------------ #
    def triage(
        self, graph: TemporalGraph, target_node: int, alert_id: str, source: str,
    ) -> Alert:
        with torch.no_grad():
            pred = self.model.predict(
                graph.node_features, graph.edge_index,
                graph.edge_features, graph.edge_times,
            )
        prob = pred["probabilities"][target_node]
        cls_idx = int(prob.argmax().item())
        conf = float(prob.max().item())

        # Build a per-alert execution graph stub so ConformalGuard can evaluate
        # cascade risk even on point-wise alerts.
        exec_graph = DynamicExecutionGraph()
        from ct_explain.conformal.execution_graph import EdgeKind, NodeKind
        exec_graph.add_node(alert_id, NodeKind.ACTION, timestamp=0.0,
                            attrs={"agent_id": source})
        # No cascade neighbours yet; the REST layer can add them when the
        # workflow evolves.

        def _p_safe(_node_id: str) -> float:
            return float(pred["safety_prob"][target_node].item())

        verdict = self.guard.monitor(alert_id, exec_graph, _p_safe) \
            if self.guard._martingale is not None else None

        alert = Alert(
            alert_id=alert_id,
            source=source,
            target_node=target_node,
            prediction=cls_idx,
            predicted_class=(
                self.class_names[cls_idx]
                if cls_idx < len(self.class_names) else str(cls_idx)
            ),
            confidence=conf,
            uncertainty=float(1 - conf),
            conformal_verdict=(verdict.verdict if verdict else "Unknown"),
            timestamp=0.0,
        )
        self.dashboard.push(alert)
        return alert

    # ------------------------------------------------------------------ #
    def investigate(
        self, graph: TemporalGraph, alert: Alert, render_dir: Optional[str] = None,
    ) -> dict:
        att = self.att_exp.explain(
            graph=graph, target_node=alert.target_node, render_dir=render_dir,
        )
        unc = self.unc_exp.explain(graph=graph, target_node=alert.target_node)
        cf = self.cf_exp.explain(graph=graph, target_node=alert.target_node)
        gt = self.gt_exp.explain(graph=graph, target_node=alert.target_node)

        calibrated = None
        if self.calibrated.state is not None:
            c = self.calibrated.explain(graph=graph, target_node=alert.target_node)
            calibrated = c.confidence_interval

        conformal = None
        if self.guard._martingale is not None:
            exec_graph = DynamicExecutionGraph()
            from ct_explain.conformal.execution_graph import NodeKind
            exec_graph.add_node(
                alert.alert_id, NodeKind.ACTION, timestamp=0.0,
                attrs={"agent_id": alert.source},
            )
            conformal = self.guard.certify(
                alert.alert_id,
                exec_graph,
                lambda _: 1.0 - alert.uncertainty,
            )

        return self.panel.build(
            alert=alert,
            attention_explanation=att,
            uncertainty_explanation=unc,
            counterfactual_explanation=cf,
            game_explanation=gt,
            calibrated_intervals=calibrated,
            conformal=conformal,
        )

    # ------------------------------------------------------------------ #
    def record_feedback(self, fb: Feedback) -> None:
        self.feedback_buffer.push(fb)
