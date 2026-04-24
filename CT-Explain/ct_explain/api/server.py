"""Flask REST server exposing CT-Explain to SIEMs and the RobustIDPS.ai SPA.

Endpoints (all under `/api/v1/ct-explain`):

    GET  /health                     – liveness + version + backbone
    POST /predict                    – detection + uncertainty
    POST /explain                    – four-way explanation (calibrated CIs)
    POST /certify                    – ConformalGuard fixed-horizon verdict
    POST /monitor                    – ConformalGuard anytime-valid monitor
    POST /calibrate                  – calibration on a provided set
    POST /feedback                   – active-learning feedback ingestion
    GET  /dashboard/top              – top-K uncertainty-ranked alerts

The server is deliberately minimal — SIEMs and the RobustIDPS.ai platform
provide the rest of the API surface (authN/Z, persistence, WebSocket,
report export). See `ct_explain.integration.robustidps` for registration.
"""
from __future__ import annotations

from typing import Optional

from flask import Flask, jsonify, request
from flask_cors import CORS

from ct_explain import __version__
from ct_explain.api.schemas import (
    CalibrationRequest,
    CertifyRequest,
    ExplainRequest,
    FeedbackRequest,
    HealthResponse,
    PredictRequest,
)
from ct_explain.conformal.execution_graph import DynamicExecutionGraph, NodeKind
from ct_explain.soc.active_learning import Feedback
from ct_explain.soc.human_ai import HumanAICollaboration
from ct_explain.soc.siem_plugin import NetFlowTranslator


def create_app(collaboration: HumanAICollaboration) -> Flask:
    app = Flask("ct_explain")
    CORS(app)

    translator = NetFlowTranslator()
    BASE = "/api/v1/ct-explain"

    # ------------------------------------------------------------------ #
    @app.get(f"{BASE}/health")
    def health():  # noqa: D401
        return jsonify(
            HealthResponse(
                status="ok",
                version=__version__,
                backbone=type(collaboration.model).__name__,
                calibrated=(collaboration.guard._martingale is not None),
            ).model_dump()
        )

    # ------------------------------------------------------------------ #
    @app.post(f"{BASE}/predict")
    def predict():
        req = PredictRequest.model_validate(request.get_json(force=True))
        graph = _events_to_graph(req.events, translator)
        result = collaboration.model.predict(
            graph.node_features, graph.edge_index,
            graph.edge_features, graph.edge_times,
        )
        probs = result["probabilities"]
        return jsonify({
            "predictions": result["prediction"].tolist(),
            "probabilities": probs.tolist(),
            "safety_probabilities": result["safety_prob"].tolist(),
            "uncertainty": (1 - probs.max(-1).values).tolist(),
        })

    # ------------------------------------------------------------------ #
    @app.post(f"{BASE}/explain")
    def explain():
        req = ExplainRequest.model_validate(request.get_json(force=True))
        graph = _events_to_graph(req.events, translator)
        alert = collaboration.triage(
            graph=graph, target_node=req.target_node,
            alert_id=f"alert_{req.target_node}", source="api",
        )
        return jsonify(collaboration.investigate(
            graph=graph, alert=alert, render_dir=req.render_dir,
        ))

    # ------------------------------------------------------------------ #
    @app.post(f"{BASE}/certify")
    def certify():
        req = CertifyRequest.model_validate(request.get_json(force=True))
        g = _deserialize_exec_graph(req.exec_graph)
        verdict = collaboration.guard.certify(
            node_id=req.action_id,
            exec_graph=g,
            p_safe_fn=lambda nid: float(req.p_safe.get(nid, 1.0)),
        )
        return jsonify({
            "verdict": verdict.verdict,
            "score": verdict.score,
            "q_hat": verdict.q_hat,
            "e_value": verdict.e_value,
            "triggered": verdict.triggered,
            "reason": verdict.reason,
        })

    # ------------------------------------------------------------------ #
    @app.post(f"{BASE}/monitor")
    def monitor():
        req = CertifyRequest.model_validate(request.get_json(force=True))
        g = _deserialize_exec_graph(req.exec_graph)
        verdict = collaboration.guard.monitor(
            node_id=req.action_id,
            exec_graph=g,
            p_safe_fn=lambda nid: float(req.p_safe.get(nid, 1.0)),
        )
        return jsonify({
            "verdict": verdict.verdict,
            "score": verdict.score,
            "q_hat": verdict.q_hat,
            "e_value": verdict.e_value,
            "triggered": verdict.triggered,
            "reason": verdict.reason,
        })

    # ------------------------------------------------------------------ #
    @app.post(f"{BASE}/calibrate")
    def calibrate():
        req = CalibrationRequest.model_validate(request.get_json(force=True))
        tuples = []
        for entry in req.calibration_set:
            g = _deserialize_exec_graph(entry["exec_graph"])
            tuples.append((
                entry["action_id"],
                g,
                (lambda m=entry["p_safe"]: (lambda nid: float(m.get(nid, 1.0))))(),
                int(entry["label"]),
            ))
        # Reconfigure guard with new parameters before calibration.
        collaboration.guard.alpha = req.alpha
        collaboration.guard.scorer.k_hops = req.k_hops
        collaboration.guard.scorer.lambda_cascade = req.lambda_cascade
        q_hat = collaboration.guard.calibrate(tuples)
        return jsonify({"q_hat": q_hat, "n_calibration": len(tuples),
                        "alpha": req.alpha})

    # ------------------------------------------------------------------ #
    @app.post(f"{BASE}/feedback")
    def feedback():
        req = FeedbackRequest.model_validate(request.get_json(force=True))
        import torch
        collaboration.record_feedback(
            Feedback(
                alert_id=req.alert_id,
                corrected_label=req.corrected_label,
                rationale=req.rationale,
                analyst_confidence=req.analyst_confidence,
                features=(torch.tensor(req.features) if req.features else None),
            )
        )
        return jsonify({"ok": True, "buffer_size": len(collaboration.feedback_buffer)})

    # ------------------------------------------------------------------ #
    @app.get(f"{BASE}/dashboard/top")
    def dashboard_top():
        k = int(request.args.get("k", 10))
        return jsonify(collaboration.dashboard.top(k))

    return app


# ---------------------------------------------------------------------- #
# Helpers
# ---------------------------------------------------------------------- #

def _events_to_graph(events, translator):
    records = translator.normalise(e.model_dump() for e in events)
    from ct_explain.data.graph_builder import ContinuousTimeGraphBuilder
    b = ContinuousTimeGraphBuilder()
    b.add_stream(records)
    return b.build()


def _deserialize_exec_graph(payload: dict) -> DynamicExecutionGraph:
    g = DynamicExecutionGraph()
    for node in payload.get("nodes", []):
        g.add_node(
            node_id=node["id"],
            kind=NodeKind(node["kind"]),
            timestamp=float(node.get("timestamp", 0.0)),
            attrs=node.get("attrs", {}),
        )
    from ct_explain.conformal.execution_graph import EdgeKind
    for edge in payload.get("edges", []):
        g.add_edge(
            src=edge["src"], dst=edge["dst"],
            kind=EdgeKind(edge["kind"]),
            timestamp=float(edge.get("timestamp", 0.0)),
        )
    return g
