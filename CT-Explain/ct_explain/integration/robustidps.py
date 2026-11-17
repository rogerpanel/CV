"""RobustIDPS.ai model-registry hook.

Exposes CT-Explain as a first-class model in the platform's registry
alongside SurrogateIDS, MambaShield, CL-RL Unified, FedGTD, CyberSecLLM,
Multi-Agent PQC-IDS, Stochastic Transformer, Neural ODE, Optimal Transport,
VAE and Adversarial Autoencoder.

The RobustIDPS.ai backend is a Flask application (see
`/robustidps_web_app/backend`). It does not yet ship a dynamic-plugin
loader — integration happens in source code. This module supplies (a) the
descriptor dict the registry expects, and (b) a `register(app)` helper that
mounts CT-Explain's REST blueprint onto an existing Flask app so that one
line of code in the platform's `backend/app.py` brings CT-Explain online.
"""
from __future__ import annotations

from typing import Any, Optional

from ct_explain import __version__
from ct_explain.api.server import create_app
from ct_explain.soc.human_ai import HumanAICollaboration

PLUGIN_METADATA: dict[str, Any] = {
    "key": "ct_explain",
    "display_name": "CT-Explain",
    "family": "continuous-time-dgnn",
    "version": __version__,
    "description": (
        "Explainable Continuous-Time Dynamic Graph Neural Network with "
        "Conformal Safety Certification (ConformalGuard)."
    ),
    "capabilities": [
        "detection",
        "uncertainty-quantification",
        "four-modal-explanation",
        "conformal-certification",
        "anytime-valid-monitoring",
        "active-learning-feedback",
    ],
    "endpoints": [
        "GET  /api/v1/ct-explain/health",
        "POST /api/v1/ct-explain/predict",
        "POST /api/v1/ct-explain/explain",
        "POST /api/v1/ct-explain/certify",
        "POST /api/v1/ct-explain/monitor",
        "POST /api/v1/ct-explain/calibrate",
        "POST /api/v1/ct-explain/feedback",
        "GET  /api/v1/ct-explain/dashboard/top",
    ],
    "datasets_supported": [
        "cic-iot-2023",
        "cse-cicids-2018",
        "unsw-nb15",
        "ms-guide-2024",
        "container-nid",
        "edge-iiot",
    ],
    "expected_latency_ms": {
        "predict": 18,
        "attention_flow": 12,
        "uncertainty_attribution": 19,
        "counterfactual": 45,
        "game_theory": 31,
        "combined_explanation": 52,
        "certify": 25,
    },
    "feature_schema": "netflow-v2",
    "feature_dim": 83,
    "class_count": 34,          # 33 attacks + benign
    "paper_reference": (
        "Anaedevha, R. N. (2026). CT-Explain: Explainable Continuous-Time "
        "Dynamic Graph Neural Networks with Conformal Safety Certification "
        "for Human-AI Collaborative Intrusion Detection in Security "
        "Operations Centers."
    ),
    "platform_doi": "10.5281/zenodo.19129512",
    "kaggle_dois": [
        "10.34740/kaggle/dsv/12483891",
        "10.34740/KAGGLE/DSV/12479689",
    ],
}


def register(
    flask_app,
    collaboration: HumanAICollaboration,
    registry: Optional[dict] = None,
) -> dict:
    """Mount CT-Explain on an existing Flask app and push its metadata
    into the RobustIDPS.ai registry dict. Returns the updated registry.

    Expected usage inside `robustidps_web_app/backend/app.py`::

        from ct_explain.integration.robustidps import register
        from ct_explain.soc.human_ai import HumanAICollaboration
        from ct_explain.models.ct_tgnn import CTTemporalGNN

        model = CTTemporalGNN(...).eval()
        collab = HumanAICollaboration(model, class_names=ATTACK_CLASSES)
        register(app, collab, registry=app.config["MODEL_REGISTRY"])
    """
    sub = create_app(collaboration)
    # Register each CT-Explain view-function on the host Flask app.
    for rule in sub.url_map.iter_rules():
        if rule.endpoint == "static":
            continue
        view = sub.view_functions[rule.endpoint]
        endpoint = f"ct_explain.{rule.endpoint}"
        methods = list(rule.methods - {"HEAD", "OPTIONS"})
        flask_app.add_url_rule(
            str(rule),
            endpoint=endpoint,
            view_func=view,
            methods=methods,
        )

    if registry is not None:
        registry[PLUGIN_METADATA["key"]] = PLUGIN_METADATA
    return PLUGIN_METADATA
