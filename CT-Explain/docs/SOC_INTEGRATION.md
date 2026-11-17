# SOC Integration

CT-Explain is packaged to plug directly into the RobustIDPS.ai platform
that already hosts the author's 13+ intrusion-detection models (see
`robustidps_web_app/`). It can also be run as a stand-alone Flask
service.

## 1. Stand-alone service

```python
from ct_explain.api.server import create_app
from ct_explain.soc.human_ai import HumanAICollaboration
from ct_explain.models.ct_tgnn import CTTemporalGNN

model = CTTemporalGNN(
    node_feat_dim=83, edge_feat_dim=83,
    hidden_dim=128, num_classes=34,
).eval()

collab = HumanAICollaboration(
    model,
    class_names=ATTACK_CLASSES,     # 33 attacks + benign
)

app = create_app(collab)
app.run(host="0.0.0.0", port=8787)
```

## 2. Mounted on RobustIDPS.ai

Inside `robustidps_web_app/backend/app.py`:

```python
from ct_explain.integration.robustidps import register
from ct_explain.soc.human_ai import HumanAICollaboration
from ct_explain.models.sde_tgnn import SDETemporalGNN

model = SDETemporalGNN(node_feat_dim=83, edge_feat_dim=83,
                       hidden_dim=128, num_classes=34).eval()
collab = HumanAICollaboration(model, class_names=ATTACK_CLASSES)
register(app, collab, registry=app.config["MODEL_REGISTRY"])
```

`register()` mounts every CT-Explain route on the host Flask app under
`/api/v1/ct-explain/…` and adds an entry to the platform's model
registry. The platform's admin dashboard will then list CT-Explain
alongside SurrogateIDS, MambaShield, CL-RL Unified, FedGTD, CyberSecLLM,
Multi-Agent PQC-IDS, Neural ODE, Optimal Transport, Stochastic
Transformer, VAE, and Adversarial Autoencoder.

## 3. SIEM plugin

`ct_explain.soc.siem_plugin.SIEMPlugin` accepts a vendor tag (`splunk`,
`qradar`, `arcsight`, or any custom string) and a `NetFlowTranslator`
instance. It converts a stream of NetFlow-v2 JSON events into a
`TemporalGraph` and emits verdicts back over REST.

```python
from ct_explain.soc.siem_plugin import SIEMPlugin, NetFlowTranslator
plugin = SIEMPlugin(vendor="qradar", translator=NetFlowTranslator())
graph = plugin.ingest(events)                  # list[dict] from QRadar API
```

## 4. SOC dashboard contract

The `InvestigationPanel.build()` method returns a JSON document with the
following shape; the RobustIDPS.ai React SPA only needs to know this
schema.

```jsonc
{
  "alert": { "alert_id": "...", "confidence": 0.93, ... },
  "attention_flow": {
    "latency_ms": 12.3,
    "edge_attention": [...],
    "stages": ["reconnaissance", "initial-access", ...],
    "colors": ["#00B4D8", "#0077B6", ...],
    "rendered": { "graph": "runs/.../attention_graph.png",
                  "heatmap": "runs/.../attention_heatmap.png" }
  },
  "uncertainty_attribution": {
    "latency_ms": 18.7,
    "per_feature_share": [...],
    "total_variance": 0.04,
    "method": "moment-closure"
  },
  "counterfactual": {
    "latency_ms": 45.2,
    "delta": [...],
    "l2_norm": 0.12,
    "original_prediction": 3,
    "perturbed_prediction": 0,
    "iterations": 27
  },
  "game_theory": {
    "latency_ms": 31.4,
    "attacker_strategy": [...],
    "defender_strategy": [...],
    "tactic_rank": [["lateral-movement", 0.42], ["exfiltration", 0.31], ...],
    "evasion_budget": 0.062,
    "narrative": "Nash-equilibrium analysis favours ..."
  },
  "calibrated_intervals": [[-0.04, 0.12], ...],
  "conformal": {
    "verdict": "Safe|Abstain|Escalate",
    "score": 0.07,
    "q_hat": 0.14,
    "e_value": 0.3,
    "triggered": false,
    "reason": "nonconformity below calibrated threshold"
  }
}
```

## 5. Active-learning feedback

Analyst corrections are captured via the `/feedback` endpoint and stored
in an `ActiveLearningBuffer`. A `BayesianUpdater` periodically drains the
buffer and performs a Gaussian-NG update on the classifier's final
linear layer — the paper's "Bayesian online updating" extension of the
UC-HGP framework. This is safe to call while the ODE backbone continues
to serve traffic; only the lightweight head is modified.
