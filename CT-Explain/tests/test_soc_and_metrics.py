import numpy as np
import torch

from ct_explain.evaluation.conformal import ConformalMetrics
from ct_explain.evaluation.detection import DetectionMetrics
from ct_explain.evaluation.explanation import ExplanationMetrics
from ct_explain.soc.active_learning import ActiveLearningBuffer, BayesianUpdater, Feedback
from ct_explain.soc.human_ai import HumanAICollaboration
from ct_explain.soc.siem_plugin import NetFlowTranslator, SIEMPlugin


def test_detection_metrics_shape():
    probs = np.array([[0.8, 0.2], [0.3, 0.7], [0.6, 0.4]])
    labels = np.array([0, 1, 0])
    r = DetectionMetrics.report(probs, labels)
    assert 0 <= r.f1 <= 1
    assert 0 <= r.ece <= 1


def test_conformal_metrics_bootstrap():
    rng = np.random.default_rng(0)
    scores = rng.uniform(size=200)
    labels = (scores < 0.9).astype(int)
    q = float(np.quantile(scores[labels == 1], 0.9))
    r = ConformalMetrics.report(scores, labels, q, alpha=0.1)
    assert len(r.bootstrap_ci) == 2


def test_explanation_metric_effective_complexity():
    a = torch.tensor([0.1, 0.5, 0.3, 0.1])
    ec = ExplanationMetrics.effective_complexity(a, mass=0.8)
    assert 1 <= ec <= 4


def test_siem_plugin_roundtrip():
    plugin = SIEMPlugin(vendor="splunk", translator=NetFlowTranslator())
    events = [
        {"src_ip": "10.0.0.1", "dst_ip": "10.0.0.2", "timestamp": 0.0, "flow_duration": 1.0},
        {"src_ip": "10.0.0.1", "dst_ip": "10.0.0.3", "timestamp": 0.1, "flow_duration": 2.0},
    ]
    g = plugin.ingest(events)
    assert g.num_edges == 2


def test_active_learning_updates_linear_head():
    linear = torch.nn.Linear(4, 2)
    updater = BayesianUpdater(linear, prior_var=1.0, noise_var=0.1)
    buf = ActiveLearningBuffer(capacity=8)
    buf.push(Feedback(alert_id="a", corrected_label=1,
                      features=torch.ones(4), analyst_confidence=1.0))
    applied = updater.apply(buf.drain())
    assert applied["applied"] == 1


def test_human_ai_collaboration_triage(ct_tgnn_model, small_graph):
    collab = HumanAICollaboration(ct_tgnn_model, class_names=["safe", "attack"])
    alert = collab.triage(graph=small_graph, target_node=0,
                          alert_id="a0", source="test")
    assert alert.predicted_class in {"safe", "attack"}
    assert 0 <= alert.confidence <= 1
