import torch

from ct_explain.explainers.attention_flow import AttentionFlowExplainer
from ct_explain.explainers.counterfactual import CounterfactualExplainer
from ct_explain.explainers.game_theory import GameTheoreticExplainer
from ct_explain.explainers.uncertainty_attribution import UncertaintyAttributionExplainer


def test_attention_flow(ct_tgnn_model, small_graph):
    exp = AttentionFlowExplainer(ct_tgnn_model).explain(
        graph=small_graph, target_node=0, solver_steps=3,
    )
    assert exp.attribution.shape[0] == small_graph.num_edges
    assert "attention_flow" in exp.supporting
    assert exp.latency_ms >= 0


def test_uncertainty_attribution(ct_tgnn_model, small_graph):
    exp = UncertaintyAttributionExplainer(ct_tgnn_model, num_samples=4).explain(
        graph=small_graph, target_node=0,
    )
    assert exp.attribution.shape[0] == small_graph.node_features.shape[1]
    assert torch.isfinite(exp.attribution).all()


def test_counterfactual(ct_tgnn_model, small_graph):
    exp = CounterfactualExplainer(ct_tgnn_model, max_iters=8).explain(
        graph=small_graph, target_node=0,
    )
    cf = exp.supporting["counterfactual"]
    assert cf.delta.shape == cf.original.shape
    assert cf.l2_norm >= 0.0


def test_game_theory(ct_tgnn_model, small_graph):
    exp = GameTheoreticExplainer(ct_tgnn_model).explain(
        graph=small_graph, target_node=0,
    )
    strat = exp.supporting["strategy"]
    assert abs(strat.attacker_strategy.sum() - 1.0) < 1e-4
    assert abs(strat.defender_strategy.sum() - 1.0) < 1e-4
