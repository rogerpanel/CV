import torch


def test_ct_tgnn_forward(ct_tgnn_model, small_graph):
    out = ct_tgnn_model(
        node_features=small_graph.node_features,
        edge_index=small_graph.edge_index,
        edge_features=small_graph.edge_features,
        edge_times=small_graph.edge_times,
    )
    assert out.hidden.shape == (small_graph.num_nodes, ct_tgnn_model.hidden_dim)
    assert out.logits.shape == (small_graph.num_nodes, ct_tgnn_model.num_classes)
    assert out.safety_prob.shape == (small_graph.num_nodes,)
    assert (out.safety_prob >= 0).all() and (out.safety_prob <= 1).all()


def test_predict_monotone(ct_tgnn_model, small_graph):
    pred = ct_tgnn_model.predict(
        node_features=small_graph.node_features,
        edge_index=small_graph.edge_index,
        edge_features=small_graph.edge_features,
        edge_times=small_graph.edge_times,
    )
    assert torch.isfinite(pred["probabilities"]).all()
    assert pred["prediction"].shape == (small_graph.num_nodes,)


def test_sde_sample(sde_tgnn_model, small_graph):
    samples = sde_tgnn_model.sample(
        node_features=small_graph.node_features,
        edge_index=small_graph.edge_index,
        edge_features=small_graph.edge_features,
        edge_times=small_graph.edge_times,
        num_samples=3,
    )
    assert samples.shape == (3, small_graph.num_nodes, sde_tgnn_model.hidden_dim)
