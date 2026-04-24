from ct_explain.data.graph_builder import ContinuousTimeGraphBuilder


def test_builder_basic():
    b = ContinuousTimeGraphBuilder(default_node_feature_dim=4)
    b.add_event("a", "b", 0.0, [0.1, 0.2, 0.3, 0.4], label=1)
    b.add_event("a", "c", 0.1, [0.2, 0.1, 0.0, 0.5], label=0)
    b.add_event("b", "c", 0.2, [0.0, 0.0, 0.1, 0.9], label=1)
    g = b.build()
    assert g.num_nodes == 3
    assert g.num_edges == 3
    assert g.edge_features.shape[1] == 4


def test_builder_window(small_graph):
    w = small_graph.window(0.0, 0.25)
    assert 0 < w.num_edges < small_graph.num_edges
