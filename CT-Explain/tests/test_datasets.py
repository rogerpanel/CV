from ct_explain.data.datasets import (
    CloudEdgeDataset,
    GeneralNetworkTrafficDataset,
    get_dataset,
    list_datasets,
)


def test_list_datasets_has_six():
    assert len(list_datasets()) == 6


def test_synthetic_general_bundle_loads():
    ds = GeneralNetworkTrafficDataset(synthetic=True, synthetic_rows=256)
    df = ds.load_training_frame()
    assert len(df) == 256
    graph = ds.build_graph(df, max_events=128)
    assert graph.num_edges == 128
    assert graph.num_nodes > 0


def test_synthetic_cloud_edge_loads():
    ds = CloudEdgeDataset(synthetic=True, synthetic_rows=256)
    df = ds.load_training_frame()
    assert len(df) == 256


def test_get_dataset_accepts_all_keys():
    for key in list_datasets():
        ds = get_dataset(key, synthetic=True, synthetic_rows=64)
        assert ds is not None
        graph = ds.build_graph(ds.load_training_frame(), max_events=32)
        assert graph.num_edges == 32
