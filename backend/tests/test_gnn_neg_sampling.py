"""Type-aware negative sampling for CompGCN."""
from __future__ import annotations

import os
os.environ.setdefault("NEO4J_URI", "bolt://localhost:7687")
os.environ.setdefault("NEO4J_PASSWORD", "test")
os.environ.setdefault("GOOGLE_API_KEY", "test-key")

import torch
from torch_geometric.data import Data


def test_build_type_pools_partitions_nodes_by_label():
    from src.gnn_module import _build_type_pools

    # 10 nodes, 3 labels: 0=Method (5 nodes), 1=Researcher (3 nodes),
    # 2=Metric (2 nodes). Sentinel -1 node omitted from pools.
    node_type = torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 2, 2, -1], dtype=torch.long)
    pools = _build_type_pools(node_type)

    assert set(pools.keys()) == {0, 1, 2}, "pools must exclude sentinel -1"
    assert sorted(pools[0].tolist()) == [0, 1, 2, 3, 4]
    assert sorted(pools[1].tolist()) == [5, 6, 7]
    assert sorted(pools[2].tolist()) == [8, 9]
    for t in pools.values():
        assert t.dtype == torch.long


def test_build_type_pools_drops_singletons():
    """Pools with <2 entries are omitted; sampler must fall back to uniform."""
    from src.gnn_module import _build_type_pools

    node_type = torch.tensor([0, 0, 1], dtype=torch.long)
    pools = _build_type_pools(node_type)
    assert set(pools.keys()) == {0}, "label 1 has only 1 node → dropped"


def test_sample_negative_edges_type_aware_preserves_labels():
    """Every corrupted endpoint must share the original endpoint's label."""
    from src.gnn_module import _sample_negative_edges, _build_type_pools

    # 20 nodes, 2 labels (10 each).
    node_type = torch.tensor([0] * 10 + [1] * 10, dtype=torch.long)
    pools = _build_type_pools(node_type)

    # 5 edges, each connecting a label-0 node to a label-1 node.
    edge_index = torch.tensor(
        [[0, 1, 2, 3, 4],
         [10, 11, 12, 13, 14]],
        dtype=torch.long,
    )
    edge_type = torch.tensor([0, 0, 0, 0, 0], dtype=torch.long)

    torch.manual_seed(42)
    neg_edges, neg_types = _sample_negative_edges(
        edge_index, edge_type, num_nodes=20, num_negatives=3,
        node_type=node_type, type_pools=pools,
    )

    # Shape: num_negatives * num_edges = 3 * 5 = 15
    assert neg_edges.size(1) == 15
    assert neg_types.size(0) == 15

    # For each corrupted edge, check that corrupted-side label matches original.
    # Original edge i has src_label=0, dst_label=1.
    for i in range(neg_edges.size(1)):
        src = neg_edges[0, i].item()
        dst = neg_edges[1, i].item()
        # The sampler corrupts either head or tail (not both). So exactly one
        # endpoint equals the original (for that edge position in the batch).
        original_col = i % 5
        orig_src = edge_index[0, original_col].item()
        orig_dst = edge_index[1, original_col].item()
        head_corrupted = src != orig_src
        tail_corrupted = dst != orig_dst
        # Either head or tail was corrupted (or same node redrawn — allowed,
        # the positive-triple filter still rejects true triples).
        if head_corrupted:
            assert node_type[src].item() == 0, f"corrupted head label mismatch at {i}"
        if tail_corrupted:
            assert node_type[dst].item() == 1, f"corrupted tail label mismatch at {i}"


def test_sample_negative_edges_uniform_mode_unchanged():
    """When node_type/type_pools absent, behavior matches pre-existing sampler."""
    from src.gnn_module import _sample_negative_edges

    edge_index = torch.tensor([[0, 1], [2, 3]], dtype=torch.long)
    edge_type = torch.tensor([0, 1], dtype=torch.long)

    torch.manual_seed(7)
    neg_a, _ = _sample_negative_edges(
        edge_index, edge_type, num_nodes=10, num_negatives=2,
    )
    torch.manual_seed(7)
    neg_b, _ = _sample_negative_edges(
        edge_index, edge_type, num_nodes=10, num_negatives=2,
        node_type=None, type_pools=None,
    )
    assert torch.equal(neg_a, neg_b), "passing None kwargs must be a no-op"


def test_evaluate_mrr_accepts_type_aware_kwargs():
    """_evaluate_mrr must accept node_type/type_pools and pass them through."""
    import inspect
    from src.gnn_module import _evaluate_mrr, _evaluate_auc

    sig_mrr = inspect.signature(_evaluate_mrr)
    assert "node_type" in sig_mrr.parameters
    assert "type_pools" in sig_mrr.parameters

    sig_auc = inspect.signature(_evaluate_auc)
    assert "node_type" in sig_auc.parameters
    assert "type_pools" in sig_auc.parameters
