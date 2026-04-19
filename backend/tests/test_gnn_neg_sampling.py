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
