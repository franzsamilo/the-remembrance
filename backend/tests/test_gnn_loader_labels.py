"""GNNLoader label-fetch behavior."""
from __future__ import annotations

import pytest
import torch
from unittest.mock import MagicMock, patch


@pytest.fixture
def fake_neo4j_session():
    """Yields a session whose .run() returns pre-canned records in order."""
    session = MagicMock()
    rel_records = [
        {"rel_id": "r1", "source": "n1", "target": "n2", "type": "USES"},
        {"rel_id": "r2", "source": "n2", "target": "n3", "type": "PROPOSES"},
    ]
    emb_records = [
        {"id": "n1", "embedding": [0.1] * 768},
        {"id": "n2", "embedding": [0.2] * 768},
        {"id": "n3", "embedding": [0.3] * 768},
    ]
    label_records = [
        {"id": "n1", "labels": ["Researcher", "__Entity__"]},
        {"id": "n2", "labels": ["Method"]},
        {"id": "n3", "labels": ["UnknownType"]},  # falls back to __Entity__ if present, else -1
    ]
    session.run.side_effect = [rel_records, emb_records, label_records]
    session.__enter__ = lambda self_: self_
    session.__exit__ = lambda *a: False
    return session


def test_fetch_graph_data_returns_node_type_and_label_map(fake_neo4j_session):
    """fetch_graph_data must return (data, rel_types, node_id_map, label_to_id)."""
    import os
    os.environ.setdefault("NEO4J_URI", "bolt://localhost:7687")
    os.environ.setdefault("NEO4J_PASSWORD", "test")
    os.environ.setdefault("GOOGLE_API_KEY", "test-key")

    from src.gnn_loader import GNNLoader

    driver = MagicMock()
    driver.session.return_value = fake_neo4j_session

    with patch("src.gnn_loader.DatabaseManager.get_driver", return_value=driver):
        loader = GNNLoader()
        result = loader.fetch_graph_data()

    assert result is not None
    assert len(result) == 4, "expected (data, rel_types, node_id_map, label_to_id)"
    data, rel_types, node_id_map, label_to_id = result

    # node_type tensor exists and aligns with node_id_map
    assert hasattr(data, "node_type")
    assert isinstance(data.node_type, torch.Tensor)
    assert data.node_type.dtype == torch.long
    assert data.node_type.size(0) == data.x.size(0)

    # label_to_id maps known schema labels (alphabetical for determinism)
    assert "Method" in label_to_id
    assert "Researcher" in label_to_id
    # Unknown-only label nodes get sentinel -1
    unknown_idx = node_id_map["n3"]
    assert data.node_type[unknown_idx].item() == -1

    # Known-label nodes get valid pool ids
    assert data.node_type[node_id_map["n1"]].item() == label_to_id["Researcher"]
    assert data.node_type[node_id_map["n2"]].item() == label_to_id["Method"]
