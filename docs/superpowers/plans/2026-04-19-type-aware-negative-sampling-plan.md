# Type-Aware Negative Sampling Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Raise CompGCN MRR from 0.886 toward the paper's >0.95 target by swapping uniform random negative corruption for schema-label-matched corruption, while preserving AUC-ROC ≥ 0.95, Grounding, and Faithfulness.

**Architecture:** Extend `GNNLoader` to fetch the primary schema label per node and emit a `data.node_type` tensor. In `gnn_module`, build per-label node pools once, then corrupt heads/tails only with same-label nodes when `COMPGCN_NEG_SAMPLING=type_aware`. Post-training, evaluate MRR twice (uniform + type-aware) for apples-to-apples comparison with the Run 6 BPR baseline. An AUC guardrail aborts the Neo4j score sync if calibration regresses.

**Tech Stack:** PyTorch + PyTorch Geometric, Neo4j (Aura Free), Python 3.11, pytest, Windows dev machine. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-04-19-type-aware-negative-sampling-design.md`

---

## File Structure

- Modify: `backend/src/config.py` — two new env-driven constants (`COMPGCN_NEG_SAMPLING`, `COMPGCN_AUC_GUARDRAIL`).
- Modify: `backend/src/gnn_loader.py` — label fetch, `data.node_type`, return tuple gains `label_to_id`.
- Modify: `backend/src/gnn_module.py` — `_build_type_pools`, `_sample_negative_edges` + `_evaluate_mrr` kwargs, dual-MRR eval in `run_audit`, AUC guardrail, extended `AuditRun` metadata, `recover_from_checkpoint` parity.
- Create: `backend/tests/test_gnn_neg_sampling.py` — unit tests for label pools and type-aware sampling.
- Modify: `backend/tests/test_gnn_neg_sampling.py` — add AUC guardrail test (same file, added later).
- Create: `backend/run_logs/type_aware_audit.py` — tagged launcher (sets env, runs audit, prints one-line summary).
- Modify: `backend/TUNING_LOG.md` — append Run 7 template, filled after full run completes.

---

## Task 1: Add config flags

**Files:**
- Modify: `backend/src/config.py:64` (insert after existing CompGCN block)

- [ ] **Step 1: Read current config surface**

Run: `grep -n "COMPGCN_" backend/src/config.py`
Expected output includes `COMPGCN_LOSS`, `COMPGCN_BPR_MARGIN` on consecutive lines near line 62–63.

- [ ] **Step 2: Insert the two new constants after `COMPGCN_BPR_MARGIN`**

Edit `backend/src/config.py`. Locate:
```python
    COMPGCN_BPR_MARGIN = float(os.getenv("COMPGCN_BPR_MARGIN", 0.0))
    LEGAL_NODE_TYPES = _parse_csv_env(
```

Replace with:
```python
    COMPGCN_BPR_MARGIN = float(os.getenv("COMPGCN_BPR_MARGIN", 0.0))
    # Negative sampling strategy: "uniform" (draw from all nodes) or
    # "type_aware" (draw only from nodes sharing the corrupted endpoint's
    # schema label). Type-aware produces harder negatives → better ranking
    # (MRR), but requires labels fetched by GNNLoader.
    COMPGCN_NEG_SAMPLING = os.getenv("COMPGCN_NEG_SAMPLING", "type_aware").lower()
    # AUC-ROC guardrail: if the trained model scores below this on validation,
    # run_audit skips the Neo4j score write-back so a regressed model never
    # clobbers production plausibility values. Paper target is 0.95.
    COMPGCN_AUC_GUARDRAIL = float(os.getenv("COMPGCN_AUC_GUARDRAIL", 0.95))
    LEGAL_NODE_TYPES = _parse_csv_env(
```

- [ ] **Step 3: Verify config loads**

Run: `cd backend && python -c "from src.config import Config; print(Config.COMPGCN_NEG_SAMPLING, Config.COMPGCN_AUC_GUARDRAIL)"`
Expected: `type_aware 0.95`

- [ ] **Step 4: Commit**

```bash
git add backend/src/config.py
git commit -m "feat(gnn): add COMPGCN_NEG_SAMPLING and COMPGCN_AUC_GUARDRAIL config flags"
```

---

## Task 2: Extend `GNNLoader` to fetch node labels

**Files:**
- Modify: `backend/src/gnn_loader.py:11-106` (add label fetch, build `node_type`, extend return tuple)

- [ ] **Step 1: Write the failing test**

Create `backend/tests/test_gnn_loader_labels.py`:
```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && pytest tests/test_gnn_loader_labels.py -v`
Expected: FAIL — current `fetch_graph_data` returns a 3-tuple, not 4-tuple; no `data.node_type` attribute.

- [ ] **Step 3: Implement the loader change**

Edit `backend/src/gnn_loader.py`. Replace the body of `fetch_graph_data` from the start of the try-block through the return statement with this version:

```python
    def fetch_graph_data(self):
        """Fetches nodes, relationships, and schema labels for GNN processing.

        Returns (data, rel_types, node_id_map, label_to_id) where
        data.node_type[i] is the integer id of node i's primary schema label
        (sentinel -1 when no label matches Config.LEGAL_NODE_TYPES).
        """
        dim = Config.EMBEDDING_DIMENSION
        zero_vec = [0.0] * dim

        try:
            with self.driver.session(database=Config.NEO4J_DATABASE) as session:
                logger.info("Fetching relationships for GNN...")
                rel_query = """
                MATCH (s)-[r]->(t)
                WHERE type(r) <> 'FROM_CHUNK'
                RETURN elementId(r) as rel_id, elementId(s) as source, elementId(t) as target, type(r) as type
                """
                rel_records = list(session.run(rel_query))

                if not rel_records:
                    logger.warning("No relationships found for GNN audit.")
                    return None

                node_ids = set()
                for rec in rel_records:
                    node_ids.add(rec["source"])
                    node_ids.add(rec["target"])

                logger.info("Fetching node embeddings for %s nodes...", len(node_ids))
                emb_query = """
                MATCH (n)
                WHERE elementId(n) IN $node_ids
                  AND n.embedding IS NOT NULL
                  AND n.embedding_model = $embedding_model
                  AND coalesce(n.embedding_dimension, 0) = $embedding_dimension
                RETURN elementId(n) as id, n.embedding as embedding
                """
                emb_result = session.run(
                    emb_query,
                    node_ids=list(node_ids),
                    embedding_model=Config.DISTILBERT_MODEL,
                    embedding_dimension=dim,
                )
                id_to_emb = {r["id"]: r["embedding"] for r in emb_result}

                logger.info("Fetching node labels for %s nodes...", len(node_ids))
                label_query = """
                MATCH (n)
                WHERE elementId(n) IN $node_ids
                RETURN elementId(n) as id, labels(n) as labels
                """
                label_result = session.run(label_query, node_ids=list(node_ids))
                id_to_labels = {r["id"]: list(r["labels"] or []) for r in label_result}

                # Deterministic label→id mapping (alphabetical) so run-to-run
                # pool indices are comparable across audits.
                schema_labels = sorted(set(Config.LEGAL_NODE_TYPES))
                label_to_id = {name: i for i, name in enumerate(schema_labels)}

                node_id_map = {}
                nodes = []
                node_type_list: list[int] = []
                for i, nid in enumerate(sorted(node_ids)):
                    node_id_map[nid] = i
                    nodes.append(id_to_emb.get(nid, zero_vec))
                    # Pick first label that's in the schema. Sentinel -1 when
                    # no match — sampler falls back to uniform for those.
                    raw_labels = id_to_labels.get(nid, [])
                    primary = next((lbl for lbl in raw_labels if lbl in label_to_id), None)
                    node_type_list.append(label_to_id[primary] if primary is not None else -1)

                rel_types = {}
                rel_type_counter = 0
                edge_index = [[], []]
                edge_type_list = []
                edge_rel_id = []

                for rec in rel_records:
                    src = rec["source"]
                    tgt = rec["target"]
                    rtype = rec["type"]
                    if src not in node_id_map or tgt not in node_id_map:
                        continue
                    edge_index[0].append(node_id_map[src])
                    edge_index[1].append(node_id_map[tgt])
                    if rtype not in rel_types:
                        rel_types[rtype] = rel_type_counter
                        rel_type_counter += 1
                    edge_type_list.append(rel_types[rtype])
                    edge_rel_id.append(rec["rel_id"])

                if not edge_type_list:
                    logger.warning("No valid edges for GNN audit.")
                    return None

                x = torch.tensor(nodes, dtype=torch.float)
                norms = x.norm(dim=1, keepdim=True).clamp_min(1e-8)
                x = x / norms
                edge_index_t = torch.tensor(edge_index, dtype=torch.long)
                edge_type_t = torch.tensor(edge_type_list, dtype=torch.long)
                node_type_t = torch.tensor(node_type_list, dtype=torch.long)

                embedded_count = sum(1 for nid in node_ids if nid in id_to_emb)
                unlabeled = int((node_type_t == -1).sum().item())
                logger.info(
                    "Loaded graph: %s nodes (%s embedded, %s unlabeled), %s edges.",
                    len(nodes), embedded_count, unlabeled, len(edge_type_list),
                )

                data = Data(x=x, edge_index=edge_index_t, edge_type=edge_type_t)
                data.node_type = node_type_t
                data.edge_rel_id = edge_rel_id
                return data, rel_types, node_id_map, label_to_id

        except Exception as e:
            logger.error(f"Error loading GNN data: {str(e)}")
            return None
```

Also update the `__main__` block at the bottom of the same file:
```python
if __name__ == "__main__":
    loader = GNNLoader()
    result = loader.fetch_graph_data()
    if result:
        data, rel_types, node_id_map, label_to_id = result
        logger.info("Graph Data: %s", data)
        logger.info("Rel Types: %s", rel_types)
        logger.info("Label map: %s", label_to_id)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd backend && pytest tests/test_gnn_loader_labels.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add backend/src/gnn_loader.py backend/tests/test_gnn_loader_labels.py
git commit -m "feat(gnn-loader): fetch node labels and emit data.node_type tensor"
```

---

## Task 3: Add `_build_type_pools` helper to `gnn_module`

**Files:**
- Create: `backend/tests/test_gnn_neg_sampling.py` (initial structure + pool test)
- Modify: `backend/src/gnn_module.py` (add `_build_type_pools` near the top with other helpers, before `_sample_negative_edges`)

- [ ] **Step 1: Write the failing test**

Create `backend/tests/test_gnn_neg_sampling.py`:
```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && pytest tests/test_gnn_neg_sampling.py -v`
Expected: FAIL — `cannot import name '_build_type_pools'`.

- [ ] **Step 3: Implement `_build_type_pools`**

Edit `backend/src/gnn_module.py`. Insert this function immediately after `_split_edge_indices` and before `_sample_negative_edges`:

```python
def _build_type_pools(node_type: torch.Tensor) -> dict[int, torch.Tensor]:
    """Partition node indices by schema-label id.

    Pools with fewer than 2 nodes are dropped — the sampler falls back to
    uniform sampling for endpoints whose label has no pool. The sentinel
    label -1 (unlabeled / not-in-schema nodes) is always excluded.
    """
    pools: dict[int, torch.Tensor] = {}
    unique_labels = torch.unique(node_type).tolist()
    for label_id in unique_labels:
        if label_id == -1:
            continue
        mask = node_type == label_id
        indices = torch.nonzero(mask, as_tuple=False).squeeze(1)
        if indices.numel() < 2:
            continue
        pools[int(label_id)] = indices.to(torch.long)
    return pools
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd backend && pytest tests/test_gnn_neg_sampling.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add backend/src/gnn_module.py backend/tests/test_gnn_neg_sampling.py
git commit -m "feat(gnn): add _build_type_pools helper for label-partitioned negatives"
```

---

## Task 4: Extend `_sample_negative_edges` with type-aware mode

**Files:**
- Modify: `backend/src/gnn_module.py:58-106` (the `_sample_negative_edges` function)
- Modify: `backend/tests/test_gnn_neg_sampling.py` (add two tests)

- [ ] **Step 1: Write the failing tests**

Append to `backend/tests/test_gnn_neg_sampling.py`:
```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd backend && pytest tests/test_gnn_neg_sampling.py -v`
Expected: FAIL — `_sample_negative_edges` does not yet accept `node_type` / `type_pools` kwargs.

- [ ] **Step 3: Replace `_sample_negative_edges` implementation**

Edit `backend/src/gnn_module.py`. Replace the entire body of `_sample_negative_edges` with:

```python
def _sample_negative_edges(
    edge_index: torch.Tensor,
    edge_type: torch.Tensor,
    num_nodes: int,
    num_negatives: int = 1,
    positive_triples: set[tuple[int, int, int]] | None = None,
    node_type: torch.Tensor | None = None,
    type_pools: dict[int, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample negatives by corrupting head or tail.

    When ``node_type`` and ``type_pools`` are both provided, the corrupted
    endpoint is drawn from the pool of nodes sharing that endpoint's schema
    label (type-aware / hard negatives). When the label has no pool (rare
    label, singleton, or sentinel -1), the sampler falls back to uniform
    corruption for that specific endpoint.

    When ``positive_triples`` is set, (src, dst, rel) matches are rejected
    and re-sampled up to 20 times per attempt.
    """
    if edge_index.size(1) == 0:
        return edge_index, edge_type

    type_aware = node_type is not None and type_pools is not None

    neg_edges_list = []
    neg_types_list = []
    rel_cpu = edge_type.tolist() if positive_triples is not None else None

    def _draw_replacements(original_nodes: torch.Tensor) -> torch.Tensor:
        """Vectorized replacement draws: uniform, or pool-indexed per label."""
        if not type_aware:
            return torch.randint(0, num_nodes, (original_nodes.size(0),), dtype=torch.long)

        replacements = torch.randint(0, num_nodes, (original_nodes.size(0),), dtype=torch.long)
        orig_labels = node_type[original_nodes]
        # For each unique label with a pool, batch-sample that label's entries.
        for label_id, pool in type_pools.items():
            mask = orig_labels == label_id
            n = int(mask.sum().item())
            if n == 0:
                continue
            pick = torch.randint(0, pool.size(0), (n,), dtype=torch.long)
            replacements[mask] = pool[pick]
        return replacements

    for _ in range(num_negatives):
        negative_edges = edge_index.clone()
        corrupt_tail_mask = torch.rand(edge_index.size(1)) > 0.5
        max_retries = 20
        for attempt in range(max_retries):
            # Head replacements drawn against the original SOURCE labels,
            # tail replacements drawn against the original TARGET labels.
            head_replacements = _draw_replacements(edge_index[0])
            tail_replacements = _draw_replacements(edge_index[1])
            negative_edges[1, corrupt_tail_mask] = tail_replacements[corrupt_tail_mask]
            negative_edges[0, ~corrupt_tail_mask] = head_replacements[~corrupt_tail_mask]
            if positive_triples is None:
                break
            srcs = negative_edges[0].tolist()
            dsts = negative_edges[1].tolist()
            bad = [
                i
                for i in range(len(rel_cpu))
                if (srcs[i], dsts[i], rel_cpu[i]) in positive_triples
            ]
            if not bad:
                break
            # Re-sample only the bad indices, honoring type_aware if active.
            if type_aware:
                # Draw one-at-a-time to preserve per-edge label-awareness.
                for i in bad:
                    if corrupt_tail_mask[i]:
                        orig = int(edge_index[1, i].item())
                        label_id = int(node_type[orig].item())
                        pool = type_pools.get(label_id)
                        if pool is not None and pool.size(0) >= 1:
                            new = int(pool[torch.randint(0, pool.size(0), (1,)).item()].item())
                        else:
                            new = int(torch.randint(0, num_nodes, (1,)).item())
                        negative_edges[1, i] = new
                    else:
                        orig = int(edge_index[0, i].item())
                        label_id = int(node_type[orig].item())
                        pool = type_pools.get(label_id)
                        if pool is not None and pool.size(0) >= 1:
                            new = int(pool[torch.randint(0, pool.size(0), (1,)).item()].item())
                        else:
                            new = int(torch.randint(0, num_nodes, (1,)).item())
                        negative_edges[0, i] = new
            else:
                for i in bad:
                    new = int(torch.randint(0, num_nodes, (1,), dtype=torch.long).item())
                    if corrupt_tail_mask[i]:
                        negative_edges[1, i] = new
                    else:
                        negative_edges[0, i] = new
        neg_edges_list.append(negative_edges)
        neg_types_list.append(edge_type.clone())

    neg_edges = torch.cat(neg_edges_list, dim=1)
    neg_types = torch.cat(neg_types_list, dim=0)
    return neg_edges, neg_types
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd backend && pytest tests/test_gnn_neg_sampling.py -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add backend/src/gnn_module.py backend/tests/test_gnn_neg_sampling.py
git commit -m "feat(gnn): type-aware negative sampling in _sample_negative_edges"
```

---

## Task 5: Thread `node_type` / `type_pools` through `_evaluate_auc` and `_evaluate_mrr`

**Files:**
- Modify: `backend/src/gnn_module.py:187-240` (`_evaluate_auc` and `_evaluate_mrr`)

- [ ] **Step 1: Write the failing test**

Append to `backend/tests/test_gnn_neg_sampling.py`:
```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && pytest tests/test_gnn_neg_sampling.py::test_evaluate_mrr_accepts_type_aware_kwargs -v`
Expected: FAIL — parameters missing.

- [ ] **Step 3: Extend both eval functions**

Edit `backend/src/gnn_module.py`. Replace `_evaluate_auc` with:

```python
def _evaluate_auc(
    model: CompGCNAuditModel,
    data,
    edge_indices: torch.Tensor,
    neg_ratio: int = 1,
    positive_triples: set[tuple[int, int, int]] | None = None,
    node_type: torch.Tensor | None = None,
    type_pools: dict[int, torch.Tensor] | None = None,
) -> float | None:
    if edge_indices.numel() == 0:
        return None

    if positive_triples is None:
        positive_triples = _build_positive_triples(data)

    encoded_nodes = model.encode(data.x, data.edge_index, data.edge_type)
    pos_logits = model.edge_logits(
        encoded_nodes,
        data.edge_index[:, edge_indices],
        data.edge_type[edge_indices],
    )
    neg_edges, neg_types = _sample_negative_edges(
        data.edge_index[:, edge_indices],
        data.edge_type[edge_indices],
        data.x.size(0),
        num_negatives=neg_ratio,
        positive_triples=positive_triples,
        node_type=node_type,
        type_pools=type_pools,
    )
    neg_logits = model.edge_logits(encoded_nodes, neg_edges, neg_types)
    labels = torch.cat([torch.ones_like(pos_logits), torch.zeros_like(neg_logits)])
    probabilities = torch.sigmoid(torch.cat([pos_logits, neg_logits]))
    return _compute_auc_roc(labels, probabilities)
```

Replace `_evaluate_mrr` with:

```python
def _evaluate_mrr(
    model: CompGCNAuditModel,
    data,
    edge_indices: torch.Tensor,
    neg_ratio: int = 5,
    positive_triples: set[tuple[int, int, int]] | None = None,
    node_type: torch.Tensor | None = None,
    type_pools: dict[int, torch.Tensor] | None = None,
) -> float | None:
    """Mean Reciprocal Rank: for each positive edge, rank it among negatives."""
    if edge_indices.numel() == 0:
        return None

    if positive_triples is None:
        positive_triples = _build_positive_triples(data)

    encoded_nodes = model.encode(data.x, data.edge_index, data.edge_type)
    reciprocal_ranks = []

    for i in range(edge_indices.numel()):
        idx = edge_indices[i].item()
        pos_edge = data.edge_index[:, idx : idx + 1]
        pos_type = data.edge_type[idx : idx + 1]
        pos_logit = model.edge_logits(encoded_nodes, pos_edge, pos_type)

        neg_edges, neg_types = _sample_negative_edges(
            pos_edge,
            pos_type,
            data.x.size(0),
            num_negatives=max(neg_ratio, 5),
            positive_triples=positive_triples,
            node_type=node_type,
            type_pools=type_pools,
        )
        neg_logits = model.edge_logits(encoded_nodes, neg_edges, neg_types)

        all_scores = torch.cat([pos_logit, neg_logits])
        sorted_desc = torch.argsort(all_scores, descending=True)
        pos_rank = (sorted_desc == 0).nonzero(as_tuple=True)[0].item() + 1
        reciprocal_ranks.append(1.0 / pos_rank)

    return float(sum(reciprocal_ranks) / len(reciprocal_ranks)) if reciprocal_ranks else None
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd backend && pytest tests/test_gnn_neg_sampling.py -v`
Expected: PASS (5 tests).

- [ ] **Step 5: Commit**

```bash
git add backend/src/gnn_module.py backend/tests/test_gnn_neg_sampling.py
git commit -m "feat(gnn): thread type-aware kwargs through _evaluate_auc/_evaluate_mrr"
```

---

## Task 6: Wire type-aware sampling into `run_audit` with dual-MRR eval

**Files:**
- Modify: `backend/src/gnn_module.py:243-517` (`run_audit`)

- [ ] **Step 1: Update the `run_audit` prologue — unpack the new loader tuple and build pools**

Edit `backend/src/gnn_module.py`. Find:
```python
    data, rel_types, _node_id_map = result
```
Replace with:
```python
    data, rel_types, _node_id_map, label_to_id = result
```

Then find:
```python
    positive_triples = _build_positive_triples(data)
```
Insert immediately after:
```python
    neg_sampling_mode = Config.COMPGCN_NEG_SAMPLING
    node_type_tensor = data.node_type if neg_sampling_mode == "type_aware" else None
    type_pools = _build_type_pools(data.node_type) if neg_sampling_mode == "type_aware" else None

    if neg_sampling_mode == "type_aware":
        pool_sizes = {
            lbl: int(type_pools.get(i, torch.empty(0)).numel())
            for lbl, i in label_to_id.items()
        }
        logger.info("Type-aware negative sampling active. Pool sizes: %s", pool_sizes)
        if not type_pools:
            logger.warning("No label pools built — falling back to uniform sampling.")
            node_type_tensor = None
            type_pools = None
    else:
        pool_sizes = {}
        logger.info("Uniform negative sampling active.")
```

- [ ] **Step 2: Pass the kwargs into every `_sample_negative_edges` / `_evaluate_auc` / `_evaluate_mrr` call inside `run_audit`**

In the same function, locate the training-loop `_sample_negative_edges` call and update it:
```python
        neg_edges, neg_types = _sample_negative_edges(
            data.edge_index[:, train_idx],
            data.edge_type[train_idx],
            data.x.size(0),
            num_negatives=neg_ratio,
            positive_triples=positive_triples,
            node_type=node_type_tensor,
            type_pools=type_pools,
        )
```

Then locate the two per-epoch `_evaluate_auc` calls and update both:
```python
        with torch.no_grad():
            current_auc = _evaluate_auc(
                model, data, val_idx, neg_ratio, positive_triples,
                node_type=node_type_tensor, type_pools=type_pools,
            )
        if current_auc is None:
            current_auc = _evaluate_auc(
                model, data, train_idx, neg_ratio, positive_triples,
                node_type=node_type_tensor, type_pools=type_pools,
            )
```

- [ ] **Step 3: Replace the post-training eval block with dual MRR + AUC guardrail**

Find the existing post-training eval block:
```python
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        encoded_nodes = model.encode(data.x, data.edge_index, data.edge_type)
        plausibility_scores = model.edge_scores(encoded_nodes, data.edge_index, data.edge_type)
        final_auc = _evaluate_auc(model, data, val_idx, neg_ratio, positive_triples)
        if final_auc is None:
            final_auc = _evaluate_auc(model, data, train_idx, neg_ratio, positive_triples)
        final_mrr = _evaluate_mrr(model, data, val_idx, neg_ratio, positive_triples)
        if final_mrr is None:
            final_mrr = _evaluate_mrr(model, data, train_idx, neg_ratio, positive_triples)
```

Replace with:
```python
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        encoded_nodes = model.encode(data.x, data.edge_index, data.edge_type)
        plausibility_scores = model.edge_scores(encoded_nodes, data.edge_index, data.edge_type)

        final_auc = _evaluate_auc(
            model, data, val_idx, neg_ratio, positive_triples,
            node_type=node_type_tensor, type_pools=type_pools,
        )
        if final_auc is None:
            final_auc = _evaluate_auc(
                model, data, train_idx, neg_ratio, positive_triples,
                node_type=node_type_tensor, type_pools=type_pools,
            )

        # Dual MRR eval: uniform (apples-to-apples vs Run 6 baseline) and
        # type-aware (intended metric). Both run regardless of training mode
        # so the thesis table shows the full story.
        uniform_pools_for_eval = None
        type_pools_for_eval = _build_type_pools(data.node_type)

        mrr_uniform = _evaluate_mrr(
            model, data, val_idx, neg_ratio, positive_triples,
            node_type=None, type_pools=None,
        )
        if mrr_uniform is None:
            mrr_uniform = _evaluate_mrr(
                model, data, train_idx, neg_ratio, positive_triples,
                node_type=None, type_pools=None,
            )

        mrr_type_aware = _evaluate_mrr(
            model, data, val_idx, neg_ratio, positive_triples,
            node_type=data.node_type, type_pools=type_pools_for_eval,
        )
        if mrr_type_aware is None:
            mrr_type_aware = _evaluate_mrr(
                model, data, train_idx, neg_ratio, positive_triples,
                node_type=data.node_type, type_pools=type_pools_for_eval,
            )

        # Pick headline MRR based on training mode: if trained type-aware,
        # the type-aware MRR is what we ship; if trained uniform, uniform.
        final_mrr = mrr_type_aware if neg_sampling_mode == "type_aware" else mrr_uniform

    auc_guardrail = Config.COMPGCN_AUC_GUARDRAIL
    guardrail_tripped = final_auc is not None and final_auc < auc_guardrail
    if guardrail_tripped:
        logger.warning(
            "AUC guardrail TRIPPED: final_auc=%.4f < threshold %.4f. "
            "Skipping Neo4j score write-back to protect production plausibility values.",
            final_auc, auc_guardrail,
        )
```

- [ ] **Step 4: Update the `_training_history` block to include the new metrics**

Find:
```python
    with _training_history_lock:
        global _training_history
        _training_history = {
            "epochs": epoch_metrics,
            "early_stop_epoch": (epoch + 1) if patience_counter >= Config.COMPGCN_PATIENCE else None,
            "best_epoch": best_epoch_idx,
            "final_auc_roc": round(final_auc, 4) if final_auc is not None else None,
            "final_mrr": round(final_mrr, 4) if final_mrr is not None else None,
            "loss_mode": loss_mode,
            "bpr_margin": bpr_margin,
        }
```

Replace with:
```python
    with _training_history_lock:
        global _training_history
        _training_history = {
            "epochs": epoch_metrics,
            "early_stop_epoch": (epoch + 1) if patience_counter >= Config.COMPGCN_PATIENCE else None,
            "best_epoch": best_epoch_idx,
            "final_auc_roc": round(final_auc, 4) if final_auc is not None else None,
            "final_mrr": round(final_mrr, 4) if final_mrr is not None else None,
            "mrr_uniform": round(mrr_uniform, 4) if mrr_uniform is not None else None,
            "mrr_type_aware": round(mrr_type_aware, 4) if mrr_type_aware is not None else None,
            "loss_mode": loss_mode,
            "bpr_margin": bpr_margin,
            "neg_sampling": neg_sampling_mode,
            "label_pool_sizes": pool_sizes,
            "auc_guardrail": auc_guardrail,
            "guardrail_tripped": guardrail_tripped,
        }
```

- [ ] **Step 5: Gate the Neo4j sync on the guardrail and extend AuditRun metadata**

Find the block starting with:
```python
    driver = DatabaseManager.refresh()
    audit_completed_at = _utc_now_iso()

    with driver.session(database=Config.NEO4J_DATABASE) as session:
        logger.info("Syncing CompGCN plausibility scores to Neo4j...")
```

Replace the entire `with driver.session(...)` block (from that `with` line down through `logger.info("Neo4j CompGCN audit sync complete.")` inclusive) with:
```python
    driver = DatabaseManager.refresh()
    audit_completed_at = _utc_now_iso()
    import json as _json  # local alias — top-level json already imported

    with driver.session(database=Config.NEO4J_DATABASE) as session:
        if guardrail_tripped:
            logger.warning(
                "Neo4j plausibility write-back SKIPPED due to AUC guardrail. "
                "AuditRun recorded as aborted_auc_guardrail."
            )
            audit_status = "aborted_auc_guardrail"
        else:
            logger.info("Syncing CompGCN plausibility scores to Neo4j...")
            audit_status = "trained_experimental"
            updates = []
            for i in range(data.edge_index.size(1)):
                updates.append({
                    "rel_id": data.edge_rel_id[i],
                    "score": float(plausibility_scores[i].item()),
                })
            batch_size = 500
            for batch_start in range(0, len(updates), batch_size):
                batch = updates[batch_start:batch_start + batch_size]
                session.run(
                    """
                    UNWIND $updates AS update
                    MATCH ()-[r]->()
                    WHERE elementId(r) = update.rel_id
                    SET r.plausibility_score = update.score,
                        r.audit_score = update.score,
                        r.experimental_score = update.score,
                        r.audit_status = 'trained_experimental',
                        r.audit_mode = 'compgcn',
                        r.audit_model = 'CompGCNAuditModel',
                        r.audit_updated_at = $updated_at
                    """,
                    updates=batch,
                    updated_at=audit_completed_at,
                )

        session.run(
            f"""
            MERGE (run:{Config.AUDIT_RUN_LABEL} {{run_id: $run_id}})
            SET run.status = $status,
                run.audit_mode = 'compgcn',
                run.audit_model = 'CompGCNAuditModel',
                run.started_at = $started_at,
                run.completed_at = $completed_at,
                run.audited_relationships = $audited_relationships,
                run.graph_nodes = $graph_nodes,
                run.graph_relationship_types = $graph_relationship_types,
                run.hidden_channels = $hidden_channels,
                run.epochs = $epochs,
                run.learning_rate = $learning_rate,
                run.weight_decay = $weight_decay,
                run.validation_split = $validation_split,
                run.patience = $patience,
                run.dropout = $dropout,
                run.label_smoothing = $label_smoothing,
                run.grad_clip = $grad_clip,
                run.neg_ratio = $neg_ratio,
                run.loss = $loss,
                run.bpr_margin = $bpr_margin,
                run.neg_sampling = $neg_sampling,
                run.label_pool_sizes = $label_pool_sizes,
                run.auc_guardrail_min = $auc_guardrail_min,
                run.auc_roc = $auc_roc,
                run.mrr = $mrr,
                run.mrr_uniform = $mrr_uniform,
                run.mrr_type_aware = $mrr_type_aware,
                run.train_loss = $train_loss
            """,
            run_id=audit_run_id,
            status=audit_status,
            started_at=audit_started_at,
            completed_at=audit_completed_at,
            audited_relationships=int(data.edge_index.size(1)),
            graph_nodes=int(data.x.size(0)),
            graph_relationship_types=int(num_rels),
            hidden_channels=Config.COMPGCN_HIDDEN_CHANNELS,
            epochs=Config.COMPGCN_EPOCHS,
            learning_rate=Config.COMPGCN_LEARNING_RATE,
            weight_decay=Config.COMPGCN_WEIGHT_DECAY,
            validation_split=Config.COMPGCN_VALIDATION_SPLIT,
            patience=Config.COMPGCN_PATIENCE,
            dropout=Config.COMPGCN_DROPOUT,
            label_smoothing=Config.COMPGCN_LABEL_SMOOTHING,
            grad_clip=Config.COMPGCN_GRAD_CLIP,
            neg_ratio=neg_ratio,
            loss=loss_mode,
            bpr_margin=bpr_margin,
            neg_sampling=neg_sampling_mode,
            label_pool_sizes=_json.dumps(pool_sizes),
            auc_guardrail_min=auc_guardrail,
            auc_roc=final_auc,
            mrr=final_mrr,
            mrr_uniform=mrr_uniform,
            mrr_type_aware=mrr_type_aware,
            train_loss=final_train_loss,
        )
        logger.info("Neo4j CompGCN audit sync complete (status=%s).", audit_status)
```

Also update the completion log line. Find:
```python
    logger.info(
        "CompGCN audit completed for %s relationships with auc_roc=%s mrr=%s",
        len(plausibility_scores),
        f"{final_auc:.4f}" if final_auc is not None else "unavailable",
        f"{final_mrr:.4f}" if final_mrr is not None else "unavailable",
    )
```

Replace with:
```python
    logger.info(
        "CompGCN audit completed for %s relationships: auc_roc=%s, "
        "mrr_uniform=%s, mrr_type_aware=%s, neg_sampling=%s, guardrail_tripped=%s",
        len(plausibility_scores),
        f"{final_auc:.4f}" if final_auc is not None else "unavailable",
        f"{mrr_uniform:.4f}" if mrr_uniform is not None else "unavailable",
        f"{mrr_type_aware:.4f}" if mrr_type_aware is not None else "unavailable",
        neg_sampling_mode,
        guardrail_tripped,
    )
```

- [ ] **Step 6: Fast smoke test — import and run with tiny synthetic data**

Run: `cd backend && python -c "from src.gnn_module import run_audit, _build_type_pools, _sample_negative_edges, _evaluate_mrr; print('imports OK')"`
Expected: `imports OK`

Then run the existing test suite to confirm nothing regressed:
`cd backend && pytest tests/ -v`
Expected: all previously passing tests + 5 new ones still pass.

- [ ] **Step 7: Commit**

```bash
git add backend/src/gnn_module.py
git commit -m "feat(gnn-audit): dual MRR eval, AUC guardrail, type-aware sampling in run_audit"
```

---

## Task 7: Mirror changes in `recover_from_checkpoint`

**Files:**
- Modify: `backend/src/gnn_module.py:526-657` (`recover_from_checkpoint`)

- [ ] **Step 1: Update the loader tuple unpack**

Find inside `recover_from_checkpoint`:
```python
    data, rel_types, _node_id_map = result
```
Replace with:
```python
    data, rel_types, _node_id_map, _label_to_id = result
```

- [ ] **Step 2: Build pools and extend post-eval**

Find:
```python
    neg_ratio = max(1, Config.COMPGCN_NEG_RATIO)
    positive_triples = _build_positive_triples(data)
    _, val_idx = _split_edge_indices(
        data.edge_index.size(1), Config.COMPGCN_VALIDATION_SPLIT
    )
    with torch.no_grad():
        encoded_nodes = model.encode(data.x, data.edge_index, data.edge_type)
        scores = model.edge_scores(encoded_nodes, data.edge_index, data.edge_type)
        final_auc = _evaluate_auc(model, data, val_idx, neg_ratio, positive_triples)
        final_mrr = _evaluate_mrr(model, data, val_idx, neg_ratio, positive_triples)
```

Replace with:
```python
    neg_ratio = max(1, Config.COMPGCN_NEG_RATIO)
    positive_triples = _build_positive_triples(data)
    _, val_idx = _split_edge_indices(
        data.edge_index.size(1), Config.COMPGCN_VALIDATION_SPLIT
    )
    type_pools = _build_type_pools(data.node_type)
    with torch.no_grad():
        encoded_nodes = model.encode(data.x, data.edge_index, data.edge_type)
        scores = model.edge_scores(encoded_nodes, data.edge_index, data.edge_type)
        final_auc = _evaluate_auc(
            model, data, val_idx, neg_ratio, positive_triples,
            node_type=data.node_type, type_pools=type_pools,
        )
        mrr_uniform = _evaluate_mrr(
            model, data, val_idx, neg_ratio, positive_triples,
            node_type=None, type_pools=None,
        )
        mrr_type_aware = _evaluate_mrr(
            model, data, val_idx, neg_ratio, positive_triples,
            node_type=data.node_type, type_pools=type_pools,
        )
        final_mrr = mrr_type_aware if mrr_type_aware is not None else mrr_uniform
```

- [ ] **Step 3: Update the recovery log + training_history + AuditRun write**

Find:
```python
    logger.info(
        "Checkpoint eval: auc=%s mrr=%s",
        f"{final_auc:.4f}" if final_auc is not None else "unavailable",
        f"{final_mrr:.4f}" if final_mrr is not None else "unavailable",
    )
```

Replace with:
```python
    logger.info(
        "Checkpoint eval: auc=%s mrr_uniform=%s mrr_type_aware=%s",
        f"{final_auc:.4f}" if final_auc is not None else "unavailable",
        f"{mrr_uniform:.4f}" if mrr_uniform is not None else "unavailable",
        f"{mrr_type_aware:.4f}" if mrr_type_aware is not None else "unavailable",
    )
```

Then find the `AuditRun` write inside `recover_from_checkpoint`:
```python
        session.run(
            f"""
            MERGE (run:{Config.AUDIT_RUN_LABEL} {{run_id: $run_id}})
            SET run.status = 'recovered_checkpoint',
                run.audit_mode = 'compgcn',
                run.audit_model = 'CompGCNAuditModel',
                run.started_at = $saved_at,
                run.completed_at = $completed_at,
                run.best_epoch = $best_epoch,
                run.auc_roc = $auc_roc,
                run.mrr = $mrr,
                run.loss = $loss
            """,
            run_id=audit_run_id,
            saved_at=meta.get("saved_at"),
            completed_at=audit_completed_at,
            best_epoch=meta.get("best_epoch"),
            auc_roc=final_auc,
            mrr=final_mrr,
            loss=meta.get("loss_mode"),
        )
```

Replace with:
```python
        session.run(
            f"""
            MERGE (run:{Config.AUDIT_RUN_LABEL} {{run_id: $run_id}})
            SET run.status = 'recovered_checkpoint',
                run.audit_mode = 'compgcn',
                run.audit_model = 'CompGCNAuditModel',
                run.started_at = $saved_at,
                run.completed_at = $completed_at,
                run.best_epoch = $best_epoch,
                run.auc_roc = $auc_roc,
                run.mrr = $mrr,
                run.mrr_uniform = $mrr_uniform,
                run.mrr_type_aware = $mrr_type_aware,
                run.loss = $loss
            """,
            run_id=audit_run_id,
            saved_at=meta.get("saved_at"),
            completed_at=audit_completed_at,
            best_epoch=meta.get("best_epoch"),
            auc_roc=final_auc,
            mrr=final_mrr,
            mrr_uniform=mrr_uniform,
            mrr_type_aware=mrr_type_aware,
            loss=meta.get("loss_mode"),
        )
```

Also update the history block at the end of `recover_from_checkpoint`. Find:
```python
        _training_history = {
            "epochs": _training_history.get("epochs", []),
            "best_epoch": meta.get("best_epoch"),
            "final_auc_roc": round(final_auc, 4) if final_auc is not None else None,
            "final_mrr": round(final_mrr, 4) if final_mrr is not None else None,
            "loss_mode": meta.get("loss_mode"),
            "recovered": True,
        }
```

Replace with:
```python
        _training_history = {
            "epochs": _training_history.get("epochs", []),
            "best_epoch": meta.get("best_epoch"),
            "final_auc_roc": round(final_auc, 4) if final_auc is not None else None,
            "final_mrr": round(final_mrr, 4) if final_mrr is not None else None,
            "mrr_uniform": round(mrr_uniform, 4) if mrr_uniform is not None else None,
            "mrr_type_aware": round(mrr_type_aware, 4) if mrr_type_aware is not None else None,
            "loss_mode": meta.get("loss_mode"),
            "recovered": True,
        }
```

Finally update the return dict:
```python
    return {
        "best_epoch": meta.get("best_epoch"),
        "final_auc_roc": final_auc,
        "final_mrr": final_mrr,
        "mrr_uniform": mrr_uniform,
        "mrr_type_aware": mrr_type_aware,
        "saved_at": meta.get("saved_at"),
    }
```

- [ ] **Step 4: Verify imports and run tests**

Run: `cd backend && python -c "from src.gnn_module import recover_from_checkpoint; print('ok')" && pytest tests/ -v`
Expected: import succeeds, all tests pass.

- [ ] **Step 5: Commit**

```bash
git add backend/src/gnn_module.py
git commit -m "feat(gnn): mirror dual-MRR + type-aware changes in recover_from_checkpoint"
```

---

## Task 8: AUC guardrail integration test

**Files:**
- Modify: `backend/tests/test_gnn_neg_sampling.py` (append guardrail test)

- [ ] **Step 1: Write the failing test**

Append to `backend/tests/test_gnn_neg_sampling.py`:
```python
def test_auc_guardrail_skips_neo4j_sync(monkeypatch):
    """If final_auc < Config.COMPGCN_AUC_GUARDRAIL, run_audit must not write scores."""
    from unittest.mock import MagicMock
    import src.gnn_module as gnn_module
    from src.config import Config

    # Build a trivial PyG-like Data object with node_type so the label-aware
    # path exercises cleanly.
    data = MagicMock()
    data.x = torch.randn(6, 4)
    data.edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long)
    data.edge_type = torch.tensor([0, 0, 1, 1], dtype=torch.long)
    data.node_type = torch.tensor([0, 0, 1, 1, 0, 1], dtype=torch.long)
    data.edge_rel_id = ["r0", "r1", "r2", "r3"]

    fake_loader = MagicMock()
    fake_loader.fetch_graph_data.return_value = (data, {"USES": 0, "PROPOSES": 1}, {}, {"A": 0, "B": 1})
    monkeypatch.setattr(gnn_module, "GNNLoader", lambda: fake_loader, raising=False)

    # Force _evaluate_auc to return a regressed value below the guardrail.
    monkeypatch.setattr(gnn_module, "_evaluate_auc", lambda *a, **kw: 0.50)
    monkeypatch.setattr(gnn_module, "_evaluate_mrr", lambda *a, **kw: 0.90)

    # Make training a no-op (1 epoch, trivial loss).
    monkeypatch.setattr(Config, "COMPGCN_EPOCHS", 1, raising=False)
    monkeypatch.setattr(Config, "COMPGCN_PATIENCE", 1, raising=False)
    monkeypatch.setattr(Config, "COMPGCN_AUC_GUARDRAIL", 0.95, raising=False)

    # Mock the Neo4j driver so we can assert the score-sync UNWIND was NOT called.
    session = MagicMock()
    session.__enter__ = lambda self_: self_
    session.__exit__ = lambda *a: False
    driver = MagicMock()
    driver.session.return_value = session
    monkeypatch.setattr(gnn_module.__import__("src.db", fromlist=["DatabaseManager"]).DatabaseManager,
                        "refresh", staticmethod(lambda: driver))

    gnn_module.run_audit()

    # Assert the UNWIND plausibility-sync query was NOT among the calls.
    sync_calls = [c for c in session.run.call_args_list
                  if "plausibility_score" in str(c)]
    assert len(sync_calls) == 0, "guardrail must block plausibility_score write-back"

    # But the AuditRun node should still be written with status aborted_auc_guardrail.
    audit_calls = [c for c in session.run.call_args_list
                   if "AuditRun" in str(c) or ":%s" % Config.AUDIT_RUN_LABEL in str(c)]
    # Just verify at least one MERGE AuditRun query ran.
    assert any("MERGE" in str(c) for c in session.run.call_args_list), \
        "AuditRun metadata must still be persisted when guardrail trips"
```

- [ ] **Step 2: Run the test — it should pass because Task 6 already added the guardrail**

Run: `cd backend && pytest tests/test_gnn_neg_sampling.py::test_auc_guardrail_skips_neo4j_sync -v`
Expected: PASS. If it fails, the Task 6 guardrail wiring is wrong — fix the guardrail branch in `run_audit` before continuing.

- [ ] **Step 3: Commit**

```bash
git add backend/tests/test_gnn_neg_sampling.py
git commit -m "test(gnn): AUC guardrail must block Neo4j score sync on regression"
```

---

## Task 9: Add the `type_aware_audit.py` launcher

**Files:**
- Create: `backend/run_logs/type_aware_audit.py`

- [ ] **Step 1: Create the launcher**

Write `backend/run_logs/type_aware_audit.py`:
```python
"""BPR + type-aware negative sampling audit run.

Sets COMPGCN_LOSS=bpr and COMPGCN_NEG_SAMPLING=type_aware, runs the audit,
prints a single-line summary suitable for greppable TUNING_LOG entries.

Rollback: re-run `bpr_audit.py` to return to uniform-sampling BPR scores.
"""
from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ["COMPGCN_LOSS"] = "bpr"
os.environ["COMPGCN_BPR_MARGIN"] = os.environ.get("COMPGCN_BPR_MARGIN", "0.0")
os.environ["COMPGCN_NEG_SAMPLING"] = "type_aware"


def main():
    from src.gnn_module import run_audit, get_training_history
    from src.evaluation import persist_gnn_metrics

    t0 = time.time()
    print("TYPE_AWARE_AUDIT_BEGIN", flush=True)
    try:
        run_audit()
        th = get_training_history()
        persist_gnn_metrics(th)
        print(
            f"TYPE_AWARE_AUDIT_DONE elapsed_min={(time.time()-t0)/60.0:.2f} "
            f"best_epoch={th.get('best_epoch')} "
            f"auc={th.get('final_auc_roc')} "
            f"mrr_uniform={th.get('mrr_uniform')} "
            f"mrr_type_aware={th.get('mrr_type_aware')} "
            f"guardrail_tripped={th.get('guardrail_tripped')} "
            f"pool_sizes={th.get('label_pool_sizes')}",
            flush=True,
        )
    except Exception as e:
        import traceback
        print(f"TYPE_AWARE_AUDIT_FAILED {type(e).__name__} {e}", flush=True)
        traceback.print_exc()


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke-import the launcher**

Run: `cd backend && python -c "import run_logs.type_aware_audit as m; print(m.__doc__.splitlines()[0])"`
Expected: `BPR + type-aware negative sampling audit run.`

- [ ] **Step 3: Commit**

```bash
git add backend/run_logs/type_aware_audit.py
git commit -m "chore(run_logs): add type_aware_audit.py launcher"
```

---

## Task 10: Full audit run + TUNING_LOG Run 7 entry

This is the single real-world validation. Run on the live Neo4j corpus.

**Files:**
- Produce: `backend/run_logs/audit_type_aware.log` (new, written by the launcher)
- Modify: `backend/TUNING_LOG.md` (append Run 7 section)

- [ ] **Step 1: Run the full audit against the live Neo4j**

Run (from `backend/`):
```bash
python run_logs/type_aware_audit.py 2>&1 | tee run_logs/audit_type_aware.log
```
Expected: log begins with `TYPE_AWARE_AUDIT_BEGIN`, ends with a single `TYPE_AWARE_AUDIT_DONE ...` line including `auc=`, `mrr_uniform=`, `mrr_type_aware=`, `pool_sizes={...}`.

If the run fails with a Windows PyTorch SymInt / SIGSEGV error, re-run once (the Apr-18 fixes reduced but did not eliminate these); if it fails twice, halt and investigate.

- [ ] **Step 2: Run the grounding / faithfulness eval chain to confirm downstream KPIs are preserved**

Run:
```bash
python run_logs/post_audit_eval.py 2>&1 | tee run_logs/eval_chain_type_aware.log
```
Expected: grounding_score ≥ 0.98 and faithfulness_score ≥ 0.95 at τ=0.95 (matching or exceeding Run 6). If grounding drops materially, investigate before claiming success.

- [ ] **Step 3: Record Run 7 in TUNING_LOG.md**

Append to `backend/TUNING_LOG.md`:
```markdown

## Run 7 — 2026-04-19: BPR + Type-Aware Negative Sampling (MRR Closure Attempt)

**Goal.** Close the final open KPI. Run 6 (BPR + uniform negatives) hit AUC 0.9688 and grounding 0.987 but MRR plateaued at 0.886. Swap uniform negative corruption for schema-label-matched corruption so every negative is a same-type candidate — directly attacking the ranking-metric gap.

**Change.** `backend/src/gnn_loader.py` now fetches `labels(n)` per node and emits `data.node_type`. `_sample_negative_edges` accepts `node_type`/`type_pools` kwargs; when both are set, corrupted endpoints are drawn from per-label node pools. `run_audit` builds pools once at the top and threads them through training + eval. Post-training MRR is evaluated twice (uniform + type-aware) for apples-to-apples comparison against Run 6. An AUC guardrail (`COMPGCN_AUC_GUARDRAIL=0.95`) blocks Neo4j score sync if calibration regressed.

**Run config (env).**
- `COMPGCN_LOSS=bpr`
- `COMPGCN_BPR_MARGIN=0.0`
- `COMPGCN_NEG_SAMPLING=type_aware`
- `COMPGCN_AUC_GUARDRAIL=0.95`
- All other hyperparameters identical to Run 6.

**Label pool sizes.** <FILL FROM LAUNCHER OUTPUT — `pool_sizes={...}`>

**Results.**

| Metric | Run 6 (BPR uniform) | Run 7 (BPR + type-aware) | Δ | Target | Status |
|--------|---------------------|---------------------------|---|--------|--------|
| AUC-ROC | 0.9688 | <FILL> | <FILL> | > 0.95 | <FILL> |
| MRR (uniform eval) | 0.8860 | <FILL> | <FILL> | > 0.95 | <FILL> |
| MRR (type-aware eval) | — | <FILL> | — | > 0.95 | <FILL> |
| Grounding @ τ=0.95 | 0.987 | <FILL> | <FILL> | > 0.98 | <FILL> |
| Faithfulness | 0.979 | <FILL> | <FILL> | high | <FILL> |
| Training wall-clock | ~2 min | <FILL> | <FILL> | — | — |

**Interpretation.** <FILL: one paragraph. If MRR hit 0.95 at type-aware eval, say so plainly. If uniform-eval MRR also improved, report magnitude. If only type-aware eval improved, note that the intervention is a ranking-within-type lift and frame as intended. If AUC dropped toward the guardrail, report the margin.>

**Headline.** <FILL: one line — e.g., "All four paper KPIs hit at their canonical thresholds" OR "MRR (type-aware eval) reached X.XXX; uniform-eval MRR Y.YYY remains short — future work: self-adversarial negatives.">

**Thesis table addition (Chapter 4).**

| Run | Loss | Neg. sampling | AUC | MRR (uniform eval) | MRR (type-aware eval) | Grounding |
|-----|------|---------------|-----|--------------------|-----------------------|-----------|
| 4 (BCE baseline) | BCE | Uniform | 0.9502 | 0.8134 | — | — |
| 6 (BPR) | BPR | Uniform | 0.9688 | 0.8860 | — | 0.987 |
| 7 (BPR + type-aware) | BPR | Same-label | <FILL> | <FILL> | <FILL> | <FILL> |
```

Fill the `<FILL>` placeholders from the launcher + eval-chain outputs.

- [ ] **Step 4: Commit the log entry and launcher logs**

```bash
git add backend/TUNING_LOG.md backend/run_logs/audit_type_aware.log backend/run_logs/eval_chain_type_aware.log
git commit -m "docs(tuning): Run 7 — BPR + type-aware negatives, <headline>"
```

Replace `<headline>` with one of:
- `MRR closed at τ-aware eval` (if mrr_type_aware ≥ 0.95)
- `MRR +X.XX lift, uniform eval still short` (if only partial)
- `type-aware sampling did not help, reverting to Run 6 model` (if AUC guardrail tripped)

---

## Task 11: Update memory snapshot for future sessions

**Files:**
- Modify: `C:\Users\Franz Samilo\.claude\projects\C--Users-Franz-Samilo-Desktop-the-remembrance\memory\MEMORY.md`
- Create: `C:\Users\Franz Samilo\.claude\projects\C--Users-Franz-Samilo-Desktop-the-remembrance\memory\project_tuning_session_apr19.md`

- [ ] **Step 1: Write the new memory file**

Save Run 7 outcomes (the filled table and headline) as a new tuning-session memory. Use the same frontmatter shape as `project_tuning_session_apr18.md`. Include: final numbers, whether MRR hit 0.95, any surprise (pool sparsity, guardrail trips, unexpected grounding movement), and the fact that the Apr-18 "MRR future work" item is closed (or partially closed).

- [ ] **Step 2: Add a line to `MEMORY.md`**

Append one bullet referencing the new file with a one-line hook. Keep under 150 characters.

- [ ] **Step 3: No commit — this is the host machine's memory dir, not part of the project repo.**

---

## Self-Review

**Spec coverage check** (against `docs/superpowers/specs/2026-04-19-type-aware-negative-sampling-design.md`):

| Spec section | Task(s) |
|--------------|---------|
| Config flags (`COMPGCN_NEG_SAMPLING`, `COMPGCN_AUC_GUARDRAIL`) | 1 |
| `GNNLoader.fetch_graph_data` label fetch + `data.node_type` + `label_to_id` | 2 |
| `_build_type_pools` helper | 3 |
| `_sample_negative_edges` extension | 4 |
| `_evaluate_auc` / `_evaluate_mrr` pass-through | 5 |
| `run_audit` — pools, dual eval, AUC guardrail, extended AuditRun | 6 |
| `recover_from_checkpoint` parity | 7 |
| Unit tests (pool, type-aware sampling, fallbacks, guardrail) | 3, 4, 8 |
| Launcher `type_aware_audit.py` | 9 |
| Full run + TUNING_LOG Run 7 | 10 |
| Memory snapshot | 11 |

No spec gaps.

**Placeholder scan:** `<FILL>` tokens only appear in Task 10's TUNING_LOG template — those are explicit fill-in-from-live-run points, not plan placeholders. All other tasks contain complete code.

**Type consistency:** `_build_type_pools` returns `dict[int, torch.Tensor]` in Task 3 and is consumed by `_sample_negative_edges` / `_evaluate_auc` / `_evaluate_mrr` with the same type in Tasks 4, 5, 6, 7. `data.node_type` is `LongTensor[num_nodes]` throughout. `label_to_id` is `dict[str, int]` in the loader (Task 2) and in `run_audit` (Task 6). The Task 6 loader unpack uses `_label_to_id` (underscore prefix — unused in the training loop but passed to `pool_sizes` construction, which *is* used). Consistent.
