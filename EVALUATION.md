# The Remembrance Vault — Evaluation & Success Criteria

## Success Criteria

| Metric | Target | Description |
|--------|--------|--------------|
| **Grounding** | ≥ 0.9 | LLM-as-judge: claims traceable to triplets (0–1 normalized) |
| **Faithfulness** | ≥ 0.9 | Ratio of claims supported by triplets |
| **GNN AUC-ROC** | ≥ 0.75 | Link-prediction ranking on held-out edges |
| **GNN MRR** | ≥ 0.75 | Mean Reciprocal Rank for edge ranking |

Run grounding/faithfulness via `POST /evaluate`; GNN metrics come from the audit run.

## Baselines (Ablation)

| Mode | Retrieval | GNN Audit | Triplets | Evidence Trail |
|------|-----------|-----------|----------|----------------|
| **Prompt Only** | Chunk RAG (vector similarity) | No | Pseudo-chunks | Minimal |
| **Graph (no GNN)** | Hybrid vector + graph | No (all edges) | All retrieved | Full |
| **Full Stack** | Hybrid vector + graph | Yes (CompGCN) | Validated only | Full |

### How to Compare

1. **Prompt Only**: Enable "Prompt Only (Ablation)" in Knowledge Discovery. Uses raw PDF chunks, no graph.
2. **Graph**: Use the default mode (graph retrieval). If the graph is unaudited, all triplets are used.
3. **Full Stack**: Run the GNN audit from Backend Config, then query. Only validated triplets are used.

### Expected Results

- **Full Stack** should score higher on grounding/faithfulness (structured extraction + audit).
- **Prompt Only** may answer broadly but lacks provenance and multi-hop discovery.
- Compare by running `POST /evaluate` after queries in each mode (evaluation uses the graph-based generator; for prompt_only comparison, run evaluation with graph mode and compare chat outputs manually).
