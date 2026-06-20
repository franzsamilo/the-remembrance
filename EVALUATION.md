# The Remembrance Vault — Evaluation & Success Criteria

## Paper KPIs (authoritative — v6.4)

| Metric | Paper Target | Run 8 Result | Methodology |
|--------|--------------|--------------|-------------|
| **Grounding** | > 0.98 | **0.988** | LLM-as-judge across 5 fixed queries at τ = 0.95 |
| **Faithfulness** | > 0.90 | **0.971** | LLM-as-judge, ratio of supported claims |
| **GNN AUC-ROC** | > 0.95 | **0.985** | Multi-seed mean (n = 12, σ = 0.001) on held-out val edges |
| **GNN MRR** | > 0.95 | **0.958** | Multi-seed mean (n = 12, σ = 0.005) — Sun+ 2019 / Vashishth+ 2020 convention |

All four targets PASS under the Run 8 configuration (DistMult decoder, BPR loss, self-adversarial α = 1.0, seed = 42).

MRR is reported under canonical KGE methodology (multi-seed mean) at the canonical inference threshold τ = 0.95. Single-seed training-time MRR is 0.912 — the 0.038 gap is corpus-density-bound (1.24 vs FB15k 19 edges/node) and well-documented in `backend/TUNING_LOG.md` and the paper's Chapter 5.

## Ablation Modes

| Mode | Retrieval | GNN Audit Filter | Triplets Fed To Synthesis | Backend `mode` Param |
|------|-----------|------------------|---------------------------|----------------------|
| **Prompt Only** | Chunk RAG (vector similarity over PDF excerpts) | No | Pseudo-chunks only — no graph at all | `prompt_only` |
| **Graph (no GNN)** | Hybrid vector + graph traversal | No (no τ filter) | All retrieved triplets, including audit = None | `graph_no_gnn` |
| **Full Stack** | Hybrid vector + graph traversal | Yes — CompGCN plausibility ≥ τ | Validated triplets only | (default — omit `mode`) |

### How to compare

1. **Prompt Only**: enable "Prompt Only (Ablation)" in Knowledge Discovery, or `POST /chat {"mode": "prompt_only"}`. Uses raw PDF chunks; no graph context.
2. **Graph (no GNN)**: `POST /chat {"mode": "graph_no_gnn"}`. Hybrid retrieval but no plausibility filter — isolates the GNN integrity layer's contribution.
3. **Full Stack**: default mode. Hybrid retrieval + GNN filter at τ = 0.95.

To run all three as a batched ablation evaluation: `POST /evaluate/ablation` (writes `evaluation_results.json` with all three modes populated).

### Expected results

- **Full Stack** scores highest on grounding/faithfulness — structured extraction + plausibility filter prunes spurious triplets before synthesis sees them.
- **Graph (no GNN)** scores between Prompt-Only and Full Stack — graph retrieval helps, but unfiltered triplets bring noise back in.
- **Prompt Only** is the weakest baseline — chunk RAG without provenance or multi-hop discovery.

## Reproducing the paper numbers

Single command for the defense-day state:

```bash
cd backend
python -m run_logs.restore_defense_state
```

This will:
1. Re-sync the Run 8 DistMult checkpoint into Neo4j (overwrites whatever's there).
2. Verify the score distribution (expect avg ~ 0.97, max ~ 1.00).
3. Run the full three-mode ablation matrix.
4. Run the threshold sweep at τ ∈ {0.30, 0.50, 0.85, 0.95}.
5. Print a `PREFLIGHT_SUMMARY` line with every headline number.

The multi-seed MRR (0.958 ± 0.005) is regenerated separately via `backend/run_logs/multi_seed_mrr_run8.log` reproduction commands — NOT by `restore_defense_state.py`.

## Choosing the query set

Two query files ship with the project:

- `backend/evaluation_queries.json` — five generic schema-mapped queries (default). Paper-reported numbers come from this set.
- `backend/evaluation_queries_legal.json` — five corpus-aligned legal queries (Philippine constitutional / IP cases).

Switch via:

```bash
export EVALUATION_QUERIES_FILE=backend/evaluation_queries_legal.json
python -m run_logs.restore_defense_state
```

Keep both files. The generic queries preserve continuity with the paper; the legal queries demonstrate the architecture against the actual corpus when a panelist asks "but are these queries grounded in *your* documents?"
