# Type-Aware Negative Sampling for CompGCN MRR Lift

**Date:** 2026-04-19
**Status:** Design approved, pending implementation plan
**Owner:** Franz Samilo
**Related runs:** `backend/TUNING_LOG.md` Runs 5 & 6 (BCE, BPR baselines)

## Problem

CompGCN link-prediction MRR plateaus at **0.886** after the April 18 BPR ablation
(Run 6). Paper target is **> 0.95**. The other three paper targets are already
hit: AUC-ROC 0.9688, Grounding 0.987, Faithfulness 0.979 — all at τ = 0.95. MRR
is the last open KPI.

Root cause analysis from TUNING_LOG Run 5: MRR is a ranking metric, and
uniform-random negative sampling across all 8 schema label types produces
negatives that are ~87% "obviously wrong type" (e.g. a `Researcher` corrupting
the tail of a `USES(Researcher → Method)` edge where the tail should be a
`Method`). The model never learns to discriminate among plausible same-type
candidates, which is exactly what MRR measures.

## Goals

1. Raise MRR meaningfully on the current corpus; target ≥ 0.95 but accept
   "best-effort single intervention" — if we land 0.92–0.94 we narrate the
   remainder as future work.
2. Preserve the other three KPIs. AUC-ROC must stay ≥ 0.95 (acts as guardrail;
   if it drops we abort the Neo4j sync for that run).
3. Produce thesis-defensible numbers: report MRR under both uniform eval
   negatives (apples-to-apples vs Run 6) and type-aware eval negatives (the
   intended metric).

## Non-goals

- Architectural changes to CompGCN (still 3-layer, LayerNorm, DistMult, 256-dim).
- Loss changes beyond BPR (already in place).
- Self-adversarial negatives, N3 regularization, margin-rank loss. These are
  explicitly scoped as future-work alternatives if this intervention
  underperforms.
- Changes to the inference pipeline or grounding threshold.

## Approach

Replace uniform random negative corruption with **schema-label-matched
corruption** in `_sample_negative_edges`. For each edge, when corrupting the
head or tail, draw the replacement node from the pool of nodes sharing the
original node's schema label, rather than from all nodes.

Example: for `USES(Researcher_A → Method_X)`, the current sampler might corrupt
the tail to `Researcher_B`, `Result_3`, `Metric_Q`, etc. Type-aware corruption
only draws from `{Method_Y, Method_Z, …}`.

Everything else — 3-layer CompGCN + LayerNorm, DistMult composition, BPR loss,
dropout, learning rate, weight decay, grad clipping, negative ratio 15 — stays.

## Components

### 1. `backend/src/gnn_loader.py` — `GNNLoader.fetch_graph_data`

Extend the node-fetch Cypher to return the primary schema label for each node:

- Add a third Cypher query fetching `labels(n)` for each node in `node_ids`.
- For each node, pick the first label that appears in the schema list from
  `Config.LEGAL_NODE_TYPES` (8 types already canonicalized there:
  `__Entity__,Entity,Method,Researcher,Dataset,Concept,Result,Metric`).
- Unknown-label and multi-label-with-no-schema-match cases get sentinel `-1`.
- Build a `label_to_id: dict[str, int]` mapping deterministically (alphabetical)
  and a `data.node_type: LongTensor[num_nodes]` aligned to the existing
  `node_id_map` order.
- Return tuple becomes `(data, rel_types, node_id_map, label_to_id)`; call sites
  updated.

### 2. `backend/src/gnn_module.py` — helpers

**`_build_type_pools(node_type: Tensor) -> dict[int, LongTensor]`**
One-time precompute invoked at the top of `run_audit` / `recover_from_checkpoint`.
For each unique label id, stores a 1-D `LongTensor` of node indices. Pools with
fewer than 2 entries are omitted (the sampler falls back to uniform for those).

**`_sample_negative_edges` — new kwargs**
Add `node_type: Tensor | None = None` and `type_pools: dict | None = None`. When
both are provided:

- For corrupt-tail positions: replacement drawn via
  `type_pools[node_type[dst_i].item()][randint(0, pool_size)]` per edge, falling
  back to `randint(0, num_nodes)` when the pool is missing or too small.
- For corrupt-head positions: symmetric.
- The existing false-negative retry/filter loop (against `positive_triples`) is
  unchanged.

Vectorize where possible: pre-build one `LongTensor` per pool, sample indices
with a single `torch.randint` per pool per corrupt-side per attempt. Per-edge
fallback to uniform is only taken when the pool is empty/too small — expected
to be rare.

**`_evaluate_mrr` — pass-through**
Same new kwargs. The function is already called for both training-val and
full-training-set eval paths; type_pools flow through unchanged.

### 3. `backend/src/gnn_module.py` — `run_audit`

- Read `Config.COMPGCN_NEG_SAMPLING ∈ {uniform, type_aware}`; default
  `type_aware`.
- Call `_build_type_pools(data.node_type)` once before the training loop.
- Pass `node_type` + `type_pools` into both training-loop `_sample_negative_edges`
  calls and the `_evaluate_auc` / `_evaluate_mrr` calls when mode is `type_aware`.
- After training, run MRR eval **twice** regardless of training mode:
  - `mrr_uniform`: `type_pools=None` — baseline-comparable metric.
  - `mrr_type_aware`: `type_pools=type_pools` — intended metric.
  Persist both.
- Log pool sizes at audit start (`label_pool_sizes={'Method': 312, …}`) so the
  thesis write-up can cite them.

### 4. `backend/src/gnn_module.py` — AUC guardrail

Before the Neo4j sync, check `final_auc`. If `final_auc < Config.COMPGCN_AUC_GUARDRAIL`
(default 0.95), log a prominent warning and **skip the score write-back**. The
`AuditRun` node is still recorded with `status='aborted_auc_guardrail'` so the
failed attempt is auditable, but production `plausibility_score` properties are
not clobbered.

### 5. `backend/src/config.py`

Add:

```python
COMPGCN_NEG_SAMPLING = os.getenv("COMPGCN_NEG_SAMPLING", "type_aware")
COMPGCN_AUC_GUARDRAIL = float(os.getenv("COMPGCN_AUC_GUARDRAIL", "0.95"))
```

`Config.LEGAL_NODE_TYPES` already holds the canonical 8-label list and is
reused. No new constant needed.

### 6. `AuditRun` Neo4j node — new properties

- `neg_sampling: str` — `'type_aware'` | `'uniform'`
- `mrr_uniform: float`
- `mrr_type_aware: float`
- `auc_roc_guardrail_min: float` — the threshold used
- `label_pool_sizes: str` — JSON-serialized `dict[label_name, count]`

### 7. `backend/run_logs/type_aware_audit.py` — helper launcher

Mirrors `bpr_audit.py`: sets `COMPGCN_LOSS=bpr`, `COMPGCN_NEG_SAMPLING=type_aware`,
runs audit, writes a tagged row to `TUNING_LOG.md` (Run 7).

## Data flow

1. `GNNLoader.fetch_graph_data` → Cypher fetch nodes + edges **+ labels** →
   `Data(x, edge_index, edge_type, node_type)` + `label_to_id`.
2. `run_audit` builds `type_pools = _build_type_pools(data.node_type)`.
3. Training loop: each epoch, `_sample_negative_edges(..., node_type,
   type_pools)` returns same-label corrupted negatives. BPR loss computed
   exactly as before. Best-AUC checkpointing unchanged.
4. Post-training: MRR evaluated twice (uniform + type-aware), AUC evaluated
   once. AUC guardrail check.
5. If guardrail passes: Neo4j sync of plausibility scores (existing logic),
   plus `AuditRun` with new properties.
6. If guardrail fails: sync skipped, `AuditRun` recorded as aborted.
7. `TUNING_LOG.md` Run 7 section appended with pool sizes, both MRRs, AUC,
   training time, early-stop epoch.

## Fallbacks & error handling

- **Sentinel label (`-1`)**: edge with unlabeled endpoint → uniform fallback for
  that endpoint only; the other endpoint uses its pool if available.
- **Empty/small pool**: if `|pool| < 2` for a given label, uniform fallback for
  corruptions touching that label.
- **All-zero `node_type`**: if the loader cannot attach labels at all (Cypher
  failure, schema mismatch), `run_audit` logs a warning and falls back to
  uniform sampling for the whole run. The run proceeds — we just get the old
  behavior.
- **Empty graph / no relationships**: unchanged — existing guards in
  `fetch_graph_data` and `run_audit` abort cleanly.

## Testing

### Unit tests

Add to `backend/tests/`:

1. **`test_gnn_type_pools.py`** — build a synthetic 3-label 10-node graph,
   assert `_build_type_pools` returns correct partitioning.
2. **`test_gnn_neg_sampling.py`** — with a fixed seed and a synthetic graph,
   assert every sampled negative has matching label and none collide with true
   triples. Also assert the fallback kicks in when pool < 2.

### Integration smoke

A 20-epoch training run with `COMPGCN_NEG_SAMPLING=type_aware`:

- Confirm training completes without Windows PyTorch regressions.
- Confirm MRR (type-aware eval) > MRR (uniform eval) > Run 6 baseline uniform
  (0.886). Any run that fails to beat 0.886 on uniform eval is investigated
  before continuing.
- Confirm AUC guardrail logic fires correctly by forcing `auc_roc < 0.95` in a
  test (monkeypatch eval function) — sync must be skipped.

### Full run

240-epoch BPR + type-aware audit via `run_logs/type_aware_audit.py`:

- Record wall-clock training time (type-aware sampling should not be
  materially slower than uniform — both are `randint` + indexing).
- Record both MRRs, AUC, grounding, faithfulness.
- Append to `TUNING_LOG.md` as Run 7 with BCE-vs-BPR-vs-TypeAware comparison
  table.

## Thesis reporting plan

Chapter 4 table gains a third row:

| Run | Loss | Neg. sampling | MRR (uniform eval) | MRR (type-aware eval) | AUC | Grounding |
|-----|------|---------------|--------------------|-----------------------|-----|-----------|
| 4 (BCE baseline) | BCE | Uniform | 0.8134 | — | 0.9502 | — |
| 6 (BPR) | BPR | Uniform | 0.8860 | — | 0.9688 | 0.987 |
| 7 (BPR + type-aware) | BPR | Same-label | TBD | TBD | TBD | TBD |

Plus a supporting table: per-label node count (pool sizes) — context for the
panel to understand where the negatives are drawn from and why some labels
fall back to uniform.

## Risks

- **Label sparsity**: if `Metric` or `Dataset` pools have ≤ 3 nodes on the real
  corpus, many edges will fall back to uniform, capping the lift. Mitigation:
  print pool sizes at audit start before the training run commits. If sparsity
  is severe, the lift ceiling is capped and we accept whatever we get.
- **Sampling overhead**: per-label indexed sampling is O(1) per draw using a
  tensor of indices per pool. Expected to match uniform sampling speed.
- **Insufficient lift**: if MRR lands in 0.88–0.91, we still have a defensible
  Chapter 4 story ("hard negatives help but pool sparsity limits the lift;
  future work: self-adversarial or per-relation sampling"). The intervention is
  motivated and the ablation is clean.
- **AUC drop**: hard negatives occasionally hurt global calibration. Guardrail
  prevents us from shipping a regressed model. If we repeatedly trip the
  guardrail, the intervention is wrong for this corpus and we revert.

## Open questions

None blocking. Schema labels are already canonicalized in
`Config.LEGAL_NODE_TYPES`; this spec consumes that list directly.
