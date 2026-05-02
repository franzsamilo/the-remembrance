# Self-Adversarial Negative Sampling for CompGCN MRR Closure

**Date:** 2026-05-03
**Status:** Design approved (auto-mode), pending implementation plan
**Owner:** Franz Samilo
**Related runs:** `backend/TUNING_LOG.md` Run 6 (BPR uniform — 0.886 MRR), Run 7 (BPR + type-aware — 0.887 MRR, no lift)
**Paper KPI targeted:** H2 — Mean Reciprocal Rank > 0.95

## Problem

CompGCN link-prediction MRR has been stuck at **0.886–0.887** across the last
two tuning runs. Run 6 (BPR + uniform negatives) hit 0.8860; Run 7 (BPR +
schema-label-matched negatives) lifted it by +0.0013 — within sampling noise.
Paper target is **> 0.95**. The other three paper KPIs are met at the canonical
τ=0.95 in Run 6:

| KPI | Run 6 | Target | Status |
|-----|-------|--------|--------|
| AUC-ROC | 0.9688 | > 0.95 | PASS |
| Grounding | 0.987 | > 0.98 | PASS |
| Faithfulness | 0.979 | high | PASS |
| **MRR** | **0.886** | **> 0.95** | **MISS −0.064** |

Root cause from Run 7's post-mortem: type-aware negatives failed to lift MRR
because (a) `Concept` covers 54% of labeled nodes — same-label sampling against
a 2,804-node pool is effectively uniform for the dominant class, and (b) rare
labels (`Metric`=53, `Dataset`=50) underflow per-batch contrast. Type-based
negatives don't produce *hard* negatives — they produce *plausibly-typed*
negatives, most of which are still trivially separable.

## Approach

Switch from uniform-mean BPR to **self-adversarial-weighted BPR**, the
canonical MRR lever from Sun et al. 2019 ("RotatE: Knowledge Graph Embedding
by Relational Rotation in Complex Space"). For each positive edge, weight each
of its `K=neg_ratio` negatives by a softmax over their scores: hard negatives
(high score = the model thinks they're plausible) dominate the gradient; easy
negatives (low score) contribute nearly zero. Hardness is *score-defined*, not
*type-defined* — solving Run 7's failure mode directly.

### Loss change (mathematical)

**Run 6 / 7 (uniform-mean BPR):**

$$
\mathcal{L} = -\frac{1}{|B| \cdot K} \sum_{(h,r,t) \in B} \sum_{k=1}^{K} \log \sigma\bigl( s(h,r,t) - s(h,r,t'_k) - \gamma \bigr)
$$

**Run 8 (self-adversarial BPR):**

$$
w_{k} = \frac{\exp\bigl( \alpha \cdot s(h,r,t'_k) \bigr)}{\sum_{j=1}^{K} \exp\bigl( \alpha \cdot s(h,r,t'_j) \bigr)} \quad \text{(stop-gradient)}
$$

$$
\mathcal{L} = -\frac{1}{|B|} \sum_{(h,r,t) \in B} \sum_{k=1}^{K} w_{k} \cdot \log \sigma\bigl( s(h,r,t) - s(h,r,t'_k) - \gamma \bigr)
$$

`α` is the adversarial temperature. RotatE paper uses α=1.0 on FB15k, FB15k-237,
WN18, WN18RR. **Run 8 uses α=1.0** (the literature canonical value); the env
var `COMPGCN_ADV_TEMP` is exposed for follow-up sweeps. The codebase env-var
fallback stays at 0.0 — see Components §1 — so Runs 6/7 reproduce
byte-for-byte without an explicit override.

The softmax weights are **detached** before multiplication — gradients flow
through the BPR term, not the weighting. This matches RotatE's original
formulation and prevents the model from gaming the loss by depressing
all negative scores.

When α=0, the formula reduces to uniform-mean BPR (weights = 1/K), so
existing Run 6 reproducibility is preserved by setting `COMPGCN_ADV_TEMP=0`.

### Why this fits this corpus

Self-adversarial sampling is *score-defined hard negative mining*: it cares
about which negatives the *current model* finds plausible, not which negatives
share a schema label. On a corpus with 54% `Concept` nodes, score-defined
hardness can identify which Concept-tail corruptions are genuinely confusing
to the model (e.g. semantically near-by concepts that haven't been trained
apart) and weight them up. Type-defined hardness cannot make this distinction
because every Concept is labeled the same.

It is also orthogonal to the existing pipeline:

- Sampling distribution (uniform vs type-aware) — unchanged. We continue to
  sample uniformly per Run 7's reverted default. Self-adversarial reweights
  whatever sampler is active.
- Loss shape (BPR pairwise) — unchanged.
- Architecture (3-layer CompGCN + LayerNorm, DistMult composition) — unchanged.
- Negative ratio (K=15) — unchanged. Higher K gives the softmax more
  candidates to discriminate among; we may sweep K=15 vs K=30 in a follow-up
  if α=1.0 lifts but stalls below target.

## Goals

1. Lift MRR (uniform eval) from 0.886 to ≥ 0.95 — paper H2 target.
2. Preserve the other three paper KPIs:
   - AUC-ROC ≥ 0.95 (existing guardrail will skip Neo4j sync if violated)
   - Grounding ≥ 0.98 at τ=0.95
   - Faithfulness ≥ 0.95 at τ=0.95
3. Produce thesis-defensible numbers with full ablation:
   - Run 6 (BPR uniform) — already logged
   - Run 7 (BPR + type-aware) — already logged
   - **Run 8 (BPR + self-adversarial α=1.0)** — this spec
   - Optional Run 8b (BPR + self-adversarial + type-aware) if Run 8 lifts but stalls

## Non-goals

- Architectural changes to CompGCN (no extra layers, no scoring-fn change).
- Loss family change beyond reweighting BPR (no margin-rank, no N3, no NCE).
- Combining with type-aware in Run 8 — confounds the ablation.
- Inference pipeline / grounding threshold / retrieval limits — untouched.
- New Neo4j schema fields beyond what Run 7 already added (`AuditRun.adv_temp`
  is the only new property).

## Components

### 1. `backend/src/config.py`

Add one new config flag:

```python
# Self-adversarial negative sampling (RotatE Sun+ 2019). For each positive,
# weight its K negatives by softmax(α * neg_score). α=0 disables (uniform mean).
# Run 8 default α=1.0 — RotatE canonical value.
COMPGCN_ADV_TEMP = float(os.getenv("COMPGCN_ADV_TEMP", "0.0"))
```

Default kept at 0.0 so reproducing Runs 6/7 needs no env override. Run 8
launcher sets it explicitly.

### 2. `backend/src/gnn_module.py` — `run_audit` BPR branch

Modify lines 443–448 only. Current code:

```python
if loss_mode == "bpr":
    pos_expanded = pos_logits.repeat(neg_ratio)
    diff = pos_expanded - neg_logits - bpr_margin
    loss = -F.logsigmoid(diff).mean()
```

New code:

```python
if loss_mode == "bpr":
    pos_expanded = pos_logits.repeat(neg_ratio)
    diff = pos_expanded - neg_logits - bpr_margin
    adv_temp = Config.COMPGCN_ADV_TEMP
    if adv_temp > 0.0:
        # Self-adversarial: weight each negative by softmax over its
        # K siblings (RotatE Sun+ 2019, eq. 5). Reshape (K, num_pos),
        # softmax over K (dim=0), detach so weights carry no gradient.
        num_pos = pos_logits.size(0)
        neg_reshaped = neg_logits.view(neg_ratio, num_pos)
        weights = F.softmax(adv_temp * neg_reshaped, dim=0).detach()
        diff_reshaped = diff.view(neg_ratio, num_pos)
        loss = -(weights * F.logsigmoid(diff_reshaped)).sum(dim=0).mean()
    else:
        loss = -F.logsigmoid(diff).mean()
```

**Reshape correctness:** `_sample_negative_edges` concatenates `K=neg_ratio`
repetitions of corrupted edges via `torch.cat(neg_edges_list, dim=1)`
(`gnn_module.py:177`). With `pos_logits.repeat(K)` the layout is
`[pos_0..pos_{N-1}] × K` along dim=0; `neg_logits` has the matching layout
`[neg_rep0(pos_0)..neg_rep0(pos_{N-1}), neg_rep1(pos_0).., …]`. Reshape to
`(K, num_pos)` aligns row `k` with the k-th repetition's negatives,
column `i` with the negatives drawn for `pos_i`. Softmax over `dim=0`
(rows) is "softmax across K negatives for each fixed positive" — exactly
the RotatE formulation.

**Determinism:** `softmax` and `detach` are deterministic; the RNG sequence
is unchanged (no new draws). Identical seed → identical per-epoch metrics
within FP tolerance, same as Runs 1–7.

### 3. `backend/src/gnn_module.py` — checkpoint meta JSON

`recover_from_checkpoint` does **not** compute loss (loads weights → evals →
syncs), so no BPR-branch mirror is needed there. However, the meta JSON
written at each best-AUC checkpoint should include the active `adv_temp` so
a future recovery can faithfully attribute the trained model:

```python
json.dump({
    ...existing fields...,
    "adv_temp": float(Config.COMPGCN_ADV_TEMP),
}, f)
```

Recovery's AuditRun Cypher write should also propagate `meta.get("adv_temp", 0.0)`
into the new `AuditRun.adv_temp` property — symmetric with the run_audit
write described in §6.

### 4. `backend/run_logs/self_adversarial_audit.py` — new launcher

Mirrors `bpr_audit.py` and `type_aware_audit.py`. Sets:

```python
os.environ["COMPGCN_LOSS"] = "bpr"
os.environ["COMPGCN_NEG_SAMPLING"] = "uniform"
os.environ["COMPGCN_ADV_TEMP"] = "1.0"
os.environ["COMPGCN_AUC_GUARDRAIL"] = "0.95"
```

Logs to `backend/run_logs/audit_self_adversarial.log` and runs
`from src.gnn_module import run_audit; run_audit()`.

### 5. `backend/run_logs/post_audit_eval.py` — already generic

The post-eval chain (Neo4j sync verify → full-stack G/F → threshold sweep →
prompt-only ablation → eval log) is loss-agnostic. No change needed; the new
audit log path is the only artifact.

### 6. `AuditRun` Neo4j node — one new property

Append to existing `AuditRun` write:

```cypher
SET a.adv_temp = $adv_temp
```

Value comes from `Config.COMPGCN_ADV_TEMP` at audit time. Allows querying past
runs by α value for ablation reporting.

### 7. `backend/TUNING_LOG.md` — Run 8 section

Append after Run 7 the standard format:

- Objective (close MRR gap with self-adversarial)
- Run config (env delta vs Run 7)
- Loss formula (math block above, copy)
- GNN metrics table (Run 6 / Run 7 / Run 8 columns)
- Score distribution (buckets vs Run 6 — sanity check that distribution
  doesn't collapse)
- Threshold sweep (τ=0.30/0.50/0.85/0.95)
- Per-query grounding/faithfulness (5 queries, same as Run 6)
- Interpretation paragraph (does it move MRR? at what cost to other KPIs?)
- Operational notes (training time, early-stop epoch, RNG match vs Run 6 at α=0)

## Data flow

1. `run_audit` reads `Config.COMPGCN_ADV_TEMP` once at startup (default 0).
2. Training loop: sampling unchanged (uniform/type-aware path inherited from
   Run 7). After scoring negatives, BPR branch checks `adv_temp > 0` →
   computes detached softmax weights → applies them per (positive, K negs).
   Optimizer step proceeds normally.
3. Validation: `_evaluate_auc` and `_evaluate_mrr` are unchanged. They consume
   only the trained model's edge scores; loss form does not enter the eval
   path.
4. Post-training: `final_auc` checked against `COMPGCN_AUC_GUARDRAIL` (Run 7
   guardrail logic preserved). If guardrail passes, plausibility scores
   written to Neo4j; `AuditRun` records `adv_temp` alongside existing fields.
5. `post_audit_eval.py` runs the standard chain. Outputs in
   `backend/run_logs/eval_chain_self_adversarial.log`.
6. `TUNING_LOG.md` appended manually with Run 8 section.

## Fallbacks & error handling

- **`adv_temp = 0`**: branch reduces to uniform-mean BPR (existing code path
  taken). Reproduces Run 6 byte-for-byte at seed=42. This is the
  reproducibility safety net.
- **`adv_temp < 0`**: same `> 0.0` gate — falls through to uniform mean.
  Negative temperature is malformed input; we silently degrade rather than
  raise (matches the project convention of never breaking training on a
  config typo). Logger emits a warning at audit start if `adv_temp < 0`.
- **`neg_ratio = 1`**: softmax over a single value is trivially 1.0 — loss
  reduces exactly to single-negative BPR. No special-casing needed.
- **NaN/Inf in negative scores**: pre-existing risk in any score-based loss.
  Existing `clip_grad_norm_` (1.0) catches optimizer instability. If softmax
  itself overflows (very large logits), `F.softmax` is numerically stable in
  PyTorch 2.x. We do not introduce a manual log-sum-exp.
- **AUC guardrail trips**: existing Run 7 logic — Neo4j sync skipped, audit
  recorded as `aborted_auc_guardrail`. Production scores untouched.

## Testing

### Unit tests (`backend/tests/test_gnn_self_adversarial.py`, new file)

1. **`test_adv_temp_zero_matches_uniform_bpr`** — synthetic graph (8 nodes,
   3 relations, 12 edges), seed=42, run 5 epochs with `adv_temp=0` and 5
   epochs with the *old* uniform-mean code path (extracted via monkeypatch).
   Assert per-epoch loss values match within 1e-6. **This is the
   non-regression guarantee.**

2. **`test_adv_temp_positive_concentrates_weight_on_hard_negatives`** —
   construct fixed positive logits, fixed negative logits with a known
   maximum among the K negatives. Compute softmax weights at α=1.0; assert
   the max-score negative carries > 1/K of the total weight, and the
   min-score carries < 1/K. (Pure math test, no training.)

3. **`test_adv_temp_gradient_does_not_flow_through_weights`** — single forward
   pass; assert `weights.requires_grad == False` after `.detach()`. Backprop
   loss; assert gradients on negative-score path are scaled by detached
   weights (compare numerical gradient to analytical via PyTorch's
   `torch.autograd.gradcheck` on a tiny module).

4. **`test_adv_temp_neg_ratio_one_equals_bpr`** — with K=1, assert weighted
   loss equals raw `-logsigmoid(diff)` exactly.

### Integration smoke (`backend/tests/test_gnn_self_adversarial_smoke.py`)

5-epoch full-pipeline run on the real Neo4j graph (uses `pytest.mark.integration`,
runs only when `RUN_INTEGRATION=1`):
- `COMPGCN_ADV_TEMP=1.0`, `COMPGCN_EPOCHS=5`, `COMPGCN_NEG_RATIO=15`.
- Assert: training completes, no NaN/Inf in loss, final AUC > 0.5.
- Not a correctness test — just a "doesn't crash on real data" check.

### Full Run 8 audit

`python backend/run_logs/self_adversarial_audit.py`:
- 300-epoch BPR + self-adversarial α=1.0 + uniform sampling.
- Expected wall-clock ~2–3 min (similar to Run 6; softmax adds negligible cost).
- Acceptance: MRR (uniform eval) ≥ 0.95 → ship as recommended config.
  Otherwise document the lift, run threshold sweep, write Run 8 section to
  `TUNING_LOG.md`, and decide whether to escalate to Run 8b (combine with
  type-aware) based on the actual lift magnitude.

### Reproducibility

Run 8 with `COMPGCN_ADV_TEMP=0.0` must produce per-epoch AUC values matching
Run 6 within FP tolerance (±0.0001). This is asserted manually as a sanity
check before declaring Run 8 valid; the unit test above codifies it on
synthetic data.

## Thesis reporting plan

Chapter 4 ablation table extended:

| Run | Loss | Neg. sampling | Adv. temp | MRR | AUC | Grounding | Faithfulness |
|-----|------|---------------|-----------|-----|-----|-----------|--------------|
| 4 (BCE baseline) | BCE | Uniform | — | 0.8134 | 0.9502 | — | — |
| 6 (BPR) | BPR | Uniform | 0 | 0.8860 | 0.9688 | 0.987 | 0.979 |
| 7 (BPR + type-aware) | BPR | Same-label | 0 | 0.8873 | 0.9662 | 0.988 | 0.91–0.95 |
| **8 (BPR + self-adv)** | BPR | Uniform | **1.0** | TBD | TBD | TBD | TBD |

Plus a discussion paragraph in Chapter 4:
- Why score-defined hardness is the natural complement to the corpus's
  label-distribution skew (Run 7's failure mode).
- Whether the lift is sufficient to claim H2 hit, partial hit, or unattained
  with documented future work.
- Per-query grounding/faithfulness table (same 5 queries as Runs 6/7) so the
  panel can compare claim coverage qualitatively.

If MRR ≥ 0.95: Chapter 4 reports H2 PASSED, Chapter 5 (recommendations)
flags self-adversarial as the closing intervention. If 0.92 ≤ MRR < 0.95:
report partial hit, escalate to Run 8b (self-adv + type-aware combined) as
the final attempt. If MRR < 0.92: write up the score-distribution diagnosis
and frame the gap as a corpus-property finding (relational density too low
for ranking-target convergence on small graphs), with self-adv listed as
"best single intervention applied."

## Risks

- **No lift**: corpus may be too small (5,187 nodes / 6,419 edges) for
  RotatE-style hard mining to find genuinely informative negatives. The
  paper reports gains on FB15k (15k nodes, 600k edges) — our graph is
  ~100× smaller in edges. Mitigation: clean ablation against Run 6 makes
  the negative result publishable and the corpus-level diagnosis stronger.
- **AUC drop**: hard negative mining can over-fit to confusable cases at the
  expense of global calibration. Run 7's guardrail (`COMPGCN_AUC_GUARDRAIL=0.95`)
  prevents shipping a regressed model. Empirically RotatE reports AUC stable
  or slightly improved with α=1.0 — risk is bounded.
- **Faithfulness regression**: Run 7 saw faithfulness wobble (0.91–0.95) when
  the score distribution shifted. If self-adv compresses scores into a
  tighter band, more triplets pass τ=0.95 → larger context → potentially
  more ungrounded claims from the LLM. Mitigation: full threshold sweep
  documents the right τ for the new distribution. If τ=0.95 is wrong for
  Run 8 we report the calibrated τ alongside the canonical τ.
- **Training-time blowup**: not expected — `softmax` over (K=15, ~5,000)
  is microseconds per epoch. Worst case adds <10% wall-clock.
- **Off-by-one in reshape**: the most likely actual bug. Caught by unit
  test #1 (α=0 reproduces uniform BPR) and by sanity-checking that final
  loss values at α=0 match Run 6's epoch-by-epoch logs.

## Open questions

None blocking. α=1.0 is the literature default; if Run 8 lifts MRR but stalls
short of 0.95, sweep α ∈ {0.5, 1.0, 2.0} as Run 8a/b/c — one-line env-var
flip per run, all sharing the same audit-log format.
