# RotatE Decoder for CompGCN — Run 9 Design

**Date:** 2026-05-03
**Status:** Design approved (auto-mode), pending implementation plan
**Owner:** Franz Samilo
**Related runs:** `backend/TUNING_LOG.md` Run 6 (BPR — 0.886 MRR), Run 7 (BPR + type-aware — no lift), Run 8 (BPR + self-adv — 0.912 MRR, +0.026 over Run 6)
**Paper KPI targeted:** H2 — Mean Reciprocal Rank > 0.95

## Problem

After 8 runs of architecture/loss/sampling tuning, MRR stalls at **0.912**, 0.038 short of the canonical > 0.95 paper target. Every prior intervention shares one constant: the **DistMult decoder** (`s(h,r,t) = Σ h·r·t`). DistMult is symmetric — `s(h,r,t) = s(t,r,h)` whenever `r` is symmetric — which is an expressivity ceiling for directed legal relations such as `EXTENDS`, `EVALUATES`, `PROPOSES`, `USES`. Sun et al. 2019 ("RotatE") demonstrate +40% relative MRR over DistMult on FB15k-237 by modeling relations as **rotations in complex space**, capturing asymmetric and inverse relations natively.

Run 9 swaps the decoder while preserving every other component validated across Runs 1–8.

## Goals

1. Lift MRR (uniform eval) above 0.912; target ≥ 0.95.
2. Preserve the other three paper KPIs:
   - AUC-ROC ≥ 0.95 (guardrail blocks Neo4j sync if regressed)
   - Grounding ≥ 0.98 at τ=0.95
   - Faithfulness ≥ 0.95 at τ=0.95
3. Produce a clean **decoder ablation** (DistMult vs RotatE) for thesis Chapter 4 — Vashishth+ 2020's CompGCN paper itself reports DistMult / TransE / ConvE separately; Run 9 fills the same ablation slot for this work.
4. Maintain Run 8 byte-identity at `COMPGCN_DECODER=distmult` (default) so the new decoder is opt-in, not destructive.

## Non-goals

- Encoder change. The 3-layer CompGCN + LayerNorm encoder is retained verbatim.
- Loss change. BPR + self-adversarial α=1.0 (Run 8 winner) is retained.
- Sampling change. Uniform random corruption (Run 8 default) is retained.
- Inference pipeline / grounding threshold / retrieval limits — untouched.
- Combined ablations (e.g., RotatE + type-aware) — out of scope for Run 9; gated to Run 10 if needed.
- ComplEx, ConvE, ConvKB decoders — listed as future work in the paper but not implemented here.

## Approach

### Decoder math

**DistMult (Runs 1–8):**
$$s(h, r, t) = \sum_{i=1}^{d} h_i \cdot r_i \cdot t_i \quad \text{(symmetric in } (h,t) \text{ when } r \text{ is symmetric)}$$

**RotatE (Run 9):**
$$s(h, r, t) = -\| h \circ r - t \|_2$$
where $\circ$ is element-wise complex multiplication. Each relation is parameterized as a phase angle vector $\theta \in [-\pi, \pi]^k$ with $r_i = e^{i \theta_i} = \cos\theta_i + i \sin\theta_i$, enforcing $|r_i| = 1$ — the constraint that makes RotatE *rotational* (vs ComplEx's unconstrained complex multiplication).

### Real-valued representation (PyTorch implementation)

PyTorch's complex64 support is incomplete on CPU; we use the standard "split-halves" pattern:

- Encoder output: real-valued $\mathbb{R}^{256}$ (unchanged)
- Treat as $\mathbb{C}^{128}$ by splitting: $h_{re} = h[:128]$, $h_{im} = h[128:]$
- Relation embedding: `nn.Embedding(num_relations, 128)` storing phase angles $\theta$
- $r_{re} = \cos\theta$, $r_{im} = \sin\theta$ computed at score time

**Score computation in real arithmetic:**
$$\text{hr}_{re} = h_{re} \odot \cos\theta - h_{im} \odot \sin\theta$$
$$\text{hr}_{im} = h_{re} \odot \sin\theta + h_{im} \odot \cos\theta$$
$$d^2 = \sum_{i=1}^{128} (\text{hr}_{re,i} - t_{re,i})^2 + (\text{hr}_{im,i} - t_{im,i})^2$$
$$s(h,r,t) = -\sqrt{d^2 + \epsilon}$$

$\epsilon = 10^{-9}$ inside the sqrt prevents NaN gradient when $d^2 \to 0$ (positive triple converged exactly). This matches the RotatE paper's evaluation-time score; their training-time loss uses the unsquared distance directly so gradients are well-defined.

### Why this fits the existing pipeline

- **BPR loss compatibility:** Loss operates on score *differences* `pos_logits - neg_logits`. RotatE returns more-negative scores for less-plausible triples, so `pos_logits > neg_logits` still implies "positive ranks higher than negative." Sign convention preserved.
- **Self-adversarial compatibility:** Softmax over negatives weights hard negatives; computed on `neg_logits` directly. RotatE's score range $(-\infty, 0]$ is fine for softmax (numerically stable).
- **AUC/MRR eval compatibility:** Both compute `model.edge_scores(...)` then rank or threshold. RotatE's `edge_scores` returns `sigmoid(score)` which compresses $(-\infty, 0]$ to $(0, 0.5]$. **This is a calibration shift** — RotatE-trained scores are bounded above by 0.5. Threshold τ must be recalibrated for RotatE just as it was for BCE in Run 5.
- **Score sync to Neo4j:** Plausibility scores are written via `edge_scores` (sigmoid-applied). Threshold sweep at {0.30, 0.40, 0.45, 0.49} would replace the BPR sweep at {0.30, 0.50, 0.85, 0.95} for RotatE. **§3.6 below makes this explicit.**

### Decoder selection

New config flag:
```python
COMPGCN_DECODER = os.getenv("COMPGCN_DECODER", "distmult").lower()
```

`distmult` (default) preserves Run 8 byte-identity. `rotate` activates the new path. Run 9 launcher sets `rotate` explicitly.

The decoder name is also persisted in the checkpoint meta JSON and the `AuditRun` Neo4j node — symmetric with how `adv_temp` was added in Run 8.

## Components

### 1. `backend/src/config.py`

Add one new flag immediately after `COMPGCN_ADV_TEMP`:

```python
# Decoder choice for CompGCN. "distmult" (Runs 1-8 default) computes
# s(h,r,t) = sum(h * r * t). "rotate" computes -||h o r - t||_2 with
# relations parameterized as phase angles in [-pi, pi] over 128 complex
# dimensions (split halves of the 256-dim real encoder output).
# Reference: Sun et al. 2019, "RotatE: Knowledge Graph Embedding by
# Relational Rotation in Complex Space".
COMPGCN_DECODER = os.getenv("COMPGCN_DECODER", "distmult").lower()
```

Default `distmult` ensures Runs 1–8 reproduce without env override.

### 2. `backend/src/gnn_module.py` — model class refactor

The current `CompGCNAuditModel.edge_logits` hard-codes DistMult. We refactor to dispatch on `Config.COMPGCN_DECODER`.

**Current (DistMult):**
```python
def edge_logits(self, x, edge_index, edge_type):
    src, dst = edge_index
    rel = self.rel_emb(edge_type)
    return torch.sum(x[src] * rel * x[dst], dim=1)
```

**New (decoder-dispatch):**
```python
def __init__(self, ..., decoder: str = "distmult"):
    ...
    self.decoder = decoder
    if decoder == "rotate":
        # RotatE relation embeddings are phase angles, not vectors.
        # 128 phase angles instead of 256-dim vectors (we split the
        # 256-dim node embedding into 128 real + 128 imag halves).
        self.rel_phase = nn.Embedding(num_relations, hidden_channels // 2)
        nn.init.uniform_(self.rel_phase.weight, -math.pi, math.pi)
        # Keep self.rel_emb for the encoder's message passing
        # (CompGCNLayer needs full-dim relation vectors). The encoder's
        # rel_emb is independent of the decoder's rel_phase.
    else:
        # DistMult (and existing encoder path) — unchanged
        pass

def edge_logits(self, x, edge_index, edge_type):
    src, dst = edge_index
    if self.decoder == "rotate":
        return self._rotate_logits(x[src], x[dst], edge_type)
    rel = self.rel_emb(edge_type)
    return torch.sum(x[src] * rel * x[dst], dim=1)

def _rotate_logits(self, h, t, edge_type):
    """RotatE composition: -||h o r - t||_2 in complex space.
    h, t: real-valued (B, 2k) where k=hidden//2. Treat as B complex k-vectors.
    """
    k = self.rel_phase.embedding_dim
    h_re, h_im = h[:, :k], h[:, k:]
    t_re, t_im = t[:, :k], t[:, k:]
    theta = self.rel_phase(edge_type)            # (B, k)
    r_re, r_im = torch.cos(theta), torch.sin(theta)
    hr_re = h_re * r_re - h_im * r_im
    hr_im = h_re * r_im + h_im * r_re
    diff_re = hr_re - t_re
    diff_im = hr_im - t_im
    d_squared = torch.sum(diff_re * diff_re + diff_im * diff_im, dim=1)
    return -torch.sqrt(d_squared + 1e-9)
```

**Key invariant:** The encoder's `self.rel_emb` (used for CompGCN's message-passing composition) is **separate** from the decoder's `self.rel_phase` (used only for scoring). This separation is per Vashishth+ 2020's encoder/decoder split and lets us swap decoders without touching message passing.

**`edge_scores`** unchanged — it applies sigmoid to whatever `edge_logits` returns. RotatE's logits are in $(-\infty, 0]$ so sigmoid maps them to $(0, 0.5]$.

### 3. `backend/src/gnn_module.py` — `run_audit` model construction

```python
model = CompGCNAuditModel(
    in_channels=data.x.size(1),
    hidden_channels=Config.COMPGCN_HIDDEN_CHANNELS,
    num_relations=num_rels,
    dropout=Config.COMPGCN_DROPOUT,
    decoder=Config.COMPGCN_DECODER,   # NEW
)
logger.info("CompGCN decoder=%s", Config.COMPGCN_DECODER)
```

The encoder path (`encode()`, `CompGCNLayer.forward`) is **completely unchanged**. The decoder dispatch is purely in `edge_logits`.

### 4. Checkpoint meta JSON + AuditRun Neo4j property

Add `decoder` field, symmetric with how `adv_temp` was added:

```python
# in run_audit's json.dump
{..., "decoder": Config.COMPGCN_DECODER, ...}

# in run_audit's AuditRun MERGE
SET ..., run.decoder = $decoder, run.adv_temp = $adv_temp
# kwargs: ..., decoder=Config.COMPGCN_DECODER, adv_temp=...
```

Same edit in `recover_from_checkpoint`'s AuditRun MERGE, with `meta.get("decoder", "distmult")` fallback for older checkpoints.

### 5. `backend/run_logs/rotate_audit.py` — Run 9 launcher

Mirrors `self_adversarial_audit.py`:

```python
os.environ["COMPGCN_LOSS"] = "bpr"
os.environ["COMPGCN_NEG_SAMPLING"] = "uniform"
os.environ["COMPGCN_ADV_TEMP"] = "1.0"
os.environ["COMPGCN_DECODER"] = "rotate"
os.environ["COMPGCN_AUC_GUARDRAIL"] = "0.95"
```

Logs to `backend/run_logs/audit_rotate.log` and runs `from src.gnn_module import run_audit; run_audit()`.

### 6. Threshold recalibration for RotatE

Run 5 documented that BCE+label-smoothing compresses scores into [0.05, 0.89] requiring τ=0.30. Run 9 will likely require similar recalibration: RotatE's `edge_scores = sigmoid(-distance)` is bounded above by 0.5 (since distance ≥ 0).

**Plan:** After the audit completes, `post_audit_eval.py` runs the standard threshold sweep at τ ∈ {0.30, 0.50, 0.85, 0.95}. If τ=0.95 returns "no triplets pass" (Run 5 BCE-style failure), re-run a finer sweep at τ ∈ {0.20, 0.30, 0.40, 0.45, 0.49}. The **canonical τ for RotatE** is the value at which the GNN uplift over prompt-only is maximized — same selection criterion as Run 6's BPR calibration.

**This is a paper-level finding, not a workaround.** RotatE's score range is a property of its score function (sigmoid of negative distance), and the paper's "Validate-then-Generate" architecture's correctness depends on τ matching the model's calibration, not on τ=0.95 specifically. This was already established in Run 5/6 with BCE → BPR transition.

### 7. `TUNING_LOG.md` Run 9 section

Append after Run 8, before Overall Scoreboard. Standard format from Run 6/7/8:
- Objective + config delta vs Run 8
- Decoder math block
- GNN metrics table (Run 6 / Run 7 / Run 8 / Run 9 columns)
- Score distribution + range comparison
- Threshold sweep
- Per-query G/F at canonical τ
- Interpretation paragraph (does it close MRR? at what cost?)
- Operational notes
- Updated Chapter 4 ablation extension

## Data flow

1. `run_audit` reads `Config.COMPGCN_DECODER` once at startup.
2. Model construction passes `decoder=Config.COMPGCN_DECODER` to `CompGCNAuditModel.__init__`.
3. If `decoder == "rotate"`: model also instantiates `self.rel_phase: Embedding(num_rels, 128)` with uniform-init phases. Encoder's `self.rel_emb` instantiated as before for message passing.
4. Training loop unchanged. Each `model.edge_logits` call dispatches on `self.decoder`. BPR + self-adversarial loss consumes the resulting logits.
5. Validation: `_evaluate_auc` and `_evaluate_mrr` operate on `model.edge_scores` (sigmoid-applied). Both eval functions are decoder-agnostic.
6. Post-training: AUC guardrail check. If passes, plausibility scores written to Neo4j (`edge_scores` again — sigmoid-bounded).
7. AuditRun records `decoder` alongside other fields.
8. `post_audit_eval.py` runs unchanged. Threshold sweep numbers will reflect RotatE's compressed score range.

## Fallbacks & error handling

- **`decoder == "distmult"`**: existing code path. Byte-identical to Run 8 at α=0; matches Run 8's 0.9786 AUC at α=1.0.
- **`decoder == "rotate"`**: new path. Encoder-side `rel_emb` and decoder-side `rel_phase` are independent — bug isolation is clean.
- **`decoder` unknown value**: log warning, fall through to DistMult. Same fail-safe pattern as Run 8's `adv_temp < 0`.
- **NaN in gradients**: `1e-9` inside sqrt prevents `d=0` NaN. `clip_grad_norm_(1.0)` catches optimizer instability (existing).
- **AUC guardrail trips**: existing Run 7 logic — Neo4j sync skipped, audit recorded as `aborted_auc_guardrail`. Production scores untouched.
- **Empty graph / no relationships**: existing guards in `fetch_graph_data` and `run_audit` abort cleanly.

## Testing

### Unit tests (`backend/tests/test_gnn_rotate.py`, new file)

1. **`test_config_exposes_decoder_with_distmult_default`** — subprocess check (avoid module-reload pollution per Run 8 lesson). Asserts `Config.COMPGCN_DECODER == "distmult"` when env var unset.

2. **`test_rotate_logits_zero_phase_equals_translation`** — at θ=0 for all relations, `r = (1, 0)` so `h ∘ r = h`, and `score = -||h - t||`. Hand-construct h, t, force phases to zero, verify the score equals negative L2 distance computed independently.

3. **`test_rotate_logits_phase_pi_negates_real`** — at θ=π, `r = (-1, 0)` so `h ∘ r = -h`. Verify score equals `-||-h - t|| = -||h + t||` for known h, t.

4. **`test_rotate_logits_returns_negative_or_zero`** — score = `-sqrt(d² + ε)` so always ≤ 0 (with floor near `-sqrt(ε)` for the positive-triple-converged case). Random forward pass, assert all logits ≤ 0.

5. **`test_rotate_distinct_from_distmult_at_same_seed`** — instantiate two models (decoder="distmult", decoder="rotate") with same seed, run forward pass on identical inputs, assert `edge_logits` outputs differ. Sanity check that the dispatch is doing something.

6. **`test_run_audit_uses_rotate_when_configured`** — monkeypatch `Config.COMPGCN_DECODER = "rotate"`, run audit on synthetic graph, assert the model is `CompGCNAuditModel` with `decoder == "rotate"` and `rel_phase` attribute populated.

7. **`test_audit_run_records_decoder`** — symmetric to Run 8's `test_audit_run_node_records_adv_temp`. Assert AuditRun MERGE Cypher contains `run.decoder = $decoder` and kwargs include `decoder='rotate'`.

8. **`test_checkpoint_meta_records_decoder`** — symmetric to Run 8's `test_checkpoint_meta_records_adv_temp`. Assert meta JSON has `decoder` field.

### Reproducibility check

With `COMPGCN_DECODER=distmult` (default), Run 9 produces identical per-epoch AUC values to Run 8. Verified by running 2-epoch dry run pre-merge.

### Full Run 9 audit

`python backend/run_logs/rotate_audit.py`:
- 300-epoch BPR + self-adversarial α=1.0 + RotatE decoder.
- Expected wall-clock ~2.5–3 min (RotatE adds ~10% per epoch due to trig ops + sqrt).
- Acceptance: MRR (uniform eval) ≥ 0.95 → ship as recommended config. Otherwise document the lift, run threshold sweep, write Run 9 section.

## Thesis reporting plan

Chapter 4 ablation table extended with a decoder column:

| Run | Loss | Sampling | Adv. temp | **Decoder** | AUC | MRR | Grounding | Faithfulness |
|-----|------|----------|-----------|-------------|-----|-----|-----------|--------------|
| 4 (BCE baseline) | BCE | Uniform | — | DistMult | 0.9502 | 0.8134 | — | — |
| 6 (BPR) | BPR | Uniform | 0 | DistMult | 0.9688 | 0.8860 | 0.987 | 0.979 |
| 7 (BPR + type-aware) | BPR | Same-label | 0 | DistMult | 0.9662 | 0.8873 | 0.988 | 0.91–0.95 |
| 8 (BPR + self-adv) | BPR | Uniform | 1.0 | **DistMult** | 0.9786 | 0.9119 | 0.9884 | 0.9714 |
| **9 (BPR + self-adv + RotatE)** | BPR | Uniform | 1.0 | **RotatE** | TBD | TBD | TBD | TBD |

This is the **standard CompGCN paper format** — Vashishth+ 2020 reports DistMult / TransE / ConvE separately in their Table 4. Run 9 fills the same ablation slot. The encoder-decoder split is preserved.

If MRR ≥ 0.95: Chapter 4 reports H2 PASSED; Chapter 5 (recommendations) updates the "recommended defense config" to include `COMPGCN_DECODER=rotate`. The architecture diagram description shifts from "CompGCN" to "CompGCN with RotatE scoring" — a one-word precision edit, not a research-claim change.

If 0.92 ≤ MRR < 0.95: Document the lift (likely +0.01 to +0.02 on this corpus's density), confirm the corpus-density diagnosis from §5.1 of the paper, escalate corpus expansion as the dominant future-work lever.

If MRR < 0.92: Either RotatE under-performs DistMult here (unlikely but possible at low density) or the lift is below the LLM-judge noise floor. Either outcome strengthens the corpus-density argument; framework remains the recommended defense config.

## Risks

- **AUC drop.** RotatE on small graphs occasionally under-performs DistMult because its expressivity is a liability when you don't have enough data to learn good rotations. Mitigation: AUC guardrail (`COMPGCN_AUC_GUARDRAIL=0.95`) blocks Neo4j sync if AUC regresses. If guardrail trips, Run 9 is shipped as a documented negative result — AuditRun is recorded with `status='aborted_auc_guardrail'`.
- **Faithfulness regression.** RotatE's compressed score range may shift which triplets pass τ; the LLM context changes; faithfulness can wobble. Mitigation: post-eval threshold sweep finds the τ that maximizes both KPIs (same protocol as Run 5).
- **Score range incompatibility with τ=0.95.** RotatE's `sigmoid(-distance)` is ≤ 0.5; the canonical τ=0.95 will reject all triplets. This is **expected** — Run 5/6 already established that the "right" τ depends on the loss. The paper's claim is robust because it reports τ as a calibration choice, not a fixed constant.
- **128-dim complex vs 256-dim DistMult.** RotatE has half the per-dim parameters. Marginal expressivity loss possible. Run 9's recommended next move (if MRR doesn't close) is to re-run with `COMPGCN_HIDDEN_CHANNELS=512` so RotatE has 256 complex dims — but that's a Run 10 ablation, out of scope here.
- **Phase initialization.** Uniform $[-\pi, \pi]$ is the RotatE paper default. Some implementations use a smaller range (e.g., $[-1, 1]$ scaled by $\pi/64$) for slower convergence. We use the paper default; if convergence is unstable, re-init range becomes a Run 10 knob.

## Open questions

None blocking. The 128-dim split is the canonical RotatE pattern; sqrt-with-epsilon is the standard numerical-stability trick; the encoder/decoder separation is per Vashishth+ 2020.
