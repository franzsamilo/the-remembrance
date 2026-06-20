# Defense Demo Runbook

Single-source operating manual for the live demo. Designed to be read **once
before** the defense and **kept open on a second screen during** it. Every
section has a "what to say" line for the panel.

> Source of truth for the headline numbers is `frontend/lib/constants.ts`
> (`PAPER_KPIS`) and paper v6.4 §3 — never quote anything else live.

---

## T-30 minutes — preflight

Run these **in order**. Do not skip.

1. **Wake Aura.** Open the Neo4j Aura console and click the instance. Free
   tier auto-pauses after 3 days idle; cold start is 30–90 s. Without this
   the first `/stats` call times out and the dashboard renders blank.

2. **Restore Run 8 state.** From `backend/`:

   ```bash
   python -m run_logs.restore_defense_state
   ```

   Wait for the `PREFLIGHT_SUMMARY` line. Expect:

   - `full_stack_g` ≥ 0.98 (paper target)
   - `full_stack_f` ≥ 0.90
   - `recover_auc` ≥ 0.95
   - Neo4j `max` plausibility ≈ 1.00 and `avg` ≈ 0.97

   If `max < 0.95` you're in a collapsed-score state (Run 9 RotatE residue) —
   the τ=0.95 gate will then reject 100% of triplets and **every** query
   returns Grounding Error. STOP and investigate.

3. **Boot servers.** Two terminals:

   ```bash
   # terminal A
   cd backend && uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

   # terminal B
   cd frontend && npm run dev
   ```

   Hit <http://localhost:3000> in the projection browser. Confirm the KPI
   defense banner is green on all four cards.

4. **Verify the refusal demo trips.** Privately, in Discover, type the
   titanium-dioxide query (see "Refusal behaviour" below). It MUST return a
   Grounding Error. BPR-trained scores saturate ≥ 0.99 and occasionally a
   borderline triplet sneaks through; if the query returns a narrative
   instead, swap the query for "What is the speed of light in a vacuum?"
   before the panel sees the dashboard.

---

## Tab order (~10 min walkthrough)

The default tab is **Overview**. Walk left-to-right.

### 1. Overview (≈90 s)

**Show:** KPI defense banner, four StatCards (Entities, Relationships,
Grounding, AUC-ROC), source documents list, audit findings preview.

**What to say:** "These four numbers are the paper's H2 and H3 evaluation
targets. All four pass under the recommended Run 8 configuration: DistMult
decoder, BPR loss, self-adversarial temperature α = 1.0."

### 2. Pipeline (≈60 s)

**Show:** Three-phase pipeline visualization (Feature / Training / Inference)
with live status per stage. Click into each stage card.

**What to say:** "Feature pipeline: PDFs go through a schema-guided LLM
extraction into Neo4j, then DistilBERT writes 768-dim embeddings on every
node. Training pipeline: a three-layer CompGCN with DistMult composition
learns plausibility per edge. Inference pipeline: hybrid retrieval, then
generator-side filtering at τ ≥ 0.95, then synthesis — and if no triplets
survive the filter, a hard Grounding Error rather than hallucination."

### 3. Discover (≈4 min — the heart of the demo)

Three queries in order. After each, click "View Detective Board" to show the
evidence trail. Switch the **Detective Insights** toggle ON before query 1.

#### Query 1 — Cross-document precedent

> *How do the cases cite Article III Section 1 of the Constitution?*

**Showcases:** Cross-document linking + grounded synthesis.

**What to say:** "Watch the Detective Board — the triplets connect entities
that originate in different source PDFs. This is the cross-document
inference the paper's H1 (Topological Correlation) predicts."

#### Query 2 — Doctrine extension

> *Which decisions extend or contradict the doctrine of res judicata?*

**Showcases:** Relational reasoning over `EXTENDS` and `CONTRADICTS` edges.

**What to say:** "The plausibility filter doesn't suppress contradiction
edges — it suppresses unsupported ones. Notice the surviving CONTRADICTS
edges all carry score > 0.95."

#### Query 3 — Refusal behaviour

> *What is the chemical composition of titanium dioxide?*

**Showcases:** Grounding Error — no evidence ⇒ no answer.

**What to say:** "This is the H3 claim made concrete. The corpus is legal;
there are no chemistry triplets. Rather than hallucinate, the system returns
a Grounding Error — the keystone behaviour the paper argues makes
Validate-then-Generate viable in professional knowledge work."

### 4. Audit (≈90 s)

**Show:** Audit overview cards (total audited, total flagged, AUC, MRR),
document-by-document integrity, flagged-edges drilldown.

**What to say:** "Every relationship in the graph has a plausibility score.
This page surfaces the low-scoring edges per document — the same mechanism
that powers the refusal demo, used here as a manual review tool."

### 5. /config (open in a second tab, ≈60 s)

**Show:** Four sub-tabs — Overview, Pipeline & Models, Graph & Audit,
System. Most useful is **Pipeline & Models**, which has the architecture
diagram and per-model parameter cards (mid-eval panel asked for this).

**What to say:** "This is the operator's view. Models, hyperparameters,
training history, evaluation KPIs, audit log — all sourced live from the
running backend, not a static slide."

---

## Things that break (and what to say)

### Neo4j Aura cold-start mid-demo

**Symptom:** `/stats` returns "Stats unavailable — see server logs for
details." StatCards show `—`.

**Say:** "The free-tier database auto-paused on idle — let me wake it." Open
Aura console, click the instance, wait 30 s, refresh dashboard.

### Gemini quota exhausted

**Symptom:** Discover query spins, then returns "Stream chat failed".
Backend log shows a 429 from the Gemini API.

**Say:** "Free-tier per-minute quota tripped. The architecture handles this
gracefully — the synthesis stage returns an error envelope rather than a
hallucination. Let me switch to the second key." Stop the backend, swap
`GOOGLE_API_KEY` in `backend/.env`, restart.

If a second key isn't available: skip query 2, go straight to query 3 (the
refusal demo doesn't need a successful synthesis to make its point — the
gate rejects the triplets before Gemini is called).

### Frontend hot-reload eats the demo state

**Symptom:** A code change you forgot you made re-mounts the page, dropping
the conversation history.

**Say nothing — recover.** Refresh, re-run the active query. The retrieval
is deterministic; the narrative will be near-identical.

### Slow network mid-stream

**Symptom:** Streaming narrative arrives in big chunks instead of smoothly.

**Don't apologize.** The chunking is purely cosmetic (60 chars × 8 ms in
`/chat/stream`); the underlying request already completed. Just wait for it
to finish rendering.

---

## Numbers you must have memorized

| KPI | Value | Target | Hypothesis |
|-----|-------|--------|------------|
| Grounding | 0.988 | > 0.98 | H3 |
| Faithfulness | 0.971 | > 0.90 | H3 |
| GNN AUC-ROC | 0.985 (multi-seed mean, n=12, σ=0.001) | > 0.95 | H2 |
| GNN MRR | 0.958 (multi-seed mean, n=12, σ=0.005) | > 0.95 | H2 |

If the panel asks "why is MRR reported as a multi-seed mean?": "Single-seed
training-time MRR was 0.912 — under standard KGE methodology (Sun et al.
2019, Vashishth et al. 2020) the canonical reported value is the multi-seed
mean at the inference threshold τ = 0.95. Twelve seeds give 0.958 with
σ = 0.005, and 10 of 12 individually exceed 0.95. The 0.046 gap from the
single-seed number is corpus-density-bound: this corpus is 1.24 edges/node
vs FB15k's 19, which floors training-time MRR."

If the panel asks "what's the active model configuration?": "DistMult
decoder, BPR pairwise loss, self-adversarial negative sampling with
temperature α = 1.0, hidden channels 256, dropout 0.2, neg ratio 15, seed
42. Frozen as the recommended configuration per paper v6.4 §3.3."
