# Pipeline Visualization Design

**Date:** 2026-04-06
**Goal:** Replace the existing PipelineStory component with an interactive, animated pipeline visualization that maps each model to its pipeline stage with live status, parameters, and "why" annotations. Addresses mid-eval panel feedback requesting more pipeline detail and frontend dynamism.

## Context

The panel said the project lacked visualizations showing where each model sits in the pipeline, what it's trained on, what the parameters are, and the justifications. The existing PipelineStory is a collapsible text-based list grouped by phase. The new component is a hybrid: compact horizontal flow strip + expandable detail panel for the selected stage.

No backend changes required — all model info and "why" text is static config. Live status comes from existing `GET /stats` polling already in page.tsx.

## Architecture

### Files

```
frontend/
  lib/
    pipelineConfig.ts        — Stage definitions, model info, params, "why" strings
  components/
    PipelineFlow.tsx          — Orchestrator: selected stage state, renders Strip + Detail
    PipelineStrip.tsx         — Horizontal strip: phase-grouped stage nodes + animated arrows
    PipelineDetail.tsx        — Expanded detail card for selected stage
    PipelineStory.tsx         — DELETED (replaced by PipelineFlow)
```

### Data Flow

- `page.tsx` already polls `GET /stats` every 2-5s. Stats are passed as a prop to `PipelineFlow` (no duplicate fetching).
- `PipelineFlow` holds `selectedStageId` state (default: first stage). Renders `PipelineStrip` above `PipelineDetail`.
- `PipelineStrip` receives `stages`, `stats`, `selectedId`, `onSelect`. Renders clickable nodes.
- `PipelineDetail` receives the selected `PipelineStage` object + `stats`. Renders the detail card.
- All static content (model names, params, "why" one-liners) lives in `pipelineConfig.ts`.

### Integration with page.tsx

Replace the current `<PipelineStory />` mount with `<PipelineFlow stats={stats} currentTask={currentTask} />`. Remove the PipelineStory import. No other page.tsx changes needed.

## pipelineConfig.ts

### Type Definition

```typescript
interface PipelineStage {
  id: string;
  order: number;
  name: string;
  phase: "feature" | "training" | "inference";
  model: { name: string; description: string };
  input: string;
  output: string;
  params: Record<string, string>;
  why: string;
  getStatus: (currentTask: string, stats: any) => "ready" | "active" | "waiting" | "error";
  getMetrics: (stats: any) => { label: string; value: string }[];
}
```

### Stage Definitions

| # | ID | Name | Phase | Model | Why |
|---|-----|------|-------|-------|-----|
| 1 | `ingest` | PDF Ingestion | feature | SimpleKGPipeline | Schema-guided extraction captures typed relations standard chunking misses. |
| 2 | `extract` | Entity Extraction | feature | Gemini 2.5 Flash | Zero-shot LLM extraction at T=0 for deterministic entity/relation output. |
| 3 | `store` | Graph Storage | feature | Neo4j Aura | Native property graph preserves multi-relational structure Cypher can traverse. |
| 4 | `embed` | Vector Embedding | feature | DistilBERT | Lightweight 768-dim vectors solve the GNN cold-start without LLM-scale compute. |
| 5 | `audit` | Integrity Audit | training | CompGCN (DistMult) | Shared relation embeddings via DistMult avoid R-GCN's O(R) parameter explosion. |
| 6 | `synthesize` | Grounded Synthesis | inference | Gemini 2.5 Flash | Generator-side filtering at τ≥0.95 ensures only validated triplets reach the LLM. |
| 7 | `evaluate` | Evaluation | inference | LLM-as-Judge (Gemini) | Gemini scores its own output against triplets for grounding/faithfulness. |

### Parameters Per Stage

**ingest:** Pipeline: SimpleKGPipeline, From: PDF, Schema: configurable node/rel types
**extract:** Model: Gemini 2.5 Flash, Temperature: 0, Entity types: 8, Relation types: 7
**store:** Database: Neo4j Aura, Protocol: Bolt, Provenance: per-node + per-edge
**embed:** Model: distilbert-base-nli-stsb-mean-tokens, Dimensions: 768, Batch size: 50, Normalization: L2
**audit:** Hidden channels: 256, Epochs: 100, LR: 0.001, Weight decay: 0.0001, Dropout: 0.2, Patience: 20, Grad clip: 1.0, Neg ratio: 10, Validation split: 20%, Label smoothing: 0.0, Composition: DistMult, Seed: 42
**synthesize:** Model: Gemini 2.5 Flash, Threshold (τ): 0.95, Retrieval: hybrid vector+graph, Top-k seeds: 5, Max hops: 2
**evaluate:** Method: LLM-as-Judge, Scorer: Gemini, Metrics: Grounding (0-1), Faithfulness (0-1)

### Status Logic

Each stage derives its status from `currentTask` string (from `GET /stats`):

- `ingest`/`extract`: active when currentTask contains "Extracting"
- `store`: active when currentTask contains "Extracting" (Neo4j writes happen during extraction — same phase)
- `embed`: active when currentTask contains "Embedding"
- `audit`: active when currentTask contains "Audit"
- `synthesize`: always "ready" when graph state is evidence_ready_graph
- `evaluate`: active when currentTask contains "Evaluation"
- All stages: "waiting" when graph_state is empty_graph; "error" when currentTask starts with "Error"

### Metrics Mapping

- `embed` → Vector Coverage: `stats.embedding_progress + "%"`
- `audit` → AUC-ROC: `stats.research_kpis.gnn_auc_roc`, MRR: `stats.research_kpis.gnn_mrr`
- `evaluate` → Grounding: `stats.research_kpis.grounding_score`, Faithfulness: `stats.research_kpis.faithfulness_score`
- Other stages: no live metrics (show stage-level counts from stats when available, e.g., node/relationship counts for store)

## PipelineStrip

### Layout

Single horizontal row of stage nodes grouped into three labeled phases:

```
  FEATURE                                    TRAINING        INFERENCE
[Ingest] → [Extract] → [Store] → [Embed]  → [Audit]      → [Synthesize] → [Evaluate]
```

Phase labels: small uppercase text above each group with the phase's accent color.

### Phase Colors

- Feature: validated-green (`#3A5A40`)
- Training: gilded-gold (`#D4AF37`)
- Inference: seal-red (`#8B1A1A`)

### Each Node Shows

- Stage name (e.g., "Entity Extraction")
- Model name in smaller muted text (e.g., "Gemini 2.5 Flash")
- Status indicator (dot or spinner — see Active States)
- Selected node: gold border, slight scale-up

### Arrows

- Default (idle): static gold arrow character or SVG line
- Active: animated flowing dots — 3-4 small gold circles traveling along a dashed path on staggered `animation-delay`. Uses CSS `@keyframes translateX` along the arrow. The `dash-flow` keyframe from globals.css is reused for the dashed underline.

### Responsive

On screens < 768px, the strip wraps into 3 rows (one per phase) with vertical arrows between phases.

### Interaction

Click any node → calls `onSelect(stageId)` → parent updates selected stage → PipelineDetail renders/animates below.

## PipelineDetail

### Layout

Card with phase-colored border. Animated entry: slide down + fade in (Framer Motion `AnimatePresence`).

**Header row:** Phase label + stage number (left), stage name large (left), status badge (right).

**2×2 grid (desktop), stacked (mobile):**

| Model | Input / Output |
|-------|---------------|
| Parameters | Why This Approach |

**Bottom full-width row:** Live Metrics (only when metrics exist for that stage). Each metric is a labeled value. Values that update get a brief gold flash animation on change.

### Content Sources

All static content from the `PipelineStage` object in pipelineConfig.ts. Live metrics from stats prop, extracted via `stage.getMetrics(stats)`.

## Active State Animations

Active states must feel alive without being distracting.

### Active Stage Node (Strip)

- Border: gold pulse animation (`gilded-pulse` from globals.css — 2s infinite)
- Status indicator: replaces static dot with a spinning ring (CSS border animation)
- Subtle shimmer on the stage name text (CSS gradient animation)
- Idle-to-active transition: quick scale bounce (1.0 → 1.05 → 1.0) via Framer Motion `animate`

### Active Arrows (Flowing Particles)

- 3-4 small gold circles (4px) traveling along the arrow path
- Staggered `animation-delay` (0s, 0.3s, 0.6s, 0.9s) for continuous flow effect
- CSS `@keyframes flowDot { 0% { transform: translateX(0) } 100% { transform: translateX(100%) } }` with ~1.5s duration
- Arrow dashed line uses `dash-flow` keyframe (already exists)

### Active Detail Card

- Thin animated progress bar along top edge: indeterminate stripe animation (repeating-linear-gradient moving left)
- Status badge text: "Running" with animated ellipsis (CSS `@keyframes ellipsis`)
- Metrics that change: gold background flash (0.3s ease-out) when value updates, via Framer Motion `animate` triggered by value change

### Reduced Motion

All animations respect `prefers-reduced-motion: reduce` (media query already in globals.css). Fallback: static gold border for active nodes, no flowing dots, no bounces. Status is communicated via text badges only.

## Testing

- Visual: verify all 7 stages render with correct phase grouping and colors
- Interaction: click each stage, verify detail panel shows correct content
- Active state: trigger ingestion via POST /ingest, verify strip nodes animate in sequence
- Responsive: verify strip wraps at mobile breakpoint
- Reduced motion: enable "reduce motion" in OS, verify no animations
- Data: verify metrics update when stats poll returns new values
