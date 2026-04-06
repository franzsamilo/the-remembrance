# Doubt Killers + UI Uplift — Design Spec

**Date:** 2026-04-07
**Context:** Panel mid-eval feedback cited (1) paper lacks pipeline detail, (2) frontend lacks dynamic pipeline visualization. The pipeline visualization has been built. This spec addresses the remaining doubt gaps and a visual/structural UI uplift to make the defense demo airtight.

**Goal:** Eliminate all remaining panel doubt through 6 targeted features + a visual modernization of the frontend. The app IS the defense — every feature must be clickable and demonstrable.

---

## 1. Visual Theme: "Modern Archival"

Keep the archival DNA (serif typography, gold accents, vault metaphor) but lose the heavy parchment. Think modern rare-books library — clean glass cases, warm lighting, deliberate typography.

### Color Palette

| Token | Variable | Old Value | New Value | Purpose |
|-------|----------|-----------|-----------|---------|
| Background | `--parchment-base` | `#F5F2E9` | `#FAFAF8` | Warm white, not yellowed |
| Surface | `--parchment-light` | `#FCFAF2` | `#FFFFFF` | Clean white cards |
| Ink Dark | `--ink-dark` | `#2B2B2B` | `#1A1A1A` | Deeper for contrast |
| Ink Medium | `--ink-medium` | `#4A4A4A` | `#525252` | Secondary text |
| Ink Light | `--ink-light` | `#6B6B6B` | `#737373` | Tertiary text |
| Gold | `--gilded-gold` | `#D4AF37` | `#C5A028` | Deeper, more dignified |
| Seal Red | `--seal-red` | `#8B1A1A` | `#7A1A1A` | Brand identity retained |
| Validated Green | `--validated-green` | `#3A5A40` | `#2D6A4F` | Slightly more vibrant |
| Danger | `--conflict-red` | `#C41E3A` | `#DC2626` | Clearer standard red |
| Border | `--border` | `#4A4A4A` | `#E5E5E3` | Light warm gray — guides, not cages |
| Muted | `--muted` | `#E8E4D9` | `#F5F5F3` | Lighter muted background |

### Typography
- **Keep:** EB Garamond (headings), Public Sans (body), JetBrains Mono (data)
- **Change:** Base body size 14px → 15px. More whitespace between sections (gap-4 → gap-6). Headers get more breathing room.

### Cards
- Drop `glass` backdrop-blur effect
- Clean white cards: `bg-white border border-[#E5E5E3] rounded-lg shadow-sm`
- Hover: `shadow-md` + `border-l-2 border-l-[#C5A028]` (gold left accent, not full border)

### Animations
- Dial back: no shimmer bars, no pulsing rings on idle elements
- Reserve animation for state changes only (task running, value updating, tab transitions)
- Keep Framer Motion spring physics for page/tab transitions

---

## 2. Tabbed Dashboard Layout

Replace the current dense 2/3 + 1/3 split layout with a tabbed single-column layout on `page.tsx`.

### Tab Structure

| Tab | Content | Demo Narrative |
|-----|---------|----------------|
| **Overview** | System status badge, document management (upload/list/delete), KPI summary cards (4 metrics), archive readiness indicator | "Here's the system state at a glance" |
| **Pipeline** | PipelineStrip + PipelineDetail (full width), execution timing summary bar, per-stage timing in detail cards | "Here's every model, parameter, and why we chose it" |
| **Discover** | Chat interface (full width hero), toggles for Explain/Prompt-Only modes, recent query display | "Let me query the knowledge graph live" |
| **Audit** | AuditOverview stats, Training Curves, Ablation Comparison, Document Integrity, Flagged Edges | "Here's the GNN integrity layer proving itself" |

### Tab Bar Design
- Horizontal, sticky below the header
- Text labels with gold underline on active tab
- Optional small icons (16px) next to labels
- Smooth animated underline transition between tabs

### Layout Changes
- **Sidebar eliminated.** Everything goes full-width within its tab.
- **Chat moves from right drawer to Discover tab.** No more overlay.
- **Evidence page (`/evidence`) remains separate** — linked from chat responses for deep-dive.
- **Audit page (`/audit`) redirects to `/?tab=audit`.**

### Header (Simplified)
- Logo + "THE REMEMBRANCE" title (smaller, less dramatic)
- System status indicator (subtle dot, not a full banner)
- Reset/Refresh buttons grouped in a compact toolbar
- Config link retained

---

## 3. Doubt-Killer Features

### 3a. Per-Triplet Plausibility Scores

**Location:** `/evidence` page — DetectiveBoard component

**UI changes:**
- Each triplet in evidence steps displays a plausibility badge: `0.97` with color coding
  - Green (`>= 0.95`): validated, passed threshold
  - Amber (`0.85 - 0.94`): borderline (shouldn't appear in validated set, but useful for rejected view)
  - Red (`< 0.85`): clearly anomalous
- New **"Filtering Summary" box** below evidence steps header:
  - "12 triplets retrieved -> 9 passed GNN validation (tau >= 0.95) -> 3 filtered out"
  - Visual: horizontal bar showing green (passed) vs red (filtered) proportions
- New **"Rejected Evidence" collapsible section:**
  - Triplets that failed the threshold, shown with red-tinted cards, scores in red, strikethrough on relation text
  - Collapsed by default, expandable to show what the system caught
  - Header: "3 Triplets Filtered (Below tau = 0.95)" with expand/collapse toggle

**Backend changes:**
- `retriever.py`: Ensure `plausibility` score is included on each triplet in the response
- `generator.py`: Return a new `filtered_triplets` array containing triplets that were below threshold, each with their score
- `synthesis.py`: Pass filtered triplets through to the response object

### 3b. Ablation Comparison

**Location:** Audit tab — new `AblationComparison` component

**Layout:** Three-column comparison card:

```
| Prompt Only         | Graph (No GNN)      | Full Stack          |
|---------------------|---------------------|---------------------|
| Chunk RAG           | Hybrid retrieval    | Hybrid + GNN filter |
| No graph, no GNN    | All edges, no audit | Validated edges only|
|                     |                     |                     |
| Grounding: 0.52     | Grounding: 0.71     | Grounding: 0.98     |
| Faithfulness: 0.61  | Faithfulness: 0.78  | Faithfulness: 0.95  |
|                     |                     |                     |
|                     | +19% grounding      | +27% grounding      |
|                     | vs Prompt Only      | vs Graph            |
```

- Full Stack column: subtle gold border + "Active Configuration" badge
- Delta indicators between columns showing improvement
- Each column: mode name, brief description, grounding score (large), faithfulness score (large), delta from previous mode
- Responsive: stacks vertically on mobile

**Backend changes:**
- Evaluation already supports three modes. Need to store/cache results from all three.
- Option A: Add `ablation_results` to the `/stats` response (preferred — avoids new endpoint)
- Option B: New `/ablation/results` endpoint
- Recommend Option A: extend `/stats` with `ablation: { prompt_only: {grounding, faithfulness}, graph: {grounding, faithfulness}, full_stack: {grounding, faithfulness} }`

### 3c. Training Curves

**Location:** Audit tab — new `TrainingCurves` component, positioned between AuditOverview and AblationComparison

**What it shows:**
- Two SVG line charts side by side (or stacked on mobile):
  1. **Loss curve:** Train loss + Validation loss over epochs. Y-axis: loss value. X-axis: epoch number.
  2. **Metrics curve:** AUC-ROC + MRR over epochs. Y-axis: score (0-1). X-axis: epoch number.
- Early stopping marker: vertical dashed line at the epoch where patience triggered, with label "Early stop @ epoch N"
- Final metrics callout: small card showing final AUC-ROC, MRR, best epoch
- Axis labels, gridlines, legend (train=solid, val=dashed)

**Implementation:** Pure SVG — same approach as existing KnowledgeGraph component. No charting library dependency. Simple polyline paths with axis ticks.

**Backend changes:**
- `gnn_module.py`: During training loop, collect per-epoch metrics into a list: `[{epoch, train_loss, val_loss, auc_roc, mrr}, ...]`
- Store in `_system_state` or a module-level variable
- New endpoint: `GET /audit/training-history` returning the epoch metrics array
- Include `early_stop_epoch` and `best_epoch` in the response

### 3d. Grounding Error Demo Path

**Location:** Discover tab (chat) + Evidence page

**Trigger:** When a query returns zero validated triplets (all retrieved triplets scored below tau threshold), the system returns a structured grounding error instead of hallucinating.

**Chat bubble treatment:**
- Distinct design: white card with `border-l-4 border-l-[#DC2626]` (red left accent)
- Shield icon + "GROUNDING ERROR" header in red
- Body: "No validated evidence found for this query. The system refuses to generate an unsupported answer."
- Below body: "Retrieved but rejected:" section showing the triplets that failed with their scores
- This is NOT an error state — it's the system working correctly. The UI should communicate confidence, not failure.

**Evidence page treatment:**
- If user clicks "View Evidence" on a grounding error response, the evidence page shows:
  - Empty Detective Board with message: "No evidence passed validation for this query"
  - Full "Rejected Evidence" section showing all retrieved-but-filtered triplets with scores
  - Knowledge Graph still renders the retrieved nodes/edges but with red-tinted styling (not validated)

**Backend changes:**
- `generator.py`: When `validated_triplets` is empty after filtering, return a structured response:
  ```python
  {
    "answer": None,
    "grounding_error": True,
    "message": "No validated evidence found. The system refuses to generate an unsupported answer.",
    "filtered_triplets": [...],  # what was retrieved but failed
    "triplets": [],  # empty — nothing passed
  }
  ```
- `synthesis.py`: Skip synthesis entirely when no validated triplets exist (don't call Gemini)

**Demo prep:** After ingesting legal documents, test queries about topics NOT in the corpus (e.g., ask about a statute or case that was never ingested). The retriever will find some vaguely related triplets, but they should all score below tau — triggering the grounding error. If this doesn't happen naturally, temporarily lower tau to 0.99 for the demo to make the filter stricter. This is the single most impressive demo moment.

### 3e. Execution Timing

**Location:** Pipeline tab

**Per-stage timing in PipelineDetail:**
- New field in each stage detail card: "Last run: 12.4s" with a clock icon
- Shown below the existing "Live Metrics" section

**Timing summary bar (top of Pipeline tab):**
- Horizontal stacked bar chart showing proportional time per stage
- Color-coded by phase (green/gold/red matching pipeline phases)
- Labels below each segment: stage name + time
- Total time displayed on the right: "Total: 68s"
- Only shown after at least one full pipeline run

**Backend changes:**
- Add timestamp logging in `_system_state` for each pipeline stage start/end
- Extend `/stats` response with `stage_timings`: `{ ingest: 45.2, embed: 12.1, audit: 8.3, synthesis: 2.8 }`
- Log timestamps in `ingestion.py`, `embed_nodes.py`, `gnn_module.py`, `generator.py`

### 3f. Paper Diagrams

**Not a frontend feature.** Noting for completeness — these are deliverables for the thesis document:

1. **System Architecture Diagram:** Three-pipeline Labarta framing (Feature / Training / Inference) showing all 7 stages, models at each stage, data flow between stages
2. **Data Flow Diagram:** PDF -> SimpleKGPipeline -> Neo4j -> DistilBERT embeddings -> CompGCN audit -> Hybrid retrieval -> Gemini synthesis
3. **Before/After Comparison:** Standard RAG (retrieve -> generate, no validation) vs Validate-then-Generate (retrieve -> GNN filter -> generate, with grounding error fallback)

These can be created with any diagramming tool (draw.io, Mermaid, etc.) and included as figures in the paper.

---

## 4. Evidence Page Uplift

The `/evidence` page is where the panel spends the most time during demo.

**Changes:**
- Plausibility badges on each triplet (from 3a)
- Filtering summary box (from 3a)
- Rejected evidence collapsible section (from 3a)
- Grounding error state (from 3d)
- KnowledgeGraph: add zoom reset button (top-right corner), retheme node colors to match new palette
- DetectiveBoard: tighter spacing, cleaner number badges matching new color tokens
- Overall: apply new card styles, color tokens, reduced animation

---

## 5. Component Architecture

### New Components

| Component | File | Purpose |
|-----------|------|---------|
| `TabShell` | `frontend/components/TabShell.tsx` | Tab bar + content wrapper, manages active tab state via URL param `?tab=` |
| `AblationComparison` | `frontend/components/AblationComparison.tsx` | Three-column mode comparison with delta indicators |
| `TrainingCurves` | `frontend/components/TrainingCurves.tsx` | SVG line charts for loss, AUC-ROC, MRR over epochs |
| `TimingSummary` | `frontend/components/TimingSummary.tsx` | Horizontal stacked bar for stage execution times |
| `GroundingError` | `frontend/components/GroundingError.tsx` | Styled grounding error chat bubble with rejected triplets |
| `FilteringSummary` | `frontend/components/FilteringSummary.tsx` | Validated vs rejected triplet counts with visual bar |
| `RejectedEvidence` | `frontend/components/RejectedEvidence.tsx` | Collapsible list of filtered-out triplets with scores |

### Modified Components

| Component | Changes |
|-----------|---------|
| `page.tsx` | Restructure into tabbed layout via TabShell. Eliminate sidebar. Move chat from drawer to Discover tab. Move audit preview to Audit tab. |
| `DetectiveBoard.tsx` | Add plausibility score badges per triplet. Retheme colors. |
| `PipelineDetail.tsx` | Add execution timing display. Retheme. |
| `PipelineStrip.tsx` | Retheme colors to new palette. |
| `PipelineFlow.tsx` | Full-width layout (no longer constrained by sidebar). |
| `AuditOverview.tsx` | Retheme card styles and colors. |
| `DocumentIntegrity.tsx` | Retheme. |
| `FlaggedEdges.tsx` | Retheme. Emphasize plausibility scores more prominently. |
| `KnowledgeGraph.tsx` | Retheme node/edge colors. Add zoom reset button. |
| `GraphInfoCard.tsx` | Retheme to match new card styles. |
| `StatCard.tsx` | New card style (white, light border, gold-left hover). |
| `Skeleton.tsx` | Update colors to match new palette. |
| `ConfirmModal.tsx` | Retheme. |
| `CustomIcons.tsx` | No changes needed. |
| `globals.css` | New color tokens, updated card styles, removed glass effect, reduced animation keyframes. |

### Backend Changes

| File | Changes |
|------|---------|
| `gnn_module.py` | Log per-epoch training metrics (loss, AUC-ROC, MRR). Store in module state. |
| `generator.py` | Return `filtered_triplets` array. Return structured grounding error when no validated triplets. |
| `retriever.py` | Include `plausibility` score on each returned triplet. |
| `synthesis.py` | Skip Gemini call when no validated triplets (grounding error path). |
| `api/main.py` | New endpoint: `GET /audit/training-history`. Extend `/stats` with `stage_timings` and `ablation` results. Add timestamp logging for stage timing. |
| `ingestion.py` | Log stage start/end timestamps to `_system_state`. |
| `embed_nodes.py` | Log stage start/end timestamps to `_system_state`. |
| `evaluation.py` | Support storing results per ablation mode. |

---

## 6. Demo Flow

The panel defense walks through tabs left-to-right:

1. **Overview tab:** "Here's the system. We have N documents ingested, N entities extracted, the archive is validated and ready."
2. **Pipeline tab:** "Here's every stage of our three-pipeline architecture. Each one shows the model, parameters, input/output, and why we chose this approach. You can see execution timing — ingestion takes 45s, embedding 12s, audit 8s, synthesis 3s."
3. **Discover tab:** "Let me query live." Run a good query — show evidence trail with plausibility scores. Then run a bad query — show the grounding error. "The system refuses to hallucinate."
4. **Audit tab:** "Here's the GNN training — loss convergence, AUC-ROC reaching 0.95. Here's the ablation: Prompt Only gets 52% grounding, Graph gets 71%, Full Stack gets 98%. The integrity layer works."

This flow directly addresses both panel feedback points and proves the thesis claims.
