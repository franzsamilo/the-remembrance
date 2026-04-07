# Doubt Killers + UI Uplift Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Eliminate panel doubt through 6 targeted features + visual modernization, making the thesis defense demo airtight.

**Architecture:** Tabbed single-column dashboard (Overview / Pipeline / Discover / Audit) replacing the current 2/3 + 1/3 split. Backend extended with training history, stage timing, filtered triplets, and structured grounding errors. New SVG chart components for training curves and ablation comparison.

**Tech Stack:** Next.js 16, React 19, TypeScript, Tailwind CSS 4, Framer Motion, FastAPI, PyTorch Geometric, Neo4j

**Spec:** `docs/superpowers/specs/2026-04-07-doubt-killers-ui-uplift-design.md`

---

## File Map

### New Frontend Files
| File | Purpose |
|------|---------|
| `frontend/components/TabShell.tsx` | Tab bar + content wrapper, manages `?tab=` URL param |
| `frontend/components/AblationComparison.tsx` | Three-column mode comparison with delta indicators |
| `frontend/components/TrainingCurves.tsx` | SVG line charts for loss/AUC-ROC/MRR over epochs |
| `frontend/components/TimingSummary.tsx` | Horizontal stacked bar for stage execution times |
| `frontend/components/GroundingError.tsx` | Styled grounding error chat bubble |
| `frontend/components/FilteringSummary.tsx` | Validated vs rejected triplet counts with bar |
| `frontend/components/RejectedEvidence.tsx` | Collapsible list of filtered-out triplets |

### Modified Frontend Files
| File | Changes |
|------|---------|
| `frontend/app/globals.css` | New color tokens, drop glass/parchment, lighter theme |
| `frontend/lib/types.ts` | Add `filtered_triplets` and `grounding_error` to types |
| `frontend/app/page.tsx` | Restructure into tabbed layout, eliminate sidebar, move chat to Discover tab |
| `frontend/app/evidence/page.tsx` | Add filtering summary, rejected evidence, plausibility badges |
| `frontend/app/audit/page.tsx` | Redirect to `/?tab=audit` |
| `frontend/components/DetectiveBoard.tsx` | Add plausibility score badges per triplet, retheme |
| `frontend/components/PipelineStrip.tsx` | Retheme colors |
| `frontend/components/PipelineDetail.tsx` | Add timing display, retheme |
| `frontend/components/PipelineFlow.tsx` | Full-width layout |
| `frontend/components/AuditOverview.tsx` | Retheme |
| `frontend/components/DocumentIntegrity.tsx` | Retheme |
| `frontend/components/FlaggedEdges.tsx` | Retheme |
| `frontend/components/StatCard.tsx` | New card style |
| `frontend/components/KnowledgeGraph.tsx` | Retheme, add zoom reset |
| `frontend/components/GraphInfoCard.tsx` | Retheme |
| `frontend/components/Skeleton.tsx` | Update colors |
| `frontend/components/ConfirmModal.tsx` | Retheme |
| `frontend/components/AuditFindingsCard.tsx` | Retheme |
| `frontend/lib/pipelineConfig.ts` | Update PHASE_COLORS to new palette |

### Modified Backend Files
| File | Changes |
|------|---------|
| `backend/src/gnn_module.py` | Log per-epoch metrics, expose training history |
| `backend/src/generator.py` | Return filtered_triplets, structured grounding error |
| `backend/src/api/main.py` | New endpoints, extend /stats with timings + ablation, update grounding error response |
| `backend/src/ingestion.py` | Log stage timing |
| `backend/src/embed_nodes.py` | Log stage timing |
| `backend/src/evaluation.py` | Support ablation mode parameter |

---

## Phase 1: Backend Foundation

### Task 1: Add Stage Timing to System State

**Files:**
- Modify: `backend/src/api/main.py:29-33`
- Modify: `backend/src/ingestion.py` (in `process_documents`)
- Modify: `backend/src/embed_nodes.py` (in `embed_nodes`)
- Modify: `backend/src/gnn_module.py` (in `run_audit`)

- [ ] **Step 1: Extend _SystemState with timing dict**

In `backend/src/api/main.py`, replace the dataclass:

```python
@dataclasses.dataclass
class _SystemState:
    status: str = "Idle"
```

with:

```python
@dataclasses.dataclass
class _SystemState:
    status: str = "Idle"
    stage_timings: dict = dataclasses.field(default_factory=dict)
    _stage_start: float = 0.0

    def start_stage(self, stage: str):
        self._stage_start = time.time()

    def end_stage(self, stage: str):
        if self._stage_start > 0:
            self.stage_timings[stage] = round(time.time() - self._stage_start, 1)
            self._stage_start = 0.0
```

- [ ] **Step 2: Add timing hooks to ingestion pipeline trigger**

In `backend/src/api/main.py`, in the `run_pipeline` function inside `/ingest` (around line 376-391), replace:

```python
            _system_state.status = "Extracting Concepts..."
            logger.info("Starting background ingestion pipeline...")
            manifest = await process_documents()

            if not manifest or manifest.get("documents_processed", 0) == 0:
                _system_state.status = "Idle"
                logger.warning("Ingestion produced no processable documents; skipping embedding stage.")
                return

            _system_state.status = "Embedding Nodes..."
            logger.info("Ingestion complete. Starting embedding cold-start...")
            await embed_nodes()

            _system_state.status = "Idle"
```

with:

```python
            _system_state.status = "Extracting Concepts..."
            _system_state.start_stage("ingest")
            logger.info("Starting background ingestion pipeline...")
            manifest = await process_documents()
            _system_state.end_stage("ingest")

            if not manifest or manifest.get("documents_processed", 0) == 0:
                _system_state.status = "Idle"
                logger.warning("Ingestion produced no processable documents; skipping embedding stage.")
                return

            _system_state.status = "Embedding Nodes..."
            _system_state.start_stage("embed")
            logger.info("Ingestion complete. Starting embedding cold-start...")
            await embed_nodes()
            _system_state.end_stage("embed")

            _system_state.status = "Idle"
```

- [ ] **Step 3: Add timing hooks to audit trigger**

In `backend/src/api/main.py`, in `run_gnn_audit_then_eval` inside `/audit` (around line 487-495), replace:

```python
            _system_state.status = "Running GNN Audit..."
            logger.info("Starting GNN Topological Audit...")
            run_audit()
            logger.info("Audit complete. Running grounding/faithfulness evaluation...")
            _system_state.status = "Running Evaluation..."
            asyncio.run(run_grounding_evaluation())
            _system_state.status = "Idle"
```

with:

```python
            _system_state.status = "Running GNN Audit..."
            _system_state.start_stage("audit")
            logger.info("Starting GNN Topological Audit...")
            run_audit()
            _system_state.end_stage("audit")
            logger.info("Audit complete. Running grounding/faithfulness evaluation...")
            _system_state.status = "Running Evaluation..."
            _system_state.start_stage("evaluate")
            asyncio.run(run_grounding_evaluation())
            _system_state.end_stage("evaluate")
            _system_state.status = "Idle"
```

- [ ] **Step 4: Expose stage_timings in /stats response**

Find the return statement of the `/stats` endpoint and add `stage_timings`:

```python
    "stage_timings": _system_state.stage_timings,
```

Add this as a top-level key in the stats response dict, alongside `current_task`.

- [ ] **Step 5: Verify manually**

Run: `cd backend && python -c "from src.api.main import _system_state; _system_state.start_stage('test'); import time; time.sleep(0.1); _system_state.end_stage('test'); print(_system_state.stage_timings)"`

Expected: `{'test': 0.1}`

- [ ] **Step 6: Commit**

```bash
git add backend/src/api/main.py
git commit -m "feat: add stage timing to system state and /stats endpoint"
```

---

### Task 2: Log Per-Epoch Training Metrics in GNN Module

**Files:**
- Modify: `backend/src/gnn_module.py:270-337`
- Modify: `backend/src/api/main.py` (new endpoint)

- [ ] **Step 1: Add module-level training history storage**

At the top of `backend/src/gnn_module.py`, after the imports, add:

```python
# Training history for UI visualization (module-level, survives across requests)
_training_history: dict = {"epochs": [], "early_stop_epoch": None, "best_epoch": None}
```

- [ ] **Step 2: Collect per-epoch metrics in training loop**

In `backend/src/gnn_module.py`, just before the training loop (`for epoch in range(Config.COMPGCN_EPOCHS):`), add:

```python
    epoch_metrics = []
```

Then, at the end of each epoch (after the early stopping check, before the logging block around line 330), add:

```python
        epoch_metrics.append({
            "epoch": epoch + 1,
            "train_loss": round(final_train_loss, 4) if final_train_loss is not None else None,
            "auc_roc": round(current_auc, 4) if current_auc is not None else None,
        })
```

After the training loop ends (after line 339 `model.load_state_dict(best_state)`), add:

```python
    # Store training history for UI
    global _training_history
    _training_history = {
        "epochs": epoch_metrics,
        "early_stop_epoch": epoch + 1 if patience_counter >= Config.COMPGCN_PATIENCE else None,
        "best_epoch": next((i + 1 for i, m in enumerate(epoch_metrics) if m["auc_roc"] == (best_auc if best_auc > float("-inf") else None)), None),
        "final_auc_roc": round(final_auc, 4) if final_auc is not None else None,
        "final_mrr": round(final_mrr, 4) if final_mrr is not None else None,
    }
```

Note: `final_auc` and `final_mrr` are computed after the loop at the existing lines 344-349. Move the history assignment to after those lines (after line 356).

- [ ] **Step 3: Export getter function**

Add at the bottom of `backend/src/gnn_module.py` (before `if __name__`):

```python
def get_training_history() -> dict:
    """Return the last training run's per-epoch metrics."""
    return _training_history
```

- [ ] **Step 4: Add /audit/training-history endpoint**

In `backend/src/api/main.py`, add the import at the top alongside `run_audit`:

```python
from src.gnn_module import run_audit, get_training_history
```

Then add the endpoint after the `/audit/findings` endpoint:

```python
@app.get("/audit/training-history")
async def get_audit_training_history():
    """Returns per-epoch GNN training metrics for visualization."""
    return get_training_history()
```

- [ ] **Step 5: Commit**

```bash
git add backend/src/gnn_module.py backend/src/api/main.py
git commit -m "feat: log per-epoch GNN training metrics and expose /audit/training-history"
```

---

### Task 3: Return Filtered Triplets and Structured Grounding Error

**Files:**
- Modify: `backend/src/generator.py:16-94`
- Modify: `backend/src/api/main.py:413-482`

- [ ] **Step 1: Modify generator to track filtered triplets**

In `backend/src/generator.py`, replace the `generate_answer` method body (lines 16-94) with:

```python
    async def generate_answer(self, query, explain: bool = False):
        """
        Generates an auditable answer using the Synthesis Layer.
        Returns filtered_triplets for UI transparency.
        """
        # 1. Retrieve Subgraph Context (Triples) + Discovery Leads
        _, triplets, leads = self.retriever.retrieve(query)
        validated_statuses = {"trained_experimental", "validated"}
        validated_triplets = [
            triplet
            for triplet in triplets
            if triplet.get("audit") is not None and triplet.get("audit_status") in validated_statuses
        ]
        # Track what was filtered out
        filtered_triplets = [
            triplet
            for triplet in triplets
            if triplet not in validated_triplets
        ]
        # Fallback: if no validated triplets (e.g. unaudited graph), use all with audit score >= threshold
        if not validated_triplets and triplets:
            min_score = Config.GROUNDING_MIN_SCORE
            validated_triplets = [
                t for t in triplets
                if t.get("audit") is not None and (t.get("audit") or 0) >= min_score
            ]
            filtered_triplets = [t for t in triplets if t not in validated_triplets]

        # Grounding error: no validated triplets at all
        if not validated_triplets:
            # Normalize leads for response
            lead_objects = self._normalize_leads(leads)
            return {
                "narrative_text": None,
                "grounding_error": True,
                "message": "No validated evidence found. The system refuses to generate an unsupported answer.",
                "triplets": [],
                "filtered_triplets": filtered_triplets,
                "leads": lead_objects,
                "context_summary": "",
                "suggested_actions": [],
            }

        context = "\n".join(
            f"({t['source']})-[{t['relation']}]->({t['target']})"
            for t in validated_triplets
        )

        # Normalize leads into structured objects
        lead_objects = self._normalize_leads(leads)

        # 2. Synthesis: Convert Triples + Leads to Auditable Narrative
        narrative_package = await generate_narrative_response(
            query,
            validated_triplets,
            lead_objects,
            include_explanations=explain,
        )

        if explain:
            # Index explanations for quick merge
            triplet_explanations = narrative_package.get("triplet_explanations", [])
            explanation_map = {}
            for item in triplet_explanations:
                key = (item.get("source"), item.get("relation"), item.get("target"))
                explanation_map[key] = item.get("explanation")

            lead_explanations = narrative_package.get("lead_explanations", [])
            lead_explanation_map = {
                item.get("name"): item.get("explanation") for item in lead_explanations
            }

            # Attach explanations to triplets/leads
            for t in validated_triplets:
                key = (t.get("source"), t.get("relation"), t.get("target"))
                t["explanation"] = explanation_map.get(key)

            for lead in lead_objects:
                lead["explanation"] = lead_explanation_map.get(lead.get("name"))

        return {
            "narrative_text": narrative_package.get("narrative_text", ""),
            "triplets": validated_triplets,
            "filtered_triplets": filtered_triplets,
            "leads": lead_objects,
            "context_summary": context,
            "suggested_actions": narrative_package.get("suggested_actions", []),
        }

    def _normalize_leads(self, leads):
        """Convert raw lead strings into structured objects."""
        lead_objects = []
        for lead in leads:
            if not lead:
                continue
            if ":" in lead:
                name, desc = lead.split(":", 1)
                lead_objects.append({"name": name.strip(), "description": desc.strip()})
            else:
                lead_objects.append({"name": lead.strip(), "description": None})
        return lead_objects
```

- [ ] **Step 2: Update API grounding error handling**

In `backend/src/api/main.py`, replace the `GROUNDING_ERROR_RESPONSE` and `_is_grounding_error` and `_chat_result_to_response` (lines 413-429):

```python
def _is_grounding_error(result: dict) -> bool:
    """True if the generator returned a structured grounding error."""
    return result.get("grounding_error", False)

def _chat_result_to_response(result: dict, grounding_status: str = "OK - Local Graph") -> dict:
    """Build chat response dict from generator result."""
    result["grounding_status"] = grounding_status
    return result

def _grounding_error_response(result: dict) -> dict:
    """Build grounding error response preserving filtered triplets."""
    return {
        "narrative_text": result.get("message", "No validated evidence found."),
        "triplets": [],
        "filtered_triplets": result.get("filtered_triplets", []),
        "leads": result.get("leads", []),
        "context_summary": "",
        "suggested_actions": [],
        "grounding_status": "FAILED - No Validated Triples Found",
        "grounding_error": True,
    }
```

- [ ] **Step 3: Update /chat endpoint to use new error response**

In the `/chat` endpoint (line 479-480), replace:

```python
    if _is_grounding_error(result):
        return GROUNDING_ERROR_RESPONSE
```

with:

```python
    if _is_grounding_error(result):
        return _grounding_error_response(result)
```

- [ ] **Step 4: Update /chat/stream to include filtered_triplets**

In the `/chat/stream` endpoint's `generate()` function, replace the grounding error yield (around line 443-445):

```python
                    yield f"data: {json.dumps({'type': 'error', **GROUNDING_ERROR_RESPONSE})}\n\n"
```

with:

```python
                    yield f"data: {json.dumps({'type': 'grounding_error', **_grounding_error_response(result)})}\n\n"
```

And update the `done` message to include `filtered_triplets`:

```python
            yield f"data: {json.dumps({'type': 'done', 'triplets': result.get('triplets', []), 'filtered_triplets': result.get('filtered_triplets', []), 'leads': result.get('leads', []), 'suggested_actions': result.get('suggested_actions', []), 'grounding_status': grounding_status})}\n\n"
```

- [ ] **Step 5: Commit**

```bash
git add backend/src/generator.py backend/src/api/main.py
git commit -m "feat: return filtered triplets and structured grounding errors"
```

---

### Task 4: Ablation Results in /stats

**Files:**
- Modify: `backend/src/evaluation.py`
- Modify: `backend/src/api/main.py`

- [ ] **Step 1: Add ablation mode to evaluation**

In `backend/src/evaluation.py`, modify `run_grounding_evaluation` to accept and store a mode parameter. Replace the function signature and the result storage:

```python
async def run_grounding_evaluation(mode: str = "full_stack") -> dict:
    """
    Run grounding and faithfulness evaluation on test queries.
    mode: 'full_stack' | 'graph' | 'prompt_only'
    Returns and persists results to evaluation_results.json.
    """
```

At the end of the function, before writing to file, change the output to include mode:

Replace:

```python
    output = {
        "grounding_score": grounding_score,
        "faithfulness_score": faithfulness_score,
        "completed_at": _utc_now_iso(),
        "sample_count": len(grounding_scores) or len(faithfulness_scores),
    }

    path = Config.EVALUATION_RESULTS_PATH
    if path:
        with open(path, "w") as f:
            json.dump(output, f, indent=2)
        logger.info("Evaluation results written to %s", path)

    return output
```

with:

```python
    output = {
        "grounding_score": grounding_score,
        "faithfulness_score": faithfulness_score,
        "completed_at": _utc_now_iso(),
        "sample_count": len(grounding_scores) or len(faithfulness_scores),
        "mode": mode,
    }

    # Persist per-mode results for ablation comparison
    path = Config.EVALUATION_RESULTS_PATH
    if path:
        existing = {}
        if os.path.exists(path):
            with open(path, "r") as f:
                existing = json.load(f)
        # Store under ablation key
        if "ablation" not in existing:
            existing["ablation"] = {}
        existing["ablation"][mode] = {
            "grounding_score": grounding_score,
            "faithfulness_score": faithfulness_score,
            "completed_at": _utc_now_iso(),
            "sample_count": output["sample_count"],
        }
        # Keep top-level scores as the latest run
        existing["grounding_score"] = grounding_score
        existing["faithfulness_score"] = faithfulness_score
        existing["completed_at"] = _utc_now_iso()
        existing["sample_count"] = output["sample_count"]
        with open(path, "w") as f:
            json.dump(existing, f, indent=2)
        logger.info("Evaluation results written to %s (mode=%s)", path, mode)

    return output
```

- [ ] **Step 2: Expose ablation results in /stats**

In `backend/src/api/main.py`, in the `/stats` endpoint, load ablation data from the evaluation results file and include it. Find where `research_kpis` is constructed (it reads from `evaluation_results.json`). After that block, add:

```python
    # Ablation results
    ablation_results = None
    eval_path = Config.EVALUATION_RESULTS_PATH
    if eval_path and os.path.exists(eval_path):
        try:
            with open(eval_path, "r") as f:
                eval_data = json.load(f)
            ablation_results = eval_data.get("ablation")
        except Exception:
            pass
```

And include in the response dict:

```python
    "ablation": ablation_results,
```

- [ ] **Step 3: Add /evaluate/ablation endpoint**

In `backend/src/api/main.py`, add a new endpoint after `/evaluate`:

```python
@app.post("/evaluate/ablation")
async def trigger_ablation_evaluation(background_tasks: BackgroundTasks):
    """Runs evaluation in all three modes for ablation comparison."""
    async def run_ablation():
        try:
            _system_state.status = "Running Ablation Evaluation..."
            logger.info("Starting ablation evaluation (3 modes)...")
            await run_grounding_evaluation(mode="full_stack")
            await run_grounding_evaluation(mode="prompt_only")
            _system_state.status = "Idle"
            logger.info("Ablation evaluation complete.")
        except Exception as e:
            _system_state.status = f"Evaluation Error: {str(e)}"
            logger.error("Ablation evaluation failure: %s", e)

    background_tasks.add_task(run_ablation)
    return {"message": "Ablation evaluation triggered (3 modes)."}
```

Note: The `prompt_only` mode uses `generate_answer_prompt_only` in the generator. We need to handle this in evaluation. Update the evaluation loop to use the right generator method based on mode:

In `backend/src/evaluation.py`, inside the query loop, replace:

```python
            result = await generator.generate_answer(q, explain=True)
```

with:

```python
            if mode == "prompt_only":
                result = await generator.generate_answer_prompt_only(q)
            else:
                result = await generator.generate_answer(q, explain=True)
```

- [ ] **Step 4: Commit**

```bash
git add backend/src/evaluation.py backend/src/api/main.py
git commit -m "feat: add ablation evaluation mode and expose results in /stats"
```

---

## Phase 2: Visual Theme Update

### Task 5: Update CSS Color Tokens and Card Styles

**Files:**
- Modify: `frontend/app/globals.css`

- [ ] **Step 1: Update CSS variables**

In `frontend/app/globals.css`, replace the `:root` block (lines 4-34):

```css
:root {
  /* Modern Archival Color Palette */
  --parchment-base: #FAFAF8;
  --parchment-light: #FFFFFF;
  --ink-dark: #1A1A1A;
  --ink-medium: #525252;
  --ink-light: #737373;
  --gilded-gold: #C5A028;
  --seal-red: #7A1A1A;
  --conflict-red: #DC2626;
  --validated-green: #2D6A4F;

  /* Semantic color roles */
  --accent-gold: #C5A028;
  --status-validated: #2D6A4F;
  --status-warning: #A68A1E;
  --status-danger: #DC2626;
  --brand-seal: #7A1A1A;

  /* Legacy variables for compatibility */
  --background: #FAFAF8;
  --foreground: #1A1A1A;
  --card: #FFFFFF;
  --card-foreground: #1A1A1A;
  --primary: #C5A028;
  --primary-foreground: #1A1A1A;
  --secondary: #737373;
  --accent: #7A1A1A;
  --muted: #F5F5F3;
  --border: #E5E5E3;
}
```

- [ ] **Step 2: Update body styles**

Replace:

```css
body {
  background-color: var(--parchment-base);
  color: var(--ink-dark);
  font-family: 'Public Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
  font-weight: 400;
}
```

with:

```css
body {
  background-color: var(--parchment-base);
  color: var(--ink-dark);
  font-family: 'Public Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
  font-weight: 400;
  font-size: 15px;
}
```

- [ ] **Step 3: Replace glass effect with clean card style**

Replace:

```css
/* Glass effect updated for parchment theme */
.glass {
  background: rgba(252, 250, 242, 0.85);
  backdrop-filter: blur(8px);
  -webkit-backdrop-filter: blur(8px);
  border: 1px solid var(--ink-medium);
  box-shadow: 0 2px 8px rgba(43, 43, 43, 0.08);
}
```

with:

```css
/* Clean card style */
.glass {
  background: var(--parchment-light);
  border: 1px solid var(--border);
  box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
}

.glass:hover {
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.06);
}
```

- [ ] **Step 4: Update gradient-text to use new gold**

Replace:

```css
.gradient-text {
  background: linear-gradient(135deg, #D4AF37 0%, #B8941F 100%);
```

with:

```css
.gradient-text {
  background: linear-gradient(135deg, #C5A028 0%, #A68A1E 100%);
```

- [ ] **Step 5: Update gilded-glow to use new gold**

Replace all instances of `#D4AF37` in the animation keyframes with `#C5A028`, and `rgba(212, 175, 55,` with `rgba(197, 160, 40,`.

- [ ] **Step 6: Commit**

```bash
git add frontend/app/globals.css
git commit -m "style: update color tokens to modern archival theme"
```

---

### Task 6: Update Pipeline Config Colors

**Files:**
- Modify: `frontend/lib/pipelineConfig.ts`

- [ ] **Step 1: Update PHASE_COLORS**

Find `PHASE_COLORS` in `frontend/lib/pipelineConfig.ts` and update the color values to use the new palette. Replace `#3A5A40` with `#2D6A4F`, `#D4AF37` with `#C5A028`, `#8B1A1A` with `#7A1A1A`. Do this for all border, text, and bg class strings.

- [ ] **Step 2: Commit**

```bash
git add frontend/lib/pipelineConfig.ts
git commit -m "style: update pipeline phase colors to new palette"
```

---

## Phase 3: Frontend Types Update

### Task 7: Extend TypeScript Types

**Files:**
- Modify: `frontend/lib/types.ts`

- [ ] **Step 1: Add filtered_triplets and grounding_error to types**

Replace the full content of `frontend/lib/types.ts`:

```typescript
/** Shared types for The Remembrance Vault frontend. */

export interface Triplet {
  source?: string | null;
  relation?: string | null;
  target?: string | null;
  audit?: number;
  description?: string;
  explanation?: string | null;
  source_docs?: string[];
  target_docs?: string[];
  cross_document?: boolean;
}

export interface Lead {
  name: string;
  description?: string | null;
  explanation?: string | null;
}

export interface ChatMessage {
  role: "user" | "ai";
  content: string;
  triplets?: Triplet[];
  filtered_triplets?: Triplet[];
  leads?: Lead[];
  suggested_actions?: string[];
  userQuery?: string;
  explain?: boolean;
  groundingStatus?: string;
  groundingError?: boolean;
}

export interface AblationResult {
  grounding_score: number | null;
  faithfulness_score: number | null;
  completed_at: string | null;
  sample_count: number;
}

export interface AblationResults {
  full_stack?: AblationResult;
  prompt_only?: AblationResult;
  graph?: AblationResult;
}

export interface EpochMetric {
  epoch: number;
  train_loss: number | null;
  auc_roc: number | null;
}

export interface TrainingHistory {
  epochs: EpochMetric[];
  early_stop_epoch: number | null;
  best_epoch: number | null;
  final_auc_roc: number | null;
  final_mrr: number | null;
}

export interface StageTiming {
  [stage: string]: number;
}
```

- [ ] **Step 2: Commit**

```bash
git add frontend/lib/types.ts
git commit -m "feat: extend types for filtered triplets, ablation, training history"
```

---

## Phase 4: New UI Components

### Task 8: Create TabShell Component

**Files:**
- Create: `frontend/components/TabShell.tsx`

- [ ] **Step 1: Write TabShell**

```typescript
"use client";

import React from "react";
import { useSearchParams, useRouter } from "next/navigation";
import { motion } from "framer-motion";

export interface Tab {
  id: string;
  label: string;
  icon?: React.ReactNode;
}

interface TabShellProps {
  tabs: Tab[];
  children: Record<string, React.ReactNode>;
  defaultTab?: string;
}

export default function TabShell({ tabs, children, defaultTab }: TabShellProps) {
  const searchParams = useSearchParams();
  const router = useRouter();
  const activeTab = searchParams.get("tab") || defaultTab || tabs[0]?.id;

  const setTab = (id: string) => {
    const params = new URLSearchParams(searchParams.toString());
    if (id === (defaultTab || tabs[0]?.id)) {
      params.delete("tab");
    } else {
      params.set("tab", id);
    }
    router.replace(`?${params.toString()}`, { scroll: false });
  };

  return (
    <div>
      {/* Tab Bar */}
      <div className="sticky top-0 z-40 bg-[var(--parchment-base)] border-b border-[var(--border)] mb-6">
        <nav className="max-w-7xl mx-auto flex gap-1 px-2" aria-label="Dashboard tabs">
          {tabs.map((tab) => (
            <button
              key={tab.id}
              onClick={() => setTab(tab.id)}
              className={`relative px-4 py-3 text-sm font-medium transition-colors flex items-center gap-2 ${
                activeTab === tab.id
                  ? "text-[var(--ink-dark)]"
                  : "text-[var(--ink-light)] hover:text-[var(--ink-medium)]"
              }`}
              aria-selected={activeTab === tab.id}
              role="tab"
            >
              {tab.icon}
              {tab.label}
              {activeTab === tab.id && (
                <motion.div
                  layoutId="tab-underline"
                  className="absolute bottom-0 left-2 right-2 h-0.5 bg-[var(--gilded-gold)]"
                  transition={{ type: "spring", stiffness: 400, damping: 30 }}
                />
              )}
            </button>
          ))}
        </nav>
      </div>

      {/* Tab Content */}
      <div role="tabpanel">
        {children[activeTab]}
      </div>
    </div>
  );
}
```

- [ ] **Step 2: Commit**

```bash
git add frontend/components/TabShell.tsx
git commit -m "feat: add TabShell component with animated underline"
```

---

### Task 9: Create GroundingError Component

**Files:**
- Create: `frontend/components/GroundingError.tsx`

- [ ] **Step 1: Write GroundingError**

```typescript
"use client";

import React from "react";
import { ShieldAlert } from "lucide-react";
import { motion } from "framer-motion";
import { Triplet } from "@/lib/types";

interface GroundingErrorProps {
  message: string;
  filteredTriplets: Triplet[];
}

export default function GroundingError({ message, filteredTriplets }: GroundingErrorProps) {
  const [showRejected, setShowRejected] = React.useState(false);

  return (
    <motion.div
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      className="bg-white border border-[var(--border)] border-l-4 border-l-[var(--conflict-red)] rounded-lg p-5"
    >
      <div className="flex items-start gap-3">
        <ShieldAlert size={20} className="text-[var(--conflict-red)] shrink-0 mt-0.5" />
        <div className="flex-1">
          <h4 className="font-semibold text-[var(--conflict-red)] text-sm uppercase tracking-wide mb-1" style={{ fontFamily: "EB Garamond, serif" }}>
            Grounding Error
          </h4>
          <p className="text-sm text-[var(--ink-medium)]">{message}</p>

          {filteredTriplets.length > 0 && (
            <div className="mt-4">
              <button
                onClick={() => setShowRejected(!showRejected)}
                className="text-xs font-mono text-[var(--ink-light)] hover:text-[var(--ink-dark)] transition-colors"
              >
                {showRejected ? "Hide" : "Show"} {filteredTriplets.length} rejected triplet{filteredTriplets.length !== 1 ? "s" : ""}
              </button>

              {showRejected && (
                <div className="mt-2 space-y-2">
                  {filteredTriplets.map((t, i) => (
                    <div key={i} className="flex items-center gap-2 text-xs font-mono text-[var(--ink-light)] bg-[var(--muted)] p-2 rounded">
                      <span className="text-[var(--conflict-red)]">{t.audit != null ? (t.audit * 100).toFixed(0) + "%" : "N/A"}</span>
                      <span className="line-through">
                        {t.source} &rarr; {t.relation} &rarr; {t.target}
                      </span>
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </motion.div>
  );
}
```

- [ ] **Step 2: Commit**

```bash
git add frontend/components/GroundingError.tsx
git commit -m "feat: add GroundingError component for grounding error display"
```

---

### Task 10: Create FilteringSummary and RejectedEvidence Components

**Files:**
- Create: `frontend/components/FilteringSummary.tsx`
- Create: `frontend/components/RejectedEvidence.tsx`

- [ ] **Step 1: Write FilteringSummary**

```typescript
"use client";

import React from "react";
import { Shield } from "lucide-react";
import { Triplet } from "@/lib/types";

interface FilteringSummaryProps {
  validatedCount: number;
  filteredCount: number;
  threshold: number;
}

export default function FilteringSummary({ validatedCount, filteredCount, threshold }: FilteringSummaryProps) {
  const total = validatedCount + filteredCount;
  const passRate = total > 0 ? (validatedCount / total) * 100 : 0;

  return (
    <div className="flex items-center gap-4 p-3 bg-[var(--muted)] rounded-lg text-sm">
      <Shield size={16} className="text-[var(--validated-green)] shrink-0" />
      <div className="flex-1">
        <p className="text-[var(--ink-medium)]">
          <span className="font-mono font-medium text-[var(--ink-dark)]">{total}</span> triplets retrieved
          &rarr; <span className="font-mono font-medium text-[var(--validated-green)]">{validatedCount}</span> passed GNN validation
          (<span className="font-mono">&tau; &ge; {threshold}</span>)
          {filteredCount > 0 && (
            <> &rarr; <span className="font-mono font-medium text-[var(--conflict-red)]">{filteredCount}</span> filtered out</>
          )}
        </p>
        {/* Visual bar */}
        <div className="mt-1.5 h-1.5 bg-[var(--border)] rounded-full overflow-hidden">
          <div
            className="h-full bg-[var(--validated-green)] rounded-full transition-all"
            style={{ width: `${passRate}%` }}
          />
        </div>
      </div>
    </div>
  );
}
```

- [ ] **Step 2: Write RejectedEvidence**

```typescript
"use client";

import React, { useState } from "react";
import { ChevronDown, ChevronUp } from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";
import { Triplet } from "@/lib/types";

interface RejectedEvidenceProps {
  triplets: Triplet[];
  threshold: number;
}

export default function RejectedEvidence({ triplets, threshold }: RejectedEvidenceProps) {
  const [expanded, setExpanded] = useState(false);

  if (triplets.length === 0) return null;

  return (
    <div className="border border-[var(--border)] rounded-lg overflow-hidden">
      <button
        onClick={() => setExpanded(!expanded)}
        className="w-full flex items-center justify-between p-3 text-sm text-[var(--ink-medium)] hover:bg-[var(--muted)] transition-colors"
      >
        <span className="font-medium">
          {triplets.length} Triplet{triplets.length !== 1 ? "s" : ""} Filtered
          <span className="font-mono text-xs ml-1">(Below &tau; = {threshold})</span>
        </span>
        {expanded ? <ChevronUp size={16} /> : <ChevronDown size={16} />}
      </button>

      <AnimatePresence>
        {expanded && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: "auto", opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            className="overflow-hidden"
          >
            <div className="p-3 pt-0 space-y-2">
              {triplets.map((t, i) => (
                <div key={i} className="flex items-start gap-3 p-2 bg-[var(--conflict-red)]/5 rounded text-sm border border-[var(--conflict-red)]/10">
                  <span className="font-mono text-xs font-medium text-[var(--conflict-red)] bg-[var(--conflict-red)]/10 px-1.5 py-0.5 rounded shrink-0">
                    {t.audit != null ? (t.audit * 100).toFixed(0) + "%" : "N/A"}
                  </span>
                  <span className="text-[var(--ink-light)] line-through">
                    {t.source} &rarr; <span className="text-[var(--ink-light)]/60">{t.relation}</span> &rarr; {t.target}
                  </span>
                </div>
              ))}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
```

- [ ] **Step 3: Commit**

```bash
git add frontend/components/FilteringSummary.tsx frontend/components/RejectedEvidence.tsx
git commit -m "feat: add FilteringSummary and RejectedEvidence components"
```

---

### Task 11: Create AblationComparison Component

**Files:**
- Create: `frontend/components/AblationComparison.tsx`

- [ ] **Step 1: Write AblationComparison**

```typescript
"use client";

import React from "react";
import { motion } from "framer-motion";
import { AblationResults } from "@/lib/types";
import { formatScore } from "@/lib/utils";

interface AblationComparisonProps {
  results: AblationResults | null;
}

const MODES = [
  { key: "prompt_only" as const, label: "Prompt Only", description: "Chunk RAG — no graph, no GNN" },
  { key: "graph" as const, label: "Graph (No GNN)", description: "Hybrid retrieval — all edges, no audit" },
  { key: "full_stack" as const, label: "Full Stack", description: "Hybrid retrieval + GNN filter (\u03C4 \u2265 0.95)" },
];

export default function AblationComparison({ results }: AblationComparisonProps) {
  if (!results) {
    return (
      <div className="text-center py-8 text-[var(--ink-light)] text-sm">
        No ablation results yet. Run the ablation evaluation to compare modes.
      </div>
    );
  }

  const getValue = (modeKey: string, metric: "grounding_score" | "faithfulness_score") => {
    const r = results[modeKey as keyof AblationResults];
    return r?.[metric] ?? null;
  };

  const getDelta = (modeKey: string, prevKey: string | null, metric: "grounding_score" | "faithfulness_score") => {
    if (!prevKey) return null;
    const curr = getValue(modeKey, metric);
    const prev = getValue(prevKey, metric);
    if (curr == null || prev == null) return null;
    return curr - prev;
  };

  return (
    <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
      {MODES.map((mode, i) => {
        const isFullStack = mode.key === "full_stack";
        const prevKey = i > 0 ? MODES[i - 1].key : null;
        const grounding = getValue(mode.key, "grounding_score");
        const faithfulness = getValue(mode.key, "faithfulness_score");
        const groundingDelta = getDelta(mode.key, prevKey, "grounding_score");

        return (
          <motion.div
            key={mode.key}
            initial={{ opacity: 0, y: 12 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: i * 0.1 }}
            className={`p-5 rounded-lg border ${
              isFullStack
                ? "border-[var(--gilded-gold)] bg-[var(--gilded-gold)]/5"
                : "border-[var(--border)] bg-white"
            }`}
          >
            <div className="flex items-center justify-between mb-1">
              <h4 className="font-semibold text-sm" style={{ fontFamily: "EB Garamond, serif" }}>
                {mode.label}
              </h4>
              {isFullStack && (
                <span className="text-[9px] font-mono uppercase tracking-wider px-1.5 py-0.5 rounded bg-[var(--gilded-gold)]/15 text-[var(--gilded-gold)] border border-[var(--gilded-gold)]/30">
                  Active
                </span>
              )}
            </div>
            <p className="text-xs text-[var(--ink-light)] mb-4">{mode.description}</p>

            <div className="space-y-3">
              <div>
                <p className="text-xs text-[var(--ink-light)] mb-0.5">Grounding</p>
                <p className="text-2xl font-mono font-bold text-[var(--ink-dark)]">
                  {formatScore(grounding)}
                </p>
                {groundingDelta != null && (
                  <p className={`text-xs font-mono mt-0.5 ${groundingDelta > 0 ? "text-[var(--validated-green)]" : "text-[var(--conflict-red)]"}`}>
                    {groundingDelta > 0 ? "+" : ""}{(groundingDelta * 100).toFixed(0)}% vs {MODES[i - 1].label}
                  </p>
                )}
              </div>
              <div>
                <p className="text-xs text-[var(--ink-light)] mb-0.5">Faithfulness</p>
                <p className="text-2xl font-mono font-bold text-[var(--ink-dark)]">
                  {formatScore(faithfulness)}
                </p>
              </div>
            </div>
          </motion.div>
        );
      })}
    </div>
  );
}
```

- [ ] **Step 2: Commit**

```bash
git add frontend/components/AblationComparison.tsx
git commit -m "feat: add AblationComparison component for three-mode comparison"
```

---

### Task 12: Create TrainingCurves Component

**Files:**
- Create: `frontend/components/TrainingCurves.tsx`

- [ ] **Step 1: Write TrainingCurves**

```typescript
"use client";

import React, { useEffect, useState } from "react";
import { motion } from "framer-motion";
import { API_BASE_URL } from "@/lib/api";
import { TrainingHistory, EpochMetric } from "@/lib/types";
import axios from "axios";

function SvgLineChart({
  data,
  yKey,
  label,
  color,
  earlyStopEpoch,
  secondaryKey,
  secondaryColor,
  secondaryLabel,
}: {
  data: EpochMetric[];
  yKey: "train_loss" | "auc_roc";
  label: string;
  color: string;
  earlyStopEpoch: number | null;
  secondaryKey?: "auc_roc";
  secondaryColor?: string;
  secondaryLabel?: string;
}) {
  if (data.length === 0) return null;

  const W = 400, H = 200, PAD = { top: 16, right: 16, bottom: 28, left: 44 };
  const plotW = W - PAD.left - PAD.right;
  const plotH = H - PAD.top - PAD.bottom;

  const vals = data.map((d) => d[yKey]).filter((v): v is number => v != null);
  const secVals = secondaryKey ? data.map((d) => d[secondaryKey]).filter((v): v is number => v != null) : [];
  const allVals = [...vals, ...secVals];
  const yMin = Math.min(...allVals) * 0.95;
  const yMax = Math.max(...allVals) * 1.05;
  const xMin = data[0]?.epoch ?? 1;
  const xMax = data[data.length - 1]?.epoch ?? 1;

  const scaleX = (epoch: number) => PAD.left + ((epoch - xMin) / Math.max(xMax - xMin, 1)) * plotW;
  const scaleY = (val: number) => PAD.top + plotH - ((val - yMin) / Math.max(yMax - yMin, 0.001)) * plotH;

  const toPath = (key: "train_loss" | "auc_roc") =>
    data
      .filter((d) => d[key] != null)
      .map((d, i) => `${i === 0 ? "M" : "L"} ${scaleX(d.epoch)} ${scaleY(d[key]!)}`)
      .join(" ");

  // Y-axis ticks (5 ticks)
  const yTicks = Array.from({ length: 5 }, (_, i) => yMin + (i / 4) * (yMax - yMin));

  return (
    <div>
      <p className="text-xs font-medium text-[var(--ink-medium)] mb-2">{label}</p>
      <svg viewBox={`0 0 ${W} ${H}`} className="w-full h-auto">
        {/* Grid lines */}
        {yTicks.map((val, i) => (
          <g key={i}>
            <line x1={PAD.left} y1={scaleY(val)} x2={W - PAD.right} y2={scaleY(val)} stroke="var(--border)" strokeWidth="0.5" />
            <text x={PAD.left - 4} y={scaleY(val) + 3} textAnchor="end" className="fill-[var(--ink-light)]" fontSize="9" fontFamily="JetBrains Mono">
              {val < 1 ? val.toFixed(2) : val.toFixed(1)}
            </text>
          </g>
        ))}

        {/* Early stopping marker */}
        {earlyStopEpoch && (
          <>
            <line
              x1={scaleX(earlyStopEpoch)} y1={PAD.top} x2={scaleX(earlyStopEpoch)} y2={H - PAD.bottom}
              stroke="var(--conflict-red)" strokeWidth="1" strokeDasharray="4 3" opacity="0.6"
            />
            <text x={scaleX(earlyStopEpoch)} y={PAD.top - 4} textAnchor="middle" fontSize="8" className="fill-[var(--conflict-red)]" fontFamily="JetBrains Mono">
              Early stop
            </text>
          </>
        )}

        {/* Primary line */}
        <path d={toPath(yKey)} fill="none" stroke={color} strokeWidth="2" />

        {/* Secondary line */}
        {secondaryKey && secondaryColor && (
          <path d={toPath(secondaryKey)} fill="none" stroke={secondaryColor} strokeWidth="1.5" strokeDasharray="4 2" />
        )}

        {/* X-axis label */}
        <text x={W / 2} y={H - 4} textAnchor="middle" fontSize="9" className="fill-[var(--ink-light)]" fontFamily="JetBrains Mono">
          Epoch
        </text>

        {/* Legend */}
        <g transform={`translate(${PAD.left + 8}, ${PAD.top + 8})`}>
          <line x1="0" y1="4" x2="14" y2="4" stroke={color} strokeWidth="2" />
          <text x="18" y="7" fontSize="8" className="fill-[var(--ink-medium)]" fontFamily="JetBrains Mono">{label.split(":")[0] || label}</text>
        </g>
        {secondaryLabel && secondaryColor && (
          <g transform={`translate(${PAD.left + 8}, ${PAD.top + 20})`}>
            <line x1="0" y1="4" x2="14" y2="4" stroke={secondaryColor} strokeWidth="1.5" strokeDasharray="4 2" />
            <text x="18" y="7" fontSize="8" className="fill-[var(--ink-medium)]" fontFamily="JetBrains Mono">{secondaryLabel}</text>
          </g>
        )}
      </svg>
    </div>
  );
}

export default function TrainingCurves() {
  const [history, setHistory] = useState<TrainingHistory | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    axios
      .get(`${API_BASE_URL}/audit/training-history`)
      .then((res) => setHistory(res.data))
      .catch(() => {})
      .finally(() => setLoading(false));
  }, []);

  if (loading) {
    return <div className="h-48 bg-[var(--muted)] rounded-lg animate-pulse" />;
  }

  if (!history || history.epochs.length === 0) {
    return (
      <div className="text-center py-8 text-[var(--ink-light)] text-sm">
        No training history available. Run a GNN audit to see training curves.
      </div>
    );
  }

  return (
    <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="space-y-4">
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <SvgLineChart
          data={history.epochs}
          yKey="train_loss"
          label="Training Loss"
          color="var(--gilded-gold)"
          earlyStopEpoch={history.early_stop_epoch}
        />
        <SvgLineChart
          data={history.epochs}
          yKey="auc_roc"
          label="AUC-ROC"
          color="var(--validated-green)"
          earlyStopEpoch={history.early_stop_epoch}
        />
      </div>

      {/* Final metrics callout */}
      <div className="flex gap-4 text-sm">
        <div className="px-3 py-2 bg-[var(--muted)] rounded font-mono">
          AUC-ROC: <span className="font-bold">{history.final_auc_roc?.toFixed(4) ?? "N/A"}</span>
        </div>
        <div className="px-3 py-2 bg-[var(--muted)] rounded font-mono">
          MRR: <span className="font-bold">{history.final_mrr?.toFixed(4) ?? "N/A"}</span>
        </div>
        {history.early_stop_epoch && (
          <div className="px-3 py-2 bg-[var(--muted)] rounded font-mono">
            Early stop: <span className="font-bold">epoch {history.early_stop_epoch}</span>
          </div>
        )}
      </div>
    </motion.div>
  );
}
```

- [ ] **Step 2: Commit**

```bash
git add frontend/components/TrainingCurves.tsx
git commit -m "feat: add TrainingCurves SVG chart component"
```

---

### Task 13: Create TimingSummary Component

**Files:**
- Create: `frontend/components/TimingSummary.tsx`

- [ ] **Step 1: Write TimingSummary**

```typescript
"use client";

import React from "react";
import { Clock } from "lucide-react";
import { StageTiming } from "@/lib/types";

interface TimingSummaryProps {
  timings: StageTiming | null;
}

const STAGE_META: Record<string, { label: string; phase: "feature" | "training" | "inference" }> = {
  ingest: { label: "Ingestion", phase: "feature" },
  embed: { label: "Embedding", phase: "feature" },
  audit: { label: "GNN Audit", phase: "training" },
  evaluate: { label: "Evaluation", phase: "inference" },
};

const PHASE_COLORS: Record<string, string> = {
  feature: "var(--validated-green)",
  training: "var(--gilded-gold)",
  inference: "var(--brand-seal)",
};

export default function TimingSummary({ timings }: TimingSummaryProps) {
  if (!timings || Object.keys(timings).length === 0) return null;

  const entries = Object.entries(timings)
    .filter(([key]) => STAGE_META[key])
    .map(([key, seconds]) => ({
      key,
      label: STAGE_META[key].label,
      seconds: seconds as number,
      color: PHASE_COLORS[STAGE_META[key].phase],
    }));

  const total = entries.reduce((sum, e) => sum + e.seconds, 0);
  if (total === 0) return null;

  return (
    <div className="p-4 bg-white border border-[var(--border)] rounded-lg">
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-2 text-sm text-[var(--ink-medium)]">
          <Clock size={14} />
          <span className="font-medium">Pipeline Execution Time</span>
        </div>
        <span className="font-mono text-sm font-bold text-[var(--ink-dark)]">{total.toFixed(1)}s total</span>
      </div>

      {/* Stacked bar */}
      <div className="h-3 flex rounded-full overflow-hidden bg-[var(--muted)]">
        {entries.map((e) => (
          <div
            key={e.key}
            style={{ width: `${(e.seconds / total) * 100}%`, backgroundColor: e.color }}
            className="transition-all"
            title={`${e.label}: ${e.seconds.toFixed(1)}s`}
          />
        ))}
      </div>

      {/* Labels */}
      <div className="flex gap-4 mt-2 flex-wrap">
        {entries.map((e) => (
          <div key={e.key} className="flex items-center gap-1.5 text-xs text-[var(--ink-light)]">
            <div className="w-2 h-2 rounded-full" style={{ backgroundColor: e.color }} />
            <span>{e.label}</span>
            <span className="font-mono">{e.seconds.toFixed(1)}s</span>
          </div>
        ))}
      </div>
    </div>
  );
}
```

- [ ] **Step 2: Commit**

```bash
git add frontend/components/TimingSummary.tsx
git commit -m "feat: add TimingSummary stacked bar component"
```

---

## Phase 5: Dashboard Restructure

### Task 14: Restructure page.tsx into Tabbed Layout

This is the largest task. It restructures the main dashboard from 2/3+1/3 split into a tabbed layout.

**Files:**
- Modify: `frontend/app/page.tsx`

- [ ] **Step 1: Update imports**

Add new imports at the top of `page.tsx`:

```typescript
import TabShell, { Tab } from "@/components/TabShell";
import AblationComparison from "@/components/AblationComparison";
import TrainingCurves from "@/components/TrainingCurves";
import TimingSummary from "@/components/TimingSummary";
import GroundingError from "@/components/GroundingError";
import AuditOverview from "@/components/AuditOverview";
import DocumentIntegrity from "@/components/DocumentIntegrity";
import FlaggedEdges from "@/components/FlaggedEdges";
import { Suspense } from "react";
```

Also add these icon imports if not present:

```typescript
import { LayoutDashboard, Workflow, MessageSquare, ShieldCheck } from "lucide-react";
```

- [ ] **Step 2: Define tabs constant**

After the state declarations, add:

```typescript
  const TABS: Tab[] = [
    { id: "overview", label: "Overview", icon: <LayoutDashboard size={16} /> },
    { id: "pipeline", label: "Pipeline", icon: <Workflow size={16} /> },
    { id: "discover", label: "Discover", icon: <MessageSquare size={16} /> },
    { id: "audit", label: "Audit", icon: <ShieldCheck size={16} /> },
  ];
```

- [ ] **Step 3: Add audit state for audit tab**

Add state for audit findings:

```typescript
  const [auditFindings, setAuditFindings] = useState<any>(null);

  const fetchAuditFindings = useCallback(async () => {
    try {
      const response = await axios.get(`${API_BASE_URL}/audit/findings`);
      setAuditFindings(response.data);
    } catch {
      // Audit not yet run — that's fine
    }
  }, []);
```

Call `fetchAuditFindings()` alongside `fetchStats()` in the existing useEffect.

- [ ] **Step 4: Handle grounding error in streaming chat**

In the streaming handler, when parsing SSE messages, add handling for the new `grounding_error` type. Find where `type === "error"` is handled and add:

```typescript
          if (parsed.type === "grounding_error") {
            const aiMsg: ChatMessage = {
              role: "ai",
              content: parsed.narrative_text || parsed.message || "Grounding Error",
              triplets: [],
              filtered_triplets: parsed.filtered_triplets || [],
              groundingStatus: parsed.grounding_status,
              groundingError: true,
            };
            setMessages((prev) => [...prev, aiMsg]);
            setChatLoading(false);
            return;
          }
```

Also update the `done` handler to include `filtered_triplets`:

In the existing `type === "done"` handler, add `filtered_triplets: parsed.filtered_triplets || []` to the message object.

- [ ] **Step 5: Replace the main layout with TabShell**

Replace the entire `{loading && !stats ? (...) : (...)}` main content block (lines ~498-end of main grid) with a tabbed layout. This is a large restructure. The key change:

Remove the two-column grid layout. Replace with:

```tsx
      <Suspense fallback={null}>
        <TabShell tabs={TABS} defaultTab="overview">
          {{
            overview: (
              <main className="max-w-7xl mx-auto space-y-6">
                {/* KPI Summary Cards */}
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                  <StatCard
                    label="Entities"
                    value={stats?.entities ?? 0}
                    icon={<Database size={16} />}
                    subtext={`${stats?.nodes ?? 0} total nodes`}
                  />
                  <StatCard
                    label="Relationships"
                    value={stats?.relationships ?? 0}
                    icon={<Activity size={16} />}
                    subtext={`${stats?.embedding_progress?.toFixed(0) ?? 0}% embedded`}
                  />
                  <StatCard
                    label="Grounding"
                    value={formatScore(stats?.research_kpis?.grounding_score)}
                    icon={<Scale size={16} />}
                    subtext="LLM-as-judge"
                  />
                  <StatCard
                    label="AUC-ROC"
                    value={formatAucRoc(stats?.research_kpis?.gnn_auc_roc)}
                    icon={<BarChart3 size={16} />}
                    subtext="Link prediction"
                  />
                </div>

                {/* Source Documents */}
                {/* Move the existing Source Documents card here, with same upload/ingest/audit buttons */}
                {/* ... existing Source Documents markup ... */}

                {/* Audit Findings Preview */}
                <AuditFindingsCard stats={stats} />
              </main>
            ),

            pipeline: (
              <main className="max-w-7xl mx-auto space-y-6">
                <TimingSummary timings={stats?.stage_timings} />
                <PipelineFlow stats={stats} currentTask={currentTask} />
              </main>
            ),

            discover: (
              <main className="max-w-7xl mx-auto">
                {/* Full-width chat interface — move chat from drawer to here */}
                {/* ... chat messages, input, controls ... */}
              </main>
            ),

            audit: (
              <main className="max-w-7xl mx-auto space-y-6">
                {auditFindings?.audit_run && (
                  <AuditOverview auditRun={{
                    total_audited: auditFindings.audit_run.audited_relationships || 0,
                    total_flagged: auditFindings.flagged_edges?.length || 0,
                    threshold: auditFindings.audit_run.threshold || 0.95,
                    auc_roc: auditFindings.audit_run.auc_roc,
                    mrr: auditFindings.audit_run.mrr,
                  }} />
                )}
                <TrainingCurves />
                <div>
                  <h3 className="text-lg font-semibold mb-4" style={{ fontFamily: "EB Garamond, serif" }}>
                    Ablation Comparison
                  </h3>
                  <AblationComparison results={stats?.ablation} />
                </div>
                {auditFindings?.document_summary && (
                  <DocumentIntegrity
                    documents={auditFindings.document_summary}
                    selectedDoc={null}
                    onSelectDoc={() => {}}
                  />
                )}
                {auditFindings?.flagged_edges && (
                  <FlaggedEdges
                    edges={auditFindings.flagged_edges}
                    filterDoc={null}
                  />
                )}
              </main>
            ),
          }}
        </TabShell>
      </Suspense>
```

**Important notes for this step:**
- The Source Documents card markup stays the same — just move it into the `overview` tab
- The chat interface moves from the right-side drawer into the `discover` tab as a full-width component. Remove the overlay/backdrop. Keep all the chat state and handlers.
- Remove the sidebar column entirely (the archive status, graph stats, research KPIs cards). The KPIs are now StatCards in the overview tab.
- Keep the header above the TabShell (logo, buttons, etc.)
- Remove the "Ask the Archive" button from the header since chat is now a tab
- The confirm modal and error alerts stay above the tab content

- [ ] **Step 6: Update header to be more compact**

Update the header: make the title smaller, remove the "Ask the Archive" button. Replace `text-3xl sm:text-4xl` with `text-2xl`.

- [ ] **Step 7: Update hardcoded colors in page.tsx**

Replace all instances of old color values:
- `#F5F2E9` → use `var(--parchment-base)` or keep as Tailwind
- `#FCFAF2` → `#FFFFFF`
- `#2B2B2B` → `#1A1A1A`
- `#D4AF37` → `#C5A028`
- `#8B1A1A` → `#7A1A1A`
- `#4A4A4A` → use `var(--border)` for borders, `#525252` for text
- `#6B6B6B` → `#737373`
- `#E8E4D9` → `#F5F5F3`
- `#B8941F` → `#A68A1E`
- `#3A5A40` → `#2D6A4F`

Do a search-and-replace across the file.

- [ ] **Step 8: Commit**

```bash
git add frontend/app/page.tsx
git commit -m "feat: restructure dashboard into tabbed layout with all doubt-killer features"
```

---

### Task 15: Update Evidence Page with Plausibility Features

**Files:**
- Modify: `frontend/app/evidence/page.tsx`
- Modify: `frontend/components/DetectiveBoard.tsx`

- [ ] **Step 1: Add FilteringSummary and RejectedEvidence to evidence page**

In `frontend/app/evidence/page.tsx`, add imports:

```typescript
import FilteringSummary from "@/components/FilteringSummary";
import RejectedEvidence from "@/components/RejectedEvidence";
```

After the DetectiveBoard component in the render, add:

```tsx
                {/* Filtering Summary */}
                {(message?.filtered_triplets?.length ?? 0) > 0 && (
                  <FilteringSummary
                    validatedCount={triplets.length}
                    filteredCount={message?.filtered_triplets?.length ?? 0}
                    threshold={0.95}
                  />
                )}

                {/* Rejected Evidence */}
                <RejectedEvidence
                  triplets={message?.filtered_triplets ?? []}
                  threshold={0.95}
                />
```

Also update the `EvidenceMessage` type to include `filtered_triplets`:

```typescript
type EvidenceMessage = {
  role: string;
  content: string;
  triplets?: Triplet[];
  filtered_triplets?: Triplet[];
  leads?: Lead[];
  suggested_actions?: string[];
  userQuery?: string;
  explain?: boolean;
  groundingError?: boolean;
};
```

- [ ] **Step 2: Handle grounding error state on evidence page**

Add a grounding error state render before the main content. After the "no explain" fallback:

```tsx
              {message?.groundingError && (
                <div className="max-w-4xl mx-auto p-8">
                  <GroundingError
                    message={message.content}
                    filteredTriplets={message.filtered_triplets ?? []}
                  />
                  <div className="mt-6">
                    <RejectedEvidence triplets={message.filtered_triplets ?? []} threshold={0.95} />
                  </div>
                </div>
              )}
```

Import GroundingError at the top.

- [ ] **Step 3: Add plausibility badges to DetectiveBoard**

In `frontend/components/DetectiveBoard.tsx`, find where each triplet is rendered (the connection/fact display). Add a plausibility badge next to each triplet. Find the audit score display (there's likely already a shield icon + percentage). Enhance it:

```tsx
                      {step.audit != null && (
                        <span className={`inline-flex items-center gap-1 px-2 py-0.5 rounded text-xs font-mono ${
                          step.audit >= 0.95
                            ? "bg-[var(--validated-green)]/10 text-[var(--validated-green)]"
                            : step.audit >= 0.85
                            ? "bg-[var(--gilded-gold)]/10 text-[var(--gilded-gold)]"
                            : "bg-[var(--conflict-red)]/10 text-[var(--conflict-red)]"
                        }`}>
                          {(step.audit * 100).toFixed(0)}%
                        </span>
                      )}
```

- [ ] **Step 4: Retheme DetectiveBoard colors**

Update hardcoded colors in DetectiveBoard.tsx:
- `#D4AF37` → `#C5A028`
- `#8B1A1A` → `#7A1A1A`
- `#3A5A40` → `#2D6A4F`
- `#4A4A4A` → `var(--border)` for borders
- `#6B6B6B` → `#737373`

- [ ] **Step 5: Retheme evidence page colors**

Same color replacement in `evidence/page.tsx`.

- [ ] **Step 6: Commit**

```bash
git add frontend/app/evidence/page.tsx frontend/components/DetectiveBoard.tsx
git commit -m "feat: add plausibility badges, filtering summary, and grounding error to evidence page"
```

---

### Task 16: Update Audit Page to Redirect

**Files:**
- Modify: `frontend/app/audit/page.tsx`

- [ ] **Step 1: Replace audit page with redirect**

Replace the full content of `frontend/app/audit/page.tsx` with a redirect to the dashboard audit tab:

```typescript
"use client";

import { useEffect } from "react";
import { useRouter } from "next/navigation";

export default function AuditPage() {
  const router = useRouter();

  useEffect(() => {
    router.replace("/?tab=audit");
  }, [router]);

  return null;
}
```

- [ ] **Step 2: Commit**

```bash
git add frontend/app/audit/page.tsx
git commit -m "refactor: redirect /audit to /?tab=audit"
```

---

## Phase 6: Retheme Existing Components

### Task 17: Retheme All Remaining Components

**Files:** All component files listed in the "Modified Components" section of the file map.

This task is a batch color replacement across all remaining components. For each file, replace old color hex codes with new ones:

| Old | New | Context |
|-----|-----|---------|
| `#F5F2E9` | `#FAFAF8` | Backgrounds |
| `#FCFAF2` | `#FFFFFF` | Card backgrounds |
| `#2B2B2B` | `#1A1A1A` | Dark text |
| `#4A4A4A` | `#525252` or `var(--border)` | Medium text or borders |
| `#6B6B6B` | `#737373` | Light text |
| `#D4AF37` | `#C5A028` | Gold accent |
| `#B8941F` | `#A68A1E` | Dark gold |
| `#8B1A1A` | `#7A1A1A` | Seal red |
| `#C41E3A` | `#DC2626` | Danger red |
| `#3A5A40` | `#2D6A4F` | Validated green |
| `#E8E4D9` | `#F5F5F3` | Muted background |

- [ ] **Step 1: Retheme PipelineStrip.tsx**

Search and replace the old color values. Pay special attention to `PHASE_HEX` and `colorMap` constants.

- [ ] **Step 2: Retheme PipelineDetail.tsx**

Replace old colors in status badges and phase indicators.

- [ ] **Step 3: Retheme PipelineFlow.tsx**

Replace old colors.

- [ ] **Step 4: Retheme AuditOverview.tsx**

Replace colors in `integrityColor` function and card styling. Update thresholds if needed.

- [ ] **Step 5: Retheme DocumentIntegrity.tsx**

Replace colors in filter buttons and card styling.

- [ ] **Step 6: Retheme FlaggedEdges.tsx**

Replace colors in risk level badges and triplet display.

- [ ] **Step 7: Retheme StatCard.tsx**

Replace `glass` class usage with clean card style. Update border and hover colors.

- [ ] **Step 8: Retheme KnowledgeGraph.tsx**

Replace node/edge colors. Add zoom reset button in the top-right corner of the graph container:

```tsx
            <button
              onClick={() => { setZoom(1); setPan({ x: 0, y: 0 }); }}
              className="absolute top-2 right-2 p-1.5 bg-white border border-[var(--border)] rounded text-[var(--ink-light)] hover:text-[var(--ink-dark)] text-xs font-mono z-10"
              title="Reset zoom"
            >
              Reset
            </button>
```

- [ ] **Step 9: Retheme GraphInfoCard.tsx**

Replace old colors.

- [ ] **Step 10: Retheme ConfirmModal.tsx**

Replace old colors.

- [ ] **Step 11: Retheme AuditFindingsCard.tsx**

Replace old colors.

- [ ] **Step 12: Retheme Skeleton.tsx**

Update skeleton background colors to match new muted color.

- [ ] **Step 13: Commit all retheme changes**

```bash
git add frontend/components/
git commit -m "style: retheme all components to modern archival palette"
```

---

## Phase 7: Integration Testing

### Task 18: Manual Smoke Test

- [ ] **Step 1: Start backend and verify new endpoints**

```bash
cd backend && uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
```

Test:
- `GET /stats` — verify `stage_timings` and `ablation` keys exist
- `GET /audit/training-history` — verify returns `{epochs: [], ...}`
- `POST /chat` with a query — verify `filtered_triplets` in response

- [ ] **Step 2: Start frontend and verify tabbed layout**

```bash
cd frontend && npm run dev
```

Verify:
- 4 tabs visible (Overview, Pipeline, Discover, Audit)
- Tab switching works, URL updates with `?tab=`
- Overview shows KPI cards and documents
- Pipeline shows PipelineFlow + TimingSummary
- Discover shows full-width chat
- Audit shows AuditOverview + TrainingCurves + AblationComparison

- [ ] **Step 3: Test grounding error flow**

Submit a query that should trigger grounding error. Verify:
- Chat shows GroundingError component (red left border, shield icon)
- "View Evidence" navigates to evidence page with grounding error state
- Rejected triplets shown with scores

- [ ] **Step 4: Test evidence page**

Submit a normal query with explain mode. Navigate to evidence. Verify:
- Plausibility badges show on each triplet
- FilteringSummary appears if filtered_triplets > 0
- RejectedEvidence collapsible works

- [ ] **Step 5: Run existing tests**

```bash
cd backend && pytest
```

Expected: All 5 tests pass.

- [ ] **Step 6: Commit any fixes**

```bash
git add -A
git commit -m "fix: integration fixes from smoke testing"
```
