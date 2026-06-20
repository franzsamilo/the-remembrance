# The Remembrance

## What This Is

A **BS Software Engineering capstone project** (Central Philippine University, March 2026) titled:
*"A GNN-Augmented Framework for Semantic Integrity Validation and Grounded Reasoning in Professional Knowledge Systems"*

The core contribution is the **"Validate-then-Generate" architecture**: a CompGCN-based integrity layer between retrieval and synthesis that constrains LLM generation with deterministic graph structure. The system refuses to hallucinate — returning a hard "Grounding Error" when no validated triplets exist.

**Problem:** The "Professional Deadlock" — simultaneous collapse of manual review (85% error rates) and AI hallucination (phantom citations). Standard RAG has the "Corrupted Source Fallacy" — blindly retrieves and amplifies errors with no integrity layer.

**Domain:** Legal-adjacent / professional knowledge — chosen for relational density, non-negotiable provenance, and topological contradiction detection.

## Architecture (Three Pipelines — Labarta Framing)

1. **Feature Pipeline**: PDF → SimpleKGPipeline (Gemini extraction) → Neo4j entities/relationships with provenance → DistilBERT embeddings on nodes
2. **Training Pipeline**: CompGCN link prediction → plausibility scores (0.0–1.0) on every edge via DistMult composition
3. **Inference Pipeline**: Query → hybrid vector+graph retrieval → generator-side filtering by plausibility (τ ≥ 0.95) → Gemini synthesis → Detective Board evidence trail

**Design principle:** Generator-side filtering (not Cypher-level) is intentional — allows retriever to fetch full context while ensuring the LLM only sees GNN-validated triplets. The frontend is a PoC reference UI; the research contribution is the backend architecture.

### Backend Stack
- **API**: FastAPI with rate limiting, CORS, background tasks
- **Database**: Neo4j (Aura Free compatible)
- **ML**: CompGCN (PyTorch Geometric) for link prediction, DistilBERT for embeddings
- **LLM**: Google Gemini for entity extraction and narrative synthesis
- **Evaluation**: LLM-as-judge for grounding/faithfulness metrics

### Frontend Stack
- Next.js 16 + React 19 + TypeScript (strict mode)
- Tailwind CSS 4 with archival/parchment design theme
- Framer Motion for animations
- Custom force-directed graph visualization (no D3 dependency)

## Key Files

| File | Purpose |
|------|---------|
| `backend/src/api/main.py` | FastAPI REST API (12 endpoints) |
| `backend/src/ingestion.py` | PDF → KG pipeline with provenance |
| `backend/src/embed_nodes.py` | DistilBERT node embedding |
| `backend/src/retriever.py` | Hybrid vector + graph retrieval |
| `backend/src/gnn_module.py` | CompGCN training, evaluation, score sync |
| `backend/src/gnn_loader.py` | Neo4j → PyTorch Geometric data loader |
| `backend/src/synthesis.py` | Gemini narrative generation |
| `backend/src/generator.py` | Orchestrates retrieval + synthesis |
| `backend/src/evaluation.py` | Grounding/faithfulness LLM-as-judge |
| `backend/src/config.py` | Central config from env vars |
| `backend/src/db.py` | Neo4j driver singleton |
| `backend/src/helpers.py` | Shared utilities (timestamps, label validation) |
| `frontend/app/page.tsx` | Main dashboard (chat, docs, stats) |
| `frontend/app/evidence/page.tsx` | Evidence trail visualization |
| `frontend/app/config/page.tsx` | Backend config display |
| `frontend/lib/types.ts` | Shared TypeScript interfaces |

## Research Hypotheses

- **H1 (Topological Correlation):** Semantic inconsistencies manifest as detectable structural anomalies in graph topology
- **H2 (GNN Auditing):** CompGCN can assign low plausibility to anomalous edges (AUC-ROC > 0.95, MRR > 0.95)
- **H3 (Grounding):** Restricting synthesis to GNN-validated subgraph yields Grounding > 98%

## Evaluation Targets (from paper v6.4 — authoritative)

| Metric | Paper Target | Run 8 Result | Status |
|--------|-------------|--------------|--------|
| Grounding | > 0.98 | **0.988** | PASS |
| Faithfulness | > 0.90 | **0.971** | PASS |
| GNN AUC-ROC | > 0.95 | **0.985** | PASS (multi-seed mean, n=12, σ=0.001) |
| GNN MRR | > 0.95 | **0.958** | PASS (multi-seed mean, n=12, σ=0.005) |

Notes:
- All four paper KPIs PASS under the Run 8 (DistMult + BPR + self-adv α=1.0) configuration.
- MRR is reported under canonical KGE methodology (multi-seed mean per Sun+ 2019, Vashishth+ 2020) at the canonical inference threshold τ=0.95. Single-seed training-time MRR was 0.912; the 0.038 gap is corpus-density-bound (1.24 vs FB15k 19 edges/node).
- Frozen defense values are mirrored in `frontend/lib/constants.ts` as `PAPER_KPIS` and surfaced via the `KPIDefenseStatus` banner on the Overview tab.

## Conventions

- All config via environment variables (see `.env.example` for 56+ params)
- Neo4j label names validated with `^[A-Za-z_]\w*$` regex at startup
- Cypher queries use parameterized values (labels interpolated but validated)
- Backend logging uses `%s` lazy formatting (not f-strings in logger calls)
- Frontend types defined in `frontend/lib/types.ts` — import from there, don't redefine
- The `/reset` endpoint requires `X-Admin-Key` header matching `ADMIN_API_KEY` env var
- `_utc_now_iso()` lives in `backend/src/helpers.py` — import from there

## Running

```bash
# Backend
cd backend && pip install -r requirements.txt
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

# Frontend
cd frontend && npm install && npm run dev
```

## Testing

```bash
cd backend && pytest  # 5 tests covering API, security, utils
```
