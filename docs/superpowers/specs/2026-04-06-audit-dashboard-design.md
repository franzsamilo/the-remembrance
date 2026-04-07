# Audit Dashboard Design — "Detective Findings"

**Date:** 2026-04-06
**Goal:** Add a dedicated `/audit` page that proactively surfaces what the GNN flagged as suspicious — low-plausibility edges, per-document integrity scores, and overall audit health. Plus a compact findings card on the main dashboard linking to it. This transforms the system from "query-driven chat wrapper" to "audit-driven digital detective."

## Context

The panel feedback is that the system looks like a GPT wrapper. The GNN audit produces plausibility scores on every edge, but the frontend never shows what it found. The detective does the work but nobody sees the findings. This feature makes the audit output a first-class citizen.

## Backend: `GET /audit/findings`

New endpoint in `backend/src/api/main.py`. No new files needed.

### Response Shape

```json
{
  "audit_run": {
    "run_id": "uuid",
    "completed_at": "ISO timestamp",
    "auc_roc": 0.89,
    "mrr": 0.87,
    "total_audited": 142,
    "total_flagged": 5,
    "threshold": 0.95
  },
  "document_summary": [
    { "document": "case_study.pdf", "total_edges": 48, "flagged": 2, "integrity": 0.958 }
  ],
  "flagged_edges": [
    {
      "source": "Entity A",
      "relation": "CONTRADICTS",
      "target": "Entity B",
      "plausibility_score": 0.42,
      "audit_status": "trained_experimental",
      "source_docs": ["case_study.pdf"],
      "target_docs": ["legal_brief.pdf"],
      "cross_document": true,
      "description": "edge description if available"
    }
  ]
}
```

### Implementation

Two Cypher queries:
1. `MATCH (run:AuditRun) RETURN run ORDER BY run.completed_at DESC LIMIT 1` — latest audit metadata
2. `MATCH (s)-[r]->(t) WHERE r.plausibility_score IS NOT NULL RETURN ...` — all audited edges with scores, names, and provenance

Document summary: computed in Python by grouping all audited edges by their source_documents, counting total vs flagged (score < threshold), computing integrity as `1 - (flagged / total)`.

Returns `{"audit_run": null, "document_summary": [], "flagged_edges": []}` when no audit has run.

## Frontend: `/audit` Page

### Layout

Full-page dashboard at `frontend/app/audit/page.tsx` with the archival theme.

**Header:** "Integrity Audit Report" with audit timestamp and overall stats.

**Section 1 — Audit Overview Strip:**
Three stat cards in a row:
- Relationships Audited (total count)
- Flagged (count, red-tinted if > 0)
- Overall Integrity (percentage = 1 - flagged/total, green if > 95%, amber if > 85%, red below)

Plus AUC-ROC and MRR badges showing model performance.

**Section 2 — Document Integrity:**
Horizontal cards, one per document. Each shows:
- Document name
- Integrity bar (visual progress bar, colored by score)
- "X of Y relationships validated"
- Flagged count badge

Clicking a document filters the flagged edges below to that document only.

**Section 3 — Flagged Relationships (the case file):**
Table/list of edges with plausibility_score < threshold, sorted by score ascending (most suspicious first):
- Source → Relation → Target (styled as triplet)
- Plausibility score (with colored bar: red < 0.5, amber < 0.8, gold < threshold)
- Source documents (badges)
- Cross-document flag (golden badge if true)

If no audit has run: empty state with "Run the GNN Audit from the pipeline to see findings."
If audit ran but 0 flagged: success state with "All relationships passed integrity validation."

### Files

```
frontend/
  app/audit/page.tsx          — The audit dashboard page
  components/AuditOverview.tsx — Overview strip (3 stat cards + model metrics)  
  components/DocumentIntegrity.tsx — Per-document integrity cards with filter
  components/FlaggedEdges.tsx  — Flagged relationships list/table
```

### Data Flow

Page fetches `GET /audit/findings` on mount. Optional: also uses stats from `GET /stats` for consistency. All data in one fetch — no polling needed (audit results are static until re-run).

## Frontend: Main Dashboard Findings Card

### Location

In `page.tsx`, below the pipeline visualization, above the chat section.

### Behavior

- Fetches flagged count from `GET /audit/findings` (or from stats if we add the count there)
- Shows a compact card: "Detective Findings" header, total audited, flagged count, integrity %, "View Full Report →" link
- If no audit has run: card says "No audit yet — run the GNN audit to detect anomalies"
- If audit clean: green card "All N relationships validated — no anomalies detected"
- If flagged > 0: amber/red card with count, links to /audit

### File

Small `AuditFindingsCard.tsx` component mounted in page.tsx.

## Phase Colors & Theme

Continues the archival theme:
- Clean relationships: validated-green (`#3A5A40`)
- Flagged/suspicious: seal-red (`#8B1A1A`) 
- Cross-document: gilded-gold (`#D4AF37`)
- Score bars: gradient from red → amber → green

## Navigation

Add "Audit" link to the navigation strip on the main page (the existing Upload → Pipeline → Audit → Chat flow). The config page's Graph & Audit tab stays as-is (developer view).
