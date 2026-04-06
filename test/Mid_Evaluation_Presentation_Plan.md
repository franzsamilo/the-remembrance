# Mid Evaluation — Presentation Plan (The Remembrance)

**Audience:** Panel / adviser mid evaluation  
**Goal:** Refresh the study, show pipeline vs standard RAG, map objectives → features → progress, justify legal-adjacent domain, end with a **guided demo** (or recorded fallback).

**Timing tip:** ~12–18 slides + 8–12 min demo = ~25–35 min total; adjust depth to your slot.

---

## Part A — Opening & problem (your first two slides)

### Slide 1 — Problem statement (refresh)
**Title:** *Professional knowledge under pressure: two failures*

**Bullets (keep tight):**
- **Volume vs. humans:** Literature and case materials outpace manual review; errors compound in synthesis (your thesis stats: systematic review error rates, time cost).
- **Generative AI risk:** LLMs optimize fluency, not truth — phantom citations, unsupported claims in high-stakes settings.
- **Gap:** Need systems that **structure evidence**, **score trust in relationships**, and **refuse to invent** when evidence is missing.

**One line takeaway:** *We are not replacing lawyers or reviewers — we are testing an architecture that makes claims accountable to a graph.*

**Optional speaker note:** Name your capstone title once: *GNN-augmented integrity + grounded generation.*

---

### Slide 2 — Standard RAG vs this pipeline (comparison)
**Title:** *From “similar chunks” to “validated graph structure”*

| Dimension | Typical RAG | This framework (Validate-then-Generate) |
|-----------|-------------|----------------------------------------|
| Retrieval unit | Text chunks | **Graph triplets** (typed entities & relations) |
| Integrity check | None / soft prompting | **GNN plausibility** on edges + audit status |
| Failure mode | Hallucination | **Grounding error** when no validated evidence |
| Provenance | Document-level or vague | **Per-triplet** source / cross-document flags |
| “Why not just prompt?” | Instructions are soft | **Scores and graph traversal are not promptable** |

**Visual:** Simple 2-column table or small diagram: PDF → Graph → Audit → Filter → LLM (only validated context).

**Takeaway:** *Same LLM family; different substrate — the graph + GNN is the trust layer.*

---

## Part A2 — Three pipelines (Labarta + Vertex framing)

**Slide:** *Feature → Training → Inference (not just a chat UI)*

Use the ready-made slide pack: **[`test/Three_Pipelines_Slide.md`](Three_Pipelines_Slide.md)** — title, subtitle, Mermaid diagram, mapping table, and links to [Labarta (feature / training / inference)](https://paulabartabajo.substack.com/p/lets-build-a-real-time-ml-system-efb) and [Google Cloud Vertex AI foundations (pipelines & MLOps thinking)](https://cloud.google.com/blog/topics/developers-practitioners/vertex-ai-foundations-secure-and-compliant-mlai-deployment).

**Talk track (30 s):** *“We modularize like production ML: **features** = PDFs to graph + embeddings; **training** = CompGCN scores on edges; **inference** = query, filter triplets, then Gemini. The website is a reference UI—the contribution is this split.”*

**POC alignment:** The main dashboard now includes an expandable **“How this system runs”** section fed by **`GET /config`** (seven stages) and live status from **`GET /stats`**, plus a labeled strip (Feature · Training · Inference).

---

## Part B — Objectives & what you built

### Slide 3 — Research objectives (from your paper)
**Title:** *What this study set out to do*

Use **3–4 bullets** aligned with Chapter 1 (shorten wording):
1. **Ingest** unstructured PDFs into a **schema-guided knowledge graph** (entities + relations).
2. **Audit** relationships with a **CompGCN-style** link model → **plausibility scores** on edges.
3. **Generate** answers **only from filtered triplets** + show an **evidence trail** (Detective Board / citations).
4. **Evaluate** with **GNN metrics** (AUC, MRR) and **generative metrics** (grounding, faithfulness via LLM-as-judge).

**Takeaway:** *Objectives = build + measure, not “make a chatbot.”*

---

### Slide 4 — Features mapped to objectives + progress
**Title:** *Implementation status (mid evaluation)*

| Objective | Feature / artifact | Status (adjust to truth) |
|-----------|-------------------|-------------------------|
| Graph from PDFs | SimpleKGPipeline + Neo4j + upload UI | **Done / demo-ready** |
| Embeddings | DistilBERT on nodes | **Done** (note: separate embed step if you say so) |
| Relationship audit | POST /audit, CompGCN, scores on edges | **Done** (time: training on laptop) |
| Grounded Q&A | DiscoveryGenerator + filter + Gemini synthesis | **Done** |
| Streaming UX | POST /chat/stream | **Done** |
| Evidence UI | Evidence page, Detective Board, optional graph view | **Done / partial** |
| Evaluation | POST /evaluate, `evaluation_results.json`, Config KPIs | **Done** (scores = baseline, not final thesis claim) |
| Optional | Prompt-only ablation, Aura agent | **Optional / WIP** |

**Honest one-liner for panel:** *Core pipeline works; evaluation numbers are a first pass — thesis work continues toward targets in EVALUATION.md.*

---

## Part C — Why law / legal-adjacent domain?

### Slide 5 — Domain choice (last defense requirement)
**Title:** *Why we stress-tested on legal-style knowledge*

**Bullets (expand beyond “complex”):**
- **Relational density:** Parties, statutes, cases, holdings, citations — naturally **multi-relational** (fits CompGCN / typed edges).
- **Time and versioning:** Facts depend on **when** something held; graphs can represent **temporal and dependency** structure better than flat chunks alone.
- **Provenance is non-negotiable:** Decisions require **which document** supported **which claim** — matches your **Detective Board / source_docs** design.
- **High cost of error:** Parallel to medicine/policy — good test for **“no evidence → no answer”** behavior.
- **Contradiction & chains:** Precedent, distinguishing cases, and **conflicting sources** are **topological** questions — aligned with “semantic integrity” and future anomaly work.
- **Not claiming legal advice:** Frame as **legal-information / archival research** corpus — methodology generalizes to policy, compliance, systematic reviews.

**Takeaway:** *Law is a hard case for structure + accountability — if the architecture is coherent here, the story travels.*

---

## Part D — Demo (slides + script)

**Order for teachers / panel (narrate the three pipelines *before* clicking chat):**  
1) **Feature** (corpus + graph + embeddings) → 2) **Training** (audit / plausibility) → 3) **Inference** (query + evidence). This matches the feedback to show *how* the pipeline does work, not only the chat drawer.

### Slide 6 — Demo overview
**Title:** *Live demo: ~8 minutes, three pipeline blocks + query*

1. **Feature** — Expand **“How this system runs”** on the main page (or scroll it). Point to stages 1–4 from the live config (PDF → extraction → Neo4j → DistilBERT). Show **Source Document Archive** + **stats row** (entities / relationships, embedding progress if visible).
2. **Training** — **Backend Config** → Overview / Graph & Audit: GNN audit, **plausibility** on relationships, optional KPIs. Say: *“This is where the model is trained on the graph and scores edges.”*
3. **Inference** — **Evidence-backed inquiry** → stream answer → **View Evidence** / Detective Board (and Graph View if wired).
4. **Optional close** — One sentence on **Evaluation** (stage 7 / `POST /evaluate`) if asked.

**Backup:** 30–60 s screen recording if live API/Neo4j fails.

---

### Slide 7 — Pre-demo checklist (for you, not on screen — or minimal “Setup” slide)
- Backend running; `.env` valid (Neo4j, `GOOGLE_API_KEY`).
- Frontend `npm run dev`; at least one PDF ingested; **Run Pipeline** completed; **Audit** run once if you want GNN scores.
- Browser: single window; zoom 100%; **do not** rely on incognito unless keys are there.
- **Test query** typed in Notes app — short, answerable from your PDFs.

---

### Demo script (speaker guide) — **Feature → Training → Inference** first

**Beat 0 — Hook (20 s)**  
- *“I’ll walk the three pipelines—feature, training, inference—then show a query.”*  
- **Do not** open the chat drawer first.

**Beat 1 — Feature pipeline (1–1.5 min)**  
- On the main page: expand **“How this system runs: Feature → Training → Inference”**; briefly name **stages 1–4** (ingest, LLM extraction, Neo4j, embeddings).  
- Scroll to **Source Document Archive**; name PDFs (reader context).  
- Point to **stats** (entity/relationship counts, readiness banner).  
- If time: **Run Pipeline** (or say you pre-ran).

**Beat 2 — Training pipeline (1 min)**  
- Open **Backend Config** → **Overview** / **Graph & Audit**: **GNN audit**, plausibility, AUC/MRR or audit state.  
- One line: *“Training here means the CompGCN pass that scores relationships—not fine-tuning Gemini.”*

**Beat 3 — Inference pipeline (2–3 min)**  
- Return to main page → **Evidence-backed inquiry**; **Explain Connections** on.  
- Ask your **prepared question**; let stream finish.  
- **View Evidence** → **Detective Board** (Graph View if available).

**Beat 4 — Close (30 s)**  
- Restate the three blocks: *features in the graph → trained scores on edges → inference only after filter.*  
- *Grounding error if no validated evidence; thesis continues on metrics and corpus.*

---

### Slide 8 (optional) — Risks & next steps
**Title:** *From mid evaluation to final defense*

- Widen or fix **corpus**; repeat **POST /evaluate**; log **GNN** snapshot with each run.  
- Tighten **grounding/faithfulness** vs targets in EVALUATION.md.  
- Optional: **user study** or **ablation slide** (Prompt only vs full stack) if panel asks.

---

## Quick Q&A prep (panel may ask)

- **“Is this legal advice?”** — No; research prototype on **information structuring and integrity**.  
- **“Why GNN?”** — Learned **edge plausibility** from graph topology; not replaceable by prompting alone.  
- **“What if the graph is wrong?”** — System validates **internal consistency** of extracted structure, not universal truth.  
- **“Numbers?”** — Show **baseline** from `evaluation_results.json` + **targets** from EVALUATION.md; say **work in progress** if below target.

---

## File reference (if panel wants artifacts)

| Artifact | Role |
|----------|------|
| `test/Three_Pipelines_Slide.md` | Slide copy + Mermaid for Feature / Training / Inference |
| `EVALUATION.md` | Targets, ablations |
| `backend/evaluation_queries.json` | Fixed evaluation questions |
| `backend/evaluation_results.json` | Last grounding/faithfulness snapshot |
| `TESTS.md` | Manual + pytest checklist |
| `frontend/components/PipelineStory.tsx` | Main-page expandable pipeline (config + stats) |

---

*End of plan — adjust slide count by merging Slides 6–8 or splitting Slide 4 if your slot is short.*

---

## Appendix — How to explain the ERD and the “other” diagram (sequence / flow)

Use this when the panel asks *“What does your ERD mean?”* or *“Walk us through the runtime graph.”*

### A. The ERD (Entity–Relationship diagram) — *what is stored*

**One-liner:**  
*“This is the **logical schema** of our Neo4j-backed archive: it ties **runs of work** to **source PDFs**, to **extracted entities and relationships**, and to **audit runs** that score whether each relationship is plausible.”*

**Tell the story left to right (or in pipeline order):**

1. **IngestionRun** — One execution of “ingest these PDFs.” It **groups** everything created in that pass so you can trace *when* structure entered the system.

2. **SourceDocument** — The actual file in the library (e.g. filename, status). This is the **anchor for provenance**: entities and relationships should be traceable back to **which PDF** they came from.

3. **Entity** (in the diagram; in the product you often have **typed** labels like Person, Concept, etc.) — A **node** with a name, **embedding** (vector for similarity search), and provenance links to documents. Embeddings are what let the retriever find **seeds** before graph expansion.

4. **Relationship** — A **directed edge** between two entities: a **type** (the relation), a **plausibility score** from the GNN audit, and **audit status**. This is the core of “validate-then-generate”: we don’t only store text—we store **typed claims** and **confidence on the link**.

5. **AuditRun** — One execution of the integrity pass. It **scores** many relationships—conceptually “this batch of edges was evaluated together.”

**Cardinality (Crow’s foot) in plain English:**  
- One ingestion run **produces** many documents / entities / relationships.  
- One document **supports** many entities and relationships (provenance).  
- One audit run **scores** many relationships.  
- Each relationship still has **one** source entity and **one** target entity.

**Closing line for the panel:**  
*“So the ERD is not ‘pretty boxes’—it’s the **contract** for **provenance** (which PDF) and **integrity** (which edge passed the model).”*

**Honest nuance (if they push):**  
The drawing may simplify labels; the live graph uses **specific node labels** from your legal/research schema and properties like `source_document` / `source_documents` on nodes and edges. The **idea** of the ERD matches the implementation.

---

### B. The other diagram — usually the **sequence diagram** (query / runtime flow)

**One-liner:**  
*“This is **what happens when the user asks a question**: not where data lives, but **which component calls which**, and where we **enforce** grounding.”*

**Walk the swimlanes:**

1. **User → Frontend → FastAPI** — Submit query (in the app, often `POST /chat/stream`).

2. **FastAPI → DiscoveryGenerator** — Start generation.

3. **DiscoveryGenerator → GraphRetriever → Neo4j** — **Vector + graph**: find similar entities, expand to neighboring triplets, optionally community “leads.”

4. **Back to DiscoveryGenerator — filter** — Keep triplets that pass **audit / plausibility rules** (e.g. audited status or score ≥ threshold). *This step is the integrity gate before the LLM sees context.*

5. **Branch:**  
   - **If no usable triplets** → **grounding error** to the user (no invented answer).  
   - **If there are triplets** → **structured prompt to Gemini** → narrative + evidence metadata for the **Detective Board**.

**Contrast with the ERD in one sentence:**  
*“The **ERD** is the **database design**; the **sequence diagram** is the **request lifecycle** for one query.”*

---

### C. If “the other graph” means the **pipeline / flowchart** (PDF → … → KPIs)

**One-liner:**  
*“This is the **end-to-end pipeline**: same story as the ERD, but as **processing stages** rather than tables.”*

Order: **PDF → ingest → Neo4j → embed → GNN audit → scores on edges → retrieve → filter → synthesize → evaluate metrics.**

---

### D. If they mean the **on-screen Knowledge Graph** (nodes and edges)

**One-liner:**  
*“That’s a **visualization of a small subgraph**—usually the triplets returned for **this answer**—so the reviewer sees **structure**, not just prose.”*

---

### E. Optional slide titles

- *Slide:* **Data model (ERD)** — *Provenance + auditable relationships*  
- *Slide:* **Query flow (sequence)** — *Retrieve → filter → generate or refuse*
