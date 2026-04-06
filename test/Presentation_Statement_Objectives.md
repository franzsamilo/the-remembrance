# Presentation: Statement of the Problem, Objectives, and Accomplished Features

---

## Statement of the Problem

The core problem this study addresses is the **architectural inability of current knowledge synthesis systems to validate semantic integrity before generation**. While tools exist for search (Google) and generation (ChatGPT), there is no integrated framework that mathematically validates the structural consistency of a professional knowledge base.

Specifically, this study addresses four critical gaps:

1. **Manual Efficiency ("The Data Review Backlog")** — Human capacity cannot keep pace with literature volume; systematic reviews take ~67 weeks.
2. **Manual Integrity ("The Unreliable Ground Truth")** — 85% error rates in systematic reviews; professionals lack a mechanism to validate internal consistency of source data.
3. **Corrupted Source Fallacy in RAG** — Standard RAG assumes retrieved documents are true; it retrieves and amplifies errors with no integrity layer.
4. **Absence of Semantic Fraud Detection** — GNNs detect financial fraud but are not yet applied to semantic integrity in professional literature.

---

## Objectives and Accomplished Features

### Objective 1 — Automate Relationship Mapping (Knowledge Ingestion)

**Goal:** Design an ingestion module capable of processing unstructured professional literature (PDFs), utilizing NLP to extract entities and latent semantic relationships to construct a raw Knowledge Graph.

**Specific Features Accomplished:**
- SimpleKGPipeline (neo4j-graphrag) for PDF text extraction and schema-guided entity/relationship extraction via Gemini LLM
- POST /upload and POST /ingest API endpoints
- Document-level provenance tagging (source_document, source_documents) on nodes and relationships
- IngestionRun metadata (nodes_created, relationships_created, documents_processed) persisted to Neo4j
- Configurable ontology via LEGAL_NODE_TYPES and LEGAL_RELATIONSHIP_TYPES environment variables

---

### Objective 2 — Proactive Conflict Detection (Addressing Manual Integrity & Semantic Fraud)

**Goal:** Implement a Graph Neural Network (CompGCN) to mathematically audit the graph topology by assigning plausibility scores to edge connections. Identify and flag semantic anomalies, contradictions, and "phantom" links.

**Specific Features Accomplished:**
- CompGCN module (gnn_module.py) with DistMult composition operator
- False-negative filtering in negative sampling to avoid training on invalid labels
- L2-normalized node features for stable training
- POST /audit endpoint triggering background GNN training
- plausibility_score, audit_score, audit_status written to every relationship in Neo4j
- AuditRun node storing auc_roc, mrr, completed_at
- GNN Audit Results tab on config page showing high/low integrity edges
- Research Evaluation tab displaying AUC-ROC and MRR with pass/fail against targets

---

### Objective 3 — Grounded Generation (Addressing the Corrupted Source Fallacy)

**Goal:** Implement a Grounded Generation engine that filters retrieved triplets by GNN plausibility scores (τ = 0.95) and synthesizes answers exclusively from validated subgraphs.

**Specific Features Accomplished:**
- DiscoveryGenerator with generator-side filtering (audit_status or plausibility_score ≥ GROUNDING_MIN_SCORE)
- Hybrid GraphRetriever (vector similarity → graph expansion → optional Leiden leads)
- Synthesis layer (synthesis.py) producing narrative + per-triplet explanations
- Hard "Grounding Error" when no validated triplets exist (refuses to hallucinate)
- Detective Board UI with source_docs, target_docs, cross_document badges, and per-triplet explanations
- POST /chat and POST /chat/stream endpoints
- POST /evaluate for LLM-as-judge grounding/faithfulness evaluation
- Research Evaluation tab with Grounding Score and Faithfulness Score

---

## Summary Table

| Objective | Key Accomplishment |
|-----------|--------------------|
| 1. Knowledge Ingestion | SimpleKGPipeline + provenance + IngestionRun |
| 2. Conflict Detection | CompGCN + plausibility scores + AuditRun + GNN Audit Results |
| 3. Grounded Generation | DiscoveryGenerator + hybrid retriever + Detective Board + Grounding Error |
