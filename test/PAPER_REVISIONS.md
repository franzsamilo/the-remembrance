# Paper Revisions — Post Mid-Evaluation

Based on panel feedback: (1) pipeline needs granular detail per model/stage, (2) formatting not formal enough, (3) visualizations need to follow ML pipeline conventions (Labarta Feature/Training/Inference framing).

---

## Part 1: Formatting & Language Corrections (Chapters 1–2)

### Issues Found

**1.1 Typos and grammatical errors:**
- "inefficiecancy" → "inefficiency" (Section 1.1, paragraph 5)
- "Delimitations:" uses bullet points (`*`) inconsistently — some sections use numbered lists, others use bullets. Standardize to numbered for major items.

**1.2 Overly informal/dramatic language for an academic paper:**
The paper uses metaphorical language that reads more like a magazine article than a capstone thesis. Examples:

| Current (informal) | Suggested (formal) |
|--------------------|--------------------|
| "contested battlespace of verifiable and fabricated claims" | "landscape where verifiable and fabricated claims coexist" |
| "the LLM's 'Id'" and "the GNN's 'Superego'" | Remove Freudian metaphor entirely — say "the GNN constrains the LLM's probabilistic output with deterministic structural validation" |
| "digital detective" | Acceptable once in the introduction as a framing device, but should not recur throughout — use "integrity validation layer" in technical sections |
| "trust ceiling" | Replace with "confidence threshold" or remove quotes |
| "forensic challenge" | "verification challenge" |
| "shattering" | "exceeding" or "surpassing" |

**1.3 Section numbering inconsistency:**
- Section 1.2.1 is titled "The Manual Efficiency" but discusses efficiency AND integrity — should be "The Manual Efficiency Gap" to distinguish from 1.2.2
- Section headers use inconsistent formatting: some have parenthetical subtitles ("The Data Review Backlog"), others don't

**1.4 Citation formatting:**
- "Guo et al. (2025)" — verify this is published/preprint, not forthcoming. If it's a 2025 preprint cited from arXiv, format as "(Guo et al., 2025, preprint)" per APA 7th
- "Barry et al. (2025)" — same issue. ACL Anthology 2025 proceedings may not be published yet at time of your March 2026 submission — verify
- "(((https://doi.org/...)))" appears in the references section — remove the triple parentheses around He et al. (2025) DOI

**1.5 Abstract issues:**
- Lists evaluation targets (AUC-ROC > 0.95, MRR > 0.95, Grounding > 98%) but does not report results. For mid-eval this is fine, but for final defense the abstract must include actual results.
- "A prototype demonstrates the feasibility" — vague. Specify: "A prototype processing N documents with M entities demonstrates..."

**1.6 Missing paragraph breaks:**
- Section 1.1 is a single massive paragraph. Break after: (a) the data velocity problem, (b) the retraction crisis, (c) the Nature ban, (d) the dual deficit definition.

**1.7 Table formatting:**
- The comparison table in Section 2.3 (Standard RAG vs Validate-then-Generate) appears as plain text with tab separators. Must be a properly formatted table in the final document.

---

## Part 2: New Section — Detailed Pipeline Architecture

**Insert as Section 3.2 (after 3.1 Research Design), OR replace the current brief pipeline description in Section 3.2.**

This section follows the Labarta three-pipeline framing (Feature / Training / Inference) that the panel requested. Each stage gets: position in pipeline, model, input/output, parameters, training data, and justification.

---

### 3.2 System Architecture: The Three-Pipeline Design

Following established ML systems architecture conventions (Labarta Bajo, 2023; Google Cloud, 2024), the system is decomposed into three modular pipelines: Feature Pipelines (data preparation), Training Pipeline (model development), and Inference Pipeline (query-time execution). This separation ensures that each pipeline can be developed, tested, and evaluated independently, and aligns with production MLOps practices where feature computation, model training, and serving are distinct operational concerns.

**Figure 3.1** illustrates the end-to-end architecture. [INSERT DIAGRAM — see Part 3 for description]

#### 3.2.1 Feature Pipelines (Stages 1–4)

The Feature Pipelines transform raw PDF documents into a queryable, embedding-enriched knowledge graph. This pipeline runs once per corpus update and produces the persistent data artifacts consumed by both the Training and Inference pipelines.

**Stage 1: PDF Ingestion**

| Attribute | Value |
|-----------|-------|
| Component | SimpleKGPipeline (neo4j-graphrag library) |
| Input | Raw PDF files from the document corpus |
| Output | Parsed text segments passed to the extraction stage |
| Key Parameters | `from_pdf=True`, `on_error=RAISE` (configurable) |
| Schema | 8 node types (Entity, Method, Researcher, Dataset, Concept, Result, Metric, __Entity__), 7 relationship types (USES, CONTRADICTS, EXTENDS, PROPOSES, EVALUATES, ACHIEVES, FROM_CHUNK) |

The ingestion stage uses the SimpleKGPipeline from the neo4j-graphrag library, which orchestrates PDF text extraction and passes the content to the LLM-based extraction stage. The pipeline is schema-guided: entity and relationship types are predefined to constrain extraction to domain-relevant concepts, preventing the LLM from generating arbitrary or hallucinated entity categories.

**Justification:** Schema-guided extraction was selected over open-ended extraction because it captures typed relations (e.g., Researcher PROPOSES Method) that standard text chunking would miss. Typed relations are essential for the CompGCN audit stage, which requires multi-relational edge types to learn compositional patterns.

**Stage 2: Entity and Relationship Extraction**

| Attribute | Value |
|-----------|-------|
| Model | Gemini 2.5 Flash (Google) |
| Input | Parsed text segments from Stage 1 |
| Output | Typed entities and relationships written to Neo4j |
| Key Parameters | Temperature: 0, Retry: exponential backoff (5 retries, 5-second base delay) |
| Entity Types | Entity, Method, Researcher, Dataset, Concept, Result, Metric |
| Relation Types | USES, CONTRADICTS, EXTENDS, PROPOSES, EVALUATES, ACHIEVES |

The LLM performs zero-shot extraction: given a text segment and the target schema, it identifies entities and their relationships without task-specific fine-tuning. Temperature is set to 0 to maximize determinism — the same input should produce the same extraction output, ensuring reproducibility across evaluation runs.

**Justification:** Gemini 2.5 Flash was selected for its balance of extraction quality and cost efficiency. Temperature 0 ensures deterministic output, which is critical for a system that claims reproducible integrity validation. The exponential backoff retry logic handles API rate limiting during large corpus ingestion.

**Stage 3: Graph Storage and Provenance**

| Attribute | Value |
|-----------|-------|
| Database | Neo4j (Aura Free tier) |
| Protocol | Bolt (neo4j+s://) |
| Provenance | Per-node `source_document` and per-edge `source_documents` properties |
| Metadata | IngestionRun nodes with timestamps, document counts, and processing status |

Every extracted entity and relationship is annotated with its source document filename, creating a complete provenance chain from generated narrative → triplet → source PDF. IngestionRun metadata nodes record when each ingestion occurred, how many documents were processed, and how many nodes/relationships were created, enabling reproducibility auditing.

**Justification:** Neo4j was selected over relational databases because professional knowledge is inherently graph-structured: entities are connected by typed, directional relationships that SQL JOIN operations cannot efficiently traverse. The Aura Free tier provides cloud persistence without cost, making the system accessible for academic research.

**Stage 4: Vector Embedding (Cold-Start Resolution)**

| Attribute | Value |
|-----------|-------|
| Model | DistilBERT (distilbert-base-nli-stsb-mean-tokens) |
| Input | Node text fields (description, summary, name) from Neo4j |
| Output | 768-dimensional L2-normalized embedding vector on each node |
| Key Parameters | Batch size: 50, Normalization: L2, Provider: sentence-transformers |

This stage solves the **GNN Cold-Start Problem**: the CompGCN model requires initial node feature vectors, but newly extracted entities have no embeddings. DistilBERT generates 768-dimensional semantic vectors from node text, which serve as initial features for the GNN. L2 normalization ensures all vectors have unit norm, preventing magnitude bias during training. Nodes without text receive zero vectors, which remain zero after normalization — the GNN treats these as uninformative but still scores their edges.

**Justification:** DistilBERT was selected over larger embedding models (e.g., OpenAI ada-002 at 1536 dimensions, or Gemini embeddings) for three reasons: (1) it runs locally without API costs, keeping the system free-tier; (2) its 768-dimensional output provides sufficient semantic resolution for similarity-based seed selection; (3) its lightweight architecture (~66M parameters vs BERT's ~110M) enables batch embedding of large graphs without GPU requirements.

#### 3.2.2 Training Pipeline (Stage 5)

The Training Pipeline takes the embedding-enriched graph from the Feature Pipelines and trains a link prediction model to score every relationship's plausibility. This pipeline runs once after each ingestion cycle and produces the integrity scores consumed by the Inference Pipeline.

**Stage 5: CompGCN Integrity Audit**

| Attribute | Value |
|-----------|-------|
| Model | CompGCNAuditModel (custom, 2-layer encoder + DistMult link predictor) |
| Input | All non-FROM_CHUNK edges + 768-dim node embeddings from Neo4j |
| Output | Plausibility score (0.0–1.0) written to every relationship in the graph |
| Architecture | 2-layer CompGCN encoder → DistMult composition → sigmoid scoring |
| Training Data | 80% of edges (train), 20% held out (validation) |
| Negative Sampling | 10 corrupted triples per positive edge (head/tail corruption) |
| Loss Function | BCEWithLogitsLoss with configurable label smoothing |

**Hyperparameters:**

| Parameter | Value | Justification |
|-----------|-------|---------------|
| Hidden Channels | 256 | Balances expressiveness with training stability for graphs of this scale |
| Epochs | 100 (max) | Upper bound; early stopping typically terminates at 30–60 epochs |
| Learning Rate | 0.001 | Standard Adam optimizer starting rate |
| Weight Decay | 0.0001 | L2 regularization to prevent overfitting on small graphs |
| Dropout | 0.2 | Applied after each CompGCN layer to prevent co-adaptation |
| Early Stopping Patience | 20 epochs | Stops training if validation AUC does not improve for 20 consecutive epochs |
| Gradient Clipping | 1.0 | Prevents exploding gradients during backpropagation |
| Negative Ratio | 10 | 10 corrupted triples per positive edge for robust negative sampling |
| Validation Split | 20% | Standard held-out proportion for link prediction evaluation |
| Label Smoothing | 0.0 | Disabled by default; configurable for noisy graphs |
| Composition Operator | DistMult (element-wise multiplication) | See justification below |
| Random Seed | 42 | Fixed for reproducible audit results across evaluation runs |

**The DistMult Composition Operator:**

The standard CompGCN layer propagation rule is:

$$h_v^{(l+1)} = \sum_{(u,r) \in \mathcal{N}(v)} W_\lambda^{(l)} \cdot \phi(h_u^{(l)}, h_r^{(l)})$$

Where $\phi$ is the composition operation. This study uses DistMult:

$$\phi(h_u, h_r) = h_u \odot h_r$$

The element-wise multiplication allows the model to learn the semantic weight of each relation type. Unlike TransE (which uses additive translation) or RotatE (which uses rotation in complex space), DistMult's multiplicative composition captures symmetric interaction patterns that are well-suited for professional knowledge relations where both entities contribute equally to the relationship's semantics.

**Justification of CompGCN over R-GCN:** The primary alternative for multi-relational graph learning is R-GCN (Schlichtkrull et al., 2018), which assigns a separate weight matrix $W_r$ to each relation type $r$. This creates $O(R)$ parameter growth where $R$ is the number of relation types. For professional knowledge graphs with 7+ relation types, this leads to parameter explosion and overfitting on small graphs. CompGCN avoids this by sharing a single set of learnable parameters across all relation types, composing entity and relation embeddings through the DistMult operator. This shared parameterization is particularly valuable for capstone-scale graphs where training data is limited.

**Training Process:**
1. Load all non-FROM_CHUNK edges and node embeddings from Neo4j via GNNLoader
2. Split edges into 80% train / 20% validation sets (random permutation)
3. For each epoch: encode nodes via 2-layer CompGCN, compute positive edge logits, sample negative edges via head/tail corruption, compute BCE loss, backpropagate with gradient clipping
4. Evaluate validation AUC-ROC after each epoch; update learning rate via ReduceLROnPlateau scheduler
5. Early stop if no improvement for 20 epochs; restore best model state
6. Score all edges with the best model; write plausibility_score to every relationship in Neo4j
7. Persist AuditRun metadata node with AUC-ROC, MRR, hyperparameters, and timestamp

#### 3.2.3 Inference Pipeline (Stages 6–7)

The Inference Pipeline runs at query time. It retrieves context from the validated graph, filters by plausibility, synthesizes a narrative, and evaluates the output quality.

**Stage 6: Grounded Synthesis**

| Attribute | Value |
|-----------|-------|
| Retriever | GraphRetriever (custom hybrid: vector similarity + graph expansion) |
| Generator | Gemini 2.5 Flash |
| Plausibility Threshold (τ) | 0.95 (configurable via GROUNDING_MIN_SCORE) |
| Seed Selection | Top-5 nodes by cosine similarity to query embedding |
| Graph Expansion | 1-hop traversal from seed nodes, excluding FROM_CHUNK edges |
| Failure Mode | Hard Grounding Error — returns error instead of hallucinating |

**Retrieval Process:**
1. Encode the user query with DistilBERT (same model as Stage 4 for embedding consistency)
2. Fetch candidate nodes with embeddings from Neo4j; compute cosine similarity in Python
3. Select top-5 seed nodes by similarity score
4. Expand 1-hop from seeds: retrieve all connected triplets (source → relation → target) with provenance
5. Filter: keep only triplets where `audit_status = 'trained_experimental'` or `plausibility_score ≥ τ`
6. If no triplets survive filtering: return a **hard Grounding Error** — the system refuses to generate an answer rather than hallucinating from unvalidated data
7. Pass validated triplets + query to Gemini for narrative synthesis

**Generator-Side Filtering (Design Decision):**
The plausibility filtering is performed at the generator level (Python), not at the Cypher query level (database). This is intentional: the retriever fetches full context including low-scoring triplets, allowing the system to log what was retrieved vs. what was filtered. This separation enables:
- Audit logging of retrieval coverage vs. generation input
- Future ablation studies comparing filtered vs. unfiltered generation
- The Audit Dashboard to display what the GNN flagged without a separate query

**Justification:** The hybrid vector + graph retrieval strategy was selected over pure vector RAG because vector similarity alone retrieves isolated text fragments. Graph expansion discovers multi-hop connections (e.g., Researcher A → PROPOSES → Method B → EVALUATES → Dataset C) that pure semantic search cannot reach. The 1-hop limit balances context richness with noise — deeper traversals on small graphs risk including the entire graph.

**Stage 7: Evaluation (LLM-as-Judge)**

| Attribute | Value |
|-----------|-------|
| Method | LLM-as-Judge (automated evaluation) |
| Scorer Model | Gemini 2.5 Flash |
| Grounding Metric | Average claim traceability score (1–5 per claim, normalized to 0–1) |
| Faithfulness Metric | Ratio of claims supported by retrieved triplets |
| Evaluation Queries | 5 fixed queries from evaluation_queries.json |

The evaluation stage uses a structured prompt to instruct Gemini to act as an impartial evaluator, scoring each factual claim in the generated narrative against the retrieved triplets. This "LLM-as-Judge" approach (Zheng et al., 2024) provides scalable evaluation without human annotators.

**Justification:** Human evaluation is the gold standard but impractical for iterative development on a capstone timeline. LLM-as-Judge provides consistent, reproducible scores across runs. The fixed query set ensures evaluation stability — the same 5 queries are used for every evaluation cycle, enabling direct comparison across pipeline configurations.

---

## Part 3: Pipeline Visualization Descriptions

The paper needs proper figures. Here are descriptions for diagrams you should create (in PowerPoint, Lucidchart, draw.io, or Mermaid):

### Figure 3.1 — Three-Pipeline Architecture (REQUIRED)

**Style:** Follow the Labarta convention — three horizontal swim lanes, one per pipeline. Each lane contains stage boxes connected by arrows. Data artifacts (graph, embeddings, scores) shown as cylinders/databases between lanes.

```
┌─────────────────────── FEATURE PIPELINES ────────────────────────┐
│                                                                   │
│  [PDF Files] → [Stage 1: Ingest] → [Stage 2: Extract] → [Stage 3: Store] → [Stage 4: Embed]  │
│               SimpleKGPipeline     Gemini 2.5 Flash     Neo4j          DistilBERT        │
│                                                                   │
└───────────────────────────────┬───────────────────────────────────┘
                                │
                        ┌───────▼───────┐
                        │  Neo4j Graph  │  (entities + relationships
                        │  + Embeddings │   + 768-dim vectors)
                        └───────┬───────┘
                                │
┌───────────────────────────────▼───────────────────────────────────┐
│                      TRAINING PIPELINE                            │
│                                                                   │
│  [Graph + Embeddings] → [Stage 5: CompGCN Audit] → [Scored Graph] │
│                          2-layer CompGCN + DistMult               │
│                          AUC-ROC, MRR evaluation                  │
│                                                                   │
└───────────────────────────────┬───────────────────────────────────┘
                                │
                        ┌───────▼───────┐
                        │  Scored Graph │  (plausibility_score
                        │               │   on every edge)
                        └───────┬───────┘
                                │
┌───────────────────────────────▼───────────────────────────────────┐
│                     INFERENCE PIPELINE                            │
│                                                                   │
│  [Query] → [Stage 6: Retrieve + Filter + Synthesize] → [Stage 7: Evaluate]  │
│             GraphRetriever + Gemini 2.5 Flash           LLM-as-Judge        │
│             τ ≥ 0.95 plausibility gate                  Grounding/Faith.    │
│                                                                   │
└───────────────────────────────────────────────────────────────────┘
```

**For your document:** Recreate this as a proper diagram with:
- Color-coded swim lanes (green for Feature, gold for Training, red for Inference)
- Model names labeled inside each stage box
- Data artifacts (Neo4j Graph, Scored Graph) as cylinder shapes between lanes
- Arrows showing data flow direction

### Figure 3.2 — CompGCN Training Loop (RECOMMENDED)

A flowchart showing the training process:

```
[Load Graph from Neo4j] → [Split Edges 80/20] → [For each epoch:]
    → [Encode nodes (2-layer CompGCN)]
    → [Compute positive edge logits]
    → [Sample negative edges (10× corruption)]
    → [Compute BCE loss]
    → [Backpropagate + gradient clip]
    → [Evaluate validation AUC]
    → {AUC improved?}
        Yes → [Save best model] → [Continue]
        No → [Increment patience counter]
            → {Patience exhausted?}
                Yes → [Early stop → Score all edges → Write to Neo4j]
                No → [Continue]
```

### Figure 3.3 — Inference Decision Flow (RECOMMENDED)

A flowchart showing the query-time decision:

```
[User Query] → [Encode with DistilBERT]
    → [Vector similarity: top-5 seeds]
    → [Graph expansion: 1-hop triplets]
    → [Filter: plausibility ≥ τ]
    → {Any triplets survive?}
        No → [GROUNDING ERROR — refuse to answer]
        Yes → [Pass to Gemini] → [Generate narrative]
            → [Return narrative + evidence trail]
```

### Figure 2.1 — Standard RAG vs Validate-then-Generate (RECOMMENDED)

Replace the current text table with a proper side-by-side diagram:

```
STANDARD RAG:                    VALIDATE-THEN-GENERATE:
[Query] → [Vector Search]        [Query] → [Vector + Graph Search]
        → [Retrieve Chunks]              → [Retrieve Triplets]
        → [LLM Generate]                 → [GNN Filter (τ ≥ 0.95)]
        → [Answer (may hallucinate)]     → {Validated triplets?}
                                            No → [GROUNDING ERROR]
                                            Yes → [LLM Generate]
                                                → [Answer (grounded)]
```

---

## Part 4: Additional Recommendations

1. **Add a "Definitions of Terms" section** after Chapter 1 — academic papers typically define key terms (Knowledge Graph, Graph Neural Network, Plausibility Score, Grounding, Faithfulness, Link Prediction, etc.)

2. **Number all figures and tables** — currently the paper references "Figure 1.1" but doesn't consistently number subsequent figures

3. **Add a "Theoretical Framework" diagram** — the conceptual framework in Section 1.7 says "Figure 1.1 presents the conceptual framework" but the figure is missing from the text file

4. **Consistent tense** — the paper switches between present tense ("the system processes") and past tense ("the system was designed to"). Academic convention: use present tense for describing the system's behavior, past tense for describing the research process

5. **References section** — has a duplicated entry (Zhang et al. 2024 appears twice) and the He et al. (2025) entry has malformed DOI with triple parentheses
