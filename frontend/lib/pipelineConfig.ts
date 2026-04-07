export type StageStatus = "ready" | "active" | "waiting" | "error";
export type Phase = "feature" | "training" | "inference";

export interface PipelineStage {
  id: string;
  order: number;
  name: string;
  phase: Phase;
  model: { name: string; description: string };
  input: string;
  output: string;
  params: Record<string, string>;
  why: string;
  getStatus: (currentTask: string, graphState: string) => StageStatus;
  getMetrics: (stats: any) => { label: string; value: string }[];
}

export const PHASE_COLORS: Record<Phase, { border: string; text: string; bg: string }> = {
  feature: { border: "border-[#2D6A4F]", text: "text-[#2D6A4F]", bg: "bg-[#2D6A4F]" },
  training: { border: "border-[#C5A028]", text: "text-[#C5A028]", bg: "bg-[#C5A028]" },
  inference: { border: "border-[#7A1A1A]", text: "text-[#7A1A1A]", bg: "bg-[#7A1A1A]" },
};

function statusFromTask(currentTask: string, graphState: string, keywords: string[]): StageStatus {
  if (currentTask.startsWith("Error") || currentTask.startsWith("Audit Error")) return "error";
  if (graphState === "empty_graph") return "waiting";
  for (const kw of keywords) {
    if (currentTask.includes(kw)) return "active";
  }
  return "ready";
}

export const PIPELINE_STAGES: PipelineStage[] = [
  {
    id: "ingest",
    order: 1,
    name: "PDF Ingestion",
    phase: "feature",
    model: { name: "SimpleKGPipeline", description: "neo4j-graphrag pipeline for PDF parsing and schema-guided extraction" },
    input: "Raw PDF documents from /documents",
    output: "Parsed text passed to LLM for entity extraction",
    params: {
      Pipeline: "SimpleKGPipeline",
      Source: "PDF files",
      "On Error": "RAISE (configurable)",
      Schema: "8 node types, 7 relationship types",
    },
    why: "Schema-guided extraction captures typed relations that standard chunking misses.",
    getStatus: (task, graph) => statusFromTask(task, graph, ["Extracting"]),
    getMetrics: () => [],
  },
  {
    id: "extract",
    order: 2,
    name: "Entity Extraction",
    phase: "feature",
    model: { name: "Gemini 2.5 Flash", description: "Zero-shot LLM for deterministic entity and relationship extraction" },
    input: "Parsed PDF text from Stage 1",
    output: "Typed entities and relationships written to Neo4j",
    params: {
      Model: "gemini-2.5-flash",
      Temperature: "0",
      "Entity Types": "Entity, Method, Researcher, Dataset, Concept, Result, Metric",
      "Relation Types": "USES, CONTRADICTS, EXTENDS, PROPOSES, EVALUATES, ACHIEVES",
      "Retry Logic": "Exponential backoff (5 retries, 5s base)",
    },
    why: "Zero-shot LLM extraction at T=0 for deterministic entity/relation output.",
    getStatus: (task, graph) => statusFromTask(task, graph, ["Extracting"]),
    getMetrics: (stats) => {
      if (stats?.nodes != null) return [{ label: "Entities", value: String(stats.nodes) }];
      return [];
    },
  },
  {
    id: "store",
    order: 3,
    name: "Graph Storage",
    phase: "feature",
    model: { name: "Neo4j Aura", description: "Cloud-native property graph database with Cypher query language" },
    input: "Extracted entities and relationships from Stage 2",
    output: "Persistent knowledge graph with provenance metadata",
    params: {
      Database: "Neo4j Aura",
      Protocol: "Bolt (neo4j+s://)",
      Provenance: "Per-node and per-edge source_document tracking",
      "Run Metadata": "IngestionRun nodes with timestamps and counts",
    },
    why: "Native property graph preserves multi-relational structure Cypher can traverse.",
    getStatus: (task, graph) => statusFromTask(task, graph, ["Extracting"]),
    getMetrics: (stats) => {
      const m: { label: string; value: string }[] = [];
      if (stats?.nodes != null) m.push({ label: "Nodes", value: String(stats.nodes) });
      if (stats?.relationships != null) m.push({ label: "Relationships", value: String(stats.relationships) });
      return m;
    },
  },
  {
    id: "embed",
    order: 4,
    name: "Vector Embedding",
    phase: "feature",
    model: { name: "DistilBERT", description: "distilbert-base-nli-stsb-mean-tokens for semantic vector encoding" },
    input: "Node text fields (description, summary, name) from Neo4j",
    output: "768-dimensional L2-normalized embedding vectors on each node",
    params: {
      Model: "distilbert-base-nli-stsb-mean-tokens",
      Dimensions: "768",
      "Batch Size": "50",
      Normalization: "L2",
      Provider: "sentence-transformers",
    },
    why: "Lightweight 768-dim vectors solve the GNN cold-start without LLM-scale compute.",
    getStatus: (task, graph) => statusFromTask(task, graph, ["Embedding"]),
    getMetrics: (stats) => {
      if (stats?.embedding_progress != null) {
        return [{ label: "Vector Coverage", value: `${Math.round(stats.embedding_progress)}%` }];
      }
      return [];
    },
  },
  {
    id: "audit",
    order: 5,
    name: "Integrity Audit",
    phase: "training",
    model: { name: "CompGCN", description: "2-layer Composition-based GCN with DistMult link predictor" },
    input: "All non-FROM_CHUNK edges + node embeddings from Neo4j",
    output: "Plausibility score (0.0\u20131.0) on every relationship",
    params: {
      Architecture: "2-layer CompGCN encoder + DistMult",
      "Hidden Channels": "256",
      Epochs: "100",
      "Learning Rate": "0.001",
      "Weight Decay": "0.0001",
      Dropout: "0.2",
      Patience: "20 (early stopping)",
      "Grad Clip": "1.0",
      "Neg Ratio": "10",
      "Val Split": "20%",
      Composition: "DistMult (element-wise multiply)",
      Seed: "42 (reproducible)",
    },
    why: "Shared relation embeddings via DistMult avoid R-GCN\u2019s O(R) parameter explosion.",
    getStatus: (task, graph) => statusFromTask(task, graph, ["Audit", "GNN"]),
    getMetrics: (stats) => {
      const m: { label: string; value: string }[] = [];
      const kpis = stats?.research_kpis;
      if (kpis?.gnn_auc_roc != null) m.push({ label: "AUC-ROC", value: Number(kpis.gnn_auc_roc).toFixed(3) });
      if (kpis?.gnn_mrr != null) m.push({ label: "MRR", value: Number(kpis.gnn_mrr).toFixed(3) });
      return m;
    },
  },
  {
    id: "synthesize",
    order: 6,
    name: "Grounded Synthesis",
    phase: "inference",
    model: { name: "Gemini 2.5 Flash", description: "LLM constrained to validated triplets for narrative generation" },
    input: "GNN-filtered triplets (\u03c4 \u2265 0.95) from hybrid retriever",
    output: "Evidence-grounded narrative with per-triplet explanations",
    params: {
      Model: "gemini-2.5-flash",
      "Threshold (\u03c4)": "0.95",
      Retrieval: "Hybrid vector + graph expansion",
      "Top-k Seeds": "5",
      "Max Hops": "2",
      "Failure Mode": "Hard Grounding Error (no hallucination)",
    },
    why: "Generator-side filtering at \u03c4\u22650.95 ensures only validated triplets reach the LLM.",
    getStatus: (task, graph) => {
      if (task.startsWith("Error")) return "error";
      if (graph === "evidence_ready_graph") return "ready";
      return "waiting";
    },
    getMetrics: () => [],
  },
  {
    id: "evaluate",
    order: 7,
    name: "Evaluation",
    phase: "inference",
    model: { name: "LLM-as-Judge", description: "Gemini scores narrative claims against retrieved triplets" },
    input: "Generated narrative + source triplets from Stage 6",
    output: "Grounding score (0\u20131) and Faithfulness score (0\u20131)",
    params: {
      Method: "LLM-as-Judge",
      Scorer: "Gemini 2.5 Flash",
      "Grounding Metric": "Average claim traceability (1\u20135 scale, normalized)",
      "Faithfulness Metric": "Ratio of supported claims",
      "Sample Queries": "5 fixed evaluation queries",
    },
    why: "Gemini scores its own output against triplets for grounding/faithfulness measurement.",
    getStatus: (task, graph) => statusFromTask(task, graph, ["Evaluation"]),
    getMetrics: (stats) => {
      const m: { label: string; value: string }[] = [];
      const kpis = stats?.research_kpis;
      if (kpis?.grounding_score != null) m.push({ label: "Grounding", value: `${Math.round(Number(kpis.grounding_score) * 100)}%` });
      if (kpis?.faithfulness_score != null) m.push({ label: "Faithfulness", value: `${Math.round(Number(kpis.faithfulness_score) * 100)}%` });
      return m;
    },
  },
];
