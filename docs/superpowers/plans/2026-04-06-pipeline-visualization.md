# Pipeline Visualization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the text-based PipelineStory with an interactive, animated pipeline visualization (horizontal strip + expandable detail panel) that maps each model to its stage with live status, parameters, and "why" annotations.

**Architecture:** Four new files replace one. `pipelineConfig.ts` holds all stage data (static). `PipelineStrip.tsx` renders the horizontal flow with animated arrows. `PipelineDetail.tsx` renders the expanded detail card. `PipelineFlow.tsx` orchestrates them and receives stats from page.tsx. No backend changes.

**Tech Stack:** React 19, TypeScript, Framer Motion, Tailwind CSS 4, existing design tokens from globals.css

**Spec:** `docs/superpowers/specs/2026-04-06-pipeline-visualization-design.md`

---

### Task 1: Create pipelineConfig.ts — Stage Definitions

**Files:**
- Create: `frontend/lib/pipelineConfig.ts`

- [ ] **Step 1: Create the config file with types and all 7 stages**

```typescript
// frontend/lib/pipelineConfig.ts

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
  feature: { border: "border-[#3A5A40]", text: "text-[#3A5A40]", bg: "bg-[#3A5A40]" },
  training: { border: "border-[#D4AF37]", text: "text-[#D4AF37]", bg: "bg-[#D4AF37]" },
  inference: { border: "border-[#8B1A1A]", text: "text-[#8B1A1A]", bg: "bg-[#8B1A1A]" },
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
```

- [ ] **Step 2: Verify TypeScript compiles**

Run: `cd frontend && npx tsc --noEmit lib/pipelineConfig.ts 2>&1 | head -20`
Expected: No errors (or only unrelated existing errors)

- [ ] **Step 3: Commit**

```bash
git add frontend/lib/pipelineConfig.ts
git commit -m "feat: add pipeline stage config with model info, params, and status logic"
```

---

### Task 2: Create PipelineStrip.tsx — Horizontal Flow

**Files:**
- Create: `frontend/components/PipelineStrip.tsx`

- [ ] **Step 1: Create the PipelineStrip component**

```tsx
// frontend/components/PipelineStrip.tsx
"use client";

import React from "react";
import { motion } from "framer-motion";
import {
  PipelineStage,
  Phase,
  PHASE_COLORS,
  StageStatus,
} from "@/lib/pipelineConfig";

interface PipelineStripProps {
  stages: PipelineStage[];
  selectedId: string;
  currentTask: string;
  graphState: string;
  onSelect: (id: string) => void;
}

const PHASE_LABELS: { phase: Phase; label: string }[] = [
  { phase: "feature", label: "Feature" },
  { phase: "training", label: "Training" },
  { phase: "inference", label: "Inference" },
];

function StatusDot({ status }: { status: StageStatus }) {
  if (status === "active") {
    return (
      <span className="relative flex h-2.5 w-2.5">
        <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-[#D4AF37] opacity-75" />
        <span className="relative inline-flex rounded-full h-2.5 w-2.5 bg-[#D4AF37]" />
      </span>
    );
  }
  const colors: Record<StageStatus, string> = {
    ready: "bg-[#3A5A40]",
    waiting: "bg-[#6B6B6B]",
    error: "bg-[#8B1A1A]",
    active: "",
  };
  return <span className={`inline-block h-2.5 w-2.5 rounded-full ${colors[status]}`} />;
}

function FlowArrow({ active }: { active: boolean }) {
  return (
    <div className="relative flex items-center w-8 shrink-0">
      {/* Dashed line */}
      <div className="absolute inset-x-0 top-1/2 -translate-y-1/2 h-px border-t border-dashed border-[#D4AF37]/40" />
      {/* Flowing dots when active */}
      {active && (
        <div className="absolute inset-x-0 top-1/2 -translate-y-1/2 overflow-hidden h-1">
          {[0, 1, 2].map((i) => (
            <span
              key={i}
              className="absolute h-1 w-1 rounded-full bg-[#D4AF37]"
              style={{
                animation: `flowDot 1.2s linear infinite`,
                animationDelay: `${i * 0.4}s`,
              }}
            />
          ))}
        </div>
      )}
      {/* Arrow head */}
      <span className="absolute right-0 text-[#D4AF37]/60 text-xs">\u203A</span>
    </div>
  );
}

export default function PipelineStrip({
  stages,
  selectedId,
  currentTask,
  graphState,
  onSelect,
}: PipelineStripProps) {
  return (
    <div className="space-y-4">
      {/* Desktop: single row */}
      <div className="hidden md:flex items-start gap-0">
        {PHASE_LABELS.map(({ phase, label }) => {
          const phaseStages = stages.filter((s) => s.phase === phase);
          const colors = PHASE_COLORS[phase];
          const anyActive = phaseStages.some(
            (s) => s.getStatus(currentTask, graphState) === "active"
          );

          return (
            <React.Fragment key={phase}>
              <div className="flex flex-col items-center gap-1.5">
                <span
                  className={`text-[9px] font-mono uppercase tracking-[0.15em] ${colors.text} font-semibold`}
                >
                  {label}
                </span>
                <div className="flex items-center gap-0">
                  {phaseStages.map((stage, idx) => {
                    const status = stage.getStatus(currentTask, graphState);
                    const isSelected = stage.id === selectedId;
                    const isActive = status === "active";

                    return (
                      <React.Fragment key={stage.id}>
                        <motion.button
                          onClick={() => onSelect(stage.id)}
                          className={`
                            relative px-3 py-2 rounded-md border text-left transition-colors
                            ${isSelected ? `${colors.border} border-2 bg-[#FCFAF2]` : "border-[#4A4A4A]/30 bg-[#FCFAF2]/60 hover:bg-[#FCFAF2]"}
                            ${isActive ? "shadow-[0_0_8px_rgba(212,175,55,0.3)]" : ""}
                          `}
                          animate={
                            isActive
                              ? { scale: [1, 1.03, 1] }
                              : isSelected
                              ? { scale: 1.02 }
                              : { scale: 1 }
                          }
                          transition={
                            isActive
                              ? { duration: 2, repeat: Infinity, ease: "easeInOut" }
                              : { duration: 0.2 }
                          }
                        >
                          {isActive && (
                            <div className="absolute inset-x-0 top-0 h-0.5 rounded-t-md overflow-hidden">
                              <div
                                className="h-full w-[200%] bg-gradient-to-r from-transparent via-[#D4AF37] to-transparent"
                                style={{ animation: "shimmerBar 1.5s linear infinite" }}
                              />
                            </div>
                          )}
                          <div className="flex items-center gap-1.5 min-w-0">
                            <StatusDot status={status} />
                            <div className="min-w-0">
                              <p className="text-xs font-semibold text-[#2B2B2B] truncate leading-tight">
                                {stage.name}
                              </p>
                              <p className="text-[10px] text-[#6B6B6B] truncate leading-tight">
                                {stage.model.name}
                              </p>
                            </div>
                          </div>
                        </motion.button>
                        {idx < phaseStages.length - 1 && (
                          <FlowArrow active={anyActive} />
                        )}
                      </React.Fragment>
                    );
                  })}
                </div>
              </div>
              {phase !== "inference" && (
                <FlowArrow
                  active={
                    anyActive ||
                    stages
                      .filter(
                        (s) =>
                          s.phase ===
                          PHASE_LABELS[PHASE_LABELS.findIndex((p) => p.phase === phase) + 1]
                            ?.phase
                      )
                      .some((s) => s.getStatus(currentTask, graphState) === "active")
                  }
                />
              )}
            </React.Fragment>
          );
        })}
      </div>

      {/* Mobile: stacked phases */}
      <div className="md:hidden space-y-3">
        {PHASE_LABELS.map(({ phase, label }) => {
          const phaseStages = stages.filter((s) => s.phase === phase);
          const colors = PHASE_COLORS[phase];

          return (
            <div key={phase}>
              <span
                className={`text-[9px] font-mono uppercase tracking-[0.15em] ${colors.text} font-semibold`}
              >
                {label}
              </span>
              <div className="flex flex-wrap items-center gap-1 mt-1">
                {phaseStages.map((stage, idx) => {
                  const status = stage.getStatus(currentTask, graphState);
                  const isSelected = stage.id === selectedId;

                  return (
                    <React.Fragment key={stage.id}>
                      <button
                        onClick={() => onSelect(stage.id)}
                        className={`
                          px-2.5 py-1.5 rounded border text-left text-xs
                          ${isSelected ? `${colors.border} border-2 bg-[#FCFAF2] font-semibold` : "border-[#4A4A4A]/30 bg-[#FCFAF2]/60"}
                        `}
                      >
                        <StatusDot status={status} />
                        <span className="ml-1.5">{stage.name}</span>
                      </button>
                      {idx < phaseStages.length - 1 && (
                        <span className="text-[#D4AF37]/50 text-xs">\u203A</span>
                      )}
                    </React.Fragment>
                  );
                })}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
```

- [ ] **Step 2: Add the flowDot and shimmerBar keyframes to globals.css**

Add after the existing `@keyframes dash-flow` block in `frontend/app/globals.css`:

```css
@keyframes flowDot {
  0% { left: 0; opacity: 0; }
  10% { opacity: 1; }
  90% { opacity: 1; }
  100% { left: calc(100% - 4px); opacity: 0; }
}

@keyframes shimmerBar {
  0% { transform: translateX(-50%); }
  100% { transform: translateX(0%); }
}
```

Also add reduced-motion overrides inside the existing `@media (prefers-reduced-motion: reduce)` block:

```css
.animate-ping { animation: none !important; }
```

- [ ] **Step 3: Verify the component builds**

Run: `cd frontend && npx next build 2>&1 | tail -10`
Expected: Build succeeds (component isn't mounted yet, so just a compile check)

- [ ] **Step 4: Commit**

```bash
git add frontend/components/PipelineStrip.tsx frontend/app/globals.css
git commit -m "feat: add PipelineStrip with animated flow arrows and phase grouping"
```

---

### Task 3: Create PipelineDetail.tsx — Expanded Detail Card

**Files:**
- Create: `frontend/components/PipelineDetail.tsx`

- [ ] **Step 1: Create the PipelineDetail component**

```tsx
// frontend/components/PipelineDetail.tsx
"use client";

import React, { useRef, useEffect, useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { PipelineStage, PHASE_COLORS, StageStatus } from "@/lib/pipelineConfig";

interface PipelineDetailProps {
  stage: PipelineStage;
  status: StageStatus;
  stats: any;
}

function StatusBadge({ status }: { status: StageStatus }) {
  const config: Record<StageStatus, { label: string; className: string }> = {
    active: {
      label: "Running",
      className: "bg-[#D4AF37] text-[#1a1a1a]",
    },
    ready: {
      label: "Ready",
      className: "bg-[#3A5A40]/20 text-[#3A5A40] border border-[#3A5A40]/40",
    },
    waiting: {
      label: "Waiting",
      className: "bg-[#6B6B6B]/20 text-[#6B6B6B] border border-[#6B6B6B]/40",
    },
    error: {
      label: "Error",
      className: "bg-[#8B1A1A]/20 text-[#8B1A1A] border border-[#8B1A1A]/40",
    },
  };
  const c = config[status];

  return (
    <span className={`px-2.5 py-0.5 rounded-full text-[10px] font-semibold tracking-wide ${c.className}`}>
      {status === "active" ? (
        <span>
          Running<span className="inline-block" style={{ animation: "ellipsis 1.2s steps(3, end) infinite" }}>...</span>
        </span>
      ) : (
        c.label
      )}
    </span>
  );
}

function MetricPill({ label, value }: { label: string; value: string }) {
  const [flash, setFlash] = useState(false);
  const prevValue = useRef(value);

  useEffect(() => {
    if (prevValue.current !== value) {
      setFlash(true);
      prevValue.current = value;
      const t = setTimeout(() => setFlash(false), 600);
      return () => clearTimeout(t);
    }
  }, [value]);

  return (
    <div
      className={`px-3 py-1.5 rounded-md border border-[#4A4A4A]/20 bg-[#F5F2E9]/80 transition-colors duration-300 ${
        flash ? "bg-[#D4AF37]/15 border-[#D4AF37]/40" : ""
      }`}
    >
      <span className="text-[9px] font-mono uppercase tracking-wider text-[#6B6B6B]">{label}</span>
      <p className="text-sm font-bold text-[#2B2B2B] mt-0.5">{value}</p>
    </div>
  );
}

export default function PipelineDetail({ stage, status, stats }: PipelineDetailProps) {
  const colors = PHASE_COLORS[stage.phase];
  const metrics = stage.getMetrics(stats);
  const paramEntries = Object.entries(stage.params);

  return (
    <AnimatePresence mode="wait">
      <motion.div
        key={stage.id}
        initial={{ opacity: 0, y: -8, height: 0 }}
        animate={{ opacity: 1, y: 0, height: "auto" }}
        exit={{ opacity: 0, y: -8, height: 0 }}
        transition={{ duration: 0.25, ease: "easeOut" }}
        className={`relative rounded-lg border-2 ${colors.border} bg-[#FCFAF2] overflow-hidden`}
      >
        {/* Active progress bar */}
        {status === "active" && (
          <div className="absolute inset-x-0 top-0 h-0.5 overflow-hidden">
            <div
              className="h-full w-[300%] bg-repeating-linear-gradient"
              style={{
                background: `repeating-linear-gradient(90deg, transparent, transparent 8px, #D4AF37 8px, #D4AF37 16px)`,
                animation: "shimmerBar 0.8s linear infinite",
              }}
            />
          </div>
        )}

        <div className="p-5">
          {/* Header */}
          <div className="flex items-start justify-between mb-4">
            <div>
              <span className={`text-[9px] font-mono uppercase tracking-[0.15em] ${colors.text}`}>
                Stage {stage.order} \u00b7 {stage.phase}
              </span>
              <h3 className="text-lg font-semibold text-[#2B2B2B] mt-0.5">{stage.name}</h3>
            </div>
            <StatusBadge status={status} />
          </div>

          {/* 2x2 Grid */}
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
            {/* Model */}
            <div>
              <span className={`text-[9px] font-mono uppercase tracking-[0.15em] ${colors.text} mb-1 block`}>
                Model
              </span>
              <p className="text-sm font-semibold text-[#2B2B2B]">{stage.model.name}</p>
              <p className="text-xs text-[#6B6B6B] mt-0.5">{stage.model.description}</p>
            </div>

            {/* Input / Output */}
            <div>
              <span className={`text-[9px] font-mono uppercase tracking-[0.15em] ${colors.text} mb-1 block`}>
                Input / Output
              </span>
              <p className="text-xs text-[#2B2B2B]">
                <span className="font-medium">In:</span> {stage.input}
              </p>
              <p className="text-xs text-[#2B2B2B] mt-1">
                <span className="font-medium">Out:</span> {stage.output}
              </p>
            </div>

            {/* Parameters */}
            <div>
              <span className={`text-[9px] font-mono uppercase tracking-[0.15em] ${colors.text} mb-1 block`}>
                Parameters
              </span>
              <div className="space-y-0.5">
                {paramEntries.map(([key, val]) => (
                  <p key={key} className="text-xs text-[#2B2B2B]">
                    <span className="text-[#6B6B6B]">{key}:</span> {val}
                  </p>
                ))}
              </div>
            </div>

            {/* Why */}
            <div>
              <span className={`text-[9px] font-mono uppercase tracking-[0.15em] ${colors.text} mb-1 block`}>
                Why This Approach
              </span>
              <p className="text-sm text-[#2B2B2B] leading-relaxed">{stage.why}</p>
            </div>
          </div>

          {/* Live Metrics */}
          {metrics.length > 0 && (
            <div className="mt-4 pt-4 border-t border-[#4A4A4A]/15">
              <span className={`text-[9px] font-mono uppercase tracking-[0.15em] ${colors.text} mb-2 block`}>
                Live Metrics
              </span>
              <div className="flex flex-wrap gap-3">
                {metrics.map((m) => (
                  <MetricPill key={m.label} label={m.label} value={m.value} />
                ))}
              </div>
            </div>
          )}
        </div>
      </motion.div>
    </AnimatePresence>
  );
}
```

- [ ] **Step 2: Add the ellipsis keyframe to globals.css**

Add after the `flowDot` keyframe added in Task 2:

```css
@keyframes ellipsis {
  0% { content: ""; width: 0; }
  33% { content: "."; width: 0.5em; }
  66% { content: ".."; width: 1em; }
  100% { content: "..."; width: 1.5em; }
}
```

- [ ] **Step 3: Verify the component builds**

Run: `cd frontend && npx next build 2>&1 | tail -10`
Expected: Build succeeds

- [ ] **Step 4: Commit**

```bash
git add frontend/components/PipelineDetail.tsx frontend/app/globals.css
git commit -m "feat: add PipelineDetail with model info, params, why, and live metrics"
```

---

### Task 4: Create PipelineFlow.tsx — Orchestrator

**Files:**
- Create: `frontend/components/PipelineFlow.tsx`

- [ ] **Step 1: Create the PipelineFlow orchestrator component**

```tsx
// frontend/components/PipelineFlow.tsx
"use client";

import React, { useState } from "react";
import { ChevronDown, ChevronUp, Workflow } from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";
import { PIPELINE_STAGES } from "@/lib/pipelineConfig";
import PipelineStrip from "@/components/PipelineStrip";
import PipelineDetail from "@/components/PipelineDetail";

interface PipelineFlowProps {
  stats: any;
  currentTask: string;
}

export default function PipelineFlow({ stats, currentTask }: PipelineFlowProps) {
  const [selectedId, setSelectedId] = useState(PIPELINE_STAGES[0].id);
  const [expanded, setExpanded] = useState(true);

  const graphState: string = stats?.graph_state ?? "empty_graph";
  const selectedStage = PIPELINE_STAGES.find((s) => s.id === selectedId) ?? PIPELINE_STAGES[0];
  const selectedStatus = selectedStage.getStatus(currentTask, graphState);

  return (
    <div className="border border-[#4A4A4A]/30 rounded-lg bg-[#FCFAF2] overflow-hidden">
      {/* Collapsible header */}
      <button
        type="button"
        onClick={() => setExpanded(!expanded)}
        className="w-full flex items-center justify-between gap-3 px-5 py-3.5 text-left hover:bg-[#E8E4D9]/40 transition-colors"
        aria-expanded={expanded}
        aria-controls="pipeline-flow-content"
      >
        <div className="flex items-center gap-2">
          <Workflow size={18} className="text-[#D4AF37]" />
          <span className="text-sm font-semibold text-[#2B2B2B]">
            System Pipeline
          </span>
          <span className="text-[9px] font-mono uppercase tracking-[0.15em] text-[#6B6B6B]">
            Feature \u00b7 Training \u00b7 Inference
          </span>
        </div>
        {expanded ? (
          <ChevronUp size={18} className="text-[#6B6B6B]" />
        ) : (
          <ChevronDown size={18} className="text-[#6B6B6B]" />
        )}
      </button>

      <AnimatePresence>
        {expanded && (
          <motion.div
            id="pipeline-flow-content"
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: "auto", opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.25 }}
            className="overflow-hidden"
          >
            <div className="px-5 pb-5 space-y-4">
              <PipelineStrip
                stages={PIPELINE_STAGES}
                selectedId={selectedId}
                currentTask={currentTask}
                graphState={graphState}
                onSelect={setSelectedId}
              />
              <PipelineDetail
                stage={selectedStage}
                status={selectedStatus}
                stats={stats}
              />
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
```

- [ ] **Step 2: Verify the component builds**

Run: `cd frontend && npx next build 2>&1 | tail -10`
Expected: Build succeeds

- [ ] **Step 3: Commit**

```bash
git add frontend/components/PipelineFlow.tsx
git commit -m "feat: add PipelineFlow orchestrator with collapsible strip + detail"
```

---

### Task 5: Integrate into page.tsx and Remove PipelineStory

**Files:**
- Modify: `frontend/app/page.tsx:34` (import swap)
- Modify: `frontend/app/page.tsx:464` (component swap)
- Delete: `frontend/components/PipelineStory.tsx`

- [ ] **Step 1: Update page.tsx — swap import**

In `frontend/app/page.tsx`, replace line 34:

```typescript
// OLD:
import PipelineStory from "@/components/PipelineStory";
// NEW:
import PipelineFlow from "@/components/PipelineFlow";
```

- [ ] **Step 2: Update page.tsx — swap component usage**

In `frontend/app/page.tsx`, replace the PipelineStory usage (around line 464):

```tsx
// OLD:
        <PipelineStory stats={stats} docCount={documents.length} />
// NEW:
        <PipelineFlow stats={stats} currentTask={currentTask} />
```

- [ ] **Step 3: Delete PipelineStory.tsx**

```bash
rm frontend/components/PipelineStory.tsx
```

- [ ] **Step 4: Verify the full app builds**

Run: `cd frontend && npx next build 2>&1 | tail -20`
Expected: Build succeeds with no errors referencing PipelineStory

- [ ] **Step 5: Manual smoke test**

Run: `cd frontend && npm run dev`

Verify in browser at http://localhost:3000:
1. "System Pipeline" section visible on main dashboard with collapsible header
2. Horizontal strip shows 7 stages grouped into Feature / Training / Inference
3. Clicking a stage shows the detail card with Model, Input/Output, Parameters, Why
4. Phase colors are correct (green, gold, red)
5. If backend is running: status dots reflect current_task from /stats
6. Mobile: strip wraps into phase rows

- [ ] **Step 6: Commit**

```bash
git add frontend/app/page.tsx
git rm frontend/components/PipelineStory.tsx
git commit -m "feat: replace PipelineStory with interactive PipelineFlow visualization"
```

---

### Task 6: Polish Animations and Verify Active States

**Files:**
- Modify: `frontend/app/globals.css` (verify all keyframes present)
- Modify: `frontend/components/PipelineStrip.tsx` (polish if needed)
- Modify: `frontend/components/PipelineDetail.tsx` (polish if needed)

- [ ] **Step 1: Verify all keyframes are in globals.css**

Read `frontend/app/globals.css` and confirm these keyframes exist:
- `flowDot` (from Task 2)
- `shimmerBar` (from Task 2)
- `ellipsis` (from Task 3)
- `gilded-pulse` (already existed)
- `dash-flow` (already existed)

Also confirm `@media (prefers-reduced-motion: reduce)` disables `animate-ping`.

- [ ] **Step 2: Test active states with backend running**

1. Start backend: `cd backend && uvicorn src.api.main:app --reload --port 8000`
2. Start frontend: `cd frontend && npm run dev`
3. Upload a PDF via the dashboard
4. Click "Run Pipeline" to trigger ingestion
5. Observe:
   - Ingest/Extract/Store nodes show gold pulsing border and spinning status dot
   - Arrows between active stages show flowing gold dots
   - Detail card for active stage shows progress bar and "Running..." badge
   - After ingestion completes, Embed stage activates
   - After embedding completes, all Feature stages show green "Ready"
6. Trigger "Run Semantic Audit" from config or API
7. Observe: Audit stage activates with same animation pattern
8. After audit completes, verify AUC-ROC and MRR appear in Audit detail metrics

- [ ] **Step 3: Test reduced motion**

Enable "Reduce motion" in Windows accessibility settings. Verify:
- No pulsing, no flowing dots, no bouncing
- Status communicated via text badges and static colored dots only

- [ ] **Step 4: Final commit if any polish changes were made**

```bash
git add -A
git commit -m "polish: refine pipeline animations and active state transitions"
```
