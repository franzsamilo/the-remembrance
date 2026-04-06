"use client";

import React, { useEffect, useState } from "react";
import Link from "next/link";
import { Workflow, Cpu, MessageSquare, ChevronDown, ChevronUp } from "lucide-react";
import axios from "axios";
import { API_BASE_URL } from "@/lib/api";

type Stage = {
  order: number;
  name: string;
  component: string;
  description: string;
};

type Stats = {
  graph_state?: string;
  embedding_progress?: number;
  nodes?: number;
  relationships?: number;
  current_task?: string;
  audit_readiness?: {
    state?: string;
    audited_relationships?: number;
    total_relationships?: number;
  };
  research_kpis?: Record<string, number | null | undefined>;
} | null;

function groupLabel(order: number): "feature" | "training" | "inference" {
  if (order <= 4) return "feature";
  if (order === 5) return "training";
  return "inference";
}

function stageStatus(
  stage: Stage,
  stats: Stats,
  docCount: number
): { label: string; tone: "muted" | "ok" | "warn" | "active" } {
  const gs = stats?.graph_state;
  const task = stats?.current_task || "";
  const emb = stats?.embedding_progress ?? 0;
  const audit = stats?.audit_readiness;

  if (stage.order <= 3) {
    if (docCount === 0)
      return { label: "Add PDFs", tone: "warn" };
    if (gs === "empty_graph")
      return { label: "Run pipeline", tone: "warn" };
    if (task.includes("Extracting") || task.includes("Embedding"))
      return { label: "Running", tone: "active" };
    if (gs && gs !== "empty_graph")
      return { label: "In graph", tone: "ok" };
    return { label: "—", tone: "muted" };
  }
  if (stage.order === 4) {
    if (gs === "empty_graph" || docCount === 0)
      return { label: "Waiting", tone: "muted" };
    if (emb >= 99)
      return { label: `${Math.round(emb)}%`, tone: "ok" };
    if (task.includes("Embedding"))
      return { label: "Embedding", tone: "active" };
    return { label: `${Math.round(emb)}%`, tone: "warn" };
  }
  if (stage.order === 5) {
    if (task.includes("Audit"))
      return { label: "Training audit", tone: "active" };
    if (audit?.state === "ready" || (audit?.audited_relationships ?? 0) > 0)
      return { label: "Scored", tone: "ok" };
    return { label: "Run audit", tone: "warn" };
  }
  if (stage.order === 6) {
    if (gs === "evidence_ready_graph")
      return { label: "Ready", tone: "ok" };
    return { label: "Needs graph", tone: "warn" };
  }
  if (stage.order === 7) {
    const k = stats?.research_kpis;
    if (k && (k.grounding_score != null || k.gnn_auc_roc != null))
      return { label: "KPIs available", tone: "ok" };
    return { label: "POST /evaluate", tone: "muted" };
  }
  return { label: "—", tone: "muted" };
}

const toneClass: Record<string, string> = {
  muted: "bg-[#E8E4D9] text-[#6B6B6B] border-[#4A4A4A]/40",
  ok: "bg-[#3A5A40]/15 text-[#3A5A40] border-[#3A5A40]/40",
  warn: "bg-amber-500/10 text-amber-900 border-amber-500/30",
  active: "bg-[#D4AF37]/20 text-[#8B1A1A] border-[#D4AF37]/40",
};

export default function PipelineStory({
  stats,
  docCount,
}: {
  stats: Stats;
  docCount: number;
}) {
  const [stages, setStages] = useState<Stage[]>([]);
  const [open, setOpen] = useState(true);
  const [loaded, setLoaded] = useState(false);

  useEffect(() => {
    axios
      .get(`${API_BASE_URL}/config`)
      .then((res) => {
        const list = res.data?.pipeline?.stages;
        if (Array.isArray(list)) setStages(list.sort((a: Stage, b: Stage) => a.order - b.order));
      })
      .catch(() => {})
      .finally(() => setLoaded(true));
  }, []);

  const groups: { key: "feature" | "training" | "inference"; title: string; blurb: string; icon: React.ReactNode }[] = [
    {
      key: "feature",
      title: "Feature pipelines",
      blurb: "Raw PDFs → graph structure → embeddings for retrieval (cf. Labarta: feature store).",
      icon: <Workflow className="text-[#D4AF37]" size={22} />,
    },
    {
      key: "training",
      title: "Training pipeline",
      blurb: "CompGCN trains on your graph; writes plausibility scores on edges (link prediction).",
      icon: <Cpu className="text-[#3A5A40]" size={22} />,
    },
    {
      key: "inference",
      title: "Inference pipeline",
      blurb: "Query → retrieve & filter triplets → Gemini narrative + evidence UI.",
      icon: <MessageSquare className="text-[#8B1A1A]" size={22} />,
    },
  ];

  return (
    <section className="max-w-7xl mx-auto mb-8" aria-labelledby="pipeline-story-heading">
      <button
        type="button"
        onClick={() => setOpen((o) => !o)}
        className="w-full flex items-center justify-between gap-3 p-4 rounded-sm border border-[#4A4A4A]/50 bg-[#FCFAF2] cut-paper text-left hover:bg-[#E8E4D9]/40 transition-colors"
      >
        <div>
          <h2
            id="pipeline-story-heading"
            className="text-lg font-semibold text-[#2B2B2B] flex items-center gap-2"
            style={{ fontFamily: "EB Garamond, serif" }}
          >
            <Workflow className="text-[#D4AF37] shrink-0" size={22} />
            How this system runs: Feature → Training → Inference
          </h2>
          <p className="text-sm text-[#6B6B6B] mt-1 max-w-3xl">
            Modular pipeline view (same idea as production ML: featurize, train a model, serve predictions). This UI is a reference implementation; the research contribution is the architecture, not the skin.
          </p>
        </div>
        {open ? <ChevronUp className="text-[#6B6B6B] shrink-0" /> : <ChevronDown className="text-[#6B6B6B] shrink-0" />}
      </button>

      {open && (
        <div className="mt-3 space-y-4 border border-[#4A4A4A]/30 rounded-sm bg-[#FCFAF2]/80 p-4 sm:p-6">
          {!loaded && <p className="text-sm text-[#6B6B6B] font-mono">Loading pipeline stages from backend…</p>}
          {loaded && stages.length === 0 && (
            <p className="text-sm text-[#6B6B6B]">Connect the backend to see live stage names from <code className="text-xs bg-[#E8E4D9] px-1 rounded">GET /config</code>.</p>
          )}
          {stages.length > 0 && (
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
              {groups.map((g) => (
                <div
                  key={g.key}
                  className="rounded-sm border border-[#4A4A4A]/40 bg-[#F5F2E9]/80 p-4 flex flex-col gap-3"
                >
                  <div className="flex items-start gap-2">
                    <div className="p-2 bg-[#FCFAF2] rounded-sm border border-[#4A4A4A]/30">{g.icon}</div>
                    <div>
                      <h3 className="font-semibold text-[#2B2B2B]" style={{ fontFamily: "EB Garamond, serif" }}>
                        {g.title}
                      </h3>
                      <p className="text-xs text-[#6B6B6B] mt-1 leading-relaxed">{g.blurb}</p>
                    </div>
                  </div>
                  <ul className="space-y-2 text-sm border-t border-[#4A4A4A]/20 pt-3">
                    {stages
                      .filter((s) => groupLabel(s.order) === g.key)
                      .map((stage) => {
                        const st = stageStatus(stage, stats, docCount);
                        return (
                          <li key={stage.order} className="flex flex-col gap-0.5">
                            <div className="flex items-center justify-between gap-2">
                              <span className="text-[#2B2B2B] font-medium">
                                {stage.order}. {stage.name}
                              </span>
                              <span
                                className={`text-[10px] font-mono uppercase tracking-wider px-2 py-0.5 rounded-sm border ${toneClass[st.tone]}`}
                              >
                                {st.label}
                              </span>
                            </div>
                            <span className="text-xs text-[#6B6B6B]">
                              {stage.component} — {stage.description}
                            </span>
                          </li>
                        );
                      })}
                  </ul>
                </div>
              ))}
            </div>
          )}
          <div className="flex flex-wrap items-center gap-3 text-xs text-[#6B6B6B] pt-2 border-t border-[#4A4A4A]/20">
            <span>
              Status uses <code className="bg-[#E8E4D9] px-1 rounded">GET /stats</code> (graph, embedding, audit task).
            </span>
            <Link href="/config" className="text-[#D4AF37] font-medium hover:underline">
              Full metrics &amp; API directory →
            </Link>
          </div>
        </div>
      )}
    </section>
  );
}
