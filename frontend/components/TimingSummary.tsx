"use client";

import React from "react";
import { Clock } from "lucide-react";
import { StageTiming } from "@/lib/types";

interface TimingSummaryProps {
  timings: StageTiming | null | undefined;
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
  inference: "var(--seal-red)",
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
