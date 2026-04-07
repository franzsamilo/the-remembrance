"use client";

import React from "react";
import { motion } from "framer-motion";
import { AblationResults } from "@/lib/types";
import { formatScore } from "@/lib/utils";

interface AblationComparisonProps {
  results: AblationResults | null;
}

const MODES = [
  { key: "prompt_only" as const, label: "Prompt Only", description: "Chunk RAG \u2014 no graph, no GNN" },
  { key: "graph" as const, label: "Graph (No GNN)", description: "Hybrid retrieval \u2014 all edges, no audit" },
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
