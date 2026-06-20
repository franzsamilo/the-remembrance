"use client";

import React from "react";
import { motion } from "framer-motion";
import { Database, ChevronRight } from "lucide-react";
import InfoTooltip from "@/components/InfoTooltip";
import type { StatsData } from "@/lib/types";

interface CorpusContextProps {
  stats: StatsData | null;
}

/** Public benchmark KGs to compare edge-density against. Used to make the
 *  "sparse-corpus / MRR-floor" argument numeric rather than hand-waved. */
const BENCHMARKS = [
  { name: "This corpus", value: null as number | null, accent: true },
  { name: "OGB-Wikikg2", value: 4.5, accent: false },
  { name: "FB15k-237", value: 18.7, accent: false },
  { name: "FB15k", value: 19.0, accent: false },
];

/**
 * Compact strip that contextualizes this study's corpus density against
 * canonical KGE benchmarks. Lives under the KPI defense banner. The point
 * it makes — "this corpus is intentionally sparser than standard benchmarks,
 * which is why single-seed MRR floors at 0.91" — is a key defensive answer.
 */
export default function CorpusContext({ stats }: CorpusContextProps) {
  const nodes = stats?.nodes ?? null;
  const rels = stats?.relationships ?? null;
  const edgesPerNode = nodes && nodes > 0 && rels !== null ? rels / nodes : null;

  // Inject the live edges/node into the comparison row.
  const comparisons = BENCHMARKS.map((b) =>
    b.name === "This corpus" ? { ...b, value: edgesPerNode } : b
  );

  return (
    <motion.section
      initial={{ opacity: 0, y: 6 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.45 }}
      className="rounded-sm border border-[#E5E5E3] bg-white"
      aria-label="Corpus context vs benchmark knowledge graphs"
    >
      <div className="flex flex-col lg:flex-row lg:items-stretch">
        {/* Left: framing */}
        <div className="p-4 lg:w-[28%] lg:border-r border-b lg:border-b-0 border-[#E5E5E3] flex flex-col justify-center">
          <p className="text-[10px] font-mono uppercase tracking-[0.18em] text-[#737373] flex items-center gap-1.5 mb-1.5">
            <Database size={11} className="text-[#C5A028]" />
            Corpus context
            <InfoTooltip label="Why this matters">
              Standard KGE benchmarks (FB15k, OGB) are <em>dense</em> — many
              edges per node. This study&apos;s corpus is sparser, which floors
              the best achievable MRR. The multi-seed methodology in the paper
              recovers MRR to 0.958; the single-seed shortfall is not a model
              failure, it&apos;s a corpus-density consequence.
            </InfoTooltip>
          </p>
          <p className="text-xs text-[#525252] leading-snug">
            14 Philippine SC decisions + IP Code, sparse by design — this
            choice floors MRR but the architecture still passes paper targets.
          </p>
        </div>

        {/* Right: the comparison row */}
        <div className="p-4 lg:flex-1 flex flex-wrap items-center gap-x-6 gap-y-2">
          <p className="text-[9px] font-mono uppercase tracking-[0.18em] text-[#737373] basis-full lg:basis-auto inline-flex items-center gap-1">
            Edges / node
            <InfoTooltip label="Edges per node">
              Average number of relationships attached to each entity. Higher
              means a denser graph — more topological signal for the integrity
              model to learn from. KGE benchmark papers train on graphs with
              4–20 edges/node.
            </InfoTooltip>
          </p>
          {comparisons.map((b) => (
            <div
              key={b.name}
              className={`flex items-center gap-2 ${
                b.accent ? "font-semibold text-[#1A1A1A]" : "text-[#525252]"
              }`}
            >
              {b.accent && (
                <ChevronRight size={11} className="text-[#C5A028]" />
              )}
              <span className="text-[11px] uppercase tracking-wider font-mono">
                {b.name}
              </span>
              <span
                className={`text-sm font-mono tabular-nums ${
                  b.accent ? "text-[#C5A028]" : ""
                }`}
              >
                {b.value !== null ? b.value.toFixed(2) : "—"}
              </span>
            </div>
          ))}
        </div>
      </div>
    </motion.section>
  );
}
