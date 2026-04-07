"use client";

import { motion, AnimatePresence } from "framer-motion";
import { AlertTriangle, FileText, Link2 } from "lucide-react";

export interface FlaggedEdge {
  source: string;
  relation: string;
  target: string;
  plausibility_score: number;
  audit_status: string;
  source_docs: string[];
  target_docs: string[];
  cross_document: boolean;
  description: string | null;
}

interface FlaggedEdgesProps {
  edges: FlaggedEdge[];
  filterDoc: string | null;
  threshold?: number;
}

function riskLevel(score: number, threshold: number) {
  if (score < 0.5)
    return {
      label: "High Risk",
      bg: "bg-[#8B1A1A]/10 border-[#8B1A1A]/30",
      text: "text-[#8B1A1A]",
      bar: "#8B1A1A",
      dot: "bg-[#8B1A1A]",
    };
  if (score < 0.8)
    return {
      label: "Moderate Risk",
      bg: "bg-[#D4AF37]/10 border-[#D4AF37]/30",
      text: "text-[#D4AF37]",
      bar: "#D4AF37",
      dot: "bg-[#D4AF37]",
    };
  return {
    label: "Low Risk",
    bg: "bg-[#D4AF37]/5 border-[#D4AF37]/20",
    text: "text-[#8B6914]",
    bar: "#B8941F",
    dot: "bg-[#D4AF37]",
  };
}

function truncate(s: string, n = 32) {
  return s.length > n ? s.slice(0, n - 1) + "…" : s;
}

export default function FlaggedEdges({
  edges,
  filterDoc,
  threshold = 0.8,
}: FlaggedEdgesProps) {
  const filtered = filterDoc
    ? edges.filter(
        (e) =>
          e.source_docs.includes(filterDoc) ||
          e.target_docs.includes(filterDoc)
      )
    : edges;

  const sorted = [...filtered].sort(
    (a, b) => a.plausibility_score - b.plausibility_score
  );

  if (sorted.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center py-12 gap-3 text-center">
        <div className="p-3 bg-[#3A5A40]/10 border border-[#3A5A40]/30 rounded-full">
          <AlertTriangle size={20} className="text-[#3A5A40]" />
        </div>
        <p className="text-sm text-[#6B6B6B] italic">
          {filterDoc
            ? "No flagged relationships for this document."
            : "No flagged relationships found."}
        </p>
      </div>
    );
  }

  return (
    <div className="space-y-2">
      <AnimatePresence mode="popLayout">
        {sorted.map((edge, i) => {
          const risk = riskLevel(edge.plausibility_score, threshold);
          const scoreDisplay = (edge.plausibility_score * 100).toFixed(1);
          const allDocs = Array.from(
            new Set([...edge.source_docs, ...edge.target_docs])
          );

          return (
            <motion.div
              key={`${edge.source}-${edge.relation}-${edge.target}`}
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -8, scale: 0.97 }}
              transition={{ delay: i * 0.04, duration: 0.28, ease: "easeOut" }}
              className={`rounded-lg border p-4 ${risk.bg} transition-all`}
            >
              <div className="flex flex-col gap-2.5">
                {/* Top row: triplet + risk badge */}
                <div className="flex flex-col sm:flex-row sm:items-start sm:justify-between gap-2">
                  {/* Triplet */}
                  <div className="flex flex-wrap items-center gap-1.5 min-w-0">
                    <span
                      className="text-sm font-semibold text-[#8B1A1A] truncate max-w-[160px]"
                      title={edge.source}
                    >
                      {truncate(edge.source)}
                    </span>
                    <div className="flex items-center gap-0.5 shrink-0">
                      <div className="w-4 h-px bg-[#D4AF37]" />
                      <div className="w-0 h-0 border-l-[4px] border-l-[#D4AF37] border-y-[2.5px] border-y-transparent" />
                    </div>
                    <span className="text-[10px] font-mono uppercase tracking-[0.1em] text-[#6B6B6B] bg-[#E8E4D9] border border-[#4A4A4A]/20 rounded px-1.5 py-0.5 shrink-0">
                      {truncate(edge.relation, 24)}
                    </span>
                    <div className="flex items-center gap-0.5 shrink-0">
                      <div className="w-4 h-px bg-[#D4AF37]" />
                      <div className="w-0 h-0 border-l-[4px] border-l-[#D4AF37] border-y-[2.5px] border-y-transparent" />
                    </div>
                    <span
                      className="text-sm font-semibold text-[#3A5A40] truncate max-w-[160px]"
                      title={edge.target}
                    >
                      {truncate(edge.target)}
                    </span>
                  </div>

                  {/* Risk badge */}
                  <div className="flex items-center gap-2 shrink-0">
                    {edge.cross_document && (
                      <span className="flex items-center gap-1 px-2 py-0.5 rounded text-[9px] font-mono uppercase tracking-[0.1em] bg-[#D4AF37]/20 text-[#8B6914] border border-[#D4AF37]/40">
                        <Link2 size={9} />
                        Cross-doc
                      </span>
                    )}
                    <span
                      className={`flex items-center gap-1.5 px-2 py-0.5 rounded border text-[9px] font-mono uppercase tracking-[0.1em] ${risk.bg} ${risk.text}`}
                    >
                      <span className={`w-1.5 h-1.5 rounded-full ${risk.dot}`} />
                      {risk.label}
                    </span>
                  </div>
                </div>

                {/* Plausibility bar + score */}
                <div className="flex items-center gap-3">
                  <span className="text-[9px] font-mono uppercase tracking-[0.12em] text-[#6B6B6B] shrink-0 w-20">
                    Plausibility
                  </span>
                  <div className="flex-1 h-2 bg-[#E8E4D9] rounded-full overflow-hidden border border-[#4A4A4A]/10">
                    <motion.div
                      initial={{ width: 0 }}
                      animate={{ width: `${edge.plausibility_score * 100}%` }}
                      transition={{
                        duration: 0.7,
                        ease: "easeOut",
                        delay: i * 0.04 + 0.15,
                      }}
                      className="h-full rounded-full"
                      style={{ background: risk.bar }}
                    />
                  </div>
                  <span
                    className={`text-xs font-bold font-mono shrink-0 w-12 text-right ${risk.text}`}
                  >
                    {scoreDisplay}%
                  </span>
                </div>

                {/* Source docs */}
                {allDocs.length > 0 && (
                  <div className="flex flex-wrap items-center gap-1.5">
                    <FileText size={11} className="text-[#6B6B6B] shrink-0" />
                    {allDocs.map((doc) => (
                      <span
                        key={doc}
                        title={doc}
                        className="px-1.5 py-0.5 rounded text-[9px] font-mono bg-[#E8E4D9] text-[#4A4A4A] border border-[#4A4A4A]/20 truncate max-w-[140px]"
                      >
                        {truncate(doc, 22)}
                      </span>
                    ))}
                  </div>
                )}

                {/* Description */}
                {edge.description && (
                  <p className="text-[11px] text-[#6B6B6B] leading-relaxed italic border-t border-[#4A4A4A]/10 pt-2">
                    {edge.description}
                  </p>
                )}
              </div>
            </motion.div>
          );
        })}
      </AnimatePresence>
    </div>
  );
}
