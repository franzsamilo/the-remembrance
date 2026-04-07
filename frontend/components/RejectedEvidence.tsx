"use client";

import React, { useState } from "react";
import { ChevronDown, ChevronUp } from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";
import { Triplet } from "@/lib/types";

interface RejectedEvidenceProps {
  triplets: Triplet[];
  threshold: number;
}

export default function RejectedEvidence({ triplets, threshold }: RejectedEvidenceProps) {
  const [expanded, setExpanded] = useState(false);

  if (triplets.length === 0) return null;

  return (
    <div className="border border-[var(--border)] rounded-lg overflow-hidden">
      <button
        onClick={() => setExpanded(!expanded)}
        className="w-full flex items-center justify-between p-3 text-sm text-[var(--ink-medium)] hover:bg-[var(--muted)] transition-colors"
      >
        <span className="font-medium">
          {triplets.length} Triplet{triplets.length !== 1 ? "s" : ""} Filtered
          <span className="font-mono text-xs ml-1">(Below &tau; = {threshold})</span>
        </span>
        {expanded ? <ChevronUp size={16} /> : <ChevronDown size={16} />}
      </button>

      <AnimatePresence>
        {expanded && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: "auto", opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            className="overflow-hidden"
          >
            <div className="p-3 pt-0 space-y-2">
              {triplets.map((t, i) => (
                <div key={i} className="flex items-start gap-3 p-2 bg-[var(--conflict-red)]/5 rounded text-sm border border-[var(--conflict-red)]/10">
                  <span className="font-mono text-xs font-medium text-[var(--conflict-red)] bg-[var(--conflict-red)]/10 px-1.5 py-0.5 rounded shrink-0">
                    {t.audit != null ? (t.audit * 100).toFixed(0) + "%" : "N/A"}
                  </span>
                  <span className="text-[var(--ink-light)] line-through">
                    {t.source} &rarr; <span className="text-[var(--ink-light)]/60">{t.relation}</span> &rarr; {t.target}
                  </span>
                </div>
              ))}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
