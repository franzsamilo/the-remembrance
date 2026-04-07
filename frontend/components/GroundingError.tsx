"use client";

import React from "react";
import { ShieldAlert } from "lucide-react";
import { motion } from "framer-motion";
import { Triplet } from "@/lib/types";

interface GroundingErrorProps {
  message: string;
  filteredTriplets: Triplet[];
}

export default function GroundingError({ message, filteredTriplets }: GroundingErrorProps) {
  const [showRejected, setShowRejected] = React.useState(false);

  return (
    <motion.div
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      className="bg-white border border-[var(--border)] border-l-4 border-l-[var(--conflict-red)] rounded-lg p-5"
    >
      <div className="flex items-start gap-3">
        <ShieldAlert size={20} className="text-[var(--conflict-red)] shrink-0 mt-0.5" />
        <div className="flex-1">
          <h4 className="font-semibold text-[var(--conflict-red)] text-sm uppercase tracking-wide mb-1" style={{ fontFamily: "EB Garamond, serif" }}>
            Grounding Error
          </h4>
          <p className="text-sm text-[var(--ink-medium)]">{message}</p>

          {filteredTriplets.length > 0 && (
            <div className="mt-4">
              <button
                onClick={() => setShowRejected(!showRejected)}
                className="text-xs font-mono text-[var(--ink-light)] hover:text-[var(--ink-dark)] transition-colors"
              >
                {showRejected ? "Hide" : "Show"} {filteredTriplets.length} rejected triplet{filteredTriplets.length !== 1 ? "s" : ""}
              </button>

              {showRejected && (
                <div className="mt-2 space-y-2">
                  {filteredTriplets.map((t, i) => (
                    <div key={i} className="flex items-center gap-2 text-xs font-mono text-[var(--ink-light)] bg-[var(--muted)] p-2 rounded">
                      <span className="text-[var(--conflict-red)]">{t.audit != null ? (t.audit * 100).toFixed(0) + "%" : "N/A"}</span>
                      <span className="line-through">
                        {t.source} &rarr; {t.relation} &rarr; {t.target}
                      </span>
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </motion.div>
  );
}
