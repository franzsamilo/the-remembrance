"use client";

import React from "react";
import { Shield } from "lucide-react";

interface FilteringSummaryProps {
  validatedCount: number;
  filteredCount: number;
  threshold: number;
}

export default function FilteringSummary({ validatedCount, filteredCount, threshold }: FilteringSummaryProps) {
  const total = validatedCount + filteredCount;
  const passRate = total > 0 ? (validatedCount / total) * 100 : 0;

  return (
    <div className="flex items-center gap-4 p-3 bg-[var(--muted)] rounded-lg text-sm">
      <Shield size={16} className="text-[var(--validated-green)] shrink-0" />
      <div className="flex-1">
        <p className="text-[var(--ink-medium)]">
          <span className="font-mono font-medium text-[var(--ink-dark)]">{total}</span> triplets retrieved
          &rarr; <span className="font-mono font-medium text-[var(--validated-green)]">{validatedCount}</span> passed GNN validation
          (<span className="font-mono">&tau; &ge; {threshold}</span>)
          {filteredCount > 0 && (
            <> &rarr; <span className="font-mono font-medium text-[var(--conflict-red)]">{filteredCount}</span> filtered out</>
          )}
        </p>
        <div className="mt-1.5 h-1.5 bg-[var(--border)] rounded-full overflow-hidden">
          <div
            className="h-full bg-[var(--validated-green)] rounded-full transition-all"
            style={{ width: `${passRate}%` }}
          />
        </div>
      </div>
    </div>
  );
}
