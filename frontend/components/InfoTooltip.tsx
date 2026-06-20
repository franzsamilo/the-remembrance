"use client";

import React from "react";
import { Info } from "lucide-react";

interface InfoTooltipProps {
  label: string;
  children: React.ReactNode;
  size?: number;
  className?: string;
}

/**
 * Small "?" affordance that reveals a plain-English explanation on hover,
 * focus, or click. Reach for this anywhere the UI shows a research term
 * (Grounding, Faithfulness, AUC-ROC, MRR, plausibility threshold) so a
 * visitor who is not in the project can self-serve the definition.
 */
export default function InfoTooltip({
  label,
  children,
  size = 12,
  className = "",
}: InfoTooltipProps) {
  const [open, setOpen] = React.useState(false);

  return (
    <span className={`relative inline-flex items-center ${className}`}>
      <button
        type="button"
        onMouseEnter={() => setOpen(true)}
        onMouseLeave={() => setOpen(false)}
        onFocus={() => setOpen(true)}
        onBlur={() => setOpen(false)}
        onClick={(e) => {
          e.stopPropagation();
          setOpen((o) => !o);
        }}
        aria-label={`What is ${label}?`}
        aria-expanded={open}
        className="inline-flex items-center justify-center text-[#737373] hover:text-[#C5A028] focus:text-[#C5A028] focus:outline-none focus-visible:ring-1 focus-visible:ring-[#C5A028] rounded-full"
      >
        <Info size={size} />
      </button>
      {open && (
        <span
          role="tooltip"
          className="absolute z-50 bottom-full left-1/2 -translate-x-1/2 mb-2 w-64 p-3 rounded-sm bg-[#1A1A1A] text-[#F5F5F3] text-[11px] leading-relaxed shadow-xl"
        >
          <span className="block text-[9px] font-mono uppercase tracking-[0.18em] text-[#C5A028] mb-1.5">
            {label}
          </span>
          {children}
        </span>
      )}
    </span>
  );
}
