"use client";

import { motion } from "framer-motion";
import { FileText, AlertTriangle, CheckCircle } from "lucide-react";

interface DocumentSummary {
  document: string;
  total_edges: number;
  flagged: number;
  integrity: number;
}

interface DocumentIntegrityProps {
  documents: DocumentSummary[];
  selectedDoc: string | null;
  onSelectDoc: (doc: string | null) => void;
}

function integrityStyle(pct: number): { bar: string; text: string; badge: string } {
  if (pct > 95)
    return {
      bar: "#3A5A40",
      text: "text-[#3A5A40]",
      badge: "bg-[#3A5A40]/10 text-[#3A5A40] border-[#3A5A40]/30",
    };
  if (pct > 85)
    return {
      bar: "#D4AF37",
      text: "text-[#D4AF37]",
      badge: "bg-[#D4AF37]/10 text-[#D4AF37] border-[#D4AF37]/30",
    };
  return {
    bar: "#8B1A1A",
    text: "text-[#8B1A1A]",
    badge: "bg-[#8B1A1A]/10 text-[#8B1A1A] border-[#8B1A1A]/30",
  };
}

function truncateDoc(name: string, max = 26): string {
  if (name.length <= max) return name;
  const base = name.replace(/\.[^/.]+$/, "");
  const ext = name.includes(".") ? "." + name.split(".").pop() : "";
  if (base.length <= max - 3 - ext.length) return name;
  return base.slice(0, max - 3 - ext.length) + "…" + ext;
}

export default function DocumentIntegrity({
  documents,
  selectedDoc,
  onSelectDoc,
}: DocumentIntegrityProps) {
  return (
    <div className="space-y-3">
      {/* Filter strip */}
      <div className="flex items-center gap-2 overflow-x-auto pb-1 scrollbar-thin">
        <button
          onClick={() => onSelectDoc(null)}
          className={`shrink-0 px-3 py-1.5 rounded-md border text-[10px] font-mono uppercase tracking-[0.12em] transition-all ${
            selectedDoc === null
              ? "bg-[#D4AF37] text-[#2B2B2B] border-[#D4AF37] shadow-sm"
              : "bg-[#FCFAF2] text-[#6B6B6B] border-[#4A4A4A]/30 hover:border-[#D4AF37]/50 hover:text-[#2B2B2B]"
          }`}
        >
          All Documents
        </button>
        {documents.map((doc) => {
          const isSelected = selectedDoc === doc.document;
          return (
            <button
              key={doc.document}
              title={doc.document}
              onClick={() => onSelectDoc(isSelected ? null : doc.document)}
              className={`shrink-0 px-3 py-1.5 rounded-md border text-[10px] font-mono uppercase tracking-[0.12em] transition-all max-w-[180px] truncate ${
                isSelected
                  ? "bg-[#D4AF37]/20 text-[#8B6914] border-[#D4AF37] shadow-sm"
                  : "bg-[#FCFAF2] text-[#6B6B6B] border-[#4A4A4A]/30 hover:border-[#D4AF37]/50 hover:text-[#2B2B2B]"
              }`}
            >
              {truncateDoc(doc.document, 22)}
            </button>
          );
        })}
      </div>

      {/* Document cards — horizontally scrollable */}
      <div className="flex gap-3 overflow-x-auto pb-2 scrollbar-thin">
        {documents.map((doc, i) => {
          const intPct = doc.integrity * 100;
          const style = integrityStyle(intPct);
          const validated = doc.total_edges - doc.flagged;
          const isSelected = selectedDoc === doc.document;

          return (
            <motion.button
              key={doc.document}
              initial={{ opacity: 0, x: 12 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: i * 0.06, duration: 0.3, ease: "easeOut" }}
              onClick={() => onSelectDoc(isSelected ? null : doc.document)}
              title={doc.document}
              className={`shrink-0 w-52 text-left bg-[#FCFAF2] rounded-lg p-4 border transition-all cursor-pointer flex flex-col gap-2 ${
                isSelected
                  ? "border-[#D4AF37] shadow-md shadow-[#D4AF37]/10"
                  : "border-[#4A4A4A]/30 hover:border-[#D4AF37]/40 hover:shadow-sm"
              }`}
            >
              {/* Header */}
              <div className="flex items-start justify-between gap-1.5">
                <div className="flex items-center gap-1.5 min-w-0">
                  <FileText size={13} className="text-[#6B6B6B] shrink-0" />
                  <span className="text-[11px] font-semibold text-[#2B2B2B] truncate leading-tight">
                    {truncateDoc(doc.document)}
                  </span>
                </div>
                {doc.flagged > 0 ? (
                  <AlertTriangle size={13} className="text-[#8B1A1A] shrink-0" />
                ) : (
                  <CheckCircle size={13} className="text-[#3A5A40] shrink-0" />
                )}
              </div>

              {/* Integrity % */}
              <div className="flex items-baseline justify-between">
                <span className={`text-xl font-bold font-mono ${style.text}`}>
                  {intPct.toFixed(1)}%
                </span>
                <span className="text-[9px] font-mono uppercase tracking-[0.12em] text-[#6B6B6B]">
                  integrity
                </span>
              </div>

              {/* Progress bar */}
              <div className="h-1.5 w-full bg-[#E8E4D9] rounded-full overflow-hidden border border-[#4A4A4A]/10">
                <motion.div
                  initial={{ width: 0 }}
                  animate={{ width: `${intPct}%` }}
                  transition={{ duration: 0.8, ease: "easeOut", delay: i * 0.06 + 0.15 }}
                  className="h-full rounded-full"
                  style={{ background: style.bar }}
                />
              </div>

              {/* Validated text */}
              <p className="text-[10px] text-[#6B6B6B] font-mono">
                {validated} of {doc.total_edges} validated
              </p>

              {/* Flagged badge */}
              {doc.flagged > 0 && (
                <span
                  className={`self-start px-1.5 py-0.5 rounded text-[9px] font-mono uppercase tracking-[0.1em] border ${style.badge}`}
                >
                  {doc.flagged} flagged
                </span>
              )}

              {/* Selected indicator */}
              {isSelected && (
                <span className="self-start px-1.5 py-0.5 rounded text-[9px] font-mono uppercase tracking-[0.1em] bg-[#D4AF37]/20 text-[#8B6914] border border-[#D4AF37]/40">
                  Filtering
                </span>
              )}
            </motion.button>
          );
        })}

        {documents.length === 0 && (
          <p className="text-sm text-[#6B6B6B] italic py-4 px-2">
            No document data available.
          </p>
        )}
      </div>
    </div>
  );
}
