"use client";

import React, { useState, useMemo } from "react";
import { motion } from "framer-motion";
import { ChevronRight, FileText, Shield, ChevronLeft, ChevronsRight } from "lucide-react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { DETECTIVE_BOARD } from "@/lib/constants";

export interface EvidenceStep {
  source: string;
  relation: string;
  target: string;
  audit?: number;
  explanation?: string | null;
  source_docs?: string[];
  target_docs?: string[];
  cross_document?: boolean;
}

interface DetectiveBoardProps {
  query: string;
  steps: EvidenceStep[];
  answerSummary?: string;
}

/**
 * Detective crime board: structured narrative flow for lawyers.
 * "Read this first → then this → then that" instead of a raw graph.
 */
export default function DetectiveBoard({
  query,
  steps,
  answerSummary,
}: DetectiveBoardProps) {
  const pageSize = DETECTIVE_BOARD.PAGE_SIZE;
  const needsPagination = steps.length > pageSize;
  const [currentPage, setCurrentPage] = useState(0);
  const totalPages = Math.ceil(steps.length / pageSize);

  const visibleSteps = useMemo(() => {
    if (!needsPagination) return steps;
    const start = currentPage * pageSize;
    return steps.slice(start, start + pageSize);
  }, [steps, currentPage, pageSize, needsPagination]);

  const pageOffset = currentPage * pageSize;

  if (steps.length === 0) return null;
  return (
    <div className="space-y-6">
      {/* Your question */}
      <div className="p-4 bg-[#7A1A1A]/5 border-l-4 border-[#7A1A1A] rounded-r-lg">
        <p className="text-[10px] font-mono uppercase tracking-widest text-[#737373] mb-1">
          Your Question
        </p>
        <p className="text-base font-medium text-[#1A1A1A]">&ldquo;{query}&rdquo;</p>
      </div>

      {/* Supporting facts: read in order */}
      <div>
        <div className="flex items-center justify-between mb-3">
          <p className="text-[10px] font-mono uppercase tracking-widest text-[#737373] flex items-center gap-2">
            <FileText size={12} />
            Supporting Facts — Read in order
            {needsPagination && (
              <span className="text-[#C5A028]">
                ({steps.length} total)
              </span>
            )}
          </p>
        </div>
        <div className="space-y-4">
          {visibleSteps.map((step, i) => {
            const globalIndex = pageOffset + i;
            return (
            <motion.div
              key={globalIndex}
              initial={{ opacity: 0, x: -10 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: i * 0.08 }}
              className="relative pl-8"
            >
              {/* Connector line to next step */}
              {i < visibleSteps.length - 1 && (
                <div className="absolute left-[11px] top-8 w-px h-8 bg-[#C5A028]/40" />
              )}
              {/* Step number */}
              <div className="absolute left-0 top-0 w-6 h-6 rounded-full bg-[#C5A028] text-[#1A1A1A] flex items-center justify-center text-xs font-bold">
                {globalIndex + 1}
              </div>
              {/* Card — lead with plain English explanation */}
              <div className="p-4 bg-[#FFFFFF] border border-[#E5E5E3] rounded-lg hover:border-[#C5A028]/40 focus-visible:border-[#C5A028]/40 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#C5A028]/50 transition-colors" tabIndex={0}>
                <div className="flex items-start justify-between gap-2">
                  <div className="min-w-0 flex-1">
                    {step.explanation ? (
                      <div className="prose prose-sm max-w-none prose-p:text-sm prose-p:text-[#1A1A1A] prose-p:leading-relaxed prose-p:my-0">
                        <ReactMarkdown remarkPlugins={[remarkGfm]}>
                          {step.explanation}
                        </ReactMarkdown>
                      </div>
                    ) : (
                      <p className="text-sm font-medium text-[#1A1A1A]">
                        {step.source}{" "}
                        <span className="text-[#737373] font-normal">
                          {String(step.relation ?? "")
                            .replace(/_/g, " ")
                            .toLowerCase()}
                        </span>{" "}
                        {step.target}
                      </p>
                    )}
                    <p className="text-[10px] text-[#737373] mt-2">
                      <span className="font-medium text-[#525252]">Connection:</span> {step.source}
                      <span className="text-[#C5A028] mx-1">→</span>
                      {String(step.relation ?? "").replace(/_/g, " ").toLowerCase()}
                      <span className="text-[#C5A028] mx-1">→</span>
                      {step.target}
                    </p>
                    {(step.source_docs?.length ?? 0) > 0 || (step.target_docs?.length ?? 0) > 0 ? (
                      <div className="mt-2 space-y-1">
                        {step.cross_document && (
                          <p className="text-[10px] font-semibold text-[#8B6914] flex items-center gap-1.5">
                            <span className="inline-block w-1.5 h-1.5 rounded-full bg-[#C5A028]" />
                            Cross-document: Links evidence from different sources
                          </p>
                        )}
                        <p className="text-[10px] text-[#737373]">
                          <span className="font-medium">Cited in:</span>{" "}
                          {[...new Set([...(step.source_docs ?? []), ...(step.target_docs ?? [])])].join(", ") || "—"}
                        </p>
                      </div>
                    ) : null}
                  </div>
                  <div className="shrink-0 flex flex-col items-end gap-1">
                    {step.cross_document && (
                      <span className="px-2.5 py-1 rounded text-[10px] font-semibold bg-[#C5A028]/25 text-[#8B6914] border border-[#C5A028]/50 flex items-center gap-1.5">
                        <span className="inline-block w-1.5 h-1.5 rounded-full bg-[#C5A028]" />
                        Cross-document
                      </span>
                    )}
                    {step.audit != null && (
                      <span className={`inline-flex items-center gap-1 px-2 py-0.5 rounded text-xs font-mono ${
                        step.audit >= 0.95
                          ? "bg-[var(--validated-green)]/10 text-[var(--validated-green)]"
                          : step.audit >= 0.85
                          ? "bg-[var(--gilded-gold)]/10 text-[var(--gilded-gold)]"
                          : "bg-[var(--conflict-red)]/10 text-[var(--conflict-red)]"
                      }`}>
                        <Shield size={10} />
                        {(step.audit * 100).toFixed(0)}%
                      </span>
                    )}
                  </div>
                </div>
              </div>
              {/* Arrow to next */}
              {i < visibleSteps.length - 1 && (
                <div className="flex justify-center py-1 text-[#C5A028]/60">
                  <ChevronRight size={16} className="rotate-90" />
                </div>
              )}
            </motion.div>
            );
          })}
        </div>

        {/* Pagination Controls */}
        {needsPagination && (
          <div className="flex items-center justify-between pt-2">
            <button
              onClick={() => setCurrentPage((p) => Math.max(0, p - 1))}
              disabled={currentPage === 0}
              className="flex items-center gap-1 px-3 py-1.5 text-xs font-medium rounded-sm border border-[#E5E5E3] bg-[#FFFFFF] hover:bg-[#E8E4D9] disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
            >
              <ChevronLeft size={14} />
              Previous
            </button>
            <span className="text-[10px] font-mono text-[#737373] uppercase tracking-wider">
              Facts {pageOffset + 1}–{Math.min(pageOffset + pageSize, steps.length)} of {steps.length}
            </span>
            <button
              onClick={() => setCurrentPage((p) => Math.min(totalPages - 1, p + 1))}
              disabled={currentPage >= totalPages - 1}
              className="flex items-center gap-1 px-3 py-1.5 text-xs font-medium rounded-sm border border-[#E5E5E3] bg-[#FFFFFF] hover:bg-[#E8E4D9] disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
            >
              Next
              <ChevronsRight size={14} />
            </button>
          </div>
        )}
      </div>

      {answerSummary && (
        <div className="p-4 bg-[#2D6A4F]/5 border-l-4 border-[#2D6A4F] rounded-r-lg">
          <p className="text-[10px] font-mono uppercase tracking-widest text-[#737373] mb-1">
            Conclusion
          </p>
          <p className="text-sm text-[#1A1A1A]">{answerSummary}</p>
        </div>
      )}
    </div>
  );
}
