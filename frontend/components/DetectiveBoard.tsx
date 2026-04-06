"use client";

import React from "react";
import { motion } from "framer-motion";
import { ChevronRight, FileText, Shield } from "lucide-react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

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
  if (steps.length === 0) return null;
  return (
    <div className="space-y-6">
      {/* Your question */}
      <div className="p-4 bg-[#8B1A1A]/5 border-l-4 border-[#8B1A1A] rounded-r-lg">
        <p className="text-[10px] font-mono uppercase tracking-widest text-[#6B6B6B] mb-1">
          Your Question
        </p>
        <p className="text-base font-medium text-[#2B2B2B]">&ldquo;{query}&rdquo;</p>
      </div>

      {/* Supporting facts: read in order */}
      <div>
        <p className="text-[10px] font-mono uppercase tracking-widest text-[#6B6B6B] mb-3 flex items-center gap-2">
          <FileText size={12} />
          Supporting Facts — Read in order
        </p>
        <div className="space-y-4">
          {steps.map((step, i) => (
            <motion.div
              key={i}
              initial={{ opacity: 0, x: -10 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: i * 0.08 }}
              className="relative pl-8"
            >
              {/* Connector line to next step */}
              {i < steps.length - 1 && (
                <div className="absolute left-[11px] top-8 w-px h-8 bg-[#D4AF37]/40" />
              )}
              {/* Step number */}
              <div className="absolute left-0 top-0 w-6 h-6 rounded-full bg-[#D4AF37] text-[#2B2B2B] flex items-center justify-center text-xs font-bold">
                {i + 1}
              </div>
              {/* Card — lead with plain English explanation */}
              <div className="p-4 bg-[#FCFAF2] border border-[#4A4A4A]/30 rounded-lg hover:border-[#D4AF37]/40 transition-colors">
                <div className="flex items-start justify-between gap-2">
                  <div className="min-w-0 flex-1">
                    {step.explanation ? (
                      <div className="prose prose-sm max-w-none prose-p:text-sm prose-p:text-[#2B2B2B] prose-p:leading-relaxed prose-p:my-0">
                        <ReactMarkdown remarkPlugins={[remarkGfm]}>
                          {step.explanation}
                        </ReactMarkdown>
                      </div>
                    ) : (
                      <p className="text-sm font-medium text-[#2B2B2B]">
                        {step.source}{" "}
                        <span className="text-[#6B6B6B] font-normal">
                          {String(step.relation ?? "")
                            .replace(/_/g, " ")
                            .toLowerCase()}
                        </span>{" "}
                        {step.target}
                      </p>
                    )}
                    <p className="text-[10px] text-[#6B6B6B] mt-2">
                      <span className="font-medium text-[#4A4A4A]">Connection:</span> {step.source}
                      <span className="text-[#D4AF37] mx-1">→</span>
                      {String(step.relation ?? "").replace(/_/g, " ").toLowerCase()}
                      <span className="text-[#D4AF37] mx-1">→</span>
                      {step.target}
                    </p>
                    {(step.source_docs?.length ?? 0) > 0 || (step.target_docs?.length ?? 0) > 0 ? (
                      <div className="mt-2 space-y-1">
                        {step.cross_document && (
                          <p className="text-[10px] font-semibold text-[#8B6914] flex items-center gap-1.5">
                            <span className="inline-block w-1.5 h-1.5 rounded-full bg-[#D4AF37]" />
                            Cross-document: Links evidence from different sources
                          </p>
                        )}
                        <p className="text-[10px] text-[#6B6B6B]">
                          <span className="font-medium">Cited in:</span>{" "}
                          {[...new Set([...(step.source_docs ?? []), ...(step.target_docs ?? [])])].join(", ") || "—"}
                        </p>
                      </div>
                    ) : null}
                  </div>
                  <div className="shrink-0 flex flex-col items-end gap-1">
                    {step.cross_document && (
                      <span className="px-2.5 py-1 rounded text-[10px] font-semibold bg-[#D4AF37]/25 text-[#8B6914] border border-[#D4AF37]/50 flex items-center gap-1.5">
                        <span className="inline-block w-1.5 h-1.5 rounded-full bg-[#D4AF37]" />
                        Cross-document
                      </span>
                    )}
                    {typeof step.audit === "number" && (
                      <div className="flex items-center gap-1 px-2 py-0.5 rounded bg-[#3A5A40]/10 border border-[#3A5A40]/20">
                        <Shield size={10} className="text-[#3A5A40]" />
                        <span className="text-[10px] font-mono font-bold text-[#3A5A40]">
                          {Math.round((step.audit ?? 0) * 100)}%
                        </span>
                      </div>
                    )}
                  </div>
                </div>
              </div>
              {/* Arrow to next */}
              {i < steps.length - 1 && (
                <div className="flex justify-center py-1 text-[#D4AF37]/60">
                  <ChevronRight size={16} className="rotate-90" />
                </div>
              )}
            </motion.div>
          ))}
        </div>
      </div>

      {answerSummary && (
        <div className="p-4 bg-[#3A5A40]/5 border-l-4 border-[#3A5A40] rounded-r-lg">
          <p className="text-[10px] font-mono uppercase tracking-widest text-[#6B6B6B] mb-1">
            Conclusion
          </p>
          <p className="text-sm text-[#2B2B2B]">{answerSummary}</p>
        </div>
      )}
    </div>
  );
}
