"use client";

import React from "react";
import { motion, AnimatePresence } from "framer-motion";
import { X } from "lucide-react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { KnowledgeGraphIcon, WaxSealIcon } from "@/components/CustomIcons";

export interface NodeInfo {
  id: string;
  name: string;
  type: "query" | "entity" | "lead";
  description?: string;
  explanation?: string;
  connections: { name: string; relation: string; direction: "in" | "out" }[];
  queryRelation?: string;
}

export interface EdgeInfo {
  id: string;
  source: string;
  target: string;
  relation: string;
  audit: number;
  description?: string;
  explanation?: string;
  cross_document?: boolean;
}

interface GraphInfoCardProps {
  nodeInfo?: NodeInfo | null;
  edgeInfo?: EdgeInfo | null;
  position: { x: number; y: number };
  onClose: () => void;
  query: string;
  variant?: "floating" | "panel";
}

export default function GraphInfoCard({
  nodeInfo,
  edgeInfo,
  position,
  onClose,
  query,
  variant = "floating",
}: GraphInfoCardProps) {
  if (!nodeInfo && !edgeInfo) return null;

  const typeBadge = (type: string) => {
    const colors: Record<string, string> = {
      query: "bg-[#D4AF37]/20 text-[#D4AF37] border-[#D4AF37]/30",
      entity: "bg-[#8B1A1A]/15 text-[#8B1A1A] border-[#8B1A1A]/30",
      lead: "bg-[#3A5A40]/15 text-[#3A5A40] border-[#3A5A40]/30",
      relation: "bg-[#6B6B6B]/15 text-[#6B6B6B] border-[#6B6B6B]/30",
    };
    return colors[type] || colors.entity;
  };

  // Clamp card position so it stays within the container
  const cardStyle: React.CSSProperties =
    variant === "panel"
      ? {
          position: "relative",
          left: "auto",
          top: "auto",
          zIndex: 1,
          width: "100%",
        }
      : {
          position: "absolute",
          left: Math.min(Math.max(position.x - 160, 8), 280),
          top: Math.min(Math.max(position.y - 20, 8), 300),
          zIndex: 100,
          width: 320,
        };

  return (
    <AnimatePresence>
      <motion.div
        initial={{ opacity: 0, scale: 0.85, y: 10 }}
        animate={{ opacity: 1, scale: 1, y: 0 }}
        exit={{ opacity: 0, scale: 0.85, y: 10 }}
        transition={{ type: "spring", damping: 25, stiffness: 350 }}
        style={cardStyle}
        className="pointer-events-auto"
      >
        <div className="graph-info-card bg-[#FCFAF2]/95 backdrop-blur-xl border border-[#D4AF37]/40 rounded-lg shadow-2xl shadow-[#2B2B2B]/20 overflow-hidden">
          {/* Gold accent top bar */}
          <div className="h-1 w-full bg-linear-to-r from-[#D4AF37] via-[#B8941F] to-[#D4AF37]" />

          <div className="p-4">
            {/* Header */}
            <div className="flex items-start justify-between mb-3">
              <div className="flex items-center gap-2 flex-1 min-w-0">
                <div className="p-1.5 bg-[#E8E4D9] rounded-md border border-[#4A4A4A]/30 shrink-0">
                  {nodeInfo ? (
                    <KnowledgeGraphIcon size={14} className="text-[#D4AF37]" />
                  ) : (
                    <WaxSealIcon size={14} className="text-[#8B1A1A]" />
                  )}
                </div>
                <div className="min-w-0">
                  <h4
                    className="text-sm font-bold text-[#2B2B2B] truncate leading-tight"
                    style={{ fontFamily: "EB Garamond, serif" }}
                  >
                    {nodeInfo ? nodeInfo.name : edgeInfo?.relation}
                  </h4>
                  <span
                    className={`inline-block mt-0.5 text-[9px] font-mono uppercase tracking-wider px-1.5 py-0.5 rounded border ${typeBadge(
                      nodeInfo ? nodeInfo.type : "relation"
                    )}`}
                  >
                    {nodeInfo ? nodeInfo.type : "Relationship"}
                  </span>
                </div>
              </div>
              <button
                onClick={onClose}
                className="p-1 hover:bg-[#E8E4D9] rounded-md transition-colors text-[#6B6B6B] hover:text-[#2B2B2B] shrink-0"
              >
                <X size={14} />
              </button>
            </div>

            {/* Divider */}
            <div className="w-full h-px bg-linear-to-r from-transparent via-[#D4AF37]/30 to-transparent mb-3" />

            {/* NODE CONTENT */}
            {nodeInfo && (
              <div className="space-y-3">
                {/* Detective Insight */}
                {(nodeInfo.explanation || nodeInfo.type !== "query") && (
                  <div>
                    <p className="text-[9px] font-mono uppercase tracking-widest text-[#6B6B6B] mb-1">
                      Detective Insight
                    </p>
                    <div className="p-2 bg-[#8B1A1A]/5 border border-[#8B1A1A]/15 rounded-md prose prose-sm max-w-none prose-p:text-xs prose-p:text-[#2B2B2B] prose-p:leading-relaxed prose-strong:text-[#2B2B2B] prose-strong:font-semibold">
                      <ReactMarkdown remarkPlugins={[remarkGfm]}>
                        {nodeInfo.explanation ||
                          "This node appears in evidence tied to your query, suggesting it plays a relevant role in the surrounding context."}
                      </ReactMarkdown>
                    </div>
                  </div>
                )}
                {/* Description */}
                {nodeInfo.description && (
                  <div>
                    <p className="text-[9px] font-mono uppercase tracking-widest text-[#6B6B6B] mb-1">
                      Description
                    </p>
                    <p className="text-xs text-[#4A4A4A] leading-relaxed">
                      {nodeInfo.description}
                    </p>
                  </div>
                )}

                {/* Query Relation */}
                {nodeInfo.type !== "query" && (
                  <div>
                    <p className="text-[9px] font-mono uppercase tracking-widest text-[#6B6B6B] mb-1">
                      Relation to Query
                    </p>
                    <div className="p-2 bg-[#D4AF37]/5 border border-[#D4AF37]/20 rounded-md">
                      <p className="text-[10px] text-[#4A4A4A] italic leading-relaxed">
                        &ldquo;{query}&rdquo;
                      </p>
                      <p className="text-xs text-[#2B2B2B] mt-1 font-medium">
                        {nodeInfo.queryRelation ||
                          `This entity was retrieved as a relevant knowledge node from the graph.`}
                      </p>
                    </div>
                  </div>
                )}

                {/* Connections */}
                {nodeInfo.connections.length > 0 && (
                  <div>
                    <p className="text-[9px] font-mono uppercase tracking-widest text-[#6B6B6B] mb-1.5">
                      Connections ({nodeInfo.connections.length})
                    </p>
                    <div className="space-y-1 max-h-28 overflow-y-auto scrollbar-thin">
                      {nodeInfo.connections.map((conn, i) => (
                        <div
                          key={i}
                          className="flex items-center gap-1.5 text-[10px] p-1.5 rounded bg-[#E8E4D9]/50 hover:bg-[#E8E4D9] transition-colors"
                        >
                          <span className="text-[#8B1A1A] font-medium truncate max-w-[80px]">
                            {conn.direction === "out"
                              ? nodeInfo.name
                              : conn.name}
                          </span>
                          <span className="text-[#D4AF37] shrink-0">→</span>
                          <span className="text-[#6B6B6B] font-mono text-[9px] truncate max-w-[70px]">
                            {conn.relation}
                          </span>
                          <span className="text-[#D4AF37] shrink-0">→</span>
                          <span className="text-[#3A5A40] font-medium truncate max-w-[80px]">
                            {conn.direction === "out"
                              ? conn.name
                              : nodeInfo.name}
                          </span>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            )}

            {/* EDGE CONTENT */}
            {edgeInfo && (
              <div className="space-y-3">
                {/* Detective Insight */}
                <div>
                  <p className="text-[9px] font-mono uppercase tracking-widest text-[#6B6B6B] mb-1">
                    Detective Insight
                  </p>
                  <div className="p-2 bg-[#D4AF37]/5 border border-[#D4AF37]/20 rounded-md prose prose-sm max-w-none prose-p:text-xs prose-p:text-[#2B2B2B] prose-p:leading-relaxed prose-strong:text-[#2B2B2B] prose-strong:font-semibold">
                    <ReactMarkdown remarkPlugins={[remarkGfm]}>
                      {edgeInfo.explanation ||
                        "This relationship was pulled as supporting evidence, indicating a meaningful link in the context of your query."}
                    </ReactMarkdown>
                  </div>
                </div>
                {/* Cross-document badge */}
                {edgeInfo.cross_document && (
                  <div className="flex items-center gap-2 px-2.5 py-1.5 rounded-md bg-[#D4AF37]/20 border border-[#D4AF37]/40">
                    <span className="inline-block w-2 h-2 rounded-full bg-[#D4AF37]" />
                    <span className="text-[10px] font-semibold text-[#8B6914]">
                      Cross-document connection — Links evidence from different sources
                    </span>
                  </div>
                )}

                {/* Source → Target */}
                <div>
                  <p className="text-[9px] font-mono uppercase tracking-widest text-[#6B6B6B] mb-1.5">
                    Connection Path
                  </p>
                  <div className="flex items-center gap-2 p-2.5 bg-[#E8E4D9]/60 border border-[#4A4A4A]/30 rounded-md">
                    <span className="text-xs font-semibold text-[#8B1A1A] truncate">
                      {edgeInfo.source}
                    </span>
                    <div className="flex items-center gap-1 shrink-0">
                      <div className="w-6 h-px bg-[#D4AF37]" />
                      <div className="w-0 h-0 border-l-[5px] border-l-[#D4AF37] border-y-[3px] border-y-transparent" />
                    </div>
                    <span className="text-xs font-semibold text-[#3A5A40] truncate">
                      {edgeInfo.target}
                    </span>
                  </div>
                </div>

                {/* Description */}
                {edgeInfo.description && (
                  <div>
                    <p className="text-[9px] font-mono uppercase tracking-widest text-[#6B6B6B] mb-1">
                      Description
                    </p>
                    <p className="text-xs text-[#4A4A4A] leading-relaxed">
                      {edgeInfo.description}
                    </p>
                  </div>
                )}

                {/* Audit Score */}
                <div>
                  <p className="text-[9px] font-mono uppercase tracking-widest text-[#6B6B6B] mb-1.5">
                    Audit Confidence
                  </p>
                  <div className="flex items-center gap-3">
                    <div className="flex-1 h-2 bg-[#E8E4D9] rounded-full overflow-hidden border border-[#4A4A4A]/20">
                      <motion.div
                        initial={{ width: 0 }}
                        animate={{
                          width: `${Math.round(edgeInfo.audit * 100)}%`,
                        }}
                        transition={{ duration: 0.8, ease: "easeOut" }}
                        className="h-full rounded-full"
                        style={{
                          background:
                            edgeInfo.audit > 0.8
                              ? "linear-gradient(90deg, #3A5A40, #4A7A50)"
                              : edgeInfo.audit > 0.5
                              ? "linear-gradient(90deg, #D4AF37, #B8941F)"
                              : "linear-gradient(90deg, #8B1A1A, #C41E3A)",
                        }}
                      />
                    </div>
                    <span className="text-xs font-mono font-bold text-[#2B2B2B] shrink-0">
                      {Math.round(edgeInfo.audit * 100)}%
                    </span>
                  </div>
                </div>

                {/* Query Context */}
                <div>
                  <p className="text-[9px] font-mono uppercase tracking-widest text-[#6B6B6B] mb-1">
                    Query Context
                  </p>
                  <div className="p-2 bg-[#D4AF37]/5 border border-[#D4AF37]/20 rounded-md">
                    <p className="text-[10px] text-[#4A4A4A] italic">
                      &ldquo;{query}&rdquo;
                    </p>
                    <p className="text-xs text-[#2B2B2B] mt-1">
                      This relationship was retrieved as supporting evidence for
                      the query.
                    </p>
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>
      </motion.div>
    </AnimatePresence>
  );
}
