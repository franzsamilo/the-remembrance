"use client";

import React, { useState, useEffect } from "react";
import { useRouter } from "next/navigation";
import Link from "next/link";
import { X, FileText, Scale, ArrowRightCircle, ChevronDown, ChevronUp, Network } from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import axios from "axios";
import DetectiveBoard from "@/components/DetectiveBoard";
import KnowledgeGraph from "@/components/KnowledgeGraph";
import { SkeletonDetectiveBoard } from "@/components/Skeleton";
import FilteringSummary from "@/components/FilteringSummary";
import RejectedEvidence from "@/components/RejectedEvidence";
import GroundingError from "@/components/GroundingError";
import { Triplet, Lead } from "@/lib/types";
import { STORAGE_KEYS } from "@/lib/constants";
import { API_BASE_URL } from "@/lib/api";

const DEFAULT_GROUNDING_THRESHOLD = 0.95;

type EvidenceMessage = {
  role: string;
  content: string;
  triplets?: Triplet[];
  filtered_triplets?: Triplet[];
  leads?: Lead[];
  suggested_actions?: string[];
  userQuery?: string;
  explain?: boolean;
  groundingError?: boolean;
};

export default function EvidencePage() {
  const router = useRouter();
  const [message, setMessage] = useState<EvidenceMessage | null>(null);
  const [graphViewExpanded, setGraphViewExpanded] = useState(false);
  const [groundingThreshold, setGroundingThreshold] = useState<number>(
    DEFAULT_GROUNDING_THRESHOLD
  );

  useEffect(() => {
    if (typeof window === "undefined") return;
    try {
      const raw = sessionStorage.getItem(STORAGE_KEYS.EVIDENCE_MESSAGE);
      if (raw) {
        const parsed = JSON.parse(raw) as EvidenceMessage;
        setMessage(parsed);
      }
      // If no data, stay on page and show fallback UI (don't redirect)
    } catch {
      // Corrupted data — stay on page with fallback
    }
  }, [router]);

  useEffect(() => {
    let cancelled = false;
    axios
      .get(`${API_BASE_URL}/stats`)
      .then((res) => {
        if (cancelled) return;
        const tau = res.data?.inference_config?.grounding_threshold;
        if (typeof tau === "number" && tau > 0 && tau <= 1) {
          setGroundingThreshold(tau);
        }
      })
      .catch(() => {
        // Backend unreachable — keep the default τ used everywhere else
      });
    return () => {
      cancelled = true;
    };
  }, []);

  // All hooks must run unconditionally (before any early return)
  const triplets = message?.triplets ?? [];
  const leads = message?.leads ?? [];
  const entityCount = React.useMemo(
    () =>
      new Set(
        triplets.flatMap((t) => [t.source, t.target]).filter((v): v is string => typeof v === "string" && v.trim().length > 0)
      ).size,
    [triplets]
  );
  const hasCrossDoc = triplets.some((t) => t.cross_document);
  const uniqueDocCount = React.useMemo(
    () => new Set(triplets.flatMap((t) => [...(t.source_docs ?? []), ...(t.target_docs ?? [])])).size,
    [triplets]
  );
  const suggestedActions = React.useMemo(() => {
    const fromBackend = message?.suggested_actions ?? [];
    if (fromBackend.length > 0) return fromBackend;
    const fallback: string[] = [];
    if (hasCrossDoc) fallback.push("Request discovery on the linked entities — cross-document evidence strengthens your case.");
    if (leads.length > 0) fallback.push(`Subpoena or depose the ${leads.length} lead(s) above; they may yield material evidence.`);
    if (uniqueDocCount > 1) fallback.push("Draft a motion to cite these sources in your brief; verify each citation against the original documents.");
    if (fallback.length === 0 && triplets.length > 0) fallback.push("Add these facts to your case file and cite the source document in your brief.");
    return fallback;
  }, [message?.suggested_actions, hasCrossDoc, leads.length, uniqueDocCount, triplets.length]);
  const sourceCitations = React.useMemo(() => {
    const byDoc = new Map<string, { facts: string[]; crossDoc: boolean }>();
    for (const t of triplets) {
      const conn = `${t.source ?? ""} ${String(t.relation ?? "").replace(/_/g, " ").toLowerCase()} ${t.target ?? ""}`.trim();
      const docs = [...new Set([...(t.source_docs ?? []), ...(t.target_docs ?? [])])];
      if (docs.length === 0) {
        const key = "(Unattributed)";
        if (!byDoc.has(key)) byDoc.set(key, { facts: [], crossDoc: false });
        byDoc.get(key)!.facts.push(conn);
      } else {
        for (const doc of docs) {
          if (!byDoc.has(doc)) byDoc.set(doc, { facts: [], crossDoc: false });
          const entry = byDoc.get(doc)!;
          if (!entry.facts.includes(conn)) entry.facts.push(conn);
          if (t.cross_document) entry.crossDoc = true;
        }
      }
    }
    return Array.from(byDoc.entries()).map(([doc, { facts, crossDoc }]) => ({ doc, facts, crossDoc }));
  }, [triplets]);

  const handleClose = () => {
    if (typeof window !== "undefined") {
      sessionStorage.removeItem(STORAGE_KEYS.EVIDENCE_MESSAGE);
      sessionStorage.setItem(STORAGE_KEYS.OPEN_CHAT, "1");
    }
    router.push("/");
  };

  const query = message?.userQuery ?? "Query";

  if (!message) {
    return (
      <div className="min-h-screen bg-[#FAFAF8] flex flex-col items-center justify-center p-8 text-center">
        <div className="p-6 bg-[#E8E4D9]/80 rounded-full border border-[#525252]/30 mb-6">
          <Network size={48} className="text-[#737373]" />
        </div>
        <h2 className="text-xl font-semibold text-[#1A1A1A] mb-2" style={{fontFamily: 'EB Garamond, serif'}}>
          No Evidence Trail Loaded
        </h2>
        <p className="text-sm text-[#737373] max-w-md mb-6">
          Ask a question from the dashboard with &ldquo;Explain Connections&rdquo; enabled, then click &ldquo;View Evidence&rdquo; to see the full trail here.
        </p>
        <Link
          href="/"
          className="inline-flex items-center gap-2 px-5 py-2.5 bg-[#C5A028] hover:bg-[#B8941F] text-[#1A1A1A] rounded-sm font-semibold shadow-lg transition-all text-sm"
        >
          Go to Dashboard
        </Link>
      </div>
    );
  }

  if (!message.explain || triplets.length === 0) {
    return (
      <div className="min-h-screen bg-[#FAFAF8] flex flex-col items-center justify-center p-8 text-center">
        <div className="p-6 bg-[#E8E4D9]/80 rounded-full border border-[#525252]/30 mb-6">
          <FileText size={48} className="text-[#737373]" />
        </div>
        <h2 className="text-xl font-semibold text-[#1A1A1A] mb-2" style={{fontFamily: 'EB Garamond, serif'}}>
          No Evidence Trail
        </h2>
        <p className="text-sm text-[#737373] max-w-md mb-6">
          This response did not include detective insights. Go back and ask a question with &ldquo;Explain Connections&rdquo; enabled.
        </p>
        <Link
          href="/"
          className="inline-flex items-center gap-2 px-5 py-2.5 bg-[#C5A028] hover:bg-[#B8941F] text-[#1A1A1A] rounded-sm font-semibold shadow-lg transition-all text-sm"
        >
          Back to Dashboard
        </Link>
      </div>
    );
  }

  const steps = triplets.map((t) => ({
    source: t.source ?? "",
    relation: t.relation ?? "",
    target: t.target ?? "",
    audit: t.audit,
    explanation: t.explanation,
    source_docs: t.source_docs,
    target_docs: t.target_docs,
    cross_document: t.cross_document,
  }));

  return (
    <div className="h-screen flex flex-col bg-[#FAFAF8] text-[#1A1A1A] overflow-hidden">
      <header className="shrink-0 z-50 bg-[#FFFFFF]/98 backdrop-blur border-b border-[#525252]/50">
        <div className="max-w-7xl mx-auto px-6 py-4 flex justify-between items-center">
          <div>
            <p className="text-[10px] font-mono uppercase tracking-widest text-[#737373]">
              Evidence Board
            </p>
            <h1 className="text-lg font-semibold text-[#1A1A1A]">
              Evidence for: &ldquo;{query}&rdquo;
            </h1>
          </div>
          <div className="flex items-center gap-1">
            <Link
              href="/"
              className="px-4 py-2 text-sm text-[#737373] hover:text-[#1A1A1A] transition-colors"
              onClick={() => {
                if (typeof window !== "undefined") {
                  sessionStorage.setItem(STORAGE_KEYS.OPEN_CHAT, "1");
                }
              }}
            >
              Back
            </Link>
            <button
              onClick={handleClose}
              className="p-2 rounded-full hover:bg-[#E8E4D9] text-[#737373] transition-colors"
              aria-label="Close"
            >
              <X size={20} />
            </button>
          </div>
        </div>
      </header>

      <main className="flex-1 min-h-0 overflow-y-auto">
        {message?.groundingError ? (
          <div className="max-w-4xl mx-auto p-8 space-y-6">
            <GroundingError
              message={message.content}
              filteredTriplets={message.filtered_triplets ?? []}
            />
            <RejectedEvidence
              triplets={message.filtered_triplets ?? []}
              threshold={groundingThreshold}
            />
          </div>
        ) : (
        <div className="max-w-7xl mx-auto grid grid-cols-1 lg:grid-cols-[1.2fr_1fr] gap-0">
        {/* Supporting Facts — Evidence Trail + Graph View */}
        <div className="p-6 border-r border-[#E5E5E3] bg-[#FAFAF8]/60 space-y-6">
          {/* Expandable Graph View */}
          <div className="border border-[#E5E5E3] rounded-lg bg-[#FFFFFF] overflow-hidden">
            <button
              type="button"
              onClick={() => setGraphViewExpanded(!graphViewExpanded)}
              className="w-full flex items-center justify-between gap-3 px-4 py-3 text-left hover:bg-[#E8E4D9]/50 transition-colors"
              aria-expanded={graphViewExpanded}
              aria-controls="graph-view-content"
              id="graph-view-toggle"
            >
              <div className="flex items-center gap-2">
                <Network size={18} className="text-[#C5A028]" />
                <span className="text-sm font-semibold text-[#1A1A1A]">Graph View</span>
                <span className="text-[10px] font-mono uppercase tracking-widest text-[#737373]">
                  {triplets.length} facts · {entityCount} entities
                </span>
              </div>
              {graphViewExpanded ? (
                <ChevronUp size={18} className="text-[#737373]" />
              ) : (
                <ChevronDown size={18} className="text-[#737373]" />
              )}
            </button>
            <AnimatePresence>
              {graphViewExpanded && (
                <motion.div
                  id="graph-view-content"
                  role="region"
                  aria-labelledby="graph-view-toggle"
                  initial={{ height: 0, opacity: 0 }}
                  animate={{ height: "auto", opacity: 1 }}
                  exit={{ height: 0, opacity: 0 }}
                  transition={{ duration: 0.25 }}
                  className="overflow-hidden"
                >
                  <div className="p-4 pt-0">
                    <div className="min-h-[400px]">
                      <KnowledgeGraph
                        triplets={triplets}
                        leads={leads}
                        query={query}
                        showInlineCard={true}
                        large={true}
                      />
                    </div>
                  </div>
                </motion.div>
              )}
            </AnimatePresence>
          </div>

          <DetectiveBoard
            query={query}
            steps={steps}
            answerSummary={
              message.content?.slice(0, 180) +
              (message.content && message.content.length > 180 ? "…" : "")
            }
          />

          {(message?.filtered_triplets?.length ?? 0) > 0 && (
            <FilteringSummary
              validatedCount={triplets.length}
              filteredCount={message?.filtered_triplets?.length ?? 0}
              threshold={groundingThreshold}
            />
          )}

          <RejectedEvidence
            triplets={message?.filtered_triplets ?? []}
            threshold={groundingThreshold}
          />
        </div>

        {/* Side panel: Suggested Actions + Source Citations + Leads (lawyer-friendly) */}
        <div className="p-6 space-y-6 bg-[#FFFFFF]">
          {/* Suggested Next Steps — actionable recommendations */}
          {suggestedActions.length > 0 && (
            <div>
              <div className="text-[10px] font-mono uppercase tracking-widest text-[#737373] mb-3 flex items-center gap-2">
                <ArrowRightCircle size={12} />
                Suggested Next Steps
              </div>
              <p className="text-xs text-[#737373] mb-3">
                To follow up on your question, take these actions:
              </p>
              <div className="space-y-2">
                {suggestedActions.map((action, idx) => (
                  <div
                    key={idx}
                    className="flex items-start gap-2 p-3 rounded-lg border border-[#2D6A4F]/50 bg-[#2D6A4F]/5 hover:border-[#2D6A4F] transition-colors"
                  >
                    <span className="text-[#2D6A4F] font-bold shrink-0 mt-0.5">{idx + 1}.</span>
                    <div className="prose prose-sm max-w-none text-sm font-medium text-[#1A1A1A] leading-relaxed prose-p:my-0 prose-strong:text-[#1A1A1A] prose-strong:font-semibold">
                      <ReactMarkdown remarkPlugins={[remarkGfm]}>
                        {action}
                      </ReactMarkdown>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Source Citations — which documents support which facts */}
          <div>
            <div className="text-[10px] font-mono uppercase tracking-widest text-[#737373] mb-3 flex items-center gap-2">
              <Scale size={12} />
              Source Citations
            </div>
            <p className="text-xs text-[#737373] mb-3">
              Trace each fact back to its source document. Use this to verify citations and build your case file.
            </p>
            <div className="space-y-4">
              {sourceCitations.map(({ doc, facts, crossDoc }, idx) => (
                <div
                  key={idx}
                  className="p-4 rounded-lg border border-[#E5E5E3] bg-[#FFFFFF] hover:border-[#C5A028]/40 transition-colors"
                >
                  <div className="flex items-center gap-2 mb-2">
                    <FileText size={14} className="text-[#7A1A1A] shrink-0" />
                    <span className="text-sm font-semibold text-[#1A1A1A] truncate">
                      {doc}
                    </span>
                    {crossDoc && (
                      <span className="px-2 py-0.5 rounded text-[9px] font-medium bg-[#C5A028]/20 text-[#8B6914] border border-[#C5A028]/40 shrink-0">
                        Cross-doc
                      </span>
                    )}
                  </div>
                  <ul className="space-y-1.5 text-xs text-[#525252] list-disc list-inside">
                    {facts.map((fact, fidx) => (
                      <li key={fidx} className="leading-relaxed">
                        {fact}
                      </li>
                    ))}
                  </ul>
                </div>
              ))}
            </div>
          </div>

          {/* Key Leads to Explore */}
          {leads.length > 0 && (
            <div>
              <div className="text-[10px] font-mono uppercase tracking-widest text-[#737373] mb-3 flex items-center gap-2">
                <FileText size={12} />
                Leads to Explore
              </div>
              <p className="text-xs text-[#737373] mb-3">
                Additional entities or concepts worth investigating for your case.
              </p>
              <div className="space-y-3">
                {leads.map((lead, lidx) => (
                  <div
                    key={lidx}
                    className="p-3 rounded-lg border border-[#C5A028]/30 bg-[#C5A028]/5"
                  >
                    <p className="text-sm font-semibold text-[#1A1A1A]">
                      {lead.name}
                    </p>
                    {lead.description && (
                      <p className="text-xs text-[#737373] mt-1">
                        {lead.description}
                      </p>
                    )}
                    {lead.explanation && (
                      <p className="text-xs text-[#1A1A1A] mt-2">
                        {lead.explanation}
                      </p>
                    )}
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
        </div>
        )}
      </main>
    </div>
  );
}
