"use client";

import React, { useState, useEffect } from "react";
import { useRouter } from "next/navigation";
import Link from "next/link";
import { X, FileText, Scale, ArrowRightCircle, ChevronDown, ChevronUp, Network } from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";
import DetectiveBoard from "@/components/DetectiveBoard";
import KnowledgeGraph from "@/components/KnowledgeGraph";

const EVIDENCE_STORAGE_KEY = "remembrance_evidence_message";

type Triplet = {
  source?: string | null;
  relation?: string | null;
  target?: string | null;
  audit?: number;
  description?: string;
  explanation?: string | null;
  source_docs?: string[];
  target_docs?: string[];
  cross_document?: boolean;
};

type Lead = {
  name: string;
  description?: string | null;
  explanation?: string | null;
};

type EvidenceMessage = {
  role: string;
  content: string;
  triplets?: Triplet[];
  leads?: Lead[];
  suggested_actions?: string[];
  userQuery?: string;
  explain?: boolean;
};

export default function EvidencePage() {
  const router = useRouter();
  const [message, setMessage] = useState<EvidenceMessage | null>(null);
  const [graphViewExpanded, setGraphViewExpanded] = useState(false);

  useEffect(() => {
    if (typeof window === "undefined") return;
    try {
      const raw = sessionStorage.getItem(EVIDENCE_STORAGE_KEY);
      if (raw) {
        const parsed = JSON.parse(raw) as EvidenceMessage;
        setMessage(parsed);
      } else {
        router.replace("/");
      }
    } catch {
      router.replace("/");
    }
  }, [router]);

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
      sessionStorage.removeItem(EVIDENCE_STORAGE_KEY);
      sessionStorage.setItem("remembrance_open_chat", "1");
    }
    router.push("/");
  };

  const query = message?.userQuery ?? "Query";

  if (!message) {
    return (
      <div className="min-h-screen bg-[#F5F2E9] flex items-center justify-center">
        <p className="text-[#6B6B6B]">Loading evidence...</p>
      </div>
    );
  }

  if (!message.explain || triplets.length === 0) {
    return (
      <div className="min-h-screen bg-[#F5F2E9] flex flex-col items-center justify-center p-8">
        <p className="text-[#6B6B6B] mb-4">No evidence trail for this message.</p>
        <Link href="/" className="text-[#D4AF37] hover:underline">
          Back to dashboard
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
    <div className="h-screen flex flex-col bg-[#F5F2E9] text-[#2B2B2B] overflow-hidden">
      <header className="shrink-0 z-50 bg-[#FCFAF2]/98 backdrop-blur border-b border-[#4A4A4A]/50">
        <div className="max-w-7xl mx-auto px-6 py-4 flex justify-between items-center">
          <div>
            <p className="text-[10px] font-mono uppercase tracking-widest text-[#6B6B6B]">
              Evidence Board
            </p>
            <h1 className="text-lg font-semibold text-[#2B2B2B]">
              Evidence for: &ldquo;{query}&rdquo;
            </h1>
          </div>
          <div className="flex items-center gap-1">
            <Link
              href="/"
              className="px-4 py-2 text-sm text-[#6B6B6B] hover:text-[#2B2B2B] transition-colors"
              onClick={() => {
                if (typeof window !== "undefined") {
                  sessionStorage.setItem("remembrance_open_chat", "1");
                }
              }}
            >
              Back
            </Link>
            <button
              onClick={handleClose}
              className="p-2 rounded-full hover:bg-[#E8E4D9] text-[#6B6B6B] transition-colors"
              aria-label="Close"
            >
              <X size={20} />
            </button>
          </div>
        </div>
      </header>

      <main className="flex-1 min-h-0 overflow-y-auto">
        <div className="max-w-7xl mx-auto grid grid-cols-1 lg:grid-cols-[1.2fr_1fr] gap-0">
        {/* Supporting Facts — Evidence Trail + Graph View */}
        <div className="p-6 border-r border-[#4A4A4A]/30 bg-[#F5F2E9]/60 space-y-6">
          {/* Expandable Graph View */}
          <div className="border border-[#4A4A4A]/30 rounded-lg bg-[#FCFAF2] overflow-hidden">
            <button
              type="button"
              onClick={() => setGraphViewExpanded(!graphViewExpanded)}
              className="w-full flex items-center justify-between gap-3 px-4 py-3 text-left hover:bg-[#E8E4D9]/50 transition-colors"
              aria-expanded={graphViewExpanded}
              aria-controls="graph-view-content"
              id="graph-view-toggle"
            >
              <div className="flex items-center gap-2">
                <Network size={18} className="text-[#D4AF37]" />
                <span className="text-sm font-semibold text-[#2B2B2B]">Graph View</span>
                <span className="text-[10px] font-mono uppercase tracking-widest text-[#6B6B6B]">
                  {triplets.length} facts · {entityCount} entities
                </span>
              </div>
              {graphViewExpanded ? (
                <ChevronUp size={18} className="text-[#6B6B6B]" />
              ) : (
                <ChevronDown size={18} className="text-[#6B6B6B]" />
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
        </div>

        {/* Side panel: Suggested Actions + Source Citations + Leads (lawyer-friendly) */}
        <div className="p-6 space-y-6 bg-[#FCFAF2]">
          {/* Suggested Next Steps — actionable recommendations */}
          {suggestedActions.length > 0 && (
            <div>
              <div className="text-[10px] font-mono uppercase tracking-widest text-[#6B6B6B] mb-3 flex items-center gap-2">
                <ArrowRightCircle size={12} />
                Suggested Next Steps
              </div>
              <p className="text-xs text-[#6B6B6B] mb-3">
                To follow up on your question, take these actions:
              </p>
              <div className="space-y-2">
                {suggestedActions.map((action, idx) => (
                  <div
                    key={idx}
                    className="flex items-start gap-2 p-3 rounded-lg border border-[#3A5A40]/50 bg-[#3A5A40]/5 hover:border-[#3A5A40] transition-colors"
                  >
                    <span className="text-[#3A5A40] font-bold shrink-0 mt-0.5">{idx + 1}.</span>
                    <p className="text-sm font-medium text-[#2B2B2B] leading-relaxed">
                      {action}
                    </p>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Source Citations — which documents support which facts */}
          <div>
            <div className="text-[10px] font-mono uppercase tracking-widest text-[#6B6B6B] mb-3 flex items-center gap-2">
              <Scale size={12} />
              Source Citations
            </div>
            <p className="text-xs text-[#6B6B6B] mb-3">
              Trace each fact back to its source document. Use this to verify citations and build your case file.
            </p>
            <div className="space-y-4">
              {sourceCitations.map(({ doc, facts, crossDoc }, idx) => (
                <div
                  key={idx}
                  className="p-4 rounded-lg border border-[#4A4A4A]/30 bg-[#FCFAF2] hover:border-[#D4AF37]/40 transition-colors"
                >
                  <div className="flex items-center gap-2 mb-2">
                    <FileText size={14} className="text-[#8B1A1A] shrink-0" />
                    <span className="text-sm font-semibold text-[#2B2B2B] truncate">
                      {doc}
                    </span>
                    {crossDoc && (
                      <span className="px-2 py-0.5 rounded text-[9px] font-medium bg-[#D4AF37]/20 text-[#8B6914] border border-[#D4AF37]/40 shrink-0">
                        Cross-doc
                      </span>
                    )}
                  </div>
                  <ul className="space-y-1.5 text-xs text-[#4A4A4A] list-disc list-inside">
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
              <div className="text-[10px] font-mono uppercase tracking-widest text-[#6B6B6B] mb-3 flex items-center gap-2">
                <FileText size={12} />
                Leads to Explore
              </div>
              <p className="text-xs text-[#6B6B6B] mb-3">
                Additional entities or concepts worth investigating for your case.
              </p>
              <div className="space-y-3">
                {leads.map((lead, lidx) => (
                  <div
                    key={lidx}
                    className="p-3 rounded-lg border border-[#D4AF37]/30 bg-[#D4AF37]/5"
                  >
                    <p className="text-sm font-semibold text-[#2B2B2B]">
                      {lead.name}
                    </p>
                    {lead.description && (
                      <p className="text-xs text-[#6B6B6B] mt-1">
                        {lead.description}
                      </p>
                    )}
                    {lead.explanation && (
                      <p className="text-xs text-[#2B2B2B] mt-2">
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
      </main>
    </div>
  );
}
