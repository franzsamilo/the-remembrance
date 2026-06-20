"use client";

import React from "react";
import { motion } from "framer-motion";
import {
  BookOpen,
  Search,
  ShieldCheck,
  MessageSquare,
  ArrowRight,
} from "lucide-react";
import { DEMO_QUERIES } from "@/lib/constants";

interface WelcomeCardProps {
  /** Fired when a visitor clicks a sample question. Parent should pre-fill the
   *  chat input AND navigate to the Discover tab so the visitor lands on the
   *  question ready to send. */
  onTryQuery: (query: string) => void;
}

/**
 * First-impression card at the top of the Overview tab. Replaces "land on
 * KPI numbers with no context" with "land on plain-English orientation,
 * three-step how-it-works, and one-click sample questions". Aimed at the
 * outside visitor — panel member, classmate, faculty walk-through — who
 * doesn't already know what Validate-then-Generate is.
 */
export default function WelcomeCard({ onTryQuery }: WelcomeCardProps) {
  return (
    <motion.section
      initial={{ opacity: 0, y: -8 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5 }}
      className="relative rounded-sm border border-[#E5E5E3] bg-gradient-to-br from-white via-white to-[#FAFAF8] overflow-hidden"
      aria-label="Welcome to The Remembrance"
    >
      <div className="absolute top-0 left-0 right-0 h-[3px] bg-gradient-to-r from-transparent via-[#C5A028] to-transparent" />

      <div className="px-6 py-6">
        <p className="text-[10px] font-mono uppercase tracking-[0.2em] text-[#C5A028] mb-2">
          New here? Start with this.
        </p>
        <h2
          className="text-2xl font-semibold mb-2 leading-tight"
          style={{ fontFamily: "EB Garamond, serif" }}
        >
          An AI assistant that refuses to make things up.
        </h2>
        <p className="text-sm text-[#525252] leading-relaxed max-w-3xl">
          The Remembrance reads professional documents — legal, medical,
          technical — and answers questions about them. When the documents
          don&apos;t contain an answer, the system tells you so instead of
          inventing one. That refusal is the point.
        </p>

        <div className="grid sm:grid-cols-3 gap-4 mt-6">
          <Step
            icon={<BookOpen size={14} />}
            title="1. Upload PDFs"
            body="The system reads them and builds a structured map of the entities and relationships inside."
          />
          <Step
            icon={<Search size={14} />}
            title="2. Ask a question"
            body="It searches the map for facts that match — even when the answer spans multiple documents."
          />
          <Step
            icon={<ShieldCheck size={14} />}
            title="3. Get a grounded answer"
            body="Every claim traces back to a source. If no evidence is found, you see a refusal — never a guess."
          />
        </div>

        <div className="mt-7 pt-5 border-t border-[#E5E5E3]">
          <p className="text-[11px] font-mono uppercase tracking-[0.15em] text-[#737373] mb-3 flex items-center gap-2">
            <MessageSquare size={12} className="text-[#C5A028]" />
            Try a sample question — opens the chat with it pre-filled
          </p>
          <div className="grid sm:grid-cols-3 gap-3">
            {DEMO_QUERIES.map((q) => (
              <button
                key={q.label}
                type="button"
                onClick={() => onTryQuery(q.query)}
                className="group relative text-left p-3.5 rounded-sm border border-[#E5E5E3] bg-white hover:border-[#C5A028]/60 hover:shadow-md hover:-translate-y-0.5 transition-all"
              >
                <div className="flex items-start justify-between gap-2 mb-1.5">
                  <p className="text-xs font-semibold text-[#1A1A1A] leading-snug">
                    {q.label}
                  </p>
                  <ArrowRight
                    size={12}
                    className="shrink-0 text-[#C5A028] opacity-0 group-hover:opacity-100 transition-opacity mt-0.5"
                  />
                </div>
                <p className="text-[11px] text-[#525252] leading-snug mb-2">
                  &ldquo;{q.query}&rdquo;
                </p>
                <p className="text-[9px] font-mono uppercase tracking-[0.15em] text-[#737373]">
                  {q.showcases}
                </p>
              </button>
            ))}
          </div>
        </div>
      </div>
    </motion.section>
  );
}

function Step({
  icon,
  title,
  body,
}: {
  icon: React.ReactNode;
  title: string;
  body: string;
}) {
  return (
    <div className="flex items-start gap-3">
      <span className="shrink-0 mt-0.5 p-1.5 rounded-sm bg-[#C5A028]/10 text-[#C5A028] border border-[#C5A028]/20">
        {icon}
      </span>
      <div>
        <p className="text-xs font-semibold text-[#1A1A1A]">{title}</p>
        <p className="text-[11px] text-[#737373] mt-0.5 leading-snug">{body}</p>
      </div>
    </div>
  );
}
