"use client";

import React from "react";
import { motion } from "framer-motion";
import { FileText, Filter, ShieldAlert, ExternalLink } from "lucide-react";

/** Names of the 14 PDFs in the corpus, organized so the panel can see
 *  the source spread and the chronological range at a glance. Kept here
 *  rather than fetched live so the answer is the same whether or not Aura
 *  is reachable mid-defense. */
const CORPUS = [
  {
    bucket: "Foundational source",
    items: ["Philippines Intellectual Property Code (RA 8293, as amended)"],
  },
  {
    bucket: "Judiciary e-library (showdocs)",
    items: [
      "elibrary_judiciary_gov_ph_thebookshelf_showdocs_1_64282.pdf",
      "elibrary_judiciary_gov_ph_thebookshelf_showdocs_1_67489.pdf",
      "elibrary_judiciary_gov_ph_thebookshelf_showdocs_2_4371.pdf",
      "elibrary_judiciary_gov_ph_thebookshelf_showdocs_2_53393.pdf",
    ],
  },
  {
    bucket: "Justice Leonen jurisprudence (recent SC)",
    items: ["gr_211850_leonen.pdf", "gr_228165_leonen.pdf"],
  },
  {
    bucket: "Lawphil.net case law (1990 – 2025)",
    items: [
      "G.R. 78325 (1990)",
      "G.R. 143993 (2004)",
      "G.R. 161693 (2005)",
      "G.R. 183404 (2010)",
      "G.R. 226444 (2021)",
      "G.R. 256091 (2023)",
      "G.R. 184661 (2025)",
    ],
  },
];

const CRITERIA = [
  "At least one foundational source — the Philippine IP Code itself.",
  "Coverage across multiple doctrines: constitutional rights, intellectual property, and procedural standards.",
  "Authorship variety, including Justice Leonen's recent jurisprudence.",
  "Chronological spread from 1990 to 2025 so temporal-citation linking is testable.",
];

/**
 * Defensive-posture panel for the defense. Surfaces the 14-PDF corpus,
 * the principled selection criteria, and the killer "refusal demo is
 * curate-proof" point — so when the panel asks "didn't you choose your
 * own files?", you flip to this tab rather than arguing from memory.
 */
export default function SelectionRationale() {
  return (
    <motion.section
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4 }}
      className="rounded-sm border border-[#E5E5E3] bg-white overflow-hidden"
      aria-label="Corpus selection rationale"
    >
      <div className="h-[3px] bg-gradient-to-r from-transparent via-[#7A1A1A]/60 to-transparent" />

      <header className="px-6 pt-5 pb-3 border-b border-[#E5E5E3]">
        <p className="text-[10px] font-mono uppercase tracking-[0.2em] text-[#7A1A1A] mb-1">
          Defensive posture · Selection rationale
        </p>
        <h3
          className="text-lg font-semibold text-[#1A1A1A] flex items-center gap-2"
          style={{ fontFamily: "EB Garamond, serif" }}
        >
          <FileText size={18} className="text-[#7A1A1A]" />
          Why these 14 documents
        </h3>
        <p className="text-xs text-[#525252] mt-1.5 max-w-3xl leading-relaxed">
          When the panel asks <em>&ldquo;didn&apos;t you choose your own files?&rdquo;</em>,
          this is the answer in one screen. The selection was principled and the
          architecture is corpus-independent of that choice.
        </p>
      </header>

      <div className="grid lg:grid-cols-2 gap-0">
        {/* Left: the 14 PDFs grouped by source */}
        <div className="p-5 border-b lg:border-b-0 lg:border-r border-[#E5E5E3]">
          <p className="text-[10px] font-mono uppercase tracking-[0.18em] text-[#737373] mb-3 flex items-center gap-1.5">
            <ExternalLink size={11} />
            The corpus — 14 documents
          </p>
          <ul className="space-y-3">
            {CORPUS.map((group) => (
              <li key={group.bucket}>
                <p className="text-[11px] font-semibold text-[#1A1A1A] mb-1">
                  {group.bucket}
                </p>
                <ul className="text-[11px] text-[#525252] space-y-0.5 pl-3">
                  {group.items.map((item) => (
                    <li
                      key={item}
                      className="font-mono text-[10px] leading-snug truncate"
                      title={item}
                    >
                      {item}
                    </li>
                  ))}
                </ul>
              </li>
            ))}
          </ul>
        </div>

        {/* Right: criteria + killer points */}
        <div className="p-5 space-y-5">
          <div>
            <p className="text-[10px] font-mono uppercase tracking-[0.18em] text-[#737373] mb-2 flex items-center gap-1.5">
              <Filter size={11} />
              Selection criteria (stated in §3 of the paper)
            </p>
            <ol className="space-y-2 text-xs text-[#525252] list-decimal pl-4 leading-relaxed">
              {CRITERIA.map((c, i) => (
                <li key={i}>{c}</li>
              ))}
            </ol>
          </div>

          <div className="border-t border-[#E5E5E3] pt-4">
            <p className="text-[10px] font-mono uppercase tracking-[0.18em] text-[#7A1A1A] mb-2 flex items-center gap-1.5">
              <ShieldAlert size={11} />
              The argument the panel can&apos;t curate around
            </p>
            <p className="text-xs text-[#525252] leading-relaxed mb-2">
              The refusal demo — &ldquo;What is the chemical composition of titanium
              dioxide?&rdquo; — is deliberately out-of-corpus. No amount of file
              selection can make the system answer a chemistry question from a
              legal corpus. Either the architecture refuses or it doesn&apos;t.
              That is a pure architectural test, not a corpus test.
            </p>
            <p className="text-xs text-[#525252] leading-relaxed">
              Two more pieces of evidence against curation:
            </p>
            <ul className="mt-2 space-y-1.5 text-xs text-[#525252] list-disc pl-4 leading-relaxed">
              <li>
                Single-seed MRR landed at <strong>0.912</strong> — below the 0.95
                target. The paper publishes it. A curator would have hidden it.
              </li>
              <li>
                Run 9 with the RotatE decoder regressed across every GNN metric.
                The paper publishes that too.
              </li>
            </ul>
          </div>

          <div className="border-t border-[#E5E5E3] pt-4">
            <p className="text-[10px] font-mono uppercase tracking-[0.18em] text-[#737373] mb-1.5">
              Honest limit (volunteer this)
            </p>
            <p className="text-xs text-[#525252] leading-relaxed">
              External validity to other professional domains — medical,
              regulatory, technical — is single-corpus, not multi-corpus. The
              architecture is corpus-agnostic; only the ingestion schema is
              domain-specific. Flagged as future work in §6.
            </p>
          </div>
        </div>
      </div>
    </motion.section>
  );
}
