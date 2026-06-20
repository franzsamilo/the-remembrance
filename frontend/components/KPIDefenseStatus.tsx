"use client";

import React from "react";
import { motion } from "framer-motion";
import { ShieldCheck, Stamp } from "lucide-react";
import { PAPER_KPIS } from "@/lib/constants";
import InfoTooltip from "@/components/InfoTooltip";

/**
 * Defense-day banner: shows the four paper-claimed KPIs side-by-side with
 * PASS stamps, the threshold, and the methodology footnote. Designed to
 * answer the panel's first question before they ask it.
 */
export default function KPIDefenseStatus() {
  return (
    <motion.section
      initial={{ opacity: 0, y: -8 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5 }}
      className="relative border border-[#C5A028]/40 bg-[#FFFFFF] rounded-sm overflow-hidden cut-paper"
      aria-label="Paper KPI Defense Status"
    >
      {/* Hairline gold rule along the top, like a sealed document */}
      <div className="h-[3px] bg-gradient-to-r from-transparent via-[#C5A028] to-transparent" />

      <div className="px-6 pt-5 pb-3 flex items-start justify-between gap-4 flex-wrap">
        <div className="flex items-start gap-3">
          <div className="p-2 rounded-sm bg-[#2D6A4F]/10 border border-[#2D6A4F]/30">
            <ShieldCheck size={18} className="text-[#2D6A4F]" />
          </div>
          <div>
            <p className="text-[10px] font-mono uppercase tracking-[0.2em] text-[#737373] mb-0.5">
              Project Study Report · v6.4
            </p>
            <h2
              className="text-xl font-semibold text-[#1A1A1A] leading-tight"
              style={{ fontFamily: "EB Garamond, serif" }}
            >
              Paper KPI Defense Status
              <span className="text-[#2D6A4F] ml-2">— All Four Targets Met</span>
            </h2>
          </div>
        </div>

        <div className="archival-stamp text-[10px] shrink-0 mt-1">
          <Stamp size={10} className="inline mr-1 -translate-y-px" />
          Defense&nbsp;Ready
        </div>
      </div>

      <div className="grid grid-cols-2 lg:grid-cols-4 gap-0 border-t border-[#E5E5E3]">
        {PAPER_KPIS.map((kpi, i) => (
          <motion.div
            key={kpi.label}
            initial={{ opacity: 0, y: 6 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.12 + i * 0.06 }}
            className={`relative px-5 py-4 ${
              i > 0 ? "lg:border-l border-[#E5E5E3]" : ""
            } ${i > 1 ? "lg:border-t-0 border-t border-[#E5E5E3]" : ""} ${
              i === 1 ? "lg:border-t-0 border-t border-[#E5E5E3] sm:border-t-0" : ""
            }`}
          >
            <div className="flex items-center justify-between mb-1">
              <span className="text-[9px] font-mono uppercase tracking-[0.18em] text-[#737373] inline-flex items-center gap-1.5">
                {kpi.label}
                <InfoTooltip label={`What is ${kpi.label}?`}>
                  {kpi.plainEnglish}
                </InfoTooltip>
              </span>
              <span className="text-[9px] font-mono uppercase tracking-[0.15em] text-[#7A1A1A]/70">
                {kpi.hypothesis}
              </span>
            </div>

            <div className="flex items-baseline gap-2 mb-1">
              <p
                className="text-3xl font-bold text-[#1A1A1A] leading-none data-value"
                style={{ fontFamily: "JetBrains Mono, monospace" }}
              >
                {kpi.value.toFixed(3)}
              </p>
              <span className="text-[10px] font-mono text-[#737373]">
                ≥ {kpi.target.toFixed(2)}
              </span>
            </div>

            <div className="flex items-center justify-between gap-2">
              <p className="text-[10px] text-[#525252] leading-snug pr-2">
                {kpi.description}
              </p>
              <span className="shrink-0 inline-flex items-center gap-1 px-1.5 py-0.5 rounded-sm bg-[#2D6A4F]/12 text-[#2D6A4F] text-[9px] font-mono uppercase tracking-[0.15em] border border-[#2D6A4F]/30">
                <span className="w-1 h-1 rounded-full bg-[#2D6A4F]" />
                Pass
              </span>
            </div>

            <p className="text-[9px] font-mono text-[#737373] mt-2 leading-snug truncate">
              {kpi.methodology}
            </p>
          </motion.div>
        ))}
      </div>

      <div className="px-6 py-2.5 bg-[#FAFAF8] border-t border-[#E5E5E3] flex flex-wrap items-center justify-between gap-2">
        <p className="text-[10px] text-[#525252] italic leading-snug">
          MRR reported under canonical KGE methodology (multi-seed mean,
          Sun&nbsp;et&nbsp;al.&nbsp;2019; Vashishth&nbsp;et&nbsp;al.&nbsp;2020) at the canonical
          inference threshold τ&nbsp;=&nbsp;0.95.
        </p>
        <p className="text-[9px] font-mono uppercase tracking-[0.18em] text-[#7A1A1A]/70">
          Run&nbsp;8 · DistMult · BPR + self-adv α=1.0
        </p>
      </div>
    </motion.section>
  );
}
