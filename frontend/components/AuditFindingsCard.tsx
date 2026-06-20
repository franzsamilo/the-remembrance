"use client";

import React, { useEffect, useState } from "react";
import Link from "next/link";
import { Shield, AlertTriangle, CheckCircle, ArrowRight } from "lucide-react";
import { motion } from "framer-motion";
import axios from "axios";
import { API_BASE_URL } from "@/lib/api";

interface AuditSummary {
  total_audited: number;
  total_flagged: number;
  integrity: number;
  has_run: boolean;
}

export default function AuditFindingsCard() {
  const [summary, setSummary] = useState<AuditSummary | null>(null);

  useEffect(() => {
    let cancelled = false;
    async function fetch() {
      try {
        const res = await axios.get(`${API_BASE_URL}/audit/findings`);
        if (cancelled) return;
        const data = res.data;
        if (!data.audit_run) {
          setSummary({ total_audited: 0, total_flagged: 0, integrity: 0, has_run: false });
          return;
        }
        const audited = data.audit_run.total_audited || 0;
        const flagged = data.audit_run.total_flagged || 0;
        const integrity = audited > 0 ? 1 - flagged / audited : 1;
        setSummary({ total_audited: audited, total_flagged: flagged, integrity, has_run: true });
      } catch {
        // Endpoint may not exist yet or backend down — silently skip
      }
    }
    fetch();
    return () => { cancelled = true; };
  }, []);

  if (!summary) return null;

  // No audit run yet
  if (!summary.has_run) {
    return (
      <motion.div
        initial={{ opacity: 0, y: 8 }}
        animate={{ opacity: 1, y: 0 }}
        className="border border-[#E5E5E3]/30 rounded-lg bg-[#FFFFFF] p-4"
      >
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Shield size={18} className="text-[#737373]" />
            <span className="text-sm font-semibold text-[#1A1A1A]">Detective Findings</span>
          </div>
          <span className="text-xs text-[#737373]">No audit yet</span>
        </div>
        <p className="text-xs text-[#737373] mt-2">
          Run the GNN Audit from the pipeline to detect anomalies in your knowledge graph.
        </p>
      </motion.div>
    );
  }

  const hasFlagged = summary.total_flagged > 0;
  const integrityPct = Math.round(summary.integrity * 100);
  const integrityColor =
    integrityPct >= 95 ? "text-[#2D6A4F]" : integrityPct >= 85 ? "text-[#A68A1E]" : "text-[#7A1A1A]";
  const integrityLabel =
    integrityPct >= 95 ? "High" : integrityPct >= 85 ? "Moderate" : "Low";
  const borderColor = hasFlagged ? "border-[#7A1A1A]/40" : "border-[#2D6A4F]/40";
  const bgTint = hasFlagged ? "bg-[#7A1A1A]/[0.03]" : "bg-[#2D6A4F]/[0.03]";

  return (
    <motion.div
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
    >
      <Link
        href="/?tab=audit"
        className={`block border ${borderColor} rounded-lg ${bgTint} p-4 hover:shadow-md transition-shadow group`}
      >
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            {hasFlagged ? (
              <AlertTriangle size={18} className="text-[#7A1A1A]" />
            ) : (
              <CheckCircle size={18} className="text-[#2D6A4F]" />
            )}
            <span className="text-sm font-semibold text-[#1A1A1A]">Detective Findings</span>
          </div>
          <div className="flex items-center gap-1 text-xs text-[#737373] group-hover:text-[#C5A028] transition-colors">
            View Full Report
            <ArrowRight size={14} />
          </div>
        </div>

        <div className="flex items-center gap-6 mt-3">
          <div>
            <span className="text-[9px] font-mono uppercase tracking-[0.15em] text-[#737373]">Audited</span>
            <p className="text-lg font-bold text-[#1A1A1A]">{summary.total_audited}</p>
          </div>
          <div>
            <span className="text-[9px] font-mono uppercase tracking-[0.15em] text-[#737373]">Flagged</span>
            <p className={`text-lg font-bold ${hasFlagged ? "text-[#7A1A1A]" : "text-[#2D6A4F]"}`}>
              {summary.total_flagged}
            </p>
          </div>
          <div>
            <span className="text-[9px] font-mono uppercase tracking-[0.15em] text-[#737373]">Integrity</span>
            <p className={`text-lg font-bold ${integrityColor}`}>
              {integrityPct}%
              <span className="text-[9px] font-normal ml-1">({integrityLabel})</span>
            </p>
          </div>
        </div>

        {hasFlagged && (
          <p className="text-xs text-[#7A1A1A]/80 mt-2">
            {summary.total_flagged} relationship{summary.total_flagged !== 1 ? "s" : ""} flagged
            as suspicious — review the audit report for details.
          </p>
        )}
        {!hasFlagged && (
          <p className="text-xs text-[#2D6A4F]/80 mt-2">
            All relationships passed integrity validation. No anomalies detected.
          </p>
        )}
      </Link>
    </motion.div>
  );
}
