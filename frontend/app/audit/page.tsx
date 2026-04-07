"use client";

import React, { useState, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Shield, ArrowLeft, AlertTriangle, CheckCircle, RefreshCw } from "lucide-react";
import Link from "next/link";
import axios from "axios";
import { API_BASE_URL } from "@/lib/api";
import AuditOverview from "@/components/AuditOverview";
import DocumentIntegrity from "@/components/DocumentIntegrity";
import FlaggedEdges, { type FlaggedEdge } from "@/components/FlaggedEdges";

interface AuditFindings {
  audit_run: {
    run_id: string;
    completed_at: string;
    auc_roc: number | null;
    mrr: number | null;
    total_audited: number;
    total_flagged: number;
    threshold: number;
  } | null;
  document_summary: {
    document: string;
    total_edges: number;
    flagged: number;
    integrity: number;
  }[];
  flagged_edges: FlaggedEdge[];
}

function SectionHeader({
  icon,
  title,
  subtitle,
}: {
  icon: React.ReactNode;
  title: string;
  subtitle?: string;
}) {
  return (
    <div className="flex items-start gap-3 mb-4">
      <div className="p-2 bg-[#F5F2E9] border border-[#4A4A4A]/30 rounded-md shrink-0">
        {icon}
      </div>
      <div>
        <h2
          className="text-lg font-semibold text-[#2B2B2B] leading-tight"
          style={{ fontFamily: "EB Garamond, serif" }}
        >
          {title}
        </h2>
        {subtitle && (
          <p className="text-[10px] font-mono uppercase tracking-[0.12em] text-[#6B6B6B] mt-0.5">
            {subtitle}
          </p>
        )}
      </div>
    </div>
  );
}

function Divider() {
  return (
    <div className="w-full h-px bg-gradient-to-r from-transparent via-[#D4AF37]/30 to-transparent my-8" />
  );
}

export default function AuditPage() {
  const [findings, setFindings] = useState<AuditFindings | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [selectedDoc, setSelectedDoc] = useState<string | null>(null);

  const fetchFindings = async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await axios.get<AuditFindings>(
        `${API_BASE_URL}/audit/findings`
      );
      setFindings(res.data);
    } catch (err: unknown) {
      if (axios.isAxiosError(err)) {
        setError(
          err.response?.data?.detail ||
            err.message ||
            "Failed to load audit findings."
        );
      } else {
        setError("An unexpected error occurred.");
      }
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchFindings();
  }, []);

  const formatTimestamp = (iso: string) => {
    try {
      return new Date(iso).toLocaleString(undefined, {
        year: "numeric",
        month: "short",
        day: "numeric",
        hour: "2-digit",
        minute: "2-digit",
      });
    } catch {
      return iso;
    }
  };

  return (
    <div className="min-h-screen bg-[#F5F2E9] text-[#2B2B2B]">
      {/* Page container */}
      <div className="max-w-5xl mx-auto px-4 sm:px-6 py-8 space-y-0">

        {/* ── Header ─────────────────────────────────────────────── */}
        <motion.header
          initial={{ opacity: 0, y: -12 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.4, ease: "easeOut" }}
          className="pb-6 border-b border-[#4A4A4A]/20"
        >
          {/* Back link */}
          <Link
            href="/"
            className="inline-flex items-center gap-1.5 text-[10px] font-mono uppercase tracking-[0.15em] text-[#6B6B6B] hover:text-[#D4AF37] transition-colors mb-5 group"
          >
            <ArrowLeft
              size={12}
              className="group-hover:-translate-x-0.5 transition-transform"
            />
            Back to Vault
          </Link>

          <div className="flex flex-col sm:flex-row sm:items-end sm:justify-between gap-3">
            <div className="flex items-center gap-3">
              <div className="p-2.5 bg-[#FCFAF2] border border-[#4A4A4A]/30 rounded-lg shadow-sm">
                <Shield size={22} className="text-[#D4AF37]" />
              </div>
              <div>
                <h1
                  className="text-3xl font-bold text-[#2B2B2B] leading-tight tracking-tight"
                  style={{ fontFamily: "EB Garamond, serif" }}
                >
                  Integrity Audit Report
                </h1>
                <p className="text-[10px] font-mono uppercase tracking-[0.15em] text-[#6B6B6B] mt-0.5">
                  GNN-powered knowledge graph analysis
                </p>
              </div>
            </div>

            <div className="flex items-center gap-3">
              {findings?.audit_run?.completed_at && (
                <p className="text-[10px] font-mono text-[#6B6B6B]">
                  Run completed{" "}
                  <span className="text-[#2B2B2B] font-semibold">
                    {formatTimestamp(findings.audit_run.completed_at)}
                  </span>
                </p>
              )}
              <button
                onClick={fetchFindings}
                disabled={loading}
                className="flex items-center gap-1.5 px-3 py-1.5 rounded-md border border-[#4A4A4A]/30 bg-[#FCFAF2] text-[10px] font-mono uppercase tracking-[0.12em] text-[#6B6B6B] hover:border-[#D4AF37]/60 hover:text-[#2B2B2B] transition-all disabled:opacity-50"
              >
                <RefreshCw size={11} className={loading ? "animate-spin" : ""} />
                Refresh
              </button>
            </div>
          </div>
        </motion.header>

        {/* ── Loading ─────────────────────────────────────────────── */}
        <AnimatePresence mode="wait">
          {loading && (
            <motion.div
              key="loading"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="flex flex-col items-center justify-center py-24 gap-4"
            >
              <div className="p-4 bg-[#FCFAF2] border border-[#4A4A4A]/30 rounded-full">
                <Shield size={28} className="text-[#D4AF37] animate-pulse" />
              </div>
              <p className="text-sm text-[#6B6B6B] font-mono uppercase tracking-[0.15em]">
                Loading audit findings…
              </p>
            </motion.div>
          )}

          {/* ── Error ─────────────────────────────────────────────── */}
          {!loading && error && (
            <motion.div
              key="error"
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0 }}
              className="flex flex-col items-center justify-center py-24 gap-4 text-center"
            >
              <div className="p-4 bg-[#8B1A1A]/10 border border-[#8B1A1A]/30 rounded-full">
                <AlertTriangle size={28} className="text-[#8B1A1A]" />
              </div>
              <div>
                <p className="text-sm font-semibold text-[#8B1A1A]">
                  Failed to load findings
                </p>
                <p className="text-xs text-[#6B6B6B] mt-1 max-w-sm">{error}</p>
              </div>
              <button
                onClick={fetchFindings}
                className="px-4 py-2 bg-[#FCFAF2] border border-[#4A4A4A]/30 rounded-md text-xs font-mono uppercase tracking-[0.12em] text-[#6B6B6B] hover:border-[#D4AF37]/60 hover:text-[#2B2B2B] transition-all"
              >
                Try Again
              </button>
            </motion.div>
          )}

          {/* ── No audit run ─────────────────────────────────────── */}
          {!loading && !error && findings && !findings.audit_run && (
            <motion.div
              key="empty"
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0 }}
              className="flex flex-col items-center justify-center py-24 gap-4 text-center"
            >
              <div className="p-4 bg-[#FCFAF2] border border-[#4A4A4A]/30 rounded-full">
                <Shield size={28} className="text-[#6B6B6B]" />
              </div>
              <div>
                <p
                  className="text-xl font-semibold text-[#2B2B2B]"
                  style={{ fontFamily: "EB Garamond, serif" }}
                >
                  No Audit Has Been Run
                </p>
                <p className="text-sm text-[#6B6B6B] mt-2 max-w-sm leading-relaxed">
                  No audit has been run yet. Run the GNN Audit from the pipeline
                  to detect anomalies.
                </p>
              </div>
              <Link
                href="/"
                className="inline-flex items-center gap-1.5 px-4 py-2 bg-[#D4AF37]/20 border border-[#D4AF37]/40 rounded-md text-xs font-mono uppercase tracking-[0.12em] text-[#8B6914] hover:bg-[#D4AF37]/30 transition-all"
              >
                <ArrowLeft size={11} />
                Go to Vault
              </Link>
            </motion.div>
          )}

          {/* ── Main Content ─────────────────────────────────────── */}
          {!loading && !error && findings && findings.audit_run && (
            <motion.div
              key="content"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ duration: 0.35 }}
              className="space-y-0 pt-8"
            >
              {/* 1. Overview */}
              <section>
                <SectionHeader
                  icon={<Shield size={16} className="text-[#D4AF37]" />}
                  title="Audit Overview"
                  subtitle={`Run ID: ${findings.audit_run.run_id}`}
                />
                <AuditOverview auditRun={findings.audit_run} />
              </section>

              <Divider />

              {/* 2. Document Integrity */}
              {findings.document_summary.length > 0 && (
                <>
                  <section>
                    <SectionHeader
                      icon={<Shield size={16} className="text-[#3A5A40]" />}
                      title="Document Integrity"
                      subtitle={`${findings.document_summary.length} source document${findings.document_summary.length !== 1 ? "s" : ""} analysed`}
                    />
                    <DocumentIntegrity
                      documents={findings.document_summary}
                      selectedDoc={selectedDoc}
                      onSelectDoc={setSelectedDoc}
                    />
                  </section>

                  <Divider />
                </>
              )}

              {/* 3. Flagged Edges */}
              <section>
                <SectionHeader
                  icon={
                    findings.flagged_edges.length > 0 ? (
                      <AlertTriangle size={16} className="text-[#8B1A1A]" />
                    ) : (
                      <CheckCircle size={16} className="text-[#3A5A40]" />
                    )
                  }
                  title="Flagged Relationships"
                  subtitle={
                    findings.flagged_edges.length > 0
                      ? `${findings.flagged_edges.length} suspicious edge${findings.flagged_edges.length !== 1 ? "s" : ""} · sorted by plausibility`
                      : "Integrity validation complete"
                  }
                />

                {/* All-clear state */}
                {findings.flagged_edges.length === 0 ? (
                  <motion.div
                    initial={{ opacity: 0, y: 8 }}
                    animate={{ opacity: 1, y: 0 }}
                    className="flex flex-col items-center justify-center py-12 gap-3 bg-[#3A5A40]/5 border border-[#3A5A40]/20 rounded-lg text-center"
                  >
                    <div className="p-3 bg-[#3A5A40]/10 border border-[#3A5A40]/30 rounded-full">
                      <CheckCircle size={22} className="text-[#3A5A40]" />
                    </div>
                    <p
                      className="text-lg font-semibold text-[#3A5A40]"
                      style={{ fontFamily: "EB Garamond, serif" }}
                    >
                      All {findings.audit_run.total_audited.toLocaleString()}{" "}
                      relationships passed integrity validation.
                    </p>
                    <p className="text-xs text-[#6B6B6B]">
                      No anomalies detected.
                    </p>
                  </motion.div>
                ) : (
                  <FlaggedEdges
                    edges={findings.flagged_edges}
                    filterDoc={selectedDoc}
                    threshold={findings.audit_run.threshold}
                  />
                )}
              </section>
            </motion.div>
          )}
        </AnimatePresence>

        {/* Footer */}
        <div className="pt-12 pb-4 flex items-center justify-center">
          <div className="w-full h-px bg-gradient-to-r from-transparent via-[#D4AF37]/20 to-transparent" />
        </div>
        <p className="text-center text-[9px] font-mono uppercase tracking-[0.2em] text-[#6B6B6B]/50 pb-8">
          The Remembrance Vault · Integrity Audit System
        </p>
      </div>
    </div>
  );
}
