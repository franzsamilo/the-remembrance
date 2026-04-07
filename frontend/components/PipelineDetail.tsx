"use client";

import React, { useRef, useEffect, useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { PipelineStage, PHASE_COLORS, StageStatus } from "@/lib/pipelineConfig";

/* ------------------------------------------------------------------ */
/* Types                                                                 */
/* ------------------------------------------------------------------ */

interface PipelineDetailProps {
  stage: PipelineStage;
  status: StageStatus;
  stats: any;
}

/* ------------------------------------------------------------------ */
/* Section label                                                         */
/* ------------------------------------------------------------------ */

function SectionLabel({
  children,
  color,
}: {
  children: React.ReactNode;
  color: string;
}) {
  return (
    <p
      className="text-[9px] font-mono uppercase tracking-[0.15em] mb-1"
      style={{ color }}
    >
      {children}
    </p>
  );
}

/* ------------------------------------------------------------------ */
/* Running ellipsis badge                                               */
/* ------------------------------------------------------------------ */

function RunningDots() {
  const [dots, setDots] = useState(0);

  useEffect(() => {
    const id = setInterval(() => setDots((d) => (d + 1) % 4), 500);
    return () => clearInterval(id);
  }, []);

  return (
    <span>
      Running
      <span className="inline-block w-[1.25em] text-left">
        {".".repeat(dots)}
      </span>
    </span>
  );
}

/* ------------------------------------------------------------------ */
/* Metric pill                                                           */
/* ------------------------------------------------------------------ */

interface MetricPillProps {
  label: string;
  value: string | number;
}

function MetricPill({ label, value }: MetricPillProps) {
  const prevRef = useRef<string | number | null>(null);
  const [flash, setFlash] = useState(false);

  useEffect(() => {
    if (prevRef.current !== null && prevRef.current !== value) {
      setFlash(true);
      const t = setTimeout(() => setFlash(false), 600);
      return () => clearTimeout(t);
    }
    prevRef.current = value;
  }, [value]);

  return (
    <span
      className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full border border-[#4A4A4A]/30 bg-[#F5F2E9] text-xs transition-colors duration-300"
      style={
        flash
          ? { borderColor: "#D4AF37", backgroundColor: "rgba(212,175,55,0.15)" }
          : {}
      }
    >
      <span className="text-[#6B6B6B]">{label}</span>
      <span className="font-bold text-[#2B2B2B]">{value}</span>
    </span>
  );
}

/* ------------------------------------------------------------------ */
/* PipelineDetail — main export                                          */
/* ------------------------------------------------------------------ */

export default function PipelineDetail({
  stage,
  status,
  stats,
}: PipelineDetailProps) {
  const colors = PHASE_COLORS[stage.phase];
  const phaseText = colors.text;
  const isActive = status === "active";

  const metrics = stage.getMetrics(stats);

  /* Status badge */
  const statusBadge =
    status === "active" ? (
      <span
        className="text-xs font-mono px-2 py-0.5 rounded-sm border"
        style={{
          color: "#8B6914",
          borderColor: "rgba(212,175,55,0.5)",
          backgroundColor: "rgba(212,175,55,0.12)",
        }}
      >
        <RunningDots />
      </span>
    ) : status === "ready" ? (
      <span className="text-xs font-mono px-2 py-0.5 rounded-sm border border-[#3A5A40]/40 bg-[#3A5A40]/10 text-[#3A5A40]">
        Ready
      </span>
    ) : status === "error" ? (
      <span className="text-xs font-mono px-2 py-0.5 rounded-sm border border-red-500/40 bg-red-500/10 text-red-700">
        Error
      </span>
    ) : (
      <span className="text-xs font-mono px-2 py-0.5 rounded-sm border border-[#4A4A4A]/30 bg-[#E8E4D9] text-[#6B6B6B]">
        Waiting
      </span>
    );

  return (
    <AnimatePresence mode="wait">
      <motion.div
        key={stage.id}
        initial={{ opacity: 0, y: -6 }}
        animate={{ opacity: 1, y: 0 }}
        exit={{ opacity: 0, y: -4 }}
        transition={{ duration: 0.2, ease: "easeOut" }}
        className={[
          "relative overflow-hidden rounded-sm border-2 bg-[#FCFAF2]",
          colors.border,
        ].join(" ")}
      >
        {/* Active progress bar */}
        {isActive && (
          <div
            className="absolute top-0 left-0 right-0 h-0.5 overflow-hidden"
            aria-hidden="true"
          >
            <div
              className="absolute inset-0 w-[200%]"
              style={{
                background: `repeating-linear-gradient(
                  90deg,
                  transparent 0px,
                  transparent 8px,
                  ${phaseText} 8px,
                  ${phaseText} 16px
                )`,
                animation: "shimmerBar 1.2s linear infinite",
              }}
            />
          </div>
        )}

        <div className="p-4 sm:p-5 space-y-4">
          {/* Header row */}
          <div className="flex items-start justify-between gap-3 flex-wrap">
            <div>
              <p
                className="text-[10px] font-mono uppercase tracking-[0.15em] mb-0.5"
                style={{ color: phaseText }}
              >
                Stage {stage.order} &middot; {stage.phase}
              </p>
              <h3 className="text-lg font-semibold text-[#2B2B2B]">
                {stage.name}
              </h3>
            </div>
            <div className="mt-0.5">{statusBadge}</div>
          </div>

          {/* 2×2 grid */}
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
            {/* 1. Model */}
            <div>
              <SectionLabel color={phaseText}>Model</SectionLabel>
              <p className="text-sm font-semibold text-[#2B2B2B]">
                {stage.model.name}
              </p>
              <p className="text-xs text-[#6B6B6B] mt-0.5">
                {stage.model.description}
              </p>
            </div>

            {/* 2. Input / Output */}
            <div>
              <SectionLabel color={phaseText}>Input / Output</SectionLabel>
              <p className="text-sm text-[#2B2B2B]">
                <span className="text-[#6B6B6B]">In:</span> {stage.input}
              </p>
              <p className="text-sm text-[#2B2B2B] mt-0.5">
                <span className="text-[#6B6B6B]">Out:</span> {stage.output}
              </p>
            </div>

            {/* 3. Parameters */}
            <div>
              <SectionLabel color={phaseText}>Parameters</SectionLabel>
              <dl className="space-y-0.5">
                {Object.entries(stage.params).map(([k, v]) => (
                  <div key={k} className="flex items-baseline gap-1.5">
                    <dt className="text-xs text-[#6B6B6B] shrink-0">{k}:</dt>
                    <dd className="text-xs text-[#2B2B2B] font-mono truncate">
                      {v}
                    </dd>
                  </div>
                ))}
              </dl>
            </div>

            {/* 4. Why This Approach */}
            <div>
              <SectionLabel color={phaseText}>Why This Approach</SectionLabel>
              <p className="text-sm leading-relaxed text-[#2B2B2B]">
                {stage.why}
              </p>
            </div>
          </div>

          {/* Bottom row: Live Metrics */}
          {metrics.length > 0 && (
            <div className="border-t border-[#4A4A4A]/20 pt-3">
              <SectionLabel color={phaseText}>Live Metrics</SectionLabel>
              <div className="flex flex-wrap gap-2">
                {metrics.map((m) => (
                  <MetricPill key={m.label} label={m.label} value={m.value} />
                ))}
              </div>
            </div>
          )}
        </div>
      </motion.div>
    </AnimatePresence>
  );
}
