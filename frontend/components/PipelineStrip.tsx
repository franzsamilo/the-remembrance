"use client";

import React, { useRef } from "react";
import { motion } from "framer-motion";
import {
  PipelineStage,
  Phase,
  PHASE_COLORS,
  StageStatus,
} from "@/lib/pipelineConfig";

/* ------------------------------------------------------------------ */
/* Types                                                                 */
/* ------------------------------------------------------------------ */

interface PipelineStripProps {
  stages: PipelineStage[];
  selectedId: string;
  currentTask: string;
  graphState: string;
  onSelect: (id: string) => void;
}

/* ------------------------------------------------------------------ */
/* Constants                                                            */
/* ------------------------------------------------------------------ */

const PHASE_ORDER: Phase[] = ["feature", "training", "inference"];

const PHASE_HEX: Record<Phase, string> = {
  feature: "#2D6A4F",
  training: "#C5A028",
  inference: "#7A1A1A",
};

const PHASE_DISPLAY: Record<Phase, string> = {
  feature: "Feature",
  training: "Training",
  inference: "Inference",
};

/* ------------------------------------------------------------------ */
/* Status dot                                                           */
/* ------------------------------------------------------------------ */

function StatusDot({ status }: { status: StageStatus }) {
  if (status === "active") {
    return (
      <span className="relative flex h-2.5 w-2.5">
        <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-[#C5A028] opacity-75" />
        <span className="relative inline-flex rounded-full h-2.5 w-2.5 bg-[#C5A028]" />
      </span>
    );
  }
  const colorMap: Record<StageStatus, string> = {
    ready: "bg-[#2D6A4F]",
    active: "bg-[#C5A028]",
    waiting: "bg-[#525252]/40",
    error: "bg-red-600",
  };
  return (
    <span
      className={`inline-flex h-2.5 w-2.5 rounded-full ${colorMap[status]}`}
    />
  );
}

/* ------------------------------------------------------------------ */
/* Arrow connector (desktop)                                            */
/* ------------------------------------------------------------------ */

interface ArrowProps {
  active: boolean;
  /** hex color for the arrow */
  color: string;
}

function ArrowConnector({ active, color }: ArrowProps) {
  return (
    <div className="relative flex items-center" style={{ width: 32, flexShrink: 0 }}>
      {/* Dashed line */}
      <svg
        width="32"
        height="12"
        viewBox="0 0 32 12"
        fill="none"
        className="absolute inset-0"
        aria-hidden="true"
      >
        <line
          x1="0"
          y1="6"
          x2="26"
          y2="6"
          stroke={color}
          strokeWidth="1.5"
          strokeDasharray="4 3"
          strokeOpacity={active ? 1 : 0.45}
          style={active ? { animation: "dash-flow 0.8s linear infinite" } : undefined}
        />
        {/* Arrowhead */}
        <polygon
          points="26,3 32,6 26,9"
          fill={color}
          fillOpacity={active ? 1 : 0.45}
        />
      </svg>

      {/* Active: 3 flowing dots */}
      {active && (
        <div className="absolute inset-0 overflow-hidden" style={{ height: 12 }}>
          {[0, 0.35, 0.7].map((delay, i) => (
            <span
              key={i}
              className="absolute top-[4px] h-1 w-1 rounded-full"
              style={{
                background: color,
                animation: `flowDot 0.9s ${delay}s linear infinite`,
              }}
            />
          ))}
        </div>
      )}
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* Single stage node                                                    */
/* ------------------------------------------------------------------ */

interface StageNodeProps {
  stage: PipelineStage;
  status: StageStatus;
  selected: boolean;
  onSelect: () => void;
}

function StageNode({ stage, status, selected, onSelect }: StageNodeProps) {
  const phase = stage.phase;
  const phaseColors = PHASE_COLORS[phase];
  const isActive = status === "active";

  return (
    <motion.button
      type="button"
      onClick={onSelect}
      animate={
        isActive
          ? { scale: [1, 1.03, 1] }
          : selected
          ? { scale: 1.02 }
          : { scale: 1 }
      }
      transition={
        isActive
          ? { duration: 2, repeat: Infinity, ease: "easeInOut" }
          : { duration: 0.15 }
      }
      className={[
        "relative flex flex-col items-start gap-0.5 px-3 py-2 rounded-sm",
        "border bg-[#FFFFFF] transition-colors focus:outline-none focus-visible:ring-2 focus-visible:ring-[#C5A028]",
        selected
          ? `border-2 ${phaseColors.border} shadow-sm`
          : "border-[#E5E5E3]/30 hover:border-[#E5E5E3]/60",
        isActive
          ? "shadow-[0_0_10px_rgba(197,160,40,0.45)]"
          : "",
      ]
        .filter(Boolean)
        .join(" ")}
      aria-pressed={selected}
      aria-label={`${stage.name} — ${status}`}
      style={{ minWidth: 96 }}
    >
      {/* Shimmer progress bar for active stage */}
      {isActive && (
        <span
          className="absolute top-0 left-0 right-0 h-0.5 overflow-hidden rounded-t-sm"
          aria-hidden="true"
        >
          <span
            className="absolute inset-0 w-[200%]"
            style={{
              background: `linear-gradient(90deg, transparent 0%, ${PHASE_HEX[phase]} 50%, transparent 100%)`,
              animation: "shimmerBar 1.2s linear infinite",
            }}
          />
        </span>
      )}

      {/* Header row: status dot + order */}
      <span className="flex items-center gap-1.5 w-full">
        <StatusDot status={status} />
        <span className="text-[10px] font-mono text-[#737373] ml-auto">
          {stage.order}
        </span>
      </span>

      {/* Stage name */}
      <span className="text-xs font-semibold text-[#1A1A1A] leading-tight text-left">
        {stage.name}
      </span>

      {/* Model name */}
      <span className="text-[10px] text-[#737373] leading-tight text-left">
        {stage.model.name}
      </span>
    </motion.button>
  );
}

/* ------------------------------------------------------------------ */
/* Phase section (desktop)                                              */
/* ------------------------------------------------------------------ */

interface PhaseSectionProps {
  phase: Phase;
  stages: PipelineStage[];
  selectedId: string;
  currentTask: string;
  graphState: string;
  onSelect: (id: string) => void;
  isLast: boolean;
}

function PhaseSection({
  phase,
  stages,
  selectedId,
  currentTask,
  graphState,
  onSelect,
  isLast,
}: PhaseSectionProps) {
  const colors = PHASE_COLORS[phase];
  const hex = PHASE_HEX[phase];

  if (stages.length === 0) return null;

  return (
    <div className="flex items-stretch gap-0">
      {/* Phase group */}
      <div className="flex flex-col gap-1">
        {/* Phase label */}
        <span
          className={`text-[10px] font-semibold uppercase tracking-widest ${colors.text} text-center`}
        >
          {PHASE_DISPLAY[phase]}
        </span>

        {/* Bordered group wrapper */}
        <div
          className={`flex items-center gap-1.5 px-2 py-1.5 rounded-sm border ${colors.border} border-opacity-40 bg-[#FAFAF8]/60`}
        >
          {stages.map((stage, idx) => {
            const status = stage.getStatus(currentTask, graphState);
            const isActiveStage = status === "active";
            const isLastInGroup = idx === stages.length - 1;

            return (
              <React.Fragment key={stage.id}>
                <StageNode
                  stage={stage}
                  status={status}
                  selected={selectedId === stage.id}
                  onSelect={() => onSelect(stage.id)}
                />
                {!isLastInGroup && (
                  <ArrowConnector active={isActiveStage} color={hex} />
                )}
              </React.Fragment>
            );
          })}
        </div>
      </div>

      {/* Inter-phase arrow */}
      {!isLast && (
        <div className="flex flex-col justify-end pb-[9px] px-1">
          <ArrowConnector active={false} color="#C5A028" />
        </div>
      )}
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* Mobile phase block                                                   */
/* ------------------------------------------------------------------ */

interface MobilePhaseProps {
  phase: Phase;
  stages: PipelineStage[];
  selectedId: string;
  currentTask: string;
  graphState: string;
  onSelect: (id: string) => void;
}

function MobilePhaseBlock({
  phase,
  stages,
  selectedId,
  currentTask,
  graphState,
  onSelect,
}: MobilePhaseProps) {
  const colors = PHASE_COLORS[phase];

  if (stages.length === 0) return null;

  return (
    <div className="flex flex-col gap-1.5">
      <span
        className={`text-[10px] font-semibold uppercase tracking-widest ${colors.text}`}
      >
        {PHASE_DISPLAY[phase]}
      </span>
      <div
        className={`flex flex-wrap gap-2 p-2 rounded-sm border ${colors.border} border-opacity-40 bg-[#FAFAF8]/60`}
      >
        {stages.map((stage) => {
          const status = stage.getStatus(currentTask, graphState);
          return (
            <StageNode
              key={stage.id}
              stage={stage}
              status={status}
              selected={selectedId === stage.id}
              onSelect={() => onSelect(stage.id)}
            />
          );
        })}
      </div>
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* PipelineStrip — main export                                          */
/* ------------------------------------------------------------------ */

export default function PipelineStrip({
  stages,
  selectedId,
  currentTask,
  graphState,
  onSelect,
}: PipelineStripProps) {
  const groupedRef = useRef<Record<Phase, PipelineStage[]>>({
    feature: [],
    training: [],
    inference: [],
  });

  // Build phase groups preserving order
  const grouped: Record<Phase, PipelineStage[]> = {
    feature: [],
    training: [],
    inference: [],
  };
  for (const s of stages) {
    grouped[s.phase].push(s);
  }
  groupedRef.current = grouped;

  const phasesWithStages = PHASE_ORDER.filter((p) => grouped[p].length > 0);

  return (
    <div className="w-full" role="region" aria-label="Pipeline stages">
      {/* ---- Desktop: single horizontal row ---- */}
      <div className="hidden md:flex items-end gap-0 overflow-x-auto pb-1 scrollbar-thin">
        {PHASE_ORDER.map((phase, phaseIdx) => {
          const isLast = phaseIdx === PHASE_ORDER.length - 1;
          return (
            <PhaseSection
              key={phase}
              phase={phase}
              stages={grouped[phase]}
              selectedId={selectedId}
              currentTask={currentTask}
              graphState={graphState}
              onSelect={onSelect}
              isLast={isLast}
            />
          );
        })}
      </div>

      {/* ---- Mobile: stacked phases ---- */}
      <div className="flex flex-col gap-3 md:hidden">
        {phasesWithStages.map((phase) => (
          <MobilePhaseBlock
            key={phase}
            phase={phase}
            stages={grouped[phase]}
            selectedId={selectedId}
            currentTask={currentTask}
            graphState={graphState}
            onSelect={onSelect}
          />
        ))}
      </div>
    </div>
  );
}
