"use client";

import React, { useState } from "react";
import { ChevronDown, ChevronUp, Workflow } from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";
import { PIPELINE_STAGES } from "@/lib/pipelineConfig";
import PipelineStrip from "@/components/PipelineStrip";
import PipelineDetail from "@/components/PipelineDetail";

interface PipelineFlowProps {
  stats: any;
  currentTask: string;
}

export default function PipelineFlow({ stats, currentTask }: PipelineFlowProps) {
  const [selectedId, setSelectedId] = useState(PIPELINE_STAGES[0].id);
  const [expanded, setExpanded] = useState(true);

  const graphState: string = stats?.graph_state ?? "empty_graph";
  const selectedStage = PIPELINE_STAGES.find((s) => s.id === selectedId) ?? PIPELINE_STAGES[0];
  const selectedStatus = selectedStage.getStatus(currentTask, graphState);

  return (
    <div className="border border-[#4A4A4A]/30 rounded-lg bg-[#FCFAF2] overflow-hidden">
      {/* Collapsible header */}
      <button
        type="button"
        onClick={() => setExpanded(!expanded)}
        className="w-full flex items-center justify-between gap-3 px-5 py-3.5 text-left hover:bg-[#E8E4D9]/40 transition-colors"
        aria-expanded={expanded}
        aria-controls="pipeline-flow-content"
      >
        <div className="flex items-center gap-2">
          <Workflow size={18} className="text-[#D4AF37]" />
          <span className="text-sm font-semibold text-[#2B2B2B]">
            System Pipeline
          </span>
          <span className="text-[9px] font-mono uppercase tracking-[0.15em] text-[#6B6B6B]">
            Feature · Training · Inference
          </span>
        </div>
        {expanded ? (
          <ChevronUp size={18} className="text-[#6B6B6B]" />
        ) : (
          <ChevronDown size={18} className="text-[#6B6B6B]" />
        )}
      </button>

      <AnimatePresence>
        {expanded && (
          <motion.div
            id="pipeline-flow-content"
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: "auto", opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.25 }}
            className="overflow-hidden"
          >
            <div className="px-5 pb-5 space-y-4">
              <PipelineStrip
                stages={PIPELINE_STAGES}
                selectedId={selectedId}
                currentTask={currentTask}
                graphState={graphState}
                onSelect={setSelectedId}
              />
              <PipelineDetail
                stage={selectedStage}
                status={selectedStatus}
                stats={stats}
              />
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
