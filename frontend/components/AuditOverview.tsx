"use client";

import { motion } from "framer-motion";
import { Shield, AlertTriangle, CheckCircle, Activity, Target, Sliders } from "lucide-react";

interface AuditOverviewProps {
  auditRun: {
    total_audited: number;
    total_flagged: number;
    threshold: number;
    auc_roc: number | null;
    mrr: number | null;
  };
}

function integrityColor(pct: number): { text: string; bg: string; bar: string } {
  if (pct > 95)
    return {
      text: "text-[#2D6A4F]",
      bg: "bg-[#2D6A4F]/10 border-[#2D6A4F]/30",
      bar: "#2D6A4F",
    };
  if (pct > 85)
    return {
      text: "text-[#C5A028]",
      bg: "bg-[#C5A028]/10 border-[#C5A028]/30",
      bar: "#C5A028",
    };
  return {
    text: "text-[#7A1A1A]",
    bg: "bg-[#7A1A1A]/10 border-[#7A1A1A]/30",
    bar: "#7A1A1A",
  };
}

const cardVariants = {
  hidden: { opacity: 0, y: 16 },
  visible: (i: number) => ({
    opacity: 1,
    y: 0,
    transition: { delay: i * 0.07, duration: 0.35, ease: "easeOut" as const },
  }),
};

export default function AuditOverview({ auditRun }: AuditOverviewProps) {
  const { total_audited, total_flagged, threshold, auc_roc, mrr } = auditRun;
  const integrityPct =
    total_audited > 0
      ? ((total_audited - total_flagged) / total_audited) * 100
      : 100;
  const ic = integrityColor(integrityPct);

  const flaggedColor =
    total_flagged > 0
      ? { text: "text-[#7A1A1A]", bg: "bg-[#7A1A1A]/10 border-[#7A1A1A]/30" }
      : { text: "text-[#2D6A4F]", bg: "bg-[#2D6A4F]/10 border-[#2D6A4F]/30" };

  const cards = [
    {
      label: "Audited",
      icon: <Activity size={16} className="text-[#C5A028]" />,
      value: total_audited.toLocaleString(),
      valueClass: "text-[#1A1A1A]",
      bg: "bg-[#FFFFFF] border-[#E5E5E3]/30",
      sub: "relationships examined",
    },
    {
      label: "Flagged",
      icon: <AlertTriangle size={16} className={flaggedColor.text} />,
      value: total_flagged.toLocaleString(),
      valueClass: flaggedColor.text,
      bg: `bg-[#FFFFFF] border ${flaggedColor.bg}`,
      sub: total_flagged === 0 ? "none detected" : "below threshold",
    },
    {
      label: "Integrity",
      icon: <Shield size={16} className={ic.text} />,
      value: `${integrityPct.toFixed(1)}%`,
      valueClass: ic.text,
      bg: `bg-[#FFFFFF] border ${ic.bg}`,
      sub: "validated relationships",
      bar: { pct: integrityPct, color: ic.bar },
    },
    {
      label: "AUC-ROC",
      icon: <Target size={16} className="text-[#737373]" />,
      value: auc_roc !== null ? Number(auc_roc).toFixed(3) : "N/A",
      valueClass: "text-[#1A1A1A]",
      bg: "bg-[#FFFFFF] border-[#E5E5E3]/30",
      sub: "model discrimination",
    },
    {
      label: "MRR",
      icon: <CheckCircle size={16} className="text-[#737373]" />,
      value: mrr !== null ? Number(mrr).toFixed(3) : "N/A",
      valueClass: "text-[#1A1A1A]",
      bg: "bg-[#FFFFFF] border-[#E5E5E3]/30",
      sub: "mean reciprocal rank",
    },
    {
      label: "Threshold (τ)",
      icon: <Sliders size={16} className="text-[#C5A028]" />,
      value: Number(threshold).toFixed(3),
      valueClass: "text-[#C5A028]",
      bg: "bg-[#FFFFFF] border-[#C5A028]/30",
      sub: "plausibility cutoff",
    },
  ];

  return (
    <motion.div
      initial="hidden"
      animate="visible"
      className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-3"
    >
      {cards.map((card, i) => (
        <motion.div
          key={card.label}
          custom={i}
          variants={cardVariants}
          className={`relative bg-[#FFFFFF] border rounded-lg p-4 ${card.bg} flex flex-col gap-1.5`}
        >
          {/* Top row: label + icon */}
          <div className="flex items-center justify-between">
            <span className="text-[9px] font-mono uppercase tracking-[0.15em] text-[#737373]">
              {card.label}
            </span>
            <div className="p-1 bg-[#FAFAF8] rounded border border-[#E5E5E3]/20">
              {card.icon}
            </div>
          </div>

          {/* Value */}
          <p
            className={`text-2xl font-bold font-mono leading-none ${card.valueClass}`}
            style={{ fontFamily: "JetBrains Mono, monospace" }}
          >
            {card.value}
          </p>

          {/* Optional integrity bar */}
          {card.bar && (
            <div className="h-1.5 w-full bg-[#F5F5F3] rounded-full overflow-hidden border border-[#E5E5E3]/10 mt-0.5">
              <motion.div
                initial={{ width: 0 }}
                animate={{ width: `${card.bar.pct}%` }}
                transition={{ duration: 0.9, ease: "easeOut", delay: i * 0.07 + 0.2 }}
                className="h-full rounded-full"
                style={{ background: card.bar.color }}
              />
            </div>
          )}

          {/* Subtext */}
          <p className="text-[9px] font-mono uppercase tracking-[0.12em] text-[#737373] leading-tight">
            {card.sub}
          </p>
        </motion.div>
      ))}
    </motion.div>
  );
}
