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
      text: "text-[#3A5A40]",
      bg: "bg-[#3A5A40]/10 border-[#3A5A40]/30",
      bar: "#3A5A40",
    };
  if (pct > 85)
    return {
      text: "text-[#D4AF37]",
      bg: "bg-[#D4AF37]/10 border-[#D4AF37]/30",
      bar: "#D4AF37",
    };
  return {
    text: "text-[#8B1A1A]",
    bg: "bg-[#8B1A1A]/10 border-[#8B1A1A]/30",
    bar: "#8B1A1A",
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
      ? { text: "text-[#8B1A1A]", bg: "bg-[#8B1A1A]/10 border-[#8B1A1A]/30" }
      : { text: "text-[#3A5A40]", bg: "bg-[#3A5A40]/10 border-[#3A5A40]/30" };

  const cards = [
    {
      label: "Audited",
      icon: <Activity size={16} className="text-[#D4AF37]" />,
      value: total_audited.toLocaleString(),
      valueClass: "text-[#2B2B2B]",
      bg: "bg-[#FCFAF2] border-[#4A4A4A]/30",
      sub: "relationships examined",
    },
    {
      label: "Flagged",
      icon: <AlertTriangle size={16} className={flaggedColor.text} />,
      value: total_flagged.toLocaleString(),
      valueClass: flaggedColor.text,
      bg: `bg-[#FCFAF2] border ${flaggedColor.bg}`,
      sub: total_flagged === 0 ? "none detected" : "below threshold",
    },
    {
      label: "Integrity",
      icon: <Shield size={16} className={ic.text} />,
      value: `${integrityPct.toFixed(1)}%`,
      valueClass: ic.text,
      bg: `bg-[#FCFAF2] border ${ic.bg}`,
      sub: "validated relationships",
      bar: { pct: integrityPct, color: ic.bar },
    },
    {
      label: "AUC-ROC",
      icon: <Target size={16} className="text-[#6B6B6B]" />,
      value: auc_roc !== null ? Number(auc_roc).toFixed(3) : "N/A",
      valueClass: "text-[#2B2B2B]",
      bg: "bg-[#FCFAF2] border-[#4A4A4A]/30",
      sub: "model discrimination",
    },
    {
      label: "MRR",
      icon: <CheckCircle size={16} className="text-[#6B6B6B]" />,
      value: mrr !== null ? Number(mrr).toFixed(3) : "N/A",
      valueClass: "text-[#2B2B2B]",
      bg: "bg-[#FCFAF2] border-[#4A4A4A]/30",
      sub: "mean reciprocal rank",
    },
    {
      label: "Threshold (τ)",
      icon: <Sliders size={16} className="text-[#D4AF37]" />,
      value: Number(threshold).toFixed(3),
      valueClass: "text-[#D4AF37]",
      bg: "bg-[#FCFAF2] border-[#D4AF37]/30",
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
          className={`relative bg-[#FCFAF2] border rounded-lg p-4 ${card.bg} flex flex-col gap-1.5`}
        >
          {/* Top row: label + icon */}
          <div className="flex items-center justify-between">
            <span className="text-[9px] font-mono uppercase tracking-[0.15em] text-[#6B6B6B]">
              {card.label}
            </span>
            <div className="p-1 bg-[#F5F2E9] rounded border border-[#4A4A4A]/20">
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
            <div className="h-1.5 w-full bg-[#E8E4D9] rounded-full overflow-hidden border border-[#4A4A4A]/10 mt-0.5">
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
          <p className="text-[9px] font-mono uppercase tracking-[0.12em] text-[#6B6B6B] leading-tight">
            {card.sub}
          </p>
        </motion.div>
      ))}
    </motion.div>
  );
}
