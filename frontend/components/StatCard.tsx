"use client";

import { motion } from "framer-motion";
import InfoTooltip from "@/components/InfoTooltip";

interface StatCardProps {
  label: string;
  value: React.ReactNode;
  icon: React.ReactNode;
  subtext: string;
  /** Optional plain-English explanation surfaced via a small info tooltip on
   *  the label — for visitors who don't know what e.g. "AUC-ROC" means. */
  explain?: string;
  delay?: number;
}

export default function StatCard({
  label,
  value,
  icon,
  subtext,
  explain,
  delay = 0,
}: StatCardProps) {
  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.95 }}
      animate={{ opacity: 1, scale: 1 }}
      transition={{ delay }}
      className="glass rounded-sm p-6 border border-[#E5E5E3] hover:border-[#C5A028] transition-all cursor-default group cut-paper"
    >
      <div className="flex justify-between items-start mb-4">
        <div className="p-3 bg-[#FAFAF8] rounded-sm group-hover:scale-110 transition-transform border border-[#E5E5E3]">
          {icon}
        </div>
      </div>
      <p className="text-xs font-medium text-[#737373] mb-1 uppercase tracking-wider inline-flex items-center gap-1.5">
        {label}
        {explain && (
          <InfoTooltip label={label}>
            {explain}
          </InfoTooltip>
        )}
      </p>
      <h4 className="text-3xl font-bold text-[#1A1A1A] mb-2 tracking-tight font-mono data-value">
        {value}
      </h4>
      <p className="text-[10px] text-[#7A1A1A] font-mono uppercase tracking-widest opacity-70">
        {subtext}
      </p>
    </motion.div>
  );
}
