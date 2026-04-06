"use client";

import { motion } from "framer-motion";

interface StatCardProps {
  label: string;
  value: React.ReactNode;
  icon: React.ReactNode;
  subtext: string;
  delay?: number;
}

export default function StatCard({
  label,
  value,
  icon,
  subtext,
  delay = 0,
}: StatCardProps) {
  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.95 }}
      animate={{ opacity: 1, scale: 1 }}
      transition={{ delay }}
      className="glass rounded-sm p-6 border border-[#4A4A4A] hover:border-[#D4AF37] transition-all cursor-default group cut-paper"
    >
      <div className="flex justify-between items-start mb-4">
        <div className="p-3 bg-[#F5F2E9] rounded-sm group-hover:scale-110 transition-transform border border-[#4A4A4A]">
          {icon}
        </div>
      </div>
      <p className="text-xs font-medium text-[#6B6B6B] mb-1 uppercase tracking-wider">
        {label}
      </p>
      <h4 className="text-3xl font-bold text-[#2B2B2B] mb-2 tracking-tight font-mono data-value">
        {value}
      </h4>
      <p className="text-[10px] text-[#8B1A1A] font-mono uppercase tracking-widest opacity-70">
        {subtext}
      </p>
    </motion.div>
  );
}
