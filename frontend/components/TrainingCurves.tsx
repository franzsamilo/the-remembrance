"use client";

import React, { useEffect, useState } from "react";
import { motion } from "framer-motion";
import { API_BASE_URL } from "@/lib/api";
import { TrainingHistory, EpochMetric } from "@/lib/types";
import axios from "axios";

function SvgLineChart({
  data,
  yKey,
  label,
  color,
  earlyStopEpoch,
}: {
  data: EpochMetric[];
  yKey: "train_loss" | "auc_roc";
  label: string;
  color: string;
  earlyStopEpoch: number | null;
}) {
  if (data.length === 0) return null;

  const W = 400, H = 200;
  const PAD = { top: 16, right: 16, bottom: 28, left: 44 };
  const plotW = W - PAD.left - PAD.right;
  const plotH = H - PAD.top - PAD.bottom;

  const vals = data.map((d) => d[yKey]).filter((v): v is number => v != null);
  if (vals.length === 0) return null;
  const yMin = Math.min(...vals) * 0.95;
  const yMax = Math.max(...vals) * 1.05;
  const xMin = data[0]?.epoch ?? 1;
  const xMax = data[data.length - 1]?.epoch ?? 1;

  const scaleX = (epoch: number) => PAD.left + ((epoch - xMin) / Math.max(xMax - xMin, 1)) * plotW;
  const scaleY = (val: number) => PAD.top + plotH - ((val - yMin) / Math.max(yMax - yMin, 0.001)) * plotH;

  const path = data
    .filter((d) => d[yKey] != null)
    .map((d, i) => `${i === 0 ? "M" : "L"} ${scaleX(d.epoch)} ${scaleY(d[yKey]!)}`)
    .join(" ");

  const yTicks = Array.from({ length: 5 }, (_, i) => yMin + (i / 4) * (yMax - yMin));

  return (
    <div>
      <p className="text-xs font-medium text-[var(--ink-medium)] mb-2">{label}</p>
      <svg viewBox={`0 0 ${W} ${H}`} className="w-full h-auto">
        {yTicks.map((val, i) => (
          <g key={i}>
            <line x1={PAD.left} y1={scaleY(val)} x2={W - PAD.right} y2={scaleY(val)} stroke="var(--border)" strokeWidth="0.5" />
            <text x={PAD.left - 4} y={scaleY(val) + 3} textAnchor="end" className="fill-[var(--ink-light)]" fontSize="9" fontFamily="JetBrains Mono">
              {val < 1 ? val.toFixed(2) : val.toFixed(1)}
            </text>
          </g>
        ))}

        {earlyStopEpoch && (
          <>
            <line
              x1={scaleX(earlyStopEpoch)} y1={PAD.top} x2={scaleX(earlyStopEpoch)} y2={H - PAD.bottom}
              stroke="var(--conflict-red)" strokeWidth="1" strokeDasharray="4 3" opacity="0.6"
            />
            <text x={scaleX(earlyStopEpoch)} y={PAD.top - 4} textAnchor="middle" fontSize="8" className="fill-[var(--conflict-red)]" fontFamily="JetBrains Mono">
              Early stop
            </text>
          </>
        )}

        <path d={path} fill="none" stroke={color} strokeWidth="2" />

        <text x={W / 2} y={H - 4} textAnchor="middle" fontSize="9" className="fill-[var(--ink-light)]" fontFamily="JetBrains Mono">
          Epoch
        </text>
      </svg>
    </div>
  );
}

export default function TrainingCurves() {
  const [history, setHistory] = useState<TrainingHistory | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    axios
      .get(`${API_BASE_URL}/audit/training-history`)
      .then((res) => setHistory(res.data))
      .catch(() => {})
      .finally(() => setLoading(false));
  }, []);

  if (loading) {
    return <div className="h-48 bg-[var(--muted)] rounded-lg animate-pulse" />;
  }

  if (!history || history.epochs.length === 0) {
    return (
      <div className="text-center py-8 text-[var(--ink-light)] text-sm">
        No training history available. Run a GNN audit to see training curves.
      </div>
    );
  }

  return (
    <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="space-y-4">
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <SvgLineChart
          data={history.epochs}
          yKey="train_loss"
          label="Training Loss"
          color="var(--gilded-gold)"
          earlyStopEpoch={history.early_stop_epoch}
        />
        <SvgLineChart
          data={history.epochs}
          yKey="auc_roc"
          label="AUC-ROC"
          color="var(--validated-green)"
          earlyStopEpoch={history.early_stop_epoch}
        />
      </div>

      <div className="flex gap-4 text-sm flex-wrap">
        <div className="px-3 py-2 bg-[var(--muted)] rounded font-mono">
          AUC-ROC: <span className="font-bold">{history.final_auc_roc?.toFixed(4) ?? "N/A"}</span>
        </div>
        <div className="px-3 py-2 bg-[var(--muted)] rounded font-mono">
          MRR: <span className="font-bold">{history.final_mrr?.toFixed(4) ?? "N/A"}</span>
        </div>
        {history.early_stop_epoch && (
          <div className="px-3 py-2 bg-[var(--muted)] rounded font-mono">
            Early stop: <span className="font-bold">epoch {history.early_stop_epoch}</span>
          </div>
        )}
      </div>
    </motion.div>
  );
}
