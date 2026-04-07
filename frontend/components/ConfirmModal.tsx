"use client";

import React, { useEffect, useRef } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { AlertTriangle, Trash2, X } from "lucide-react";

export type ConfirmVariant = "danger" | "warning" | "default";

export interface ConfirmModalProps {
  open: boolean;
  title: string;
  message: string;
  confirmLabel?: string;
  cancelLabel?: string;
  variant?: ConfirmVariant;
  onConfirm: () => void | Promise<void>;
  onCancel: () => void;
  loading?: boolean;
}

const variantStyles: Record<
  ConfirmVariant,
  { icon: React.ReactNode; confirmClass: string; accentClass: string }
> = {
  danger: {
    icon: <Trash2 size={24} className="text-[#8B1A1A]" />,
    confirmClass: "bg-[#8B1A1A] hover:bg-[#6B1515] text-white border-[#8B1A1A]",
    accentClass: "border-[#8B1A1A]/30 bg-[#8B1A1A]/5",
  },
  warning: {
    icon: <AlertTriangle size={24} className="text-[#D4AF37]" />,
    confirmClass: "bg-[#D4AF37] hover:bg-[#B8941F] text-[#2B2B2B] border-[#D4AF37]",
    accentClass: "border-[#D4AF37]/30 bg-[#D4AF37]/5",
  },
  default: {
    icon: <AlertTriangle size={24} className="text-[#6B6B6B]" />,
    confirmClass: "bg-[#4A4A4A] hover:bg-[#2B2B2B] text-white border-[#4A4A4A]",
    accentClass: "border-[#4A4A4A]/30 bg-[#E8E4D9]",
  },
};

export default function ConfirmModal({
  open,
  title,
  message,
  confirmLabel = "Confirm",
  cancelLabel = "Cancel",
  variant = "danger",
  onConfirm,
  onCancel,
  loading = false,
}: ConfirmModalProps) {
  const styles = variantStyles[variant];
  const cancelRef = useRef<HTMLButtonElement>(null);
  const modalRef = useRef<HTMLDivElement>(null);

  // Focus the cancel button when the modal opens
  useEffect(() => {
    if (open) {
      cancelRef.current?.focus();
    }
  }, [open]);

  // Escape key closes modal
  useEffect(() => {
    if (!open) return;
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Escape" && !loading) {
        onCancel();
      }
      // Focus trap: Tab within modal
      if (e.key === "Tab" && modalRef.current) {
        const focusable = modalRef.current.querySelectorAll<HTMLElement>(
          'button:not([disabled]), [tabindex]:not([tabindex="-1"])'
        );
        if (focusable.length === 0) return;
        const first = focusable[0];
        const last = focusable[focusable.length - 1];
        if (e.shiftKey && document.activeElement === first) {
          e.preventDefault();
          last.focus();
        } else if (!e.shiftKey && document.activeElement === last) {
          e.preventDefault();
          first.focus();
        }
      }
    };
    document.addEventListener("keydown", handleKeyDown);
    return () => document.removeEventListener("keydown", handleKeyDown);
  }, [open, loading, onCancel]);

  const handleConfirm = async () => {
    try {
      await onConfirm();
    } finally {
      onCancel();
    }
  };

  return (
    <AnimatePresence>
      {open && (
        <>
          {/* Backdrop */}
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            onClick={loading ? undefined : onCancel}
            className="fixed inset-0 z-[200] bg-[#2B2B2B]/50 backdrop-blur-sm"
            aria-hidden="true"
          />
          {/* Modal */}
          <motion.div
            ref={modalRef}
            role="dialog"
            aria-modal="true"
            aria-labelledby="confirm-modal-title"
            aria-describedby="confirm-modal-message"
            initial={{ opacity: 0, scale: 0.95, y: 10 }}
            animate={{ opacity: 1, scale: 1, y: 0 }}
            exit={{ opacity: 0, scale: 0.95, y: 10 }}
            transition={{ type: "spring", damping: 25, stiffness: 350 }}
            className="fixed left-1/2 top-1/2 z-[201] w-full max-w-md -translate-x-1/2 -translate-y-1/2 px-4"
          >
            <div className="graph-info-card bg-[#FCFAF2] border border-[#4A4A4A] rounded-lg shadow-2xl overflow-hidden">
              <div className="h-1 w-full bg-linear-to-r from-[#D4AF37] via-[#B8941F] to-[#D4AF37]" />
              <div className="p-5">
                <div className="flex items-start gap-4">
                  <div
                    className={`flex shrink-0 items-center justify-center rounded-lg border p-3 ${styles.accentClass}`}
                  >
                    {styles.icon}
                  </div>
                  <div className="min-w-0 flex-1">
                    <h3
                      id="confirm-modal-title"
                      className="text-lg font-bold text-[#2B2B2B]"
                      style={{ fontFamily: "EB Garamond, serif" }}
                    >
                      {title}
                    </h3>
                    <p id="confirm-modal-message" className="mt-2 text-sm text-[#6B6B6B] leading-relaxed">
                      {message}
                    </p>
                  </div>
                  <button
                    onClick={onCancel}
                    disabled={loading}
                    className="shrink-0 rounded p-1.5 text-[#6B6B6B] hover:bg-[#E8E4D9] hover:text-[#2B2B2B] transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#D4AF37]"
                    aria-label={`Close ${title} dialog`}
                  >
                    <X size={18} />
                  </button>
                </div>
                <div className="mt-6 flex items-center justify-between">
                  <span className="text-[10px] text-[#6B6B6B] flex items-center gap-1">
                    <span className="kbd-hint">Esc</span> to cancel
                  </span>
                  <div className="flex gap-3">
                  <button
                    ref={cancelRef}
                    onClick={onCancel}
                    disabled={loading}
                    className="rounded-md border border-[#4A4A4A] bg-transparent px-4 py-2 text-sm font-medium text-[#4A4A4A] hover:bg-[#E8E4D9] transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#D4AF37]"
                  >
                    {cancelLabel}
                  </button>
                  <button
                    onClick={handleConfirm}
                    disabled={loading}
                    className={`rounded-md border px-4 py-2 text-sm font-medium transition-colors disabled:opacity-50 disabled:cursor-not-allowed focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#D4AF37] ${styles.confirmClass}`}
                  >
                    {loading ? "Processing..." : confirmLabel}
                  </button>
                  </div>
                </div>
              </div>
            </div>
          </motion.div>
        </>
      )}
    </AnimatePresence>
  );
}
