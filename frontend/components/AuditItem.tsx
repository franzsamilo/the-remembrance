"use client";

interface AuditItemProps {
  label: string;
  success: boolean;
}

export default function AuditItem({ label, success }: AuditItemProps) {
  return (
    <div className="flex items-center gap-3">
      <div
        className={`w-1.5 h-1.5 rounded-full ${
          success
            ? "bg-[#2D6A4F] shadow-[0_0_8px_rgba(58,90,64,0.5)]"
            : "bg-[#F5F5F3]"
        }`}
      />
      <span className={`text-sm ${success ? "text-[#1A1A1A]" : "text-[#737373]"}`}>
        {label}
      </span>
    </div>
  );
}
