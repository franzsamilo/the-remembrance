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
            ? "bg-[#3A5A40] shadow-[0_0_8px_rgba(58,90,64,0.5)]"
            : "bg-[#E8E4D9]"
        }`}
      />
      <span className={`text-sm ${success ? "text-[#2B2B2B]" : "text-[#6B6B6B]"}`}>
        {label}
      </span>
    </div>
  );
}
