"use client";

import React from "react";

interface TabIntroProps {
  eyebrow: string;
  description: string;
}

/**
 * One-line plain-English "here's what you're looking at" banner that sits
 * at the top of every tab. Targets the visitor who landed here cold and
 * needs to understand the page before parsing any data on it.
 */
export default function TabIntro({ eyebrow, description }: TabIntroProps) {
  return (
    <div className="mb-2 px-1">
      <p className="text-[10px] font-mono uppercase tracking-[0.2em] text-[#C5A028] mb-1">
        {eyebrow}
      </p>
      <p className="text-sm text-[#525252] leading-snug max-w-3xl">
        {description}
      </p>
    </div>
  );
}
