"use client";

import React from "react";
import { useSearchParams, useRouter } from "next/navigation";
import { motion } from "framer-motion";

export interface Tab {
  id: string;
  label: string;
  icon?: React.ReactNode;
}

type TabContent = React.ReactNode | (() => React.ReactNode);

interface TabShellProps {
  tabs: Tab[];
  // Pass `() => <Tab/>` for lazy tabs — only the active one's builder runs,
  // so hidden tabs never construct their JSX. Plain ReactNode still works.
  children: Record<string, TabContent>;
  defaultTab?: string;
}

export default function TabShell({ tabs, children, defaultTab }: TabShellProps) {
  const searchParams = useSearchParams();
  const router = useRouter();
  const activeTab = searchParams.get("tab") || defaultTab || tabs[0]?.id;

  const activeContent = React.useMemo(() => {
    const entry = children[activeTab];
    return typeof entry === "function" ? entry() : entry;
  }, [children, activeTab]);

  const setTab = (id: string) => {
    const params = new URLSearchParams(searchParams.toString());
    if (id === (defaultTab || tabs[0]?.id)) {
      params.delete("tab");
    } else {
      params.set("tab", id);
    }
    router.replace(`?${params.toString()}`, { scroll: false });
  };

  return (
    <div>
      <div className="sticky top-0 z-40 bg-[var(--parchment-base)] border-b border-[var(--border)] mb-6">
        <nav className="max-w-7xl mx-auto flex gap-1 px-2" aria-label="Dashboard tabs">
          {tabs.map((tab) => (
            <button
              key={tab.id}
              onClick={() => setTab(tab.id)}
              className={`relative px-4 py-3 text-sm font-medium transition-colors flex items-center gap-2 ${
                activeTab === tab.id
                  ? "text-[var(--ink-dark)]"
                  : "text-[var(--ink-light)] hover:text-[var(--ink-medium)]"
              }`}
              aria-selected={activeTab === tab.id}
              role="tab"
            >
              {tab.icon}
              {tab.label}
              {activeTab === tab.id && (
                <motion.div
                  layoutId="tab-underline"
                  className="absolute bottom-0 left-2 right-2 h-0.5 bg-[var(--gilded-gold)]"
                  transition={{ type: "spring", stiffness: 400, damping: 30 }}
                />
              )}
            </button>
          ))}
        </nav>
      </div>

      <div role="tabpanel">
        {activeContent}
      </div>
    </div>
  );
}
