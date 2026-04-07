"use client";

import React from "react";

/** Reusable skeleton loading placeholders. */

export function SkeletonLine({ className = "" }: { className?: string }) {
  return <div className={`skeleton skeleton-text ${className}`} />;
}

export function SkeletonBlock({ className = "" }: { className?: string }) {
  return <div className={`skeleton ${className}`} />;
}

/** Skeleton for a StatCard-sized element */
export function SkeletonStatCard() {
  return (
    <div className="card-raised p-5 space-y-3">
      <div className="flex items-center gap-3">
        <div className="skeleton skeleton-circle w-8 h-8 shrink-0" />
        <SkeletonLine className="w-24" />
      </div>
      <SkeletonLine className="skeleton-text-lg w-16" />
      <SkeletonLine className="w-32" />
    </div>
  );
}

/** Skeleton for the sidebar status cards on the dashboard */
export function SkeletonSidebarCard() {
  return (
    <div className="card-raised p-5 space-y-3">
      <div className="flex items-center gap-3">
        <div className="skeleton skeleton-circle w-2.5 h-2.5 shrink-0" />
        <SkeletonLine className="w-28" />
      </div>
      <SkeletonLine className="w-full" />
      <SkeletonLine className="w-20" />
    </div>
  );
}

/** Skeleton for the document list area */
export function SkeletonDocumentList() {
  return (
    <div className="card-raised overflow-hidden">
      <div className="p-5 border-b border-[#4A4A4A]/20">
        <div className="flex justify-between items-center mb-4">
          <SkeletonLine className="w-36 skeleton-text-lg" />
          <SkeletonBlock className="w-24 h-8" />
        </div>
        <div className="flex gap-3">
          <SkeletonBlock className="flex-1 h-10" />
          <SkeletonBlock className="flex-1 h-10" />
        </div>
      </div>
      <div className="divide-y divide-[#4A4A4A]/10">
        {[1, 2, 3].map((i) => (
          <div key={i} className="px-5 py-3.5 flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="skeleton skeleton-circle w-4 h-4 shrink-0" />
              <SkeletonLine className="w-48" />
            </div>
            <div className="skeleton skeleton-circle w-5 h-5" />
          </div>
        ))}
      </div>
    </div>
  );
}

/** Skeleton for the Detective Board evidence steps */
export function SkeletonDetectiveBoard() {
  return (
    <div className="space-y-6">
      <div className="p-4 bg-[#8B1A1A]/5 border-l-4 border-[#E8E4D9] rounded-r-lg">
        <SkeletonLine className="w-20 mb-2" />
        <SkeletonLine className="skeleton-text-lg w-3/4" />
      </div>
      <div>
        <SkeletonLine className="w-48 mb-4" />
        <div className="space-y-4">
          {[1, 2, 3].map((i) => (
            <div key={i} className="relative pl-8">
              <div className="absolute left-0 top-0 skeleton skeleton-circle w-6 h-6" />
              <div className="card-flat p-4 space-y-2">
                <SkeletonLine className="w-full" />
                <SkeletonLine className="w-3/4" />
                <SkeletonLine className="w-1/2" />
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

/** Skeleton for config page cards grid */
export function SkeletonConfigGrid() {
  return (
    <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 gap-4">
      {Array.from({ length: 8 }).map((_, i) => (
        <SkeletonStatCard key={i} />
      ))}
    </div>
  );
}
