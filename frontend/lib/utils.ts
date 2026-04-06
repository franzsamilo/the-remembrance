/** Format AUC-ROC value for display, or "N/A" if null/undefined. */
export function formatAucRoc(value: number | null | undefined): string {
  if (value === null || value === undefined) return "N/A";
  return Number(value).toFixed(3);
}

/** Format 0-1 score as percentage for display. */
export function formatScore(value: number | null | undefined): string {
  if (value === null || value === undefined) return "N/A";
  return `${Math.round(Number(value) * 100)}%`;
}
