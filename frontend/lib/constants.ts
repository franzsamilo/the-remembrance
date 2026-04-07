/** Shared constants for The Remembrance Vault frontend. */

// ── SessionStorage Keys ──
export const STORAGE_KEYS = {
  EVIDENCE_MESSAGE: "remembrance_evidence_message",
  OPEN_CHAT: "remembrance_open_chat",
} as const;

// ── Polling ──
export const POLLING = {
  ACTIVE_INTERVAL_MS: 2000,
  IDLE_INTERVAL_MS: 5000,
  BACKOFF_MULTIPLIER: 1.5,
  MAX_INTERVAL_MS: 15000,
} as const;

// ── Streaming ──
export const STREAMING = {
  BATCH_INTERVAL_MS: 50,
} as const;

// ── Detective Board ──
export const DETECTIVE_BOARD = {
  PAGE_SIZE: 10,
} as const;
