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

// ── Paper KPI Defense Targets ──
// Values reported in Project Study Report v6.4 (Run 8 + multi-seed methodology).
// MRR is the 12-seed mean (Sun+ 2019, Vashishth+ 2020 standard); other metrics
// are the single-eval values from the Run 8 DistMult checkpoint.
export const PAPER_KPIS = [
  {
    label: "Grounding",
    value: 0.988,
    target: 0.98,
    hypothesis: "H3",
    methodology: "LLM-as-judge, n=5 queries, τ=0.95",
    description: "Synthesis restricted to GNN-validated triplets",
    plainEnglish:
      "Out of every claim the system makes, this share can be traced directly back to a fact in your documents. 0.988 means almost none of it is invented.",
  },
  {
    label: "Faithfulness",
    value: 0.971,
    target: 0.90,
    hypothesis: "H3",
    methodology: "LLM-as-judge, n=5 queries, τ=0.95",
    description: "Claims supported by retrieved evidence",
    plainEnglish:
      "How well each sentence the system writes is actually supported by the evidence it pulled. Higher means tighter coupling between the answer and the source documents.",
  },
  {
    label: "AUC-ROC",
    value: 0.985,
    target: 0.95,
    hypothesis: "H2",
    methodology: "Multi-seed mean, n=12 (σ=0.001)",
    description: "GNN discrimination on validation edges",
    plainEnglish:
      "How well the integrity model can tell a real fact apart from a fake one. 1.0 is perfect, 0.5 is random; 0.985 means the model is very confident about what's real.",
  },
  {
    label: "MRR",
    value: 0.958,
    target: 0.95,
    hypothesis: "H2",
    methodology: "Multi-seed mean, n=12 (σ=0.005)",
    description: "Mean reciprocal rank on link prediction",
    plainEnglish:
      "When the system is asked to rank real facts against fake ones, how high up the real fact lands on average. Closer to 1.0 means the right answer almost always wins.",
  },
] as const;

// ── Demo Queries (Discover empty state) ──
// Curated to showcase the system's strengths on the loaded Philippine
// legal corpus. Each demonstrates a different reasoning pattern.
export const DEMO_QUERIES = [
  {
    label: "Cross-document precedent",
    query: "How do the cases cite Article III Section 1 of the Constitution?",
    showcases: "Cross-document linking · grounded synthesis",
  },
  {
    label: "Doctrine extension",
    query: "Which decisions extend or contradict the doctrine of res judicata?",
    showcases: "Relational reasoning · CONTRADICTS edges",
  },
  {
    label: "Refusal behaviour",
    query: "What is the chemical composition of titanium dioxide?",
    showcases: "Grounding Error · no evidence ⇒ no answer",
  },
] as const;
