/** Shared types for The Remembrance Vault frontend. */

export interface Triplet {
  source?: string | null;
  relation?: string | null;
  target?: string | null;
  audit?: number;
  description?: string;
  explanation?: string | null;
  source_docs?: string[];
  target_docs?: string[];
  cross_document?: boolean;
}

export interface Lead {
  name: string;
  description?: string | null;
  explanation?: string | null;
}

export interface ChatMessage {
  role: "user" | "ai";
  content: string;
  triplets?: Triplet[];
  filtered_triplets?: Triplet[];
  leads?: Lead[];
  suggested_actions?: string[];
  userQuery?: string;
  explain?: boolean;
  groundingStatus?: string;
  groundingError?: boolean;
}

export interface AblationResult {
  grounding_score: number | null;
  faithfulness_score: number | null;
  completed_at: string | null;
  sample_count: number;
}

export interface AblationResults {
  full_stack?: AblationResult;
  prompt_only?: AblationResult;
  graph_no_gnn?: AblationResult;
}

export interface EpochMetric {
  epoch: number;
  train_loss: number | null;
  auc_roc: number | null;
}

export interface TrainingHistory {
  epochs: EpochMetric[];
  early_stop_epoch: number | null;
  best_epoch: number | null;
  final_auc_roc: number | null;
  final_mrr: number | null;
}

export interface StageTiming {
  [stage: string]: number;
}

export interface GraphReadiness {
  source_documents?: number;
  provenance_covered_nodes?: number;
  latest_ingestion_status?: string;
  latest_documents_processed?: number;
  latest_documents_failed?: number;
  latest_completed_at?: string | null;
  embedding_model?: string;
  embedding_dimension?: number;
}

export interface AuditReadiness {
  state?: string;
  audited_relationships?: number;
  total_relationships?: number;
  latest_audit_status?: string;
  latest_audit_mode?: string | null;
  latest_audit_completed_at?: string | null;
  latest_auc_roc?: number | null;
  latest_mrr?: number | null;
}

export interface ResearchKpis {
  gnn_auc_roc?: number | null;
  gnn_mrr?: number | null;
  grounding_score?: number | null;
  faithfulness_score?: number | null;
}

export interface InferenceConfig {
  grounding_threshold: number;
  retrieval_seed_limit: number;
  retrieval_expansion_limit: number;
}

export interface StatsData {
  status?: string;
  message?: string;
  nodes?: number;
  entities?: number;
  feature_complete?: number;
  relationships?: number;
  embedding_progress?: number;
  current_task?: string;
  graph_state?: string;
  graph_readiness?: GraphReadiness;
  audit_readiness?: AuditReadiness;
  research_kpis?: ResearchKpis;
  ablation?: AblationResults | null;
  stage_timings?: StageTiming | null;
  inference_config?: InferenceConfig;
}

export interface AuditRunSummary {
  run_id?: string | null;
  completed_at?: string | null;
  auc_roc?: number | null;
  mrr?: number | null;
  total_audited?: number;
  total_flagged?: number;
  threshold?: number;
}

export interface FlaggedEdge {
  source: string;
  relation: string;
  target: string;
  plausibility_score: number;
  audit_status: string;
  source_docs: string[];
  target_docs: string[];
  cross_document: boolean;
  description: string | null;
}

export interface DocumentSummary {
  document: string;
  total_edges: number;
  flagged: number;
  integrity: number;
}

export interface AuditFindings {
  audit_run: AuditRunSummary | null;
  document_summary: DocumentSummary[];
  flagged_edges: FlaggedEdge[];
}
