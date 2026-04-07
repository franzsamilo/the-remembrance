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
  graph?: AblationResult;
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
