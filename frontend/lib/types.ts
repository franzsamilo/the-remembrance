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
  leads?: Lead[];
  suggested_actions?: string[];
  userQuery?: string;
  explain?: boolean;
  groundingStatus?: string;
}
