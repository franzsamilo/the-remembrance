"use client";

import React, { useState, useEffect, useCallback, Suspense } from "react";
import {
  Activity,
  RefreshCw,
  Trash2,
  Upload,
  Send,
  AlertCircle,
  Play,
  Database,
  BarChart3,
  Scale,
  LayoutDashboard,
  Workflow,
  MessageSquare,
  ShieldCheck,
} from "lucide-react";
import {
  KnowledgeGraphIcon,
  ManuscriptIcon,
  SettingsGearIcon,
} from "@/components/CustomIcons";
import { motion, AnimatePresence } from "framer-motion";
import axios from "axios";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import Link from "next/link";
import { useRouter } from "next/navigation";
import ConfirmModal from "@/components/ConfirmModal";
import PipelineFlow from "@/components/PipelineFlow";
import AuditFindingsCard from "@/components/AuditFindingsCard";
import StatCard from "@/components/StatCard";
import TabShell, { Tab } from "@/components/TabShell";
import AblationComparison from "@/components/AblationComparison";
import TrainingCurves from "@/components/TrainingCurves";
import TimingSummary from "@/components/TimingSummary";
import GroundingError from "@/components/GroundingError";
import AuditOverview from "@/components/AuditOverview";
import DocumentIntegrity from "@/components/DocumentIntegrity";
import FlaggedEdges from "@/components/FlaggedEdges";
import { SkeletonDocumentList, SkeletonSidebarCard } from "@/components/Skeleton";
import { API_BASE_URL } from "@/lib/api";
import { formatAucRoc, formatScore } from "@/lib/utils";
import { STORAGE_KEYS, POLLING, STREAMING } from "@/lib/constants";
import type { Triplet, Lead, ChatMessage } from "@/lib/types";

const TABS: Tab[] = [
  { id: "overview", label: "Overview", icon: <LayoutDashboard size={16} /> },
  { id: "pipeline", label: "Pipeline", icon: <Workflow size={16} /> },
  { id: "discover", label: "Discover", icon: <MessageSquare size={16} /> },
  { id: "audit", label: "Audit", icon: <ShieldCheck size={16} /> },
];

export default function FrameworkDashboard() {
  const router = useRouter();
  const [stats, setStats] = useState<any>(null);
  const [documents, setDocuments] = useState<string[]>([]);
  const [loading, setLoading] = useState(true);
  const [ingesting, setIngesting] = useState(false);
  const [auditing, setAuditing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [auditFindings, setAuditFindings] = useState<any>(null);
  const [selectedAuditDoc, setSelectedAuditDoc] = useState<string | null>(null);

  // Chat State
  const [query, setQuery] = useState("");
  const [enableDetectiveInsights, setEnableDetectiveInsights] = useState(false);
  const [promptOnlyMode, setPromptOnlyMode] = useState(false);

  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [chatLoading, setChatLoading] = useState(false);

  const openEvidencePage = (msg: {
    role: string;
    content: string;
    triplets?: Triplet[];
    leads?: Lead[];
    suggested_actions?: string[];
    userQuery?: string;
    explain?: boolean;
  }) => {
    if (typeof window !== "undefined") {
      sessionStorage.setItem(STORAGE_KEYS.EVIDENCE_MESSAGE, JSON.stringify(msg));
      router.push("/evidence");
    }
  };

  // Custom confirm modal (replaces native confirm())
  const [confirmModal, setConfirmModal] = useState<{
    open: boolean;
    title: string;
    message: string;
    confirmLabel?: string;
    cancelLabel?: string;
    variant?: "danger" | "warning" | "default";
    onConfirm: () => void | Promise<void>;
  }>({ open: false, title: "", message: "", onConfirm: () => {} });
  const [confirmInProgress, setConfirmInProgress] = useState(false);
  const currentTask = stats?.current_task || "Idle";
  const graphState = stats?.graph_state;
  const isIngestionActive =
    typeof currentTask === "string" &&
    (currentTask.includes("Extracting") || currentTask.includes("Embedding"));
  const isAuditActive =
    typeof currentTask === "string" && currentTask.includes("Audit");
  const hasTaskError =
    typeof currentTask === "string" &&
    (currentTask.startsWith("Error:") || currentTask.startsWith("Audit Error:"));
  const showActiveTaskBanner = currentTask !== "Idle" && !hasTaskError;
  const canRunPipeline = !isIngestionActive && !isAuditActive;
  const isReady = graphState === "evidence_ready_graph";

  const fetchStats = useCallback(async () => {
    try {
      const response = await axios.get(`${API_BASE_URL}/stats`);
      setStats(response.data);
      setError(null);
    } catch (err) {
      console.error("Error fetching stats:", err);
      setError("Backend API is unreachable. Is the server running?");
    } finally {
      setLoading(false);
    }
  }, []);

  const fetchDocs = async () => {
    try {
      const response = await axios.get(`${API_BASE_URL}/documents`);
      setDocuments(response.data.documents);
    } catch (err) {
      console.error("Error fetching docs:", err);
    }
  };

  const fetchAuditFindings = useCallback(async () => {
    try {
      const response = await axios.get(`${API_BASE_URL}/audit/findings`);
      setAuditFindings(response.data);
    } catch {
      // Audit not yet run — that's fine
    }
  }, []);

  const pollIntervalRef = React.useRef<number>(POLLING.IDLE_INTERVAL_MS);
  const idleTicksRef = React.useRef(0);

  useEffect(() => {
    fetchStats();
    fetchDocs();
    fetchAuditFindings();

    if (showActiveTaskBanner) {
      pollIntervalRef.current = POLLING.ACTIVE_INTERVAL_MS;
      idleTicksRef.current = 0;
    }

    const tick = () => {
      fetchStats();
      if (!showActiveTaskBanner) {
        idleTicksRef.current++;
        if (idleTicksRef.current > 6) {
          pollIntervalRef.current = Math.min(
            pollIntervalRef.current * POLLING.BACKOFF_MULTIPLIER,
            POLLING.MAX_INTERVAL_MS
          );
        }
      }
    };

    const interval = setInterval(
      tick,
      showActiveTaskBanner ? POLLING.ACTIVE_INTERVAL_MS : pollIntervalRef.current
    );
    return () => clearInterval(interval);
  }, [showActiveTaskBanner, fetchStats, fetchAuditFindings]);

  useEffect(() => {
    setIngesting(isIngestionActive);
    setAuditing(isAuditActive);
  }, [isIngestionActive, isAuditActive]);

  const triggerIngestion = async () => {
    try {
      setError(null);
      await axios.post(`${API_BASE_URL}/ingest`);
      await fetchStats();
      await fetchDocs();
    } catch (err) {
      console.error("Error triggering ingestion:", err);
      setError(
        "Ingestion pipeline failed to start. Ensure documents are uploaded and the backend is running."
      );
      setIngesting(false);
    }
  };

  const triggerAudit = async () => {
    try {
      setError(null);
      await axios.post(`${API_BASE_URL}/audit`);
      await fetchStats();
    } catch (err) {
      console.error("Error triggering audit:", err);
      setError(
        "GNN audit failed to start. Run the ingestion pipeline first to populate the knowledge graph."
      );
      setAuditing(false);
    }
  };

  const handleUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    if (!e.target.files || e.target.files.length === 0) return;
    const file = e.target.files[0];
    const formData = new FormData();
    formData.append("file", file);

    try {
      setLoading(true);
      await axios.post(`${API_BASE_URL}/upload`, formData);
      await fetchDocs();
    } catch (err: unknown) {
      console.error("Upload failed:", err);
      const detail =
        axios.isAxiosError(err) && err.response?.data?.detail
          ? String(err.response.data.detail)
          : null;
      if (axios.isAxiosError(err) && err.response?.status === 413) {
        setError("File is too large. Try a smaller PDF.");
      } else if (detail) {
        setError(`Upload failed: ${detail}`);
      } else {
        setError("File upload failed. Is the backend running on port 8000?");
      }
    } finally {
      setLoading(false);
    }
  };

  const handleChat = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!query.trim()) return;

    const userMsg = query;
    setQuery("");
    setMessages((prev) => [
      ...prev,
      {
        role: "user",
        content: userMsg,
        explain: enableDetectiveInsights,
      } as ChatMessage,
    ]);
    setChatLoading(true);

    try {
      const useStream = true;
      if (useStream) {
        const res = await fetch(`${API_BASE_URL}/chat/stream`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            query: userMsg,
            explain: enableDetectiveInsights,
            mode: promptOnlyMode ? "prompt_only" : "graph",
          }),
        });
        if (!res.ok) throw new Error(res.statusText);
        const reader = res.body?.getReader();
        const decoder = new TextDecoder();
        let buffer = "";
        let narrative = "";
        let triplets: Triplet[] = [];
        let filteredTriplets: Triplet[] = [];
        let leads: Lead[] = [];
        let suggested_actions: string[] = [];
        let groundingStatus = "";
        let groundingErrored = false;
        let batchTimer: ReturnType<typeof setTimeout> | null = null;
        let pendingFlush = false;

        const flushNarrative = () => {
          pendingFlush = false;
          batchTimer = null;
          setMessages((prev) => {
            const next = [...prev];
            const lastIdx = next.length - 1;
            if (next[lastIdx]?.role === "ai")
              next[lastIdx] = { ...next[lastIdx], content: narrative };
            return next;
          });
        };

        const scheduleFlush = () => {
          if (!pendingFlush) {
            pendingFlush = true;
            batchTimer = setTimeout(flushNarrative, STREAMING.BATCH_INTERVAL_MS);
          }
        };

        setMessages((prev) => [
          ...prev,
          {
            role: "ai",
            content: "",
            triplets: [],
            leads: [],
            suggested_actions: [],
            userQuery: userMsg,
            explain: enableDetectiveInsights,
          } as ChatMessage,
        ]);
        while (reader) {
          const { done, value } = await reader.read();
          if (done) break;
          buffer += decoder.decode(value, { stream: true });
          const lines = buffer.split("\n\n");
          buffer = lines.pop() ?? "";
          for (const line of lines) {
            if (line.startsWith("data: ")) {
              try {
                const data = JSON.parse(line.slice(6));
                if (data.type === "chunk") {
                  narrative += data.text ?? "";
                  scheduleFlush();
                } else if (data.type === "done") {
                  triplets = data.triplets ?? [];
                  filteredTriplets = data.filtered_triplets ?? [];
                  leads = data.leads ?? [];
                  suggested_actions = data.suggested_actions ?? [];
                  groundingStatus = data.grounding_status ?? "";
                } else if (data.type === "grounding_error") {
                  groundingErrored = true;
                  narrative = data.narrative_text ?? data.message ?? "Grounding Error";
                  filteredTriplets = data.filtered_triplets ?? [];
                  groundingStatus = data.grounding_status ?? "FAILED";
                } else if (data.type === "error") {
                  narrative = data.narrative_text ?? "Error.";
                  triplets = data.triplets ?? [];
                  filteredTriplets = data.filtered_triplets ?? [];
                  leads = data.leads ?? [];
                  suggested_actions = data.suggested_actions ?? [];
                  groundingStatus = data.grounding_status ?? "FAILED";
                }
              } catch (_) {}
            }
          }
        }
        if (batchTimer) clearTimeout(batchTimer);
        setMessages((prev) => {
          const next = [...prev];
          const lastIdx = next.length - 1;
          if (next[lastIdx]?.role === "ai") {
            next[lastIdx] = {
              ...next[lastIdx],
              content: narrative,
              triplets,
              filtered_triplets: filteredTriplets,
              leads,
              suggested_actions,
              groundingStatus,
              groundingError: groundingErrored,
            };
          }
          return next;
        });
      } else {
        const response = await axios.post(`${API_BASE_URL}/chat`, {
          query: userMsg,
          explain: enableDetectiveInsights,
          mode: promptOnlyMode ? "prompt_only" : "graph",
        });
        setMessages((prev) => [
          ...prev,
          {
            role: "ai",
            content: response.data.narrative_text,
            triplets: response.data.triplets,
            filtered_triplets: response.data.filtered_triplets ?? [],
            leads: response.data.leads,
            suggested_actions: response.data.suggested_actions ?? [],
            userQuery: userMsg,
            explain: enableDetectiveInsights,
            groundingStatus: response.data.grounding_status,
          },
        ]);
      }
    } catch (err: unknown) {
      console.error("Chat failed:", err);
      const msg =
        axios.isAxiosError(err) && err.response?.data?.detail
          ? String(err.response.data.detail)
          : "Discovery failed. Is the backend running?";
      setMessages((prev) => [...prev, { role: "ai", content: msg }]);
    } finally {
      setChatLoading(false);
    }
  };

  const handleReset = () => {
    setConfirmModal({
      open: true,
      title: "Reset Database",
      message:
        "This will permanently wipe all nodes and relationships from your Neo4j instance. This cannot be undone.",
      confirmLabel: "Yes, reset",
      cancelLabel: "Cancel",
      variant: "danger",
      onConfirm: async () => {
        setConfirmInProgress(true);
        try {
          setLoading(true);
          await axios.post(`${API_BASE_URL}/reset`);
          await fetchStats();
          await fetchDocs();
          await fetchAuditFindings();
        } catch (err) {
          console.error("Reset failed:", err);
          setError(
            "Database reset failed. Ensure the ADMIN_API_KEY is configured and the backend is running."
          );
        } finally {
          setLoading(false);
          setConfirmInProgress(false);
        }
      },
    });
  };

  const handleDeleteDoc = (filename: string) => {
    setConfirmModal({
      open: true,
      title: "Remove Document",
      message: `Are you sure you want to remove "${filename}" from the source library? This will not delete the file from disk, only remove it from the framework.`,
      confirmLabel: "Remove",
      cancelLabel: "Cancel",
      variant: "warning",
      onConfirm: async () => {
        setConfirmInProgress(true);
        try {
          await axios.delete(`${API_BASE_URL}/documents/${filename}`);
          await fetchDocs();
        } catch (err) {
          console.error("Delete failed:", err);
          setError(
            "Could not remove document. The backend may be processing another request."
          );
        } finally {
          setConfirmInProgress(false);
        }
      },
    });
  };

  // ===== Tab content builders =====

  const overviewTab = (
    <div className="max-w-7xl mx-auto space-y-6 px-2">
      {/* KPI cards */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
        <StatCard
          label="Entities"
          value={(stats?.nodes ?? 0).toLocaleString()}
          icon={<Database size={20} className="text-[#C5A028]" />}
          subtext="Knowledge graph nodes"
          delay={0}
        />
        <StatCard
          label="Relationships"
          value={(stats?.relationships ?? 0).toLocaleString()}
          icon={<BarChart3 size={20} className="text-[#C5A028]" />}
          subtext="Audited edges"
          delay={0.05}
        />
        <StatCard
          label="Grounding"
          value={
            stats?.research_kpis?.grounding_score != null
              ? formatScore(stats.research_kpis.grounding_score)
              : "—"
          }
          icon={<ShieldCheck size={20} className="text-[#2D6A4F]" />}
          subtext="LLM-as-judge"
          delay={0.1}
        />
        <StatCard
          label="AUC-ROC"
          value={
            stats?.research_kpis?.gnn_auc_roc != null
              ? formatAucRoc(stats.research_kpis.gnn_auc_roc)
              : "—"
          }
          icon={<Activity size={20} className="text-[#2D6A4F]" />}
          subtext="GNN audit quality"
          delay={0.15}
        />
      </div>

      {/* Source Documents card (preserved) */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="card-raised rounded-sm overflow-hidden"
      >
        <div className="p-5 border-b border-[#E5E5E3] bg-[#FFFFFF]">
          <div className="flex justify-between items-center mb-4">
            <h3
              className="flex items-center gap-2 font-semibold text-lg"
              style={{ fontFamily: "EB Garamond, serif" }}
            >
              <ManuscriptIcon size={20} className="text-[#C5A028]" />
              Source Documents
            </h3>
            <label className="flex items-center gap-2 px-3 py-1.5 rounded-sm text-sm font-medium bg-[#FAFAF8] hover:bg-[#F5F5F3] text-[#525252] transition-all cursor-pointer border border-[#E5E5E3]">
              <Upload size={14} />
              Upload PDF
              <input
                type="file"
                accept=".pdf"
                className="hidden"
                onChange={handleUpload}
              />
            </label>
          </div>
          <div className="flex items-center gap-3">
            <button
              onClick={triggerIngestion}
              disabled={!canRunPipeline}
              className={`flex-1 flex items-center justify-center gap-2 px-4 py-2.5 rounded-sm font-medium transition-all border text-sm ${
                ingesting
                  ? "bg-[#F5F5F3] text-[#737373] cursor-not-allowed border-[#E5E5E3]"
                  : "bg-[#C5A028] hover:bg-[#A68A1E] text-[#1A1A1A] shadow-lg border-[#E5E5E3]"
              }`}
            >
              {ingesting ? (
                <>
                  <RefreshCw size={14} className="animate-spin" /> Processing...
                </>
              ) : (
                <>
                  <Play size={14} fill="currentColor" /> Run Pipeline
                </>
              )}
            </button>

            <button
              onClick={triggerAudit}
              disabled={!canRunPipeline}
              className={`flex-1 relative flex items-center justify-center gap-2 px-4 py-2.5 rounded-sm font-medium transition-all border text-sm ${
                auditing
                  ? "bg-[#2D6A4F]/10 text-[#2D6A4F] cursor-not-allowed border-[#2D6A4F]/40 shadow-[0_0_12px_rgba(45,106,79,0.15)]"
                  : "bg-[#FFFFFF] hover:bg-[#F5F5F3] text-[#1A1A1A] border-[#E5E5E3] hover:border-[#C5A028]"
              }`}
            >
              {auditing && (
                <span className="absolute inset-0 rounded-sm border border-[#2D6A4F]/30 animate-pulse" />
              )}
              {auditing ? (
                <>
                  <Scale size={14} className="animate-[spin_3s_linear_infinite]" />
                  <span>Auditing</span>
                  <span className="flex gap-0.5 ml-1">
                    {[0, 1, 2].map((i) => (
                      <motion.span
                        key={i}
                        className="w-1 h-1 rounded-full bg-[#2D6A4F]"
                        animate={{ opacity: [0.3, 1, 0.3] }}
                        transition={{
                          repeat: Infinity,
                          duration: 1.2,
                          delay: i * 0.3,
                        }}
                      />
                    ))}
                  </span>
                </>
              ) : (
                <>
                  <Scale size={14} /> Run Audit
                </>
              )}
            </button>
          </div>
        </div>
        <div className="divide-y divide-[#E5E5E3]">
          {documents.length > 0 ? (
            documents.map((doc, idx) => (
              <div
                key={idx}
                className="px-5 py-3.5 flex items-center justify-between hover:bg-[#F5F5F3] transition-colors"
              >
                <div className="flex items-center gap-3">
                  <ManuscriptIcon size={16} className="text-[#737373] shrink-0" />
                  <span className="text-sm font-medium">{doc}</span>
                </div>
                <button
                  onClick={() => handleDeleteDoc(doc)}
                  className="p-1.5 hover:bg-[#7A1A1A]/10 text-[#737373] hover:text-[#7A1A1A] rounded-sm transition-all"
                  title="Remove document"
                >
                  <Trash2 size={14} />
                </button>
              </div>
            ))
          ) : (
            <div className="p-8 text-center text-[#737373] italic text-sm">
              No documents uploaded yet
            </div>
          )}
        </div>
      </motion.div>

      {/* Audit findings preview */}
      <motion.div
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.2 }}
      >
        <AuditFindingsCard />
      </motion.div>
    </div>
  );

  const pipelineTab = (
    <div className="max-w-7xl mx-auto space-y-6 px-2">
      <TimingSummary timings={stats?.stage_timings ?? null} />
      <motion.div
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.1 }}
      >
        <PipelineFlow stats={stats} currentTask={currentTask} />
      </motion.div>
    </div>
  );

  const discoverTab = (
    <div className="max-w-5xl mx-auto px-2 flex flex-col" style={{ minHeight: "70vh" }}>
      <div className="flex-1 overflow-y-auto p-6 space-y-8 scrollbar-thin">
        {messages.length === 0 && (
          <div className="h-full flex flex-col items-center justify-center text-center space-y-6 opacity-60 py-20">
            <div className="p-6 bg-[#F5F5F3] rounded-full border border-[#E5E5E3]">
              <KnowledgeGraphIcon size={48} className="text-[#1A1A1A]" />
            </div>
            <div>
              <p className="text-lg font-medium text-[#1A1A1A]">Ready to Explore</p>
              <p className="text-base text-[#525252] mt-2 max-w-md mx-auto">
                Ask a question about your case. Answers are grounded in your
                ingested documents — no evidence, no answer.
              </p>
            </div>
          </div>
        )}

        {messages.map((msg, idx) => {
          const triplets = msg.triplets || [];
          const leads = msg.leads || [];
          const entityCount = new Set(
            triplets
              .flatMap((t) => [t.source, t.target])
              .filter(
                (val): val is string =>
                  typeof val === "string" && val.trim().length > 0
              )
          ).size;

          if (msg.role === "ai" && msg.groundingError) {
            return (
              <div key={idx} className="flex flex-col items-start">
                <GroundingError
                  message={msg.content}
                  filteredTriplets={msg.filtered_triplets || []}
                />
              </div>
            );
          }

          return (
            <div
              key={idx}
              className={`flex flex-col ${
                msg.role === "user" ? "items-end" : "items-start"
              } space-y-4`}
            >
              <div
                className={`relative max-w-[90%] ${
                  msg.role === "user"
                    ? "bg-[#7A1A1A] text-white p-5 rounded-sm shadow-lg shadow-[#7A1A1A]/20"
                    : ""
                }`}
              >
                {msg.role === "ai" ? (
                  <div
                    className="prose prose-sm max-w-none
                          prose-p:leading-relaxed prose-p:text-[#1A1A1A]
                          prose-headings:text-[#C5A028] prose-headings:font-semibold prose-headings:mt-6 prose-headings:mb-3
                          prose-strong:text-[#1A1A1A] prose-strong:font-semibold
                          prose-ul:my-4 prose-li:marker:text-[#C5A028]/50
                          prose-hr:border-[#E5E5E3] prose-hr:my-6"
                  >
                    <ReactMarkdown remarkPlugins={[remarkGfm]}>
                      {msg.content}
                    </ReactMarkdown>
                  </div>
                ) : (
                  <p className="text-sm font-medium leading-relaxed">
                    {msg.content}
                  </p>
                )}
              </div>

              {msg.role === "ai" && (
                <div className="w-full space-y-4 pl-4 border-l-2 border-[#E5E5E3] ml-2">
                  {triplets.length > 0 && msg.explain && (
                    <div className="flex items-center gap-3">
                      <button
                        onClick={() => openEvidencePage(msg)}
                        className="graph-view-toggle flex items-center gap-2 px-3 py-1.5 rounded-md text-[10px] font-mono uppercase tracking-widest border bg-transparent border-[#E5E5E3] text-[#737373] hover:text-[#C5A028] hover:border-[#C5A028]/50"
                      >
                        <KnowledgeGraphIcon size={12} />
                        View Evidence
                      </button>
                      <span className="text-[9px] text-[#737373] italic">
                        {triplets.length} facts · {entityCount} entities
                      </span>
                    </div>
                  )}

                  {leads.length > 0 && (
                    <div className="flex flex-wrap gap-2 animate-in fade-in slide-in-from-bottom-2 duration-500 delay-100">
                      <div className="w-full text-[10px] font-mono uppercase tracking-widest text-[#737373] mb-1 flex items-center gap-2">
                        <KnowledgeGraphIcon size={10} />
                        Active Discovery Leads
                      </div>
                      {leads.map((lead, lidx) => (
                        <div
                          key={lidx}
                          className="px-4 py-2 bg-[#C5A028]/10 border border-[#C5A028]/20 rounded-full text-xs text-[#C5A028] cursor-default"
                        >
                          {lead.name}
                        </div>
                      ))}
                    </div>
                  )}

                  {triplets.length > 0 && !msg.explain && (
                    <div className="animate-in fade-in slide-in-from-bottom-4 duration-500 delay-200">
                      <div className="text-[10px] font-mono uppercase tracking-widest text-[#737373] mb-3 flex items-center gap-2">
                        <Database size={10} />
                        Verified Facts
                      </div>
                      <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
                        {triplets.slice(0, 6).map((t: any, tidx: number) => (
                          <div
                            key={tidx}
                            className="p-3 bg-[#F5F5F3] border border-[#E5E5E3] rounded-lg group"
                          >
                            <div className="flex items-center gap-2 text-[10px] text-[#737373] mb-1 font-mono">
                              <span className="group-hover:text-[#7A1A1A] transition-colors">
                                {t.source}
                              </span>
                              <span className="text-[#737373]">&rarr;</span>
                              <span className="group-hover:text-[#C5A028] transition-colors">
                                {t.target}
                              </span>
                            </div>
                            <div className="text-xs text-[#737373] truncate font-medium">
                              {t.relation}
                            </div>
                          </div>
                        ))}
                        {triplets.length > 6 && (
                          <div className="p-3 border border-dashed border-[#E5E5E3] rounded-lg flex items-center justify-center text-xs text-[#737373] italic">
                            + {triplets.length - 6} more facts
                          </div>
                        )}
                      </div>
                    </div>
                  )}
                </div>
              )}
            </div>
          );
        })}

        {chatLoading && (
          <div className="flex items-center gap-4 animate-pulse pl-4">
            <div className="w-8 h-8 rounded-full bg-[#C5A028]/10 flex items-center justify-center">
              <Activity size={16} className="text-[#C5A028] animate-spin" />
            </div>
            <div className="space-y-2">
              <div className="h-2 w-24 bg-[#F5F5F3] rounded-full" />
              <div className="h-2 w-32 bg-[#F5F5F3] rounded-full" />
            </div>
          </div>
        )}
      </div>

      <form
        onSubmit={handleChat}
        className="p-4 bg-[#FFFFFF] border-t border-[#E5E5E3] sticky bottom-0"
      >
        <div className="max-w-4xl mx-auto space-y-3">
          <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-2 text-[10px] text-[#737373]">
            <div className="flex items-center gap-3 flex-wrap">
              <label className="flex items-center gap-2 font-mono uppercase tracking-widest cursor-pointer px-3 py-1.5 rounded-full border border-[#E5E5E3] bg-[#F5F5F3]">
                <input
                  type="checkbox"
                  checked={enableDetectiveInsights}
                  onChange={(e) => setEnableDetectiveInsights(e.target.checked)}
                  disabled={promptOnlyMode}
                  className="accent-[#C5A028]"
                />
                Explain Connections
              </label>
              <label className="flex items-center gap-2 font-mono uppercase tracking-widest cursor-pointer px-3 py-1.5 rounded-full border border-amber-500/40 bg-amber-500/10">
                <input
                  type="checkbox"
                  checked={promptOnlyMode}
                  onChange={(e) => setPromptOnlyMode(e.target.checked)}
                  className="accent-amber-600"
                />
                Prompt Only (Ablation)
              </label>
              {!promptOnlyMode && (
                <span className="font-mono uppercase tracking-widest">
                  Evidence Board + ordered narrative
                </span>
              )}
            </div>
            <span className="text-[10px] italic">
              {promptOnlyMode
                ? "Chunk RAG only — no graph, no GNN. Compare grounding."
                : "Adds extra analysis time for richer explanations."}
            </span>
          </div>
          <div className="relative">
            <input
              type="text"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Ask a deep question..."
              className="w-full bg-[#FAFAF8] hover:bg-[#F5F5F3] border border-[#E5E5E3] rounded-2xl py-4 pl-6 pr-16 text-base text-[#1A1A1A] placeholder:text-[#737373] focus:outline-none focus:ring-2 focus:ring-[#C5A028]/50 focus:border-[#C5A028] transition-all shadow-inner"
            />
            <button
              type="submit"
              disabled={chatLoading || !query.trim()}
              className="absolute right-3 top-1/2 -translate-y-1/2 p-2 bg-[#C5A028] hover:bg-[#A68A1E] rounded-xl text-white transition-all disabled:opacity-50 disabled:cursor-not-allowed shadow-lg active:scale-95"
            >
              {chatLoading ? (
                <RefreshCw size={20} className="animate-spin" />
              ) : (
                <Send size={20} />
              )}
            </button>
          </div>
          <p className="text-center text-[10px] text-[#737373]">
            Unlike a standard chatbot: answers come only from validated facts in
            your documents. No evidence = no answer.
          </p>
        </div>
      </form>
    </div>
  );

  const auditRunData = auditFindings?.audit_run
    ? {
        total_audited: auditFindings.audit_run.total_audited ?? 0,
        total_flagged: auditFindings.audit_run.total_flagged ?? 0,
        threshold: auditFindings.audit_run.threshold ?? 0.95,
        auc_roc: auditFindings.audit_run.auc_roc ?? null,
        mrr: auditFindings.audit_run.mrr ?? null,
      }
    : null;

  const auditTab = (
    <div className="max-w-7xl mx-auto space-y-6 px-2">
      {auditRunData ? (
        <AuditOverview auditRun={auditRunData} />
      ) : (
        <div className="card-raised rounded-sm p-8 text-center text-sm text-[#737373]">
          No audit run yet. Run the GNN audit from the Overview tab to populate
          findings.
        </div>
      )}

      <TrainingCurves />

      <div>
        <h3
          className="text-lg font-semibold mb-3"
          style={{ fontFamily: "EB Garamond, serif" }}
        >
          Ablation Comparison
        </h3>
        <AblationComparison results={stats?.ablation ?? null} />
      </div>

      {auditFindings?.document_summary && (
        <DocumentIntegrity
          documents={auditFindings.document_summary}
          selectedDoc={selectedAuditDoc}
          onSelectDoc={setSelectedAuditDoc}
        />
      )}

      {auditFindings?.flagged_edges && (
        <FlaggedEdges
          edges={auditFindings.flagged_edges}
          filterDoc={selectedAuditDoc}
          threshold={auditFindings?.audit_run?.threshold ?? 0.95}
        />
      )}
    </div>
  );

  return (
    <div className="min-h-screen bg-[#FAFAF8] text-[#1A1A1A] p-4 sm:p-6 paper-texture">
      {/* Compact Header */}
      <header className="max-w-7xl mx-auto flex justify-between items-center mb-4 relative z-50">
        <motion.div
          initial={{ opacity: 0, x: -20 }}
          animate={{ opacity: 1, x: 0 }}
          className="flex items-center gap-3"
        >
          <div className="p-2 bg-[#7A1A1A] rounded-sm shadow-lg border border-[#E5E5E3]">
            <KnowledgeGraphIcon className="text-[#FAFAF8]" size={22} />
          </div>
          <div>
            <h1
              className="text-2xl font-bold tracking-tight gradient-text"
              style={{ fontFamily: "EB Garamond, serif" }}
            >
              THE REMEMBRANCE VAULT
            </h1>
            <p className="text-[#737373] text-[10px] font-mono uppercase tracking-widest mt-0.5">
              Structured documents &rarr; knowledge graph &rarr; auditable answers
            </p>
          </div>
        </motion.div>

        <div className="flex items-center gap-2">
          <button
            onClick={handleReset}
            className="p-2 hover:bg-[#7A1A1A]/10 text-[#7A1A1A]/60 hover:text-[#7A1A1A] rounded-sm transition-all border border-transparent hover:border-[#7A1A1A]/30"
            title="Wipe Graph"
            aria-label="Reset database and wipe graph"
          >
            <Trash2 size={18} />
          </button>
          <button
            onClick={fetchStats}
            className="p-2 hover:bg-[#F5F5F3] rounded-sm text-[#737373] hover:text-[#1A1A1A] transition-colors border border-transparent hover:border-[#E5E5E3]"
            title="Refresh Data"
            aria-label="Refresh statistics"
          >
            <RefreshCw size={18} className={loading ? "animate-spin" : ""} />
          </button>
          <Link href="/config">
            <button className="flex items-center gap-2 px-3 py-2 bg-[#FFFFFF] hover:bg-[#FAFAF8] text-[#525252] hover:text-[#1A1A1A] rounded-sm transition-all border border-[#E5E5E3] hover:border-[#C5A028] text-sm">
              <SettingsGearIcon size={16} />
              <span className="font-medium hidden sm:inline">Config</span>
            </button>
          </Link>
        </div>
      </header>

      {/* Live System Status Banner — simplified single-line indicator */}
      <AnimatePresence>
        {showActiveTaskBanner && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: "auto" }}
            exit={{ opacity: 0, height: 0 }}
            className="max-w-7xl mx-auto overflow-hidden mb-4"
          >
            <div
              className={`flex items-center gap-3 px-4 py-2.5 rounded-sm border ${
                isAuditActive
                  ? "bg-[#2D6A4F]/5 border-[#2D6A4F]/30"
                  : "bg-[#C5A028]/5 border-[#C5A028]/30"
              }`}
            >
              {isAuditActive ? (
                <Scale size={16} className="text-[#2D6A4F] animate-[spin_3s_linear_infinite]" />
              ) : (
                <Activity size={16} className="text-[#C5A028] animate-spin-slow" />
              )}
              <span
                className="font-semibold text-xs"
                style={{ fontFamily: "EB Garamond, serif" }}
              >
                {isAuditActive
                  ? "Semantic Audit in Progress"
                  : "Archival Process Active"}
              </span>
              <span className="text-[#737373] font-mono text-[10px]">
                &gt; {currentTask}
              </span>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Error Alerts */}
      {(error ||
        hasTaskError ||
        ((stats?.status === "error" || stats?.status === "unavailable") &&
          stats?.message)) && (
        <div className="max-w-7xl mx-auto mb-4 space-y-2">
          {error && (
            <div className="flex items-start gap-3 p-3 bg-rose-500/10 border border-rose-500/20 rounded-sm">
              <AlertCircle
                size={18}
                className="text-rose-500 shrink-0 mt-0.5"
              />
              <p className="text-sm text-rose-700">{error}</p>
            </div>
          )}
          {(stats?.status === "error" || stats?.status === "unavailable") &&
            stats?.message && (
              <div className="flex items-start gap-3 p-3 bg-amber-500/10 border border-amber-500/20 rounded-sm">
                <AlertCircle
                  size={18}
                  className="text-amber-500 shrink-0 mt-0.5"
                />
                <p className="text-sm text-amber-800">{stats.message}</p>
              </div>
            )}
          {hasTaskError && (
            <div className="flex items-start gap-3 p-3 bg-amber-500/10 border border-amber-500/20 rounded-sm">
              <AlertCircle
                size={18}
                className="text-amber-500 shrink-0 mt-0.5"
              />
              <p className="text-sm text-amber-800 font-mono">{currentTask}</p>
            </div>
          )}
        </div>
      )}

      {/* Tabbed Main Content */}
      {loading && !stats ? (
        <main className="max-w-7xl mx-auto grid grid-cols-1 lg:grid-cols-3 gap-6">
          <div className="lg:col-span-2 space-y-6">
            <SkeletonDocumentList />
          </div>
          <div className="space-y-6">
            <SkeletonSidebarCard />
            <SkeletonSidebarCard />
            <SkeletonSidebarCard />
          </div>
        </main>
      ) : (
        <Suspense fallback={null}>
          <TabShell tabs={TABS} defaultTab="overview">
            {{
              overview: overviewTab,
              pipeline: pipelineTab,
              discover: discoverTab,
              audit: auditTab,
            }}
          </TabShell>
        </Suspense>
      )}

      {/* Custom confirm modal */}
      <ConfirmModal
        open={confirmModal.open}
        title={confirmModal.title}
        message={confirmModal.message}
        confirmLabel={confirmModal.confirmLabel}
        cancelLabel={confirmModal.cancelLabel}
        variant={confirmModal.variant}
        onConfirm={confirmModal.onConfirm}
        onCancel={() => setConfirmModal((p) => ({ ...p, open: false }))}
        loading={confirmInProgress}
      />
    </div>
  );
}
