"use client";

import React, { useState, useEffect, useCallback } from "react";
import { 
  Activity, 
  RefreshCw, 
  Trash2,
  Upload,
  Send,
  AlertCircle,
  Play,
  X,
  Database,
  BarChart3,
  Scale
} from "lucide-react";
import { 
  KnowledgeGraphIcon,
  ManuscriptIcon,
  SettingsGearIcon,
  QuillPenIcon
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
import { SkeletonDocumentList, SkeletonSidebarCard } from "@/components/Skeleton";
import { API_BASE_URL } from "@/lib/api";
import { formatAucRoc, formatScore } from "@/lib/utils";
import { STORAGE_KEYS, POLLING, STREAMING } from "@/lib/constants";
import type { Triplet, Lead, ChatMessage } from "@/lib/types";

export default function FrameworkDashboard() {
  const router = useRouter();
  const [stats, setStats] = useState<any>(null);
  const [documents, setDocuments] = useState<string[]>([]);
  const [loading, setLoading] = useState(true);
  const [ingesting, setIngesting] = useState(false);
  const [auditing, setAuditing] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Chat State
  const [showChat, setShowChat] = useState(false);
  const [query, setQuery] = useState("");
  const [enableDetectiveInsights, setEnableDetectiveInsights] = useState(false);
  const [promptOnlyMode, setPromptOnlyMode] = useState(false);

  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [chatLoading, setChatLoading] = useState(false);

  const openEvidencePage = (msg: { role: string; content: string; triplets?: Triplet[]; leads?: Lead[]; suggested_actions?: string[]; userQuery?: string; explain?: boolean }) => {
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
  const showActiveTaskBanner =
    currentTask !== "Idle" && !hasTaskError;
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

  const pollIntervalRef = React.useRef<number>(POLLING.IDLE_INTERVAL_MS);
  const idleTicksRef = React.useRef(0);

  useEffect(() => {
    fetchStats();
    fetchDocs();

    if (showActiveTaskBanner) {
      // Active task: fast fixed polling, reset backoff
      pollIntervalRef.current = POLLING.ACTIVE_INTERVAL_MS;
      idleTicksRef.current = 0;
    }

    const tick = () => {
      fetchStats();
      if (!showActiveTaskBanner) {
        idleTicksRef.current++;
        // Backoff after 6 idle ticks (~30s at 5s interval)
        if (idleTicksRef.current > 6) {
          pollIntervalRef.current = Math.min(
            pollIntervalRef.current * POLLING.BACKOFF_MULTIPLIER,
            POLLING.MAX_INTERVAL_MS
          );
        }
      }
    };

    const interval = setInterval(tick, showActiveTaskBanner ? POLLING.ACTIVE_INTERVAL_MS : pollIntervalRef.current);
    return () => clearInterval(interval);
  }, [showActiveTaskBanner, fetchStats]);

  useEffect(() => {
    setIngesting(isIngestionActive);
    setAuditing(isAuditActive);
  }, [isIngestionActive, isAuditActive]);

  useEffect(() => {
    if (typeof window !== "undefined" && sessionStorage.getItem(STORAGE_KEYS.OPEN_CHAT) === "1") {
      sessionStorage.removeItem(STORAGE_KEYS.OPEN_CHAT);
      setShowChat(true);
    }
  }, []);

  const triggerIngestion = async () => {
    try {
      setError(null);
      await axios.post(`${API_BASE_URL}/ingest`);
      await fetchStats();
      await fetchDocs();
    } catch (err) {
      console.error("Error triggering ingestion:", err);
      setError("Ingestion pipeline failed to start. Ensure documents are uploaded and the backend is running.");
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
      setError("GNN audit failed to start. Run the ingestion pipeline first to populate the knowledge graph.");
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
        const detail = axios.isAxiosError(err) && err.response?.data?.detail
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
    setMessages(prev => [...prev, { role: 'user', content: userMsg, explain: enableDetectiveInsights, promptOnly: promptOnlyMode }]);
    setChatLoading(true);

    try {
        const useStream = true;
        if (useStream) {
            const res = await fetch(`${API_BASE_URL}/chat/stream`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ query: userMsg, explain: enableDetectiveInsights, mode: promptOnlyMode ? "prompt_only" : "graph" }),
            });
            if (!res.ok) throw new Error(res.statusText);
            const reader = res.body?.getReader();
            const decoder = new TextDecoder();
            let buffer = "";
            let narrative = "";
            let triplets: Triplet[] = [];
            let leads: Lead[] = [];
            let suggested_actions: string[] = [];
            let groundingStatus = "";
            let batchTimer: ReturnType<typeof setTimeout> | null = null;
            let pendingFlush = false;

            const flushNarrative = () => {
              pendingFlush = false;
              batchTimer = null;
              setMessages(prev => {
                const next = [...prev];
                const lastIdx = next.length - 1;
                if (next[lastIdx]?.role === "ai") next[lastIdx] = { ...next[lastIdx], content: narrative };
                return next;
              });
            };

            const scheduleFlush = () => {
              if (!pendingFlush) {
                pendingFlush = true;
                batchTimer = setTimeout(flushNarrative, STREAMING.BATCH_INTERVAL_MS);
              }
            };

            setMessages(prev => [...prev, { role: "ai", content: "", triplets: [], leads: [], suggested_actions: [], userQuery: userMsg, explain: enableDetectiveInsights, promptOnly: promptOnlyMode }]);
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
                                leads = data.leads ?? [];
                                suggested_actions = data.suggested_actions ?? [];
                                groundingStatus = data.grounding_status ?? "";
                            } else if (data.type === "error") {
                                narrative = data.narrative_text ?? "Error.";
                                triplets = data.triplets ?? [];
                                leads = data.leads ?? [];
                                suggested_actions = data.suggested_actions ?? [];
                                groundingStatus = data.grounding_status ?? "FAILED";
                            }
                        } catch (_) {}
                    }
                }
            }
            // Flush any remaining batched content
            if (batchTimer) clearTimeout(batchTimer);
            setMessages(prev => {
                const next = [...prev];
                const lastIdx = next.length - 1;
                if (next[lastIdx]?.role === "ai") {
                    next[lastIdx] = { ...next[lastIdx], content: narrative, triplets, leads, suggested_actions, groundingStatus };
                }
                return next;
            });
        } else {
            const response = await axios.post(`${API_BASE_URL}/chat`, {
                query: userMsg,
                explain: enableDetectiveInsights,
                mode: promptOnlyMode ? "prompt_only" : "graph",
            });
            setMessages(prev => [...prev, { 
                role: 'ai', 
                content: response.data.narrative_text,
                triplets: response.data.triplets,
                leads: response.data.leads,
                suggested_actions: response.data.suggested_actions ?? [],
                userQuery: userMsg,
                explain: enableDetectiveInsights,
                groundingStatus: response.data.grounding_status
            }]);
        }
    } catch (err: unknown) {
        console.error("Chat failed:", err);
        const msg = axios.isAxiosError(err) && err.response?.data?.detail
          ? String(err.response.data.detail)
          : "Discovery failed. Is the backend running?";
        setMessages(prev => [...prev, { role: 'ai', content: msg }]);
    } finally {
        setChatLoading(false);
    }
  };

  const handleReset = () => {
    setConfirmModal({
      open: true,
      title: "Reset Database",
      message: "This will permanently wipe all nodes and relationships from your Neo4j instance. This cannot be undone.",
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
        } catch (err) {
          console.error("Reset failed:", err);
          setError("Database reset failed. Ensure the ADMIN_API_KEY is configured and the backend is running.");
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
          setError("Could not remove document. The backend may be processing another request.");
        } finally {
          setConfirmInProgress(false);
        }
      },
    });
  };

  return (
    <div className="min-h-screen bg-[#F5F2E9] text-[#2B2B2B] p-6 sm:p-8 paper-texture">
      {/* Header */}
      <header className="max-w-7xl mx-auto flex justify-between items-center mb-6 relative z-50">
        <motion.div
          initial={{ opacity: 0, x: -20 }}
          animate={{ opacity: 1, x: 0 }}
          className="flex items-center gap-4"
        >
          <div className="p-3 bg-[#8B1A1A] rounded-sm shadow-lg border border-[#4A4A4A]">
            <KnowledgeGraphIcon className="text-[#F5F2E9]" size={28} />
          </div>
          <div>
            <h1 className="text-3xl sm:text-4xl font-bold tracking-tight gradient-text" style={{fontFamily: 'EB Garamond, serif'}}>
              THE REMEMBRANCE VAULT
            </h1>
            <p className="text-[#6B6B6B] text-xs font-mono uppercase tracking-widest mt-1">
              Structured documents &rarr; knowledge graph &rarr; auditable answers
            </p>
          </div>
        </motion.div>

        <div className="flex items-center gap-3">
           <button
             onClick={handleReset}
             className="p-2 hover:bg-[#8B1A1A]/10 text-[#8B1A1A]/60 hover:text-[#8B1A1A] rounded-sm transition-all border border-transparent hover:border-[#8B1A1A]/30"
             title="Wipe Graph"
             aria-label="Reset database and wipe graph"
           >
             <Trash2 size={18} />
           </button>
           <button
             onClick={fetchStats}
             className="p-2 hover:bg-[#E8E4D9] rounded-sm text-[#6B6B6B] hover:text-[#2B2B2B] transition-colors border border-transparent hover:border-[#4A4A4A]"
             title="Refresh Data"
             aria-label="Refresh statistics"
           >
             <RefreshCw size={18} className={loading ? "animate-spin" : ""} />
           </button>
           <Link href="/config">
             <button className="flex items-center gap-2 px-3 py-2 bg-[#FCFAF2] hover:bg-[#F5F2E9] text-[#4A4A4A] hover:text-[#2B2B2B] rounded-sm transition-all border border-[#4A4A4A] hover:border-[#D4AF37] text-sm">
               <SettingsGearIcon size={16} />
               <span className="font-medium hidden sm:inline">Config</span>
             </button>
           </Link>
           <button
            onClick={() => setShowChat(true)}
            title="Answers cite graph evidence"
            className="flex items-center gap-2 px-4 py-2 bg-[#D4AF37] hover:bg-[#B8941F] text-[#2B2B2B] rounded-sm font-semibold shadow-lg transition-all transform hover:scale-105 border border-[#4A4A4A] text-sm"
          >
            <QuillPenIcon size={16} />
            <span className="hidden sm:inline">Ask the Archive</span>
          </button>
        </div>
      </header>

      {/* Live System Status Banner */}
      <AnimatePresence>
        {showActiveTaskBanner && (
            <motion.div
                initial={{ opacity: 0, height: 0 }}
                animate={{ opacity: 1, height: 'auto' }}
                exit={{ opacity: 0, height: 0 }}
                className="max-w-7xl mx-auto overflow-hidden mb-6"
            >
                <div className={`relative w-full p-0.5 rounded-sm ${isAuditActive ? "bg-linear-to-r from-[#3A5A40] via-[#2D4A34] to-[#3A5A40]" : "bg-linear-to-r from-[#D4AF37] via-[#B8941F] to-[#D4AF37]"}`}>
                    <div className="bg-[#FCFAF2] rounded-sm px-6 py-4 flex items-center justify-between relative overflow-hidden">
                        <div className="relative z-10 flex items-center gap-4">
                            <div className="relative">
                                <div className={`absolute inset-0 blur-xl opacity-30 animate-pulse ${isAuditActive ? "bg-[#3A5A40]" : "bg-[#D4AF37]"}`} />
                                {isAuditActive ? (
                                  <Scale size={24} className="text-[#3A5A40] relative z-10 animate-[spin_3s_linear_infinite]" />
                                ) : (
                                  <Activity size={24} className="text-[#D4AF37] relative z-10 animate-spin-slow" />
                                )}
                            </div>
                            <div>
                                <p className="font-semibold text-[#2B2B2B] text-sm flex items-center gap-2" style={{fontFamily: 'EB Garamond, serif'}}>
                                    {isAuditActive ? "Semantic Audit in Progress" : "Archival Process Active"}
                                    <span className={`px-1.5 py-0.5 rounded-sm text-[9px] font-mono border uppercase tracking-wider ${isAuditActive ? "bg-[#3A5A40]/15 text-[#3A5A40] border-[#3A5A40]/30" : "bg-[#D4AF37]/20 text-[#8B1A1A] border-[#D4AF37]/30"}`}>
                                        {isAuditActive ? "GNN Training" : "Cataloging"}
                                    </span>
                                </p>
                                <p className="text-[#6B6B6B] font-mono text-xs mt-0.5">
                                    &gt; {currentTask}
                                </p>
                            </div>
                        </div>
                        <div className="flex gap-1 items-end h-6">
                            {[...Array(5)].map((_, i) => (
                                <motion.div
                                    key={i}
                                    animate={{ height: [6, 18, 6] }}
                                    transition={{ repeat: Infinity, duration: 1, delay: i * 0.1 }}
                                    className={`w-1 rounded-full ${isAuditActive ? "bg-[#3A5A40]/50" : "bg-[#D4AF37]/50"}`}
                                />
                            ))}
                        </div>
                    </div>
                </div>
            </motion.div>
        )}
      </AnimatePresence>

      {/* Error Alerts */}
      {(error || hasTaskError || ((stats?.status === "error" || stats?.status === "unavailable") && stats?.message)) && (
        <div className="max-w-7xl mx-auto mb-6 space-y-2">
          {error && (
            <div className="flex items-start gap-3 p-3 bg-rose-500/10 border border-rose-500/20 rounded-sm">
              <AlertCircle size={18} className="text-rose-500 shrink-0 mt-0.5" />
              <p className="text-sm text-rose-700">{error}</p>
            </div>
          )}
          {(stats?.status === "error" || stats?.status === "unavailable") && stats?.message && (
            <div className="flex items-start gap-3 p-3 bg-amber-500/10 border border-amber-500/20 rounded-sm">
              <AlertCircle size={18} className="text-amber-500 shrink-0 mt-0.5" />
              <p className="text-sm text-amber-800">{stats.message}</p>
            </div>
          )}
          {hasTaskError && (
            <div className="flex items-start gap-3 p-3 bg-amber-500/10 border border-amber-500/20 rounded-sm">
              <AlertCircle size={18} className="text-amber-500 shrink-0 mt-0.5" />
              <p className="text-sm text-amber-800 font-mono">{currentTask}</p>
            </div>
          )}
        </div>
      )}

      {/* Main Two-Column Layout */}
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
      <main className="max-w-7xl mx-auto grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* LEFT: Primary workspace (2/3) */}
        <div className="lg:col-span-2 space-y-6">
          {/* Source Document Archive */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="card-raised rounded-sm overflow-hidden"
          >
            <div className="p-5 border-b border-[#4A4A4A] bg-[#FCFAF2]">
              <div className="flex justify-between items-center mb-4">
                <h3 className="flex items-center gap-2 font-semibold text-lg" style={{fontFamily: 'EB Garamond, serif'}}>
                  <ManuscriptIcon size={20} className="text-[#D4AF37]" />
                  Source Documents
                </h3>
                <label className="flex items-center gap-2 px-3 py-1.5 rounded-sm text-sm font-medium bg-[#F5F2E9] hover:bg-[#E8E4D9] text-[#4A4A4A] transition-all cursor-pointer border border-[#4A4A4A]">
                  <Upload size={14} />
                  Upload PDF
                  <input type="file" accept=".pdf" className="hidden" onChange={handleUpload} />
                </label>
              </div>
              <div className="flex items-center gap-3">
                <button
                    onClick={triggerIngestion}
                    disabled={!canRunPipeline}
                    className={`flex-1 flex items-center justify-center gap-2 px-4 py-2.5 rounded-sm font-medium transition-all border text-sm ${ingesting ? "bg-[#E8E4D9] text-[#6B6B6B] cursor-not-allowed border-[#4A4A4A]" : "bg-[#D4AF37] hover:bg-[#B8941F] text-[#2B2B2B] shadow-lg border-[#4A4A4A]"}`}
                >
                    {ingesting ? (
                    <><RefreshCw size={14} className="animate-spin" /> Processing...</>
                    ) : (
                    <><Play size={14} fill="currentColor" /> Run Pipeline</>
                    )}
                </button>

                <button
                    onClick={triggerAudit}
                    disabled={!canRunPipeline}
                    className={`flex-1 relative flex items-center justify-center gap-2 px-4 py-2.5 rounded-sm font-medium transition-all border text-sm ${auditing ? "bg-[#3A5A40]/10 text-[#3A5A40] cursor-not-allowed border-[#3A5A40]/40 shadow-[0_0_12px_rgba(58,90,64,0.15)]" : "bg-[#FCFAF2] hover:bg-[#E8E4D9] text-[#2B2B2B] border-[#4A4A4A] hover:border-[#D4AF37]"}`}
                >
                    {auditing && (
                      <span className="absolute inset-0 rounded-sm border border-[#3A5A40]/30 animate-pulse" />
                    )}
                    {auditing ? (
                    <>
                        <Scale size={14} className="animate-[spin_3s_linear_infinite]" />
                        <span>Auditing</span>
                        <span className="flex gap-0.5 ml-1">
                          {[0, 1, 2].map(i => (
                            <motion.span key={i} className="w-1 h-1 rounded-full bg-[#3A5A40]" animate={{ opacity: [0.3, 1, 0.3] }} transition={{ repeat: Infinity, duration: 1.2, delay: i * 0.3 }} />
                          ))}
                        </span>
                    </>
                    ) : (
                    <><Scale size={14} /> Run Audit</>
                    )}
                </button>
              </div>
            </div>
            <div className="divide-y divide-[#4A4A4A]/30">
              {documents.length > 0 ? documents.map((doc, idx) => (
                <div key={idx} className="px-5 py-3.5 flex items-center justify-between hover:bg-[#E8E4D9]/50 transition-colors">
                  <div className="flex items-center gap-3">
                    <ManuscriptIcon size={16} className="text-[#6B6B6B] shrink-0" />
                    <span className="text-sm font-medium">{doc}</span>
                  </div>
                  <button
                      onClick={() => handleDeleteDoc(doc)}
                      className="p-1.5 hover:bg-[#8B1A1A]/10 text-[#6B6B6B] hover:text-[#8B1A1A] rounded-sm transition-all"
                      title="Remove document"
                  >
                      <Trash2 size={14} />
                  </button>
                </div>
              )) : (
                <div className="p-8 text-center text-[#6B6B6B] italic text-sm">
                  No documents uploaded yet
                </div>
              )}
            </div>
          </motion.div>

          {/* Pipeline Detail (collapsible) */}
          <motion.div
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.15 }}
          >
            <PipelineFlow stats={stats} currentTask={currentTask} />
          </motion.div>
        </div>

        {/* RIGHT: Status sidebar (1/3) */}
        <div className="space-y-6">
          {/* Archive Status */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.05 }}
            className={`card-raised rounded-sm p-5 ${isReady ? "border-[#3A5A40]/50 bg-[#3A5A40]/5" : "border-[#D4AF37]/50 bg-[#D4AF37]/5"}`}
          >
            <div className="flex items-center gap-3 mb-3">
              <div className="relative flex items-center justify-center" aria-hidden="true">
                <div className={`w-2.5 h-2.5 rounded-full ${isReady ? "bg-[#3A5A40]" : "bg-[#D4AF37]"}`} />
                {isReady && <div className="absolute inset-0 w-2.5 h-2.5 rounded-full bg-[#3A5A40] animate-ping opacity-30" />}
              </div>
              <p className="font-semibold text-sm text-[#2B2B2B]">
                {isReady ? "Archive Ready" : "Not Ready"}
              </p>
              <span className={`text-[9px] font-mono uppercase tracking-wider px-1.5 py-0.5 rounded border ${isReady ? "bg-[#3A5A40]/10 text-[#3A5A40] border-[#3A5A40]/20" : "bg-[#D4AF37]/10 text-[#B8941F] border-[#D4AF37]/20"}`}>
                {isReady ? "Validated" : "Pending"}
              </span>
            </div>
            <p className="text-xs text-[#6B6B6B] leading-relaxed">
              {isReady ? "Knowledge graph is validated. You can query with evidence-backed answers." : "Upload PDFs, run the pipeline, then run audit to enable queries."}
            </p>
            <Link href="/config" className="inline-block mt-3 text-xs font-medium text-[#D4AF37] hover:underline">
              System details &rarr;
            </Link>
          </motion.div>

          {/* Graph Stats */}
          {stats && (stats.nodes > 0 || stats.relationships > 0) && (
            <motion.div
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.1 }}
              className="card-raised rounded-sm p-5"
            >
              <h4 className="text-[10px] font-mono uppercase tracking-widest text-[#6B6B6B] mb-3">Knowledge Graph</h4>
              <div className="space-y-3">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <Database size={14} className="text-[#D4AF37]" />
                    <span className="text-xs text-[#6B6B6B]">Entities</span>
                  </div>
                  <span className="text-sm font-bold text-[#2B2B2B]">{(stats.nodes ?? 0).toLocaleString()}</span>
                </div>
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <BarChart3 size={14} className="text-[#D4AF37]" />
                    <span className="text-xs text-[#6B6B6B]">Relationships</span>
                  </div>
                  <span className="text-sm font-bold text-[#2B2B2B]">{(stats.relationships ?? 0).toLocaleString()}</span>
                </div>
              </div>
            </motion.div>
          )}

          {/* Research KPIs */}
          {(stats?.research_kpis && (stats.research_kpis.gnn_auc_roc != null || stats.research_kpis.grounding_score != null)) && (
            <motion.div
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.15 }}
              className="card-raised rounded-sm p-5"
            >
              <div className="flex items-center justify-between mb-3">
                <h4 className="text-[10px] font-mono uppercase tracking-widest text-[#6B6B6B]">Research KPIs</h4>
                <Link href="/config" className="text-[10px] font-medium text-[#D4AF37] hover:underline">Details</Link>
              </div>
              <div className="grid grid-cols-2 gap-2">
                {stats.research_kpis.gnn_auc_roc != null && (
                  <div className="p-2.5 rounded-sm border border-[#3A5A40]/20 bg-[#3A5A40]/5">
                    <p className="text-[9px] font-mono uppercase tracking-widest text-[#6B6B6B]">AUC-ROC</p>
                    <p className="text-base font-bold text-[#3A5A40] mt-0.5">{formatAucRoc(stats.research_kpis.gnn_auc_roc)}</p>
                  </div>
                )}
                {stats.research_kpis.gnn_mrr != null && (
                  <div className="p-2.5 rounded-sm border border-[#3A5A40]/20 bg-[#3A5A40]/5">
                    <p className="text-[9px] font-mono uppercase tracking-widest text-[#6B6B6B]">MRR</p>
                    <p className="text-base font-bold text-[#3A5A40] mt-0.5">{formatAucRoc(stats.research_kpis.gnn_mrr)}</p>
                  </div>
                )}
                {stats.research_kpis.grounding_score != null && (
                  <div className="p-2.5 rounded-sm border border-[#3A5A40]/20 bg-[#3A5A40]/5">
                    <p className="text-[9px] font-mono uppercase tracking-widest text-[#6B6B6B]">Grounding</p>
                    <p className="text-base font-bold text-[#3A5A40] mt-0.5">{formatScore(stats.research_kpis.grounding_score)}</p>
                  </div>
                )}
                {stats.research_kpis.faithfulness_score != null && (
                  <div className="p-2.5 rounded-sm border border-[#3A5A40]/20 bg-[#3A5A40]/5">
                    <p className="text-[9px] font-mono uppercase tracking-widest text-[#6B6B6B]">Faithfulness</p>
                    <p className="text-base font-bold text-[#3A5A40] mt-0.5">{formatScore(stats.research_kpis.faithfulness_score)}</p>
                  </div>
                )}
              </div>
            </motion.div>
          )}

          {/* Audit Findings */}
          <motion.div
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.2 }}
          >
            <AuditFindingsCard />
          </motion.div>
        </div>
      </main>
      )}

      {/* Chat Drawer Side Panel */}
      <AnimatePresence>
        {showChat && (
          <>
            <motion.div 
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              onClick={() => setShowChat(false)}
              className="fixed inset-0 bg-[#2B2B2B]/80 backdrop-blur-sm z-40"
            />
            <motion.div 
              initial={{ x: '100%' }}
              animate={{ x: 0 }}
              exit={{ x: '100%' }}
              transition={{ type: 'spring', damping: 30, stiffness: 300 }}
              className="fixed right-0 top-0 h-full w-full max-w-3xl bg-[#FCFAF2]/98 backdrop-blur-xl border-l border-[#4A4A4A] z-50 flex flex-col shadow-2xl"
            >
              <div className="p-6 border-b border-[#4A4A4A]/50 flex justify-between items-center bg-transparent sticky top-0 z-10">
                <div className="flex items-center gap-4">
                    <div className="p-2 bg-[#D4AF37]/10 rounded-lg">
                        <QuillPenIcon className="text-[#D4AF37]" size={20} />
                    </div>
                    <div>
                        <h2 className="font-bold text-lg tracking-tight">Evidence-backed inquiry</h2>
                        <p className="text-xs text-[#6B6B6B] font-mono uppercase tracking-widest">Answers cite graph evidence</p>
                    </div>
                </div>
                <button onClick={() => setShowChat(false)} className="p-2 hover:bg-[#E8E4D9] rounded-full transition-colors text-[#6B6B6B]" aria-label="Close">
                    <X size={20} />
                </button>
              </div>

              <div className="flex-1 overflow-y-auto p-8 space-y-8 scrollbar-thin scrollbar-thumb-slate-700 scrollbar-track-transparent">
                {messages.length === 0 && (
                    <div className="h-full flex flex-col items-center justify-center text-center space-y-6 opacity-50">
                        <div className="p-6 bg-[#E8E4D9]/80 rounded-full border border-[#4A4A4A]">
                            <KnowledgeGraphIcon size={48} className="text-[#2B2B2B]" />
                        </div>
                        <div>
                            <p className="text-lg font-medium text-[#2B2B2B]">Ready to Explore</p>
                            <p className="text-base text-[#2B2B2B]/40 mt-2 max-w-xs mx-auto">Ask a question about your case. Answers are grounded in your ingested documents—no evidence, no answer.</p>
                        </div>
                    </div>
                )}
                
                {messages.map((msg, idx) => {
                    const triplets = msg.triplets || [];
                    const leads = msg.leads || [];
                    const entityCount = new Set(
                        triplets
                            .flatMap((t) => [t.source, t.target])
                            .filter((val): val is string => typeof val === "string" && val.trim().length > 0)
                    ).size;

                    return (
                    <div key={idx} className={`flex flex-col ${msg.role === 'user' ? 'items-end' : 'items-start'} space-y-4`}>
                        
                        {/* Message Content */}
                        <div className={`relative max-w-[90%] ${msg.role === 'user' ? 'bg-[#8B1A1A] text-white p-5 rounded-sm rounded-tr-sm shadow-lg shadow-[#8B1A1A]/20' : ''}`}>
                            {msg.role === 'ai' ? (
                                <div className="prose prose-invert prose-sm max-w-none 
                                    prose-p:leading-relaxed prose-p:text-[#2B2B2B]
                                    prose-headings:text-[#D4AF37] prose-headings:font-semibold prose-headings:mt-6 prose-headings:mb-3
                                    prose-strong:text-[#2B2B2B] prose-strong:font-semibold
                                    prose-ul:my-4 prose-li:marker:text-[#D4AF37]/50
                                    prose-hr:border-[#4A4A4A] prose-hr:my-6">
                                    <ReactMarkdown remarkPlugins={[remarkGfm]}>
                                        {msg.content}
                                    </ReactMarkdown>
                                </div>
                            ) : (
                                <p className="text-sm font-medium leading-relaxed">{msg.content}</p>
                            )}
                        </div>

                        {/* Evidence & Leads Grid (Only for AI) */}
                        {msg.role === 'ai' && (
                            <div className="w-full space-y-4 pl-4 border-l-2 border-[#4A4A4A]/50 ml-2">
                                
                                {/* View Evidence link */}
                                {triplets.length > 0 && msg.explain && (
                                    <div className="flex items-center gap-3">
                                        <button
                                            onClick={() => openEvidencePage(msg)}
                                            className="graph-view-toggle flex items-center gap-2 px-3 py-1.5 rounded-md text-[10px] font-mono uppercase tracking-widest border bg-transparent border-[#4A4A4A]/30 text-[#6B6B6B] hover:text-[#D4AF37] hover:border-[#D4AF37]/50"
                                        >
                                            <KnowledgeGraphIcon size={12} />
                                            View Evidence
                                        </button>
                                        <span className="text-[9px] text-[#6B6B6B]/50 italic">
                                            {triplets.length} facts · {entityCount} entities
                                        </span>
                                    </div>
                                )}

                                {/* Leads Chips */}
                                {leads.length > 0 && (
                                    <div className="flex flex-wrap gap-2 animate-in fade-in slide-in-from-bottom-2 duration-500 delay-100">
                                        <div className="w-full text-[10px] font-mono uppercase tracking-widest text-[#6B6B6B] mb-1 flex items-center gap-2">
                                            <KnowledgeGraphIcon size={10} />
                                            Active Discovery Leads
                                        </div>
                                        {leads.map((lead, lidx) => (
                                            <div key={lidx} className="px-4 py-2 bg-[#D4AF37]/10 border border-[#D4AF37]/20 rounded-full text-xs text-[#D4AF37] hover:bg-[#D4AF37]/10 transition-colors cursor-default">
                                                {lead.name}
                                            </div>
                                        ))}
                                    </div>
                                )}

                                {/* Non-explain: simple facts grid */}
                                {triplets.length > 0 && !msg.explain && (
                                    <div className="animate-in fade-in slide-in-from-bottom-4 duration-500 delay-200">
                                        <div className="text-[10px] font-mono uppercase tracking-widest text-[#6B6B6B] mb-3 flex items-center gap-2">
                                            <Database size={10} />
                                            Verified Facts
                                        </div>
                                        <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
                                            {triplets.slice(0, 6).map((t: any, tidx: number) => (
                                                <div key={tidx} className="p-3 bg-[#E8E4D9]/50 border border-[#4A4A4A] rounded-lg hover:border-[#4A4A4A] transition-colors group">
                                                    <div className="flex items-center gap-2 text-[10px] text-[#6B6B6B] mb-1 font-mono">
                                                        <span className="group-hover:text-[#8B1A1A] transition-colors">{t.source}</span>
                                                        <span className="text-[#6B6B6B]">&rarr;</span>
                                                        <span className="group-hover:text-[#D4AF37] transition-colors">{t.target}</span>
                                                    </div>
                                                    <div className="text-xs text-[#6B6B6B] truncate font-medium">
                                                        {t.relation}
                                                    </div>
                                                </div>
                                            ))}
                                            {triplets.length > 6 && (
                                                <div className="p-3 border border-dashed border-[#4A4A4A] rounded-lg flex items-center justify-center text-xs text-[#6B6B6B] italic">
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
                        <div className="w-8 h-8 rounded-full bg-[#D4AF37]/10 flex items-center justify-center">
                            <Activity size={16} className="text-[#D4AF37] animate-spin" />
                        </div>
                        <div className="space-y-2">
                            <div className="h-2 w-24 bg-[#E8E4D9] rounded-full" />
                            <div className="h-2 w-32 bg-[#E8E4D9] rounded-full" />
                        </div>
                    </div>
                )}
                
                <div className="h-12" /> {/* Bottom Spacer */}
              </div>

              <form onSubmit={handleChat} className="p-6 bg-[#FCFAF2] border-t border-[#4A4A4A]">
                <div className="max-w-4xl mx-auto space-y-3">
                    <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-2 text-[10px] text-[#6B6B6B]">
                        <div className="flex items-center gap-3 flex-wrap">
                            <label className="flex items-center gap-2 font-mono uppercase tracking-widest cursor-pointer px-3 py-1.5 rounded-full border border-[#4A4A4A]/40 bg-[#E8E4D9]/60">
                                <input
                                    type="checkbox"
                                    checked={enableDetectiveInsights}
                                    onChange={(e) => setEnableDetectiveInsights(e.target.checked)}
                                    disabled={promptOnlyMode}
                                    className="accent-[#D4AF37]"
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
                            {promptOnlyMode ? "Chunk RAG only — no graph, no GNN. Compare grounding." : "Adds extra analysis time for richer explanations."}
                        </span>
                    </div>
                    <div className="relative">
                      <input 
                          type="text" 
                          value={query}
                          onChange={(e) => setQuery(e.target.value)}
                          placeholder="Ask a deep question..."
                          className="w-full bg-[#E8E4D9]/80 hover:bg-[#E8E4D9] border border-[#4A4A4A] rounded-2xl py-5 pl-6 pr-16 text-base text-[#2B2B2B]-200 placeholder:text-[#6B6B6B] focus:outline-none focus:ring-2 focus:ring-[#D4AF37]/50 focus:border-[#D4AF37] transition-all shadow-inner"
                      />
                      <button 
                          type="submit"
                          disabled={chatLoading || !query.trim()}
                          className="absolute right-3 top-1/2 -translate-y-1/2 p-2 bg-linear-to-r bg-[#D4AF37] hover:bg-[#B8941F] rounded-xl text-white transition-all disabled:opacity-50 disabled:cursor-not-allowed shadow-lg shadow-[#8B1A1A]/20 active:scale-95"
                      >
                          {chatLoading ? <RefreshCw size={20} className="animate-spin" /> : <Send size={20} />}
                      </button>
                    </div>
                    <p className="text-center text-[10px] text-[#6B6B6B] mt-3">
                        Unlike a standard chatbot: answers come only from validated facts in your documents. No evidence = no answer.
                    </p>
                </div>
              </form>
            </motion.div>
          </>
        )}
      </AnimatePresence>

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

