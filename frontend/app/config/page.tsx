"use client";

import React, { useState, useEffect } from "react";
import { 
  ArrowLeft, 
  ChevronDown,
  ChevronRight,
  Cpu, 
  Database, 
  Layers, 
  ShieldCheck, 
  Settings, 
  Network,
  Zap,
  Code,
  Server,
  GitBranch,
  Activity,
  AlertTriangle,
  CheckCircle2,
  TrendingUp,
  Box,
  Workflow,
  RefreshCw,
  BarChart3
} from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";
import axios from "axios";
import Link from "next/link";
import { API_BASE_URL } from "@/lib/api";
import StatCard from "@/components/StatCard";
import AuditItem from "@/components/AuditItem";
import { SkeletonConfigGrid } from "@/components/Skeleton";
import { formatAucRoc, formatScore } from "@/lib/utils";

type ConfigData = {
  models: any;
  pipeline: any;
  synthesis: any;
  neo4j: any;
  connections: any;
  audit: any;
  evaluation?: {
    gnn: { auc_roc?: number; mrr?: number; target_auc: number; target_mrr: number; auc_pass?: boolean; mrr_pass?: boolean; completed_at?: string };
    generative: { grounding_score?: number; faithfulness_score?: number; target_grounding: number; grounding_pass?: boolean; completed_at?: string };
  };
  tech_stack: any[];
  api_endpoints: any[];
  environment: any;
};

type StatsData = {
  entities?: number;
  relationships?: number;
  embedding_progress?: number;
  feature_complete?: number;
  graph_state?: string;
  graph_readiness?: { source_documents?: number; provenance_covered_nodes?: number; latest_ingestion_status?: string; latest_documents_processed?: number; latest_documents_failed?: number };
  audit_readiness?: { state?: string; audited_relationships?: number; total_relationships?: number; latest_audit_mode?: string; latest_auc_roc?: number };
  status?: string;
  current_task?: string;
  message?: string;
  research_kpis?: { gnn_auc_roc?: number; gnn_mrr?: number; grounding_score?: number; faithfulness_score?: number };
} | null;

export default function BackendConfigPage() {
  const [config, setConfig] = useState<ConfigData | null>(null);
  const [stats, setStats] = useState<StatsData>(null);
  const [loading, setLoading] = useState(true);
  const [activeTab, setActiveTab] = useState(0);
  const [auditing, setAuditing] = useState(false);

  const tabs = [
    { id: 0, label: "Overview", icon: BarChart3 },
    { id: 1, label: "Pipeline & Models", icon: Workflow },
    { id: 2, label: "Graph & Audit", icon: ShieldCheck },
    { id: 3, label: "System", icon: Settings }
  ];

  const fetchStats = React.useCallback(async () => {
    try {
      const response = await axios.get(`${API_BASE_URL}/stats`);
      setStats(response.data);
    } catch (err) {
      console.error("Failed to fetch stats:", err);
      setStats(null);
    }
  }, []);

  useEffect(() => {
    const fetchConfig = async () => {
      try {
        const [configRes, statsRes] = await Promise.all([
          axios.get(`${API_BASE_URL}/config`),
          axios.get(`${API_BASE_URL}/stats`).catch(() => ({ data: null }))
        ]);
        setConfig(configRes.data);
        setStats(statsRes?.data ?? null);
      } catch (err) {
        console.error("Failed to fetch config:", err);
      } finally {
        setLoading(false);
      }
    };

    fetchConfig();
  }, []);

  const handleAudit = async () => {
    try {
      setAuditing(true);
      await axios.post(`${API_BASE_URL}/audit`);
      await fetchStats();
    } catch (err) {
      console.error("Audit failed:", err);
    } finally {
      setAuditing(false);
    }
  };

  if (loading) {
    return (
      <div className="min-h-screen bg-[#F5F2E9] text-[#2B2B2B] p-8 paper-texture">
        <header className="max-w-7xl mx-auto mb-8">
          <Link href="/" className="inline-flex items-center gap-2 text-[#6B6B6B] hover:text-[#D4AF37] transition-colors mb-6 group">
            <ArrowLeft size={20} className="group-hover:-translate-x-1 transition-transform" />
            <span className="text-sm font-medium">Back to Dashboard</span>
          </Link>
          <div className="skeleton skeleton-text-lg w-64 mb-2" />
          <div className="skeleton skeleton-text w-96" />
        </header>
        <div className="max-w-7xl mx-auto">
          <SkeletonConfigGrid />
        </div>
      </div>
    );
  }

  if (!config) {
    return (
      <div className="min-h-screen bg-[#F5F2E9] text-[#2B2B2B] flex flex-col items-center justify-center paper-texture p-8">
        <div className="p-6 bg-[#E8E4D9]/80 rounded-full border border-[#4A4A4A]/30 mb-6">
          <AlertTriangle className="text-[#8B1A1A]" size={48} />
        </div>
        <h2 className="text-xl font-semibold text-[#2B2B2B] mb-2" style={{fontFamily: 'EB Garamond, serif'}}>
          Configuration Unavailable
        </h2>
        <p className="text-sm text-[#6B6B6B] max-w-md text-center mb-6">
          Could not load backend configuration. Ensure the backend is running on port 8000 and try again.
        </p>
        <Link href="/" className="inline-flex items-center gap-2 px-5 py-2.5 bg-[#D4AF37] hover:bg-[#B8941F] text-[#2B2B2B] rounded-sm font-semibold shadow-lg transition-all text-sm">
          Back to Dashboard
        </Link>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-[#F5F2E9] text-[#2B2B2B] p-8 paper-texture">
      {/* Header */}
      <header className="max-w-7xl mx-auto mb-8">
        <Link href="/" className="inline-flex items-center gap-2 text-[#6B6B6B] hover:text-[#D4AF37] transition-colors mb-6 group">
          <ArrowLeft size={20} className="group-hover:-translate-x-1 transition-transform" />
          <span className="font-medium">Back to Archive</span>
        </Link>
        
        <motion.div 
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          className="flex items-center gap-4"
        >
          <div className="p-3 bg-[#8B1A1A] rounded-sm shadow-lg border border-[#4A4A4A]">
            <Settings className="text-[#F5F2E9]" size={32} />
          </div>
          <div>
            <h1 className="text-3xl font-bold tracking-tight gradient-text" style={{fontFamily: 'EB Garamond, serif'}}>
              Archive Configuration
            </h1>
            <p className="text-[#6B6B6B] text-sm font-mono uppercase tracking-widest mt-1">
              Technical Specifications & System Details
            </p>
          </div>
        </motion.div>
      </header>

      {/* Tab Navigation */}
      <div className="max-w-7xl mx-auto mb-8" role="tablist" aria-label="Configuration sections">
        <div className="glass rounded-sm p-2 flex gap-2 overflow-x-auto border border-[#4A4A4A] cut-paper">
          {tabs.map((tab) => {
            const Icon = tab.icon;
            const isSelected = activeTab === tab.id;
            return (
              <button
                key={tab.id}
                role="tab"
                aria-selected={isSelected}
                aria-controls={`panel-${tab.id}`}
                id={`tab-${tab.id}`}
                tabIndex={isSelected ? 0 : -1}
                onClick={() => setActiveTab(tab.id)}
                className={`flex items-center gap-2 px-6 py-3 rounded-sm font-medium transition-all whitespace-nowrap border ${
                  isSelected
                    ? "bg-[#D4AF37] text-[#2B2B2B] shadow-lg border-[#4A4A4A] font-semibold"
                    : "text-[#6B6B6B] hover:text-[#2B2B2B] hover:bg-[#E8E4D9] border-transparent"
                }`}
              >
                <Icon size={18} />
                <span>{tab.label}</span>
              </button>
            );
          })}
        </div>
      </div>

      {/* Tab Content */}
      <main className="max-w-7xl mx-auto">
        <AnimatePresence mode="wait">
          <motion.div
            key={activeTab}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
            transition={{ duration: 0.3 }}
          >
            {activeTab === 0 && <OverviewTab stats={stats} config={config} onRefresh={fetchStats} onAudit={handleAudit} auditing={auditing} />}
            {activeTab === 1 && <PipelineModelsTab pipeline={config.pipeline} models={config.models} />}
            {activeTab === 2 && <GraphAuditTab connections={config.connections} pipeline={config.pipeline} neo4j={config.neo4j} audit={config.audit} evaluation={config.evaluation} />}
            {activeTab === 3 && <SystemTab techStack={config.tech_stack} endpoints={config.api_endpoints} environment={config.environment} synthesis={config.synthesis} />}
          </motion.div>
        </AnimatePresence>
      </main>
    </div>
  );
}

// Tab 0: Overview — system health at a glance
function OverviewTab({ stats, config, onRefresh, onAudit, auditing }: { stats: StatsData; config: ConfigData; onRefresh: () => void; onAudit: () => void; auditing: boolean }) {
  const graphState = stats?.graph_state;
  const graphReadiness = stats?.graph_readiness;
  const auditReadiness = stats?.audit_readiness;
  const canRunAudit = graphState === "evidence_ready_graph" && !auditing;

  if (!stats) {
    return (
      <div className="glass rounded-sm p-8 text-center">
        <AlertTriangle className="text-amber-400 mx-auto mb-4" size={32} />
        <p className="text-[#6B6B6B]">Could not load system status. Is the backend running?</p>
        <button onClick={onRefresh} className="mt-4 px-4 py-2 bg-[#D4AF37]/20 text-[#2B2B2B] rounded-sm border border-[#D4AF37]/40 hover:bg-[#D4AF37]/30 transition-colors">
          Retry
        </button>
      </div>
    );
  }

  const hasBackendError = stats?.status === "error" || stats?.status === "unavailable";

  return (
    <div className="space-y-6">
      {hasBackendError && stats?.message && (
        <div className="glass rounded-sm p-4 border-l-4 border-amber-500 bg-amber-500/5 flex items-start gap-3">
          <AlertTriangle className="text-amber-500 shrink-0" size={20} />
          <div>
            <p className="font-semibold text-amber-800">System status unavailable</p>
            <p className="text-sm text-[#6B6B6B] mt-1">{stats.message}</p>
          </div>
        </div>
      )}

      {/* Top row: key numbers + refresh */}
      <div className="flex justify-between items-start">
        <div className="grid grid-cols-3 gap-4 flex-1 mr-4">
          <StatCard label="Entities" value={stats?.entities ?? 0} icon={<Database className="text-[#D4AF37]" size={24} />} subtext={graphState === "evidence_ready_graph" ? "Graph ready" : graphState ?? "—"} delay={0} />
          <StatCard label="Relationships" value={stats?.relationships ?? 0} icon={<Layers className="text-[#8B1A1A]" size={24} />} subtext={`${Math.round(stats?.embedding_progress ?? 0)}% embedded`} delay={0.05} />
          <StatCard label="Audit" value={(auditReadiness?.state ?? "absent").toUpperCase()} icon={<ShieldCheck className="text-[#3A5A40]" size={24} />} subtext={formatAucRoc(auditReadiness?.latest_auc_roc) !== "—" ? `AUC-ROC: ${formatAucRoc(auditReadiness?.latest_auc_roc)}` : "Not run yet"} delay={0.1} />
        </div>
        <button onClick={onRefresh} className="flex items-center gap-2 px-3 py-2 bg-[#E8E4D9] hover:bg-[#D4AF37]/20 rounded-sm border border-[#4A4A4A] text-[#2B2B2B] transition-colors text-sm shrink-0">
          <RefreshCw size={14} />
          Refresh
        </button>
      </div>

      {/* Research KPIs — compact */}
      {stats?.research_kpis && (stats.research_kpis.gnn_auc_roc != null || stats.research_kpis.grounding_score != null) && (
        <div className="glass rounded-sm p-5 border border-[#4A4A4A]/50">
          <h3 className="text-[10px] font-mono uppercase tracking-widest text-[#6B6B6B] mb-3">Research KPIs</h3>
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
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
        </div>
      )}

      {/* Encoding progress + Audit trigger */}
      <div className="glass rounded-sm p-5 border border-[#4A4A4A]">
        <div className="flex justify-between items-center mb-3">
          <h3 className="flex items-center gap-2 text-sm font-semibold">
            <Activity size={16} className="text-[#D4AF37]" />
            Semantic Encoding
          </h3>
          <div className="flex items-center gap-3">
            <span className="text-xs font-mono text-[#6B6B6B]">{stats?.feature_complete ?? 0} / {stats?.entities ?? 0}</span>
            <button
              onClick={onAudit}
              disabled={!canRunAudit}
              className={`flex items-center gap-2 px-3 py-1 rounded-sm text-[10px] uppercase font-bold tracking-widest transition-all border ${auditing ? "bg-[#E8E4D9] text-[#6B6B6B] border-[#4A4A4A]" : "bg-[#8B1A1A]/10 text-[#8B1A1A] border-[#8B1A1A]/30 hover:bg-[#8B1A1A]/20"}`}
            >
              {auditing ? <RefreshCw size={10} className="animate-spin" /> : <ShieldCheck size={10} />}
              {auditing ? "Auditing..." : "Run Audit"}
            </button>
          </div>
        </div>
        <div className="w-full bg-[#E8E4D9] rounded-sm h-2.5 overflow-hidden border border-[#4A4A4A]">
          <motion.div
            initial={{ width: 0 }}
            animate={{ width: `${stats?.embedding_progress ?? 0}%` }}
            className="h-full bg-linear-to-r from-[#D4AF37] to-[#B8941F]"
          />
        </div>
      </div>

      {/* Readiness — two columns */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div className="glass rounded-sm p-5">
          <h4 className="text-[10px] font-mono uppercase tracking-widest text-[#6B6B6B] mb-3">Graph Readiness</h4>
          <p className="font-semibold text-sm text-[#2B2B2B] mb-2">{graphState ?? "unknown"}</p>
          <div className="space-y-1 text-xs text-[#6B6B6B]">
            <p>Documents: {graphReadiness?.source_documents ?? 0} | Provenance: {graphReadiness?.provenance_covered_nodes ?? 0}</p>
            <p>Ingestion: {graphReadiness?.latest_ingestion_status ?? "unknown"} | Processed: {graphReadiness?.latest_documents_processed ?? 0}</p>
          </div>
        </div>
        <div className="glass rounded-sm p-5">
          <h4 className="text-[10px] font-mono uppercase tracking-widest text-[#6B6B6B] mb-3">Audit Readiness</h4>
          <p className="font-semibold text-sm text-[#2B2B2B] mb-2">{auditReadiness?.state ?? "unknown"}</p>
          <div className="space-y-1 text-xs text-[#6B6B6B]">
            <p>Coverage: {auditReadiness?.audited_relationships ?? 0} / {auditReadiness?.total_relationships ?? 0}</p>
            <p>Mode: {auditReadiness?.latest_audit_mode ?? "—"} | AUC-ROC: {formatAucRoc(auditReadiness?.latest_auc_roc)}</p>
          </div>
        </div>
      </div>

      {/* Checklist */}
      <div className="glass rounded-sm p-5">
        <h4 className="text-[10px] font-mono uppercase tracking-widest text-[#6B6B6B] mb-3">System Checklist</h4>
        <div className="space-y-3">
          <AuditItem label="Neo4j Connected" success={stats?.status === "healthy"} />
          <AuditItem label="Graph Evidence Ready" success={graphState === "evidence_ready_graph"} />
          <AuditItem label="Document Provenance" success={(graphReadiness?.provenance_covered_nodes ?? 0) > 0} />
          <AuditItem label="Vector Coverage" success={stats?.embedding_progress === 100} />
          <AuditItem label="CompGCN Audit" success={auditReadiness?.state === "ready"} />
        </div>
      </div>
    </div>
  );
}

// Tab 1: Pipeline & Models (merged, collapsible)
function PipelineModelsTab({ pipeline, models }: { pipeline: any; models: any }) {
  const [openSection, setOpenSection] = useState<string>("llm");
  const hasModels = models?.llm && models?.embedder && models?.gnn;

  return (
    <div className="space-y-6">
      <div className="glass rounded-sm p-6">
        <h2 className="text-2xl font-bold mb-2 flex items-center gap-3">
          <Workflow className="text-[#D4AF37]" size={28} />
          Pipeline & Models
        </h2>
        <p className="text-[#6B6B6B] mb-6">End-to-end flow from PDF ingestion to evaluation</p>
        <div className="flex flex-wrap gap-2 mb-6">
          {pipeline?.stages?.map((stage: any, idx: number) => (
            <span key={idx} className="px-3 py-1.5 bg-[#E8E4D9] border border-[#4A4A4A] rounded-sm text-xs font-medium text-[#2B2B2B]">
              {stage.order}. {stage.name}
            </span>
          ))}
        </div>

        {/* Mermaid in details */}
        <details className="mt-4">
          <summary className="cursor-pointer text-sm font-medium text-[#6B6B6B] hover:text-[#D4AF37]">Copy architecture diagram (Mermaid)</summary>
          <pre className="mt-2 bg-[#2B2B2B] text-[#E8E4D9] p-4 rounded-sm text-xs overflow-x-auto font-mono">
{`flowchart TB
  PDF[PDF Documents] --> S1
  S1[1. PDF Ingestion] --> S2[2. Entity Extraction]
  S2 --> S3[3. Graph Storage]
  S3 --> Neo4j[(Neo4j)]
  Neo4j --> S4[4. Vector Embedding]
  Neo4j --> S5[5. Integrity Validation]
  S5 --> S6[6. Synthesis]
  S6 --> S7[7. Evaluation]
  S7 --> KPIs[AUC-ROC, MRR, Grounding, Faithfulness]`}
          </pre>
        </details>
      </div>

      {!hasModels ? (
        <div className="glass rounded-sm p-8 text-center">
          <p className="text-[#6B6B6B]">Model configuration not available</p>
        </div>
      ) : (
        <>
          {/* LLM - default open */}
          <AccordionSection
            id="llm"
            title={models.llm.name}
            subtitle={models.llm.model}
            isOpen={openSection === "llm"}
            onToggle={() => setOpenSection(openSection === "llm" ? "" : "llm")}
            icon={<Zap className="text-[#D4AF37]" size={24} />}
          >
            <p className="text-sm text-[#6B6B6B] mb-4">{models.llm.purpose}</p>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {models.llm.hyperparameters && (
                <div className="space-y-2">
                  <h4 className="text-xs font-semibold text-[#D4AF37] uppercase">Hyperparameters</h4>
                  <DetailRow label="Temperature" value={models.llm.hyperparameters.temperature} />
                  <DetailRow label="Max Output Tokens" value={models.llm.hyperparameters.max_output_tokens?.toLocaleString()} />
                </div>
              )}
              {models.llm.extraction_config && (
                <div className="space-y-2">
                  <h4 className="text-xs font-semibold text-[#D4AF37] uppercase">Extraction</h4>
                  <DetailRow label="Entity Types" value={Array.isArray(models.llm.extraction_config.entity_types) ? models.llm.extraction_config.entity_types.slice(0, 3).join(", ") + "…" : "-"} />
                </div>
              )}
            </div>
          </AccordionSection>

          {/* Embedder - collapsed */}
          <AccordionSection
            id="embedder"
            title={models.embedder.name}
            subtitle={models.embedder.full_model_name}
            isOpen={openSection === "embedder"}
            onToggle={() => setOpenSection(openSection === "embedder" ? "" : "embedder")}
            icon={<Layers className="text-[#8B1A1A]" size={24} />}
          >
            <p className="text-sm text-[#6B6B6B] mb-4">{models.embedder.purpose}</p>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {models.embedder.architecture && (
                <div className="space-y-2">
                  <DetailRow label="Hidden Size" value={models.embedder.architecture.hidden_size} />
                  <DetailRow label="Output Dim" value={models.embedder.hyperparameters?.output_dim} />
                </div>
              )}
            </div>
          </AccordionSection>

          {/* GNN - default open per plan */}
          <AccordionSection
            id="gnn"
            title={models.gnn.name}
            subtitle="Relationship plausibility scoring"
            isOpen={openSection === "gnn"}
            onToggle={() => setOpenSection(openSection === "gnn" ? "" : "gnn")}
            icon={<ShieldCheck className="text-[#3A5A40]" size={24} />}
          >
            <p className="text-sm text-[#6B6B6B] mb-4">{models.gnn.purpose}</p>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {models.gnn.training && (
                <div className="space-y-2">
                  <DetailRow label="Epochs" value={models.gnn.training.epochs} />
                  <DetailRow label="Evaluation" value={models.gnn.training.evaluation} />
                </div>
              )}
            </div>
          </AccordionSection>

          {/* Retriever - collapsed */}
          {models.retriever && (
            <AccordionSection
              id="retriever"
              title={models.retriever.name}
              subtitle={models.retriever.type}
              isOpen={openSection === "retriever"}
              onToggle={() => setOpenSection(openSection === "retriever" ? "" : "retriever")}
              icon={<Network className="text-[#6B6B6B]" size={24} />}
            >
              <p className="text-sm text-[#6B6B6B]">{models.retriever.purpose}</p>
            </AccordionSection>
          )}

          {/* Ingestion - collapsed */}
          {models.ingestion && (
            <AccordionSection
              id="ingestion"
              title={models.ingestion.name}
              subtitle={models.ingestion.pipeline}
              isOpen={openSection === "ingestion"}
              onToggle={() => setOpenSection(openSection === "ingestion" ? "" : "ingestion")}
              icon={<Box className="text-[#6B6B6B]" size={24} />}
            >
              <p className="text-sm text-[#6B6B6B]">{models.ingestion.purpose}</p>
            </AccordionSection>
          )}
        </>
      )}
    </div>
  );
}

function AccordionSection({ id, title, subtitle, isOpen, onToggle, icon, children }: { id: string; title: string; subtitle?: string; isOpen: boolean; onToggle: () => void; icon: React.ReactNode; children: React.ReactNode }) {
  return (
    <div className="glass rounded-sm border border-[#4A4A4A]/50 overflow-hidden">
      <button
        type="button"
        onClick={onToggle}
        className="w-full flex items-center justify-between gap-3 p-4 text-left hover:bg-[#E8E4D9]/50 transition-colors"
      >
        <div className="flex items-center gap-3">
          <div className="p-2 bg-[#F5F2E9] rounded-sm">{icon}</div>
          <div>
            <h3 className="font-semibold text-[#2B2B2B]">{title}</h3>
            {subtitle && <p className="text-xs text-[#6B6B6B] font-mono">{subtitle}</p>}
          </div>
        </div>
        {isOpen ? <ChevronDown size={18} className="text-[#6B6B6B]" /> : <ChevronRight size={18} className="text-[#6B6B6B]" />}
      </button>
      {isOpen && <div className="p-4 pt-0 border-t border-[#4A4A4A]/30">{children}</div>}
    </div>
  );
}

// Tab 2: Graph & Audit (merged Knowledge Graph + GNN Audit + Evaluation)
function GraphAuditTab({ connections, pipeline, neo4j, audit, evaluation }: { connections: any; pipeline: any; neo4j: any; audit: any; evaluation?: ConfigData["evaluation"] }) {
  const maxCount = connections?.relationship_distribution?.[0]?.count || 1;
  const isConnected = neo4j?.driver_status === "Connected";
  const results = audit?.results ?? [];
  const threshold = audit?.low_confidence_threshold ?? 0.95;
  const lowConfidence = results.filter((r: any) => r.score < threshold);
  const highConfidence = results.filter((r: any) => r.score >= threshold);
  const gnn = evaluation?.gnn;
  const gen = evaluation?.generative;

  return (
    <div className="space-y-6">
      {!isConnected && (
        <div className="glass rounded-sm p-4 border-l-4 border-amber-500 bg-amber-500/5">
          <div className="flex items-center gap-3">
            <AlertTriangle className="text-amber-400" size={20} />
            <p className="font-semibold text-amber-400">Neo4j Database Disconnected</p>
          </div>
        </div>
      )}

      {/* Neo4j + Schema */}
      <div className="glass rounded-sm p-6">
        <h2 className="text-xl font-bold mb-4 flex items-center gap-3">
          <Database className="text-emerald-400" size={24} />
          Graph Database
        </h2>
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-6">
          <InfoCard label="Type" value={neo4j?.type} />
          <InfoCard label="Status" value={neo4j?.driver_status} />
        </div>
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <div>
            <h3 className="text-sm font-semibold text-[#6B6B6B] uppercase mb-2">Node Types</h3>
            <div className="flex flex-wrap gap-2">
              {pipeline?.schema?.node_types?.map((type: string, idx: number) => (
                <span key={idx} className="px-2 py-1 bg-[#D4AF37]/10 text-[#D4AF37] rounded text-xs">{type}</span>
              ))}
            </div>
          </div>
          <div>
            <h3 className="text-sm font-semibold text-[#6B6B6B] uppercase mb-2">Relationship Types</h3>
            <div className="flex flex-wrap gap-2">
              {pipeline?.schema?.relationship_types?.map((type: string, idx: number) => (
                <span key={idx} className="px-2 py-1 bg-[#8B1A1A]/10 text-[#8B1A1A] rounded text-xs">{type}</span>
              ))}
            </div>
          </div>
        </div>
      </div>

      {/* Relationship Distribution */}
      {connections?.relationship_distribution?.length > 0 && (
        <div className="glass rounded-sm p-6">
          <h3 className="font-semibold mb-4">Relationship Distribution</h3>
          <div className="space-y-2">
            {connections.relationship_distribution.slice(0, 10).map((rel: any, idx: number) => (
              <div key={idx} className="flex justify-between items-center text-sm">
                <span className="font-mono">{rel.type}</span>
                <span className="text-[#6B6B6B]">{rel.count?.toLocaleString()}</span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Evaluation KPIs (GNN metrics / Generative metrics - no H1/H2/H3) */}
      {evaluation && (
        <div className="glass rounded-sm p-6">
          <h3 className="font-semibold mb-4">Evaluation Metrics</h3>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
            {gnn?.auc_roc != null && <InfoCard label="GNN AUC-ROC" value={formatAucRoc(gnn.auc_roc)} />}
            {gnn?.mrr != null && <InfoCard label="GNN MRR" value={formatAucRoc(gnn.mrr)} />}
            {gen?.grounding_score != null && <InfoCard label="Grounding" value={formatScore(gen.grounding_score)} />}
            {gen?.faithfulness_score != null && <InfoCard label="Faithfulness" value={formatScore(gen.faithfulness_score)} />}
          </div>
        </div>
      )}

      {/* Audit Results Summary + Low Confidence Table */}
      <div className="glass rounded-sm p-6">
        <h3 className="font-semibold mb-4 flex items-center gap-2">
          <ShieldCheck size={18} className="text-[#D4AF37]" />
          GNN Audit Results
        </h3>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
          <InfoCard label="Total Audited" value={audit?.total_audited ?? 0} />
          <InfoCard label="High Integrity" value={highConfidence.length} />
          <InfoCard label="Flagged Edges" value={lowConfidence.length} />
        </div>
        {lowConfidence.length > 0 && (
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-[#4A4A4A]">
                  <th className="text-left py-2 px-3 text-[#6B6B6B]">Source</th>
                  <th className="text-left py-2 px-3 text-[#6B6B6B]">Relation</th>
                  <th className="text-left py-2 px-3 text-[#6B6B6B]">Target</th>
                  <th className="text-right py-2 px-3 text-[#6B6B6B]">Score</th>
                </tr>
              </thead>
              <tbody>
                {lowConfidence.slice(0, 15).map((edge: any, idx: number) => (
                  <tr key={idx} className="border-b border-[#4A4A4A]/50">
                    <td className="py-2 px-3 font-mono text-xs">{edge.source}</td>
                    <td className="py-2 px-3 text-[#6B6B6B]">{edge.relation}</td>
                    <td className="py-2 px-3 font-mono text-xs">{edge.target}</td>
                    <td className="py-2 px-3 text-right text-amber-500 font-mono">{edge.score?.toFixed(3)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
}

// Tab 3: System
function SystemTab({ techStack, endpoints, environment, synthesis }: { techStack: any[]; endpoints: any[]; environment: any; synthesis: any }) {
  return (
    <div className="space-y-6">
      {/* Environment */}
      <div className="glass rounded-sm p-6">
        <h2 className="text-2xl font-bold mb-4 flex items-center gap-3">
          <Server className="text-[#D4AF37]" size={28} />
          Environment
        </h2>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <InfoCard label="API Port" value={environment.port} />
          <InfoCard label="Debug Mode" value={environment.debug ? "Enabled" : "Disabled"} />
          <InfoCard label="Documents Path" value={environment.docs_dir} />
        </div>
      </div>

      {/* Synthesis Config */}
      <div className="glass rounded-sm p-6">
        <h3 className="font-semibold text-lg mb-4 flex items-center gap-2">
          <Zap className="text-[#8B1A1A]" size={20} />
          Synthesis Layer Configuration
        </h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <DetailRow label="Persona" value={synthesis.persona} />
          <DetailRow label="Temperature" value={synthesis.temperature} />
          <DetailRow label="Response Structure" value={synthesis.response_structure} />
          <DetailRow label="Strategy" value={synthesis.strategy} />
        </div>
        <div className="mt-4">
          <p className="text-xs text-[#6B6B6B] mb-2">Features:</p>
          <div className="flex flex-wrap gap-2">
            {synthesis.features.map((feature: string, idx: number) => (
              <span key={idx} className="text-xs bg-[#8B1A1A]/10 text-[#8B1A1A] px-3 py-1 rounded-sm border border-[#8B1A1A]/20">
                {feature}
              </span>
            ))}
          </div>
        </div>
      </div>

      {/* Tech Stack */}
      <div className="glass rounded-sm p-6">
        <h3 className="font-semibold text-lg mb-4 flex items-center gap-2">
          <Code className="text-emerald-400" size={20} />
          Technology Stack
        </h3>
        <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-3">
          {techStack.map((tech, idx) => (
            <motion.div
              key={idx}
              initial={{ opacity: 0, scale: 0.9 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ delay: idx * 0.03 }}
              className="bg-[#E8E4D9]/80 p-3 rounded-sm border border-[#4A4A4A] hover:border-[#D4AF37]/50 transition-all group"
            >
              <h4 className="font-semibold text-sm mb-1 group-hover:text-[#D4AF37] transition-colors">{tech.name}</h4>
              <p className="text-xs text-[#6B6B6B] mb-1">{tech.version}</p>
              <p className="text-xs text-[#6B6B6B]">{tech.role}</p>
            </motion.div>
          ))}
        </div>
      </div>

      {/* API Endpoints */}
      <div className="glass rounded-sm p-6">
        <h3 className="font-semibold text-lg mb-4 flex items-center gap-2">
          <GitBranch className="text-amber-400" size={20} />
          API Endpoints Directory
        </h3>
        <div className="space-y-2">
          {endpoints.map((endpoint, idx) => (
            <motion.div
              key={idx}
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: idx * 0.02 }}
              className="flex items-center gap-4 p-3 bg-[#E8E4D9]/50 rounded-sm hover:bg-[#E8E4D9]/80 transition-colors"
            >
              <span className={`px-2 py-1 rounded text-xs font-mono font-bold ${
                endpoint.method === "GET" ? "bg-emerald-500/10 text-emerald-400" :
                endpoint.method === "POST" ? "bg-[#D4AF37]/10 text-[#D4AF37]" :
                "bg-rose-500/10 text-rose-400"
              }`}>
                {endpoint.method}
              </span>
              <code className="text-sm font-mono text-[#2B2B2B] flex-1">{endpoint.path}</code>
              <span className="text-xs text-[#6B6B6B]">{endpoint.description}</span>
            </motion.div>
          ))}
        </div>
      </div>
    </div>
  );
}

// Helper Components
function InfoCard({ label, value, icon }: { label: string; value: any; icon?: React.ReactNode }) {
  return (
    <div className="bg-[#E8E4D9]/80 p-4 rounded-sm border border-[#4A4A4A]">
      <div className="flex items-center gap-2 mb-1">
        {icon}
        <p className="text-xs text-[#6B6B6B]">{label}</p>
      </div>
      <p className="font-semibold text-sm truncate">{value}</p>
    </div>
  );
}

function DetailRow({ label, value }: { label: string; value: any }) {
  return (
    <div className="flex justify-between items-center py-2 border-b border-[#4A4A4A]">
      <span className="text-xs text-[#6B6B6B]">{label}</span>
      <span className="text-sm font-medium">{value}</span>
    </div>
  );
}
