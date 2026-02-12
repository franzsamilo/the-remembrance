"use client";

import React, { useState, useEffect } from "react";
import { 
  Activity, 
  Database, 
  FileText, 
  Network, 
  Play, 
  RefreshCw, 
  ShieldCheck, 
  Zap, 
  AlertCircle,
  Layers,
  Cpu,
  Trash2,
  Upload
} from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";
import axios from "axios";

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export default function FrameworkDashboard() {
  const [stats, setStats] = useState<any>(null);
  const [documents, setDocuments] = useState<string[]>([]);
  const [loading, setLoading] = useState(true);
  const [ingesting, setIngesting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const fetchStats = async () => {
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
  };

  const fetchDocs = async () => {
    try {
      const response = await axios.get(`${API_BASE_URL}/documents`);
      setDocuments(response.data.documents);
    } catch (err) {
      console.error("Error fetching docs:", err);
    }
  };

  useEffect(() => {
    fetchStats();
    fetchDocs();
    const interval = setInterval(fetchStats, 5000);
    return () => clearInterval(interval);
  }, []);

  const triggerIngestion = async () => {
    setIngesting(true);
    try {
      await axios.post(`${API_BASE_URL}/ingest`);
      setTimeout(fetchStats, 2000);
    } catch (err) {
      console.error("Error triggering ingestion:", err);
    } finally {
      setTimeout(() => setIngesting(false), 3000);
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
    } catch (err) {
        console.error("Upload failed:", err);
        setError("File upload failed. Check backend logs.");
    } finally {
        setLoading(false);
    }
  };

  const handleReset = async () => {
    if (!confirm("CRITICAL WARNING: This will permanently wipe all nodes and relationships from your Neo4j instance. Continue?")) return;
    
    try {
        setLoading(true);
        await axios.post(`${API_BASE_URL}/reset`);
        await fetchStats();
        await fetchDocs();
    } catch (err) {
        console.error("Reset failed:", err);
        setError("Failed to clear database.");
    } finally {
        setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-[#0f172a] text-slate-200 p-8 font-sans">
      {/* Header */}
      <header className="max-w-7xl mx-auto flex justify-between items-center mb-12">
        <motion.div 
          initial={{ opacity: 0, x: -20 }}
          animate={{ opacity: 1, x: 0 }}
          className="flex items-center gap-3"
        >
          <div className="p-2 bg-sky-500 rounded-lg glow">
            <Network className="text-slate-900" size={28} />
          </div>
          <div>
            <h1 className="text-2xl font-bold tracking-tight">THE REMEMBRANCE</h1>
            <p className="text-sky-400 text-xs font-mono uppercase tracking-widest">GNN Audit Framework</p>
          </div>
        </motion.div>

        <div className="flex items-center gap-6">
          <div className={`flex items-center gap-2 px-3 py-1 rounded-full text-xs font-medium border ${stats?.status === 'healthy' ? 'bg-emerald-500/10 border-emerald-500/50 text-emerald-400' : 'bg-rose-500/10 border-rose-500/50 text-rose-400'}`}>
            <div className={`w-2 h-2 rounded-full ${stats?.status === 'healthy' ? 'bg-emerald-500 animate-pulse' : 'bg-rose-500'}`} />
            {stats?.status === 'healthy' ? 'Aura Online' : 'Aura Offline'}
          </div>
          <button 
            onClick={handleReset}
            className="flex items-center gap-2 px-3 py-1 rounded-full text-xs font-medium border bg-rose-500/10 border-rose-500/50 text-rose-400 hover:bg-rose-500/20 transition-all"
            title="Wipe Graph"
          >
            <Trash2 size={14} />
            Reset Framework
          </button>

          <button 
            onClick={fetchStats}
            className="p-2 hover:bg-slate-800 rounded-full transition-colors"
          >
            <RefreshCw size={20} className={loading ? "animate-spin" : ""} />
          </button>
        </div>
      </header>

      <main className="max-w-7xl mx-auto grid grid-cols-1 lg:grid-cols-3 gap-8">
        
        {/* Left Column: System Stats */}
        <div className="lg:col-span-2 space-y-8">
          
          {/* Main Metrics */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            <StatCard 
              label="Total Entity Nodes" 
              value={stats?.entities || 0} 
              icon={<Database className="text-sky-400" />}
              subtext="EP-RSR Core Knowledge"
              delay={0.1}
            />
            <StatCard 
              label="Triplet Facts" 
              value={stats?.relationships || 0} 
              icon={<Zap className="text-amber-400" />}
              subtext="Extracted Relationships"
              delay={0.2}
            />
            <StatCard 
              label="Vector Coverage" 
              value={`${Math.round(stats?.embedding_progress || 0)}%`} 
              icon={<Cpu className="text-rose-400" />}
              subtext="GNN-Ready Nodes"
              delay={0.3}
            />
          </div>

          {/* GNN Readiness Progress Bar */}
          <motion.div 
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.4 }}
            className="glass rounded-2xl p-6 glow"
          >
            <div className="flex justify-between items-center mb-4">
              <h3 className="flex items-center gap-2 font-semibold">
                <Layers size={18} className="text-sky-400" />
                Mathematic Weighting Progress (DistilBERT)
              </h3>
              <span className="text-xs font-mono text-slate-400">{stats?.feature_complete || 0} / {stats?.entities || 0} Nodes</span>
            </div>
            <div className="w-full bg-slate-800 rounded-full h-3 mb-2 overflow-hidden">
              <motion.div 
                initial={{ width: 0 }}
                animate={{ width: `${stats?.embedding_progress || 0}%` }}
                className="h-full bg-gradient-to-r from-sky-600 to-indigo-500"
              />
            </div>
            <p className="text-xs text-slate-400">
              {stats?.embedding_progress === 100 
                ? "Framework is mathematically prepared for Phase 2 GNN Logic." 
                : "Embedding cold-start vectors to satisfy CompGCN semantic formula..."}
            </p>
          </motion.div>

          {/* Ingestion Console */}
          <motion.div 
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.5 }}
            className="glass rounded-2xl overflow-hidden"
          >
            <div className="p-6 border-b border-slate-800 flex justify-between items-center">
              <h3 className="flex items-center gap-2 font-semibold">
                <FileText size={18} className="text-sky-400" />
                Inference Source Library
              </h3>
              
              <div className="flex items-center gap-4">
                <label className="flex items-center gap-2 px-4 py-2 rounded-lg font-medium bg-slate-800 hover:bg-slate-700 text-slate-300 transition-all cursor-pointer">
                    <Upload size={16} />
                    Upload PDF
                    <input type="file" accept=".pdf" className="hidden" onChange={handleUpload} />
                </label>

                <button 
                    onClick={triggerIngestion}
                    disabled={ingesting}
                    className={`flex items-center gap-2 px-4 py-2 rounded-lg font-medium transition-all ${ingesting ? "bg-slate-800 text-slate-500 cursor-not-allowed" : "bg-sky-600 hover:bg-sky-500 text-white shadow-lg shadow-sky-900/20"}`}
                >
                    {ingesting ? (
                    <>
                        <RefreshCw size={16} className="animate-spin" />
                        Processing...
                    </>
                    ) : (
                    <>
                        <Play size={16} fill="currentColor" />
                        Run Pipeline
                    </>
                    )}
                </button>
              </div>
            </div>
            <div className="divide-y divide-slate-800">
              {documents.length > 0 ? documents.map((doc, idx) => (
                <div key={idx} className="p-4 flex items-center justify-between hover:bg-slate-800/50 transition-colors">
                  <div className="flex items-center gap-3">
                    <div className="p-2 bg-slate-800 rounded-md">
                      <FileText size={16} className="text-slate-400" />
                    </div>
                    <span className="text-sm">{doc}</span>
                  </div>
                  <div className="text-[10px] font-mono uppercase bg-slate-800 px-2 py-1 rounded text-slate-500">
                    Source PDF
                  </div>
                </div>
              )) : (
                <div className="p-8 text-center text-slate-500 italic text-sm">
                  No documents found in backend/documents/
                </div>
              )}
            </div>
          </motion.div>
        </div>

        {/* Right Column: Alerts & Details */}
        <div className="space-y-8">
          <motion.div 
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: 0.6 }}
            className="glass rounded-2xl p-6"
          >
            <h3 className="flex items-center gap-2 font-semibold mb-6">
              <ShieldCheck size={18} className="text-emerald-400" />
              Audit Readiness Status
            </h3>
            
            <div className="space-y-4">
              <AuditItem label="Neo4j Aura Protocol" success={stats?.status === 'healthy'} />
              <AuditItem label="EP-RSR Schema Validation" success={stats?.entities > 0} />
              <AuditItem label="High-Dim Vector Coverage" success={stats?.embedding_progress > 95} />
              <AuditItem label="GNN Topology Map" success={stats?.relationships > 0} />
            </div>

            {error && (
              <div className="mt-8 flex items-start gap-3 p-4 bg-rose-500/10 border border-rose-500/20 rounded-xl">
                <AlertCircle size={20} className="text-rose-500 shrink-0" />
                <p className="text-xs text-rose-200 leading-relaxed">{error}</p>
              </div>
            )}
          </motion.div>

          <motion.div 
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: 0.7 }}
            className="bg-indigo-600/10 border border-indigo-500/20 rounded-2xl p-6"
          >
            <p className="text-xs text-indigo-300 leading-relaxed mb-4 italic">
              "The Semantic Integrity Audit relies on the mathematical equivalence of Triplet Facts across the manifold."
            </p>
            <div className="flex items-center gap-2">
              <div className="h-4 w-1 bg-indigo-500 rounded-full" />
              <span className="text-[10px] font-mono text-indigo-400 uppercase">Phase 1 Complete</span>
            </div>
          </motion.div>
        </div>
      </main>
    </div>
  );
}

function StatCard({ label, value, icon, subtext, delay }: { label: string, value: any, icon: React.ReactNode, subtext: string, delay: number }) {
  return (
    <motion.div 
      initial={{ opacity: 0, scale: 0.95 }}
      animate={{ opacity: 1, scale: 1 }}
      transition={{ delay }}
      className="glass rounded-2xl p-6 border-b-2 border-transparent hover:border-sky-500/50 transition-all cursor-default group"
    >
      <div className="flex justify-between items-start mb-4">
        <div className="p-3 bg-slate-800 rounded-xl group-hover:scale-110 transition-transform">
          {icon}
        </div>
      </div>
      <p className="text-xs font-medium text-slate-400 mb-1">{label}</p>
      <h4 className="text-3xl font-bold text-white mb-2 tracking-tight">{value}</h4>
      <p className="text-[10px] text-slate-500 font-mono uppercase tracking-widest">{subtext}</p>
    </motion.div>
  );
}

function AuditItem({ label, success }: { label: string, success: boolean }) {
  return (
    <div className="flex items-center gap-3">
      <div className={`w-1.5 h-1.5 rounded-full ${success ? "bg-emerald-500 shadow-[0_0_8px_rgba(16,185,129,0.5)]" : "bg-slate-700"}`} />
      <span className={`text-sm ${success ? "text-slate-300" : "text-slate-500"}`}>{label}</span>
    </div>
  );
}
