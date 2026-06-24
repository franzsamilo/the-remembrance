"""Refresh the H1 ablation (full-stack vs graph-no-GNN vs prompt-only) on the live
system + shipped retrieval config, so the uplift percentages match the recovered
grounding/faithfulness reported in paper v6.11.

The paper's +45% Grounding / +204% Faithfulness uplift was measured on the OLD graph,
5 generic queries, old retrieval config. This re-measures all three ablation modes on
the rebuilt graph, the 18-query corpus-aligned set, and the shipped retriever
(seed=6000 exhaustive NN, exp=10), N runs each, and reports uplift = (full - base)/base.

Usage (from backend/):  python -m run_logs.reroll_ablation_refresh
"""
from __future__ import annotations

import asyncio
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_BUNDLE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".grpc-ssl-bundle.pem")
if os.path.exists(_BUNDLE):
    os.environ.setdefault("GRPC_DEFAULT_SSL_ROOTS_FILE_PATH", _BUNDLE)
try:
    import pip_system_certs.wrapt_requests  # noqa: F401
except ImportError:
    pass

for _k, _v in {
    "RETRIEVAL_SEED_LIMIT": "6000", "RETRIEVAL_EXPANSION_LIMIT": "10",   # shipped config
    "COMPGCN_DECODER": "distmult", "COMPGCN_LOSS": "bpr", "COMPGCN_ADV_TEMP": "1.0",
    "COMPGCN_HIDDEN_CHANNELS": "256", "COMPGCN_DROPOUT": "0.2", "COMPGCN_NEG_RATIO": "15",
    "GROUNDING_MIN_SCORE": "0.95", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
}.items():
    os.environ[_k] = _v

TAU = 0.95
RUNS = 3
MODES = ["full_stack", "graph_no_gnn", "prompt_only"]


async def _preflight(Config):
    from neo4j import GraphDatabase
    d = GraphDatabase.driver(Config.NEO4J_URI, auth=(Config.NEO4J_USERNAME, Config.NEO4J_PASSWORD),
                             connection_acquisition_timeout=25)
    try:
        d.verify_connectivity()
        with d.session() as s:
            return s.run("MATCH ()-[r]->() WHERE r.plausibility_score IS NOT NULL RETURN count(r) AS c").single()["c"]
    finally:
        d.close()


async def main() -> int:
    from src.config import Config
    from src.evaluation import run_grounding_evaluation

    scored = await _preflight(Config)
    if scored < 1000:
        print(f"PREFLIGHT_FAIL scored_edges={scored}", flush=True); return 2
    print(f"PREFLIGHT_OK scored_edges={scored} seed={Config.RETRIEVAL_SEED_LIMIT} exp={Config.RETRIEVAL_EXPANSION_LIMIT}", flush=True)
    print(f"ABLATION_BEGIN tau={TAU} runs={RUNS} modes={MODES}", flush=True)

    agg = {}
    for mode in MODES:
        gs, fs, ns = [], [], []
        for i in range(RUNS):
            r = await run_grounding_evaluation(mode=mode, grounding_threshold=TAU, persist_to_ablation=False)
            g, f, n = r.get("grounding_score"), r.get("faithfulness_score"), r.get("sample_count")
            if g is None:
                print(f"  {mode} run{i+1} FAILED", flush=True); continue
            gs.append(g); fs.append(f); ns.append(n)
            print(f"  {mode} run{i+1} grounding={g:.4f} faith={f:.4f} answered={n}", flush=True)
        if gs:
            agg[mode] = (statistics.mean(gs), statistics.mean(fs), statistics.mean(ns))
            print(f"  {mode} MEAN grounding={agg[mode][0]:.4f} faith={agg[mode][1]:.4f} answered={agg[mode][2]:.1f}", flush=True)

    print("\nABLATION_SUMMARY:", flush=True)
    for m in MODES:
        if m in agg:
            print(f"  {m:<14} G={agg[m][0]:.4f} F={agg[m][1]:.4f} answered={agg[m][2]:.1f}", flush=True)

    if "full_stack" in agg:
        fg, ff, _ = agg["full_stack"]
        for base in ("prompt_only", "graph_no_gnn"):
            if base in agg:
                bg, bf, _ = agg[base]
                ug = (fg - bg) / bg * 100 if bg else float("inf")
                uf = (ff - bf) / bf * 100 if bf else float("inf")
                print(f"UPLIFT full_stack vs {base}: Grounding +{ug:.0f}%  Faithfulness +{uf:.0f}%  "
                      f"(G {bg:.3f}->{fg:.3f}, F {bf:.3f}->{ff:.3f})", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
