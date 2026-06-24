"""Retrieval-lever sweep to lift grounding on broad cross-document queries.

Baseline (seed=100 arbitrary / expansion=10 / top_k=5) grounds at ~0.964 on the
18-query corpus-aligned set; broad cross-doc queries (0.83-0.87) drag the mean while
narrow factual queries ground at 1.0. Two suspected causes:
  (B) seed QUALITY: seeds_query LIMITs an ARBITRARY 100 of ~5,500 nodes (Aura-free has
      no vector index), then ranks in python -> top-5 among an arbitrary subset, not the
      true nearest neighbours.
  (A) breadth: expansion_limit=10 + top_k=5 is thin for broad questions.

This sweeps both. Levers are read from Config at query time, so we set them in-process
(top_k via a thin monkeypatch since generator.py calls retrieve() with the default).
The synthesis.py no-leads fix is active underneath (current best config).

Whatever this produces is what gets reported — including "no lever helps".

Usage (from backend/):  python -m run_logs.reroll_retrieval_sweep
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
    "COMPGCN_DECODER": "distmult", "COMPGCN_LOSS": "bpr", "COMPGCN_ADV_TEMP": "1.0",
    "COMPGCN_HIDDEN_CHANNELS": "256", "COMPGCN_DROPOUT": "0.2", "COMPGCN_NEG_RATIO": "15",
    "GROUNDING_MIN_SCORE": "0.95", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
}.items():
    os.environ[_k] = _v

TAU = 0.95
N_RUNS = 2  # per config; the winner gets a 3-run confirm afterwards
CONFIGS = [
    # name,           seed_limit, expansion_limit, top_k
    ("C0_baseline",       100,   10,  5),
    ("C1_trueNN_seeds",  6000,   10,  5),
    ("C2_seeds+breadth", 6000,   20,  8),
    ("C3_seeds+wide",    6000,   40, 12),
]


def _patch_top_k(Config):
    import src.retriever as rmod
    if getattr(rmod.GraphRetriever, "_topk_patched", False):
        return
    _orig = rmod.GraphRetriever.retrieve

    def _patched(self, query, top_k=None):
        if top_k is None:
            top_k = getattr(Config, "RETRIEVAL_TOP_K", 5)
        return _orig(self, query, top_k=top_k)

    rmod.GraphRetriever.retrieve = _patched
    rmod.GraphRetriever._topk_patched = True


def _per_query(r):
    """Best-effort extraction of per-query rows across return shapes."""
    if isinstance(r.get("per_query"), list):
        return r["per_query"]
    abl = (r.get("ablation") or {}).get("full_stack") or {}
    if isinstance(abl.get("per_query"), list):
        return abl["per_query"]
    return []


async def _preflight(Config):
    from neo4j import GraphDatabase
    d = GraphDatabase.driver(Config.NEO4J_URI, auth=(Config.NEO4J_USERNAME, Config.NEO4J_PASSWORD),
                             connection_acquisition_timeout=25)
    try:
        d.verify_connectivity()
        with d.session() as s:
            n = s.run("MATCH ()-[r]->() WHERE r.plausibility_score IS NOT NULL RETURN count(r) AS c").single()["c"]
    finally:
        d.close()
    return n


async def main() -> int:
    from src.config import Config
    from src.evaluation import run_grounding_evaluation

    scored = await _preflight(Config)
    if scored < 1000:
        print(f"PREFLIGHT_FAIL scored_edges={scored} (graph not ready)", flush=True)
        return 2
    print(f"PREFLIGHT_OK scored_edges={scored}", flush=True)
    _patch_top_k(Config)

    print(f"SWEEP_BEGIN tau={TAU} n_runs={N_RUNS} configs={len(CONFIGS)}", flush=True)
    summary = []
    for name, seed_lim, exp_lim, top_k in CONFIGS:
        Config.RETRIEVAL_SEED_LIMIT = seed_lim
        Config.RETRIEVAL_EXPANSION_LIMIT = exp_lim
        Config.RETRIEVAL_TOP_K = top_k
        gs, fs, ns = [], [], []
        first_pq = None
        for i in range(N_RUNS):
            r = await run_grounding_evaluation(mode="full_stack", grounding_threshold=TAU, persist_to_ablation=False)
            g, f, n = r.get("grounding_score"), r.get("faithfulness_score"), r.get("sample_count")
            if g is None:
                print(f"  {name} run{i+1} FAILED", flush=True); continue
            gs.append(g); fs.append(f); ns.append(n)
            if first_pq is None:
                first_pq = _per_query(r)
            print(f"  {name} run{i+1} grounding={g:.4f} faith={f:.4f} answered={n}/18", flush=True)
        if not gs:
            continue
        gm = statistics.mean(gs); fm = statistics.mean(fs)
        summary.append((name, seed_lim, exp_lim, top_k, gm, fm, min(gs), max(gs), statistics.mean(ns)))
        print(f"  {name} MEAN grounding={gm:.4f} faith={fm:.4f} (g range {min(gs):.4f}-{max(gs):.4f}, avg_answered={statistics.mean(ns):.1f})", flush=True)
        # show the worst-grounding queries for this config (diagnostic on broad queries)
        if first_pq:
            rows = sorted(first_pq, key=lambda x: x.get("grounding_score", 1.0))[:5]
            for q in rows:
                print(f"      low-g {q.get('grounding_score'):.3f}  {q.get('query','')[:70]}", flush=True)

    print("\nSWEEP_SUMMARY (sorted by grounding mean):", flush=True)
    for name, sl, el, tk, gm, fm, lo, hi, an in sorted(summary, key=lambda x: -x[4]):
        verdict = "PASS>=0.98" if gm >= 0.98 else ("PASS>=0.95" if gm >= 0.95 else "FAIL")
        print(f"  {name:<18} seed={sl:<5} exp={el:<3} topk={tk:<3} | grounding={gm:.4f} faith={fm:.4f} answered={an:.1f} [{verdict}]", flush=True)
    if summary:
        best = max(summary, key=lambda x: x[4])
        print(f"\nBEST {best[0]} grounding={best[4]:.4f} faith={best[5]:.4f} "
              f"(seed={best[1]} exp={best[2]} topk={best[3]})", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
