"""5-run confirm of the winning retrieval config before shipping it.

Sweep finding: seed QUALITY was the grounding bottleneck. The seeds_query LIMITed an
ARBITRARY 100 of ~5,500 nodes (Aura-free has no vector index), so the python cosine
ranking chose the top-5 among an arbitrary subset, not the true nearest neighbours.
Ranking ALL candidates (seed=6000) + tight expansion (exp=10) gave grounding 0.999 /
faith 0.983 / 16-of-18 answered in 2 sweep runs. This pins that exact config and runs
it 5x to report a robust mean +/- std for the paper (H3).

NOTE: pins BOTH levers via env BEFORE importing src — the live .env had
RETRIEVAL_EXPANSION_LIMIT=25 (a workaround for the bad seeds), which we override to 10.

Usage (from backend/):  python -m run_logs.reroll_grounding_confirm
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

# Winning config — pinned BEFORE src import so Config picks them up.
for _k, _v in {
    "RETRIEVAL_SEED_LIMIT": "6000",        # rank ALL candidates -> true nearest-neighbour seeds
    "RETRIEVAL_EXPANSION_LIMIT": "10",     # tight expansion (override .env's 25)
    "COMPGCN_DECODER": "distmult", "COMPGCN_LOSS": "bpr", "COMPGCN_ADV_TEMP": "1.0",
    "COMPGCN_HIDDEN_CHANNELS": "256", "COMPGCN_DROPOUT": "0.2", "COMPGCN_NEG_RATIO": "15",
    "GROUNDING_MIN_SCORE": "0.95", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
}.items():
    os.environ[_k] = _v

TAU = 0.95
RUNS = 5


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


def _per_query(r):
    if isinstance(r.get("per_query"), list):
        return r["per_query"]
    abl = (r.get("ablation") or {}).get("full_stack") or {}
    return abl.get("per_query") or []


async def main() -> int:
    from src.config import Config
    from src.evaluation import run_grounding_evaluation

    print(f"CONFIRM config seed={Config.RETRIEVAL_SEED_LIMIT} exp={Config.RETRIEVAL_EXPANSION_LIMIT} tau={TAU} runs={RUNS}", flush=True)
    scored = await _preflight(Config)
    if scored < 1000:
        print(f"PREFLIGHT_FAIL scored_edges={scored}", flush=True)
        return 2
    print(f"PREFLIGHT_OK scored_edges={scored}", flush=True)

    gs, fs, ns = [], [], []
    worst = None
    for i in range(RUNS):
        r = await run_grounding_evaluation(mode="full_stack", grounding_threshold=TAU, persist_to_ablation=False)
        g, f, n = r.get("grounding_score"), r.get("faithfulness_score"), r.get("sample_count")
        if g is None:
            print(f"RUN {i+1} FAILED", flush=True); continue
        gs.append(g); fs.append(f); ns.append(n)
        print(f"RUN {i+1} grounding={g:.4f} faith={f:.4f} answered={n}/18", flush=True)
        pq = _per_query(r)
        if pq:
            lo = min(pq, key=lambda x: x.get("grounding_score", 1.0))
            if worst is None or lo.get("grounding_score", 1.0) < worst[0]:
                worst = (lo.get("grounding_score", 1.0), lo.get("query", ""))

    if not gs:
        print("CONFIRM_FATAL no successful runs", flush=True)
        return 3
    gm, gsd = statistics.mean(gs), statistics.pstdev(gs)
    fm, fsd = statistics.mean(fs), statistics.pstdev(fs)
    print(f"\nGROUNDING    mean={gm:.4f} std={gsd:.4f} min={min(gs):.4f} max={max(gs):.4f} n={len(gs)}", flush=True)
    print(f"FAITHFULNESS mean={fm:.4f} std={fsd:.4f} min={min(fs):.4f} max={max(fs):.4f}", flush=True)
    print(f"ANSWERED     mean={statistics.mean(ns):.1f}/18", flush=True)
    if worst:
        print(f"WORST_SINGLE_QUERY grounding={worst[0]:.3f} :: {worst[1][:70]}", flush=True)
    print(f"VERDICT grounding>=0.98:{'PASS' if gm >= 0.98 else 'FAIL'} "
          f"grounding>=0.95:{'PASS' if gm >= 0.95 else 'FAIL'} "
          f"faith>=0.90:{'PASS' if fm >= 0.90 else 'FAIL'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
