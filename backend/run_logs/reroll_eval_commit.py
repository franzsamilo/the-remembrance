"""Re-run the 18-query corpus-aligned eval on the LIVE system (train13 + post-fix
retriever) WITH per-query results persisted to a committed JSON — so the repo backs
the paper's 0.984 grounding claim, and Table 5.10 can be rebuilt from real legal-query
per-query scores (Option B).

Fixes the original gap: the earlier 18-query drivers used persist_to_ablation=False and
printed only aggregates, so per-query data was never saved. This writes per-query G/F
(per run + mean across runs) to backend/run_logs/reroll_eval_commit.json incrementally,
so a mid-run segfault (free-tier Aura under heavy seed=6000 retrieval) preserves
completed runs.

Config pinned to the SHIPPED retriever: seed=6000 (exhaustive NN), exp=10, tau=0.95.

Usage (from backend/):  python -m run_logs.reroll_eval_commit
"""
from __future__ import annotations

import asyncio
import json
import os
import statistics
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_BUNDLE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".grpc-ssl-bundle.pem")
if os.path.exists(_BUNDLE):
    os.environ.setdefault("GRPC_DEFAULT_SSL_ROOTS_FILE_PATH", _BUNDLE)
try:
    import pip_system_certs.wrapt_requests  # noqa: F401
except ImportError:
    pass

for _k, _v in {
    "RETRIEVAL_SEED_LIMIT": "6000", "RETRIEVAL_EXPANSION_LIMIT": "10",
    "COMPGCN_DECODER": "distmult", "COMPGCN_LOSS": "bpr", "COMPGCN_ADV_TEMP": "1.0",
    "COMPGCN_HIDDEN_CHANNELS": "256", "COMPGCN_DROPOUT": "0.2", "COMPGCN_NEG_RATIO": "15",
    "GROUNDING_MIN_SCORE": "0.95", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
}.items():
    os.environ[_k] = _v

TAU = 0.95
RUNS = 5
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reroll_eval_commit.json")

# The 5 queries promoted to Table 5.10 (Option B — corpus-aligned IP-law subset of the 18).
TABLE_510 = [
    "What is the dominancy test for determining trademark infringement?",
    "What is the holistic test in trademark law?",
    "What constitutes unfair competition under Philippine intellectual property law?",
    "What is copyright infringement?",
    "What remedies are available for intellectual property infringement?",
]


def _per_query(r):
    if isinstance(r.get("per_query"), list):
        return r["per_query"]
    abl = (r.get("ablation") or {}).get("full_stack") or {}
    return abl.get("per_query") or []


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


def _write(payload):
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


async def main() -> int:
    from src.config import Config
    from src.evaluation import run_grounding_evaluation

    scored = await _preflight(Config)
    if scored < 1000:
        print(f"PREFLIGHT_FAIL scored_edges={scored}", flush=True); return 2
    print(f"PREFLIGHT_OK scored_edges={scored} seed={Config.RETRIEVAL_SEED_LIMIT} exp={Config.RETRIEVAL_EXPANSION_LIMIT}", flush=True)
    print(f"EVAL_COMMIT_BEGIN tau={TAU} runs={RUNS} queries_file={os.path.basename(Config.EVALUATION_QUERIES_FILE)}", flush=True)

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "config": {"seed_limit": 6000, "expansion_limit": 10, "tau": TAU,
                   "checkpoint": "compgcn_best.pt (train13, epoch 228)", "runs_planned": RUNS},
        "runs": [], "per_query_mean": {}, "aggregate": {},
    }
    g_runs, f_runs, n_runs = [], [], []
    pq_accum = {}  # query -> {"g": [...], "f": [...]}

    for i in range(RUNS):
        r = await run_grounding_evaluation(mode="full_stack", grounding_threshold=TAU, persist_to_ablation=False)
        g, f, n = r.get("grounding_score"), r.get("faithfulness_score"), r.get("sample_count")
        pq = _per_query(r)
        if g is None:
            print(f"RUN {i+1} FAILED", flush=True); continue
        g_runs.append(g); f_runs.append(f); n_runs.append(n)
        for q in pq:
            qt = q.get("query")
            pq_accum.setdefault(qt, {"g": [], "f": []})
            if q.get("grounding_score") is not None:
                pq_accum[qt]["g"].append(q["grounding_score"])
            if q.get("faithfulness_score") is not None:
                pq_accum[qt]["f"].append(q["faithfulness_score"])
        payload["runs"].append({"run": i + 1, "grounding": g, "faithfulness": f, "answered": n,
                                "per_query": pq})
        # incremental aggregate + write so a crash preserves progress
        payload["aggregate"] = {
            "grounding_mean": round(statistics.mean(g_runs), 4),
            "grounding_std": round(statistics.pstdev(g_runs), 4) if len(g_runs) > 1 else 0.0,
            "faithfulness_mean": round(statistics.mean(f_runs), 4),
            "faithfulness_std": round(statistics.pstdev(f_runs), 4) if len(f_runs) > 1 else 0.0,
            "answered_mean": round(statistics.mean(n_runs), 1), "n_runs": len(g_runs),
            "grounding_runs": [round(x, 4) for x in g_runs],
        }
        payload["per_query_mean"] = {
            qt: {"grounding": round(statistics.mean(v["g"]), 4) if v["g"] else None,
                 "faithfulness": round(statistics.mean(v["f"]), 4) if v["f"] else None,
                 "n": len(v["g"])}
            for qt, v in pq_accum.items()
        }
        _write(payload)
        print(f"RUN {i+1} grounding={g:.4f} faith={f:.4f} answered={n}/18  [json updated]", flush=True)

    if not g_runs:
        print("EVAL_COMMIT_FATAL no successful runs", flush=True); return 3

    agg = payload["aggregate"]
    print(f"\nAGGREGATE grounding={agg['grounding_mean']}±{agg['grounding_std']} "
          f"faith={agg['faithfulness_mean']}±{agg['faithfulness_std']} answered={agg['answered_mean']}/18 n={agg['n_runs']}", flush=True)
    print("\nTABLE 5.10 (5 corpus-aligned IP queries, per-query mean across runs):", flush=True)
    for q in TABLE_510:
        m = payload["per_query_mean"].get(q)
        if m:
            print(f"  G={m['grounding']} F={m['faithfulness']} (n={m['n']}) :: {q[:60]}", flush=True)
        else:
            print(f"  !! NOT FOUND in per-query results :: {q[:60]}", flush=True)
    print(f"\nWROTE {OUT}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
