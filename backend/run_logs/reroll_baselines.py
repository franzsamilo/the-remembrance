"""H1 baselines for the uplift refresh — prompt_only (essential) + graph_no_gnn (best-effort).

full_stack is already measured robustly (reroll_grounding_confirm.py: G=0.984+/-0.013,
F=0.983+/-0.011, 5-run). This measures the baselines so uplift = (full - base)/base can be
refreshed to the live system. prompt_only is chunk-RAG (light on Aura) and runs FIRST so it
is secured/flushed before the heavier graph_no_gnn (seed=6000 retrieval), which can crash the
Neo4j driver natively when free-tier Aura returns corrupt frames under load.

Usage (from backend/):  python -m run_logs.reroll_baselines
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
    "RETRIEVAL_SEED_LIMIT": "6000", "RETRIEVAL_EXPANSION_LIMIT": "10",
    "COMPGCN_DECODER": "distmult", "COMPGCN_LOSS": "bpr", "COMPGCN_ADV_TEMP": "1.0",
    "COMPGCN_HIDDEN_CHANNELS": "256", "COMPGCN_DROPOUT": "0.2", "COMPGCN_NEG_RATIO": "15",
    "GROUNDING_MIN_SCORE": "0.95", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
}.items():
    os.environ[_k] = _v

TAU = 0.95
# full_stack reference (reroll_grounding_confirm.py, 5-run mean) — for uplift.
FULL_G, FULL_F = 0.984, 0.983


async def run_mode(mode, n):
    from src.evaluation import run_grounding_evaluation
    gs, fs, ns = [], [], []
    for i in range(n):
        r = await run_grounding_evaluation(mode=mode, grounding_threshold=TAU, persist_to_ablation=False)
        g, f, c = r.get("grounding_score"), r.get("faithfulness_score"), r.get("sample_count")
        if g is None:
            print(f"  {mode} run{i+1} FAILED", flush=True); continue
        gs.append(g); fs.append(f); ns.append(c)
        print(f"  {mode} run{i+1} grounding={g:.4f} faith={f:.4f} answered={c}", flush=True)
    if gs:
        gm, fm = statistics.mean(gs), statistics.mean(fs)
        print(f"  {mode} MEAN grounding={gm:.4f} faith={fm:.4f} answered={statistics.mean(ns):.1f}", flush=True)
        return gm, fm
    return None


def uplift(label, bg, bf):
    ug = (FULL_G - bg) / bg * 100 if bg else float("inf")
    uf = (FULL_F - bf) / bf * 100 if bf else float("inf")
    print(f"UPLIFT full_stack vs {label}: Grounding +{ug:.0f}%  Faithfulness +{uf:.0f}%  "
          f"(G {bg:.3f}->{FULL_G}, F {bf:.3f}->{FULL_F})", flush=True)


async def main() -> int:
    from src.config import Config
    print(f"BASELINES_BEGIN tau={TAU} full_ref G={FULL_G} F={FULL_F} "
          f"seed={Config.RETRIEVAL_SEED_LIMIT} exp={Config.RETRIEVAL_EXPANSION_LIMIT}", flush=True)

    print("PROMPT_ONLY (essential, light):", flush=True)
    po = await run_mode("prompt_only", 3)

    print("GRAPH_NO_GNN (best-effort, heavy — may segfault):", flush=True)
    gn = await run_mode("graph_no_gnn", 2)

    print("\nUPLIFT_SUMMARY:", flush=True)
    if po:
        uplift("prompt_only", po[0], po[1])
    if gn:
        uplift("graph_no_gnn", gn[0], gn[1])
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
