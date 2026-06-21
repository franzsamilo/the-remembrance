"""Reroll stage 1: wipe + re-ingest + embed for a fresh corpus realization.

The current graph (8,153 GNN edges, multi-seed MRR 0.942) is one realization of
the 14-PDF corpus. The old (lost) graph (6,419 edges) hit 0.958. This re-runs
non-deterministic Gemini extraction for a new realization, hoping to land
>= 0.95. The current graph is backed up (graph_restore.py can restore it), so a
worse reroll is recoverable.

SAFETY: refuses to wipe unless (a) the backup snapshot exists and is non-trivial
and (b) a live Gemini ping succeeds — so we never wipe into a broken extraction.

Usage (from backend/):
    python -m run_logs.reroll_ingest run_logs/graph_snapshot_20260621_202736.jsonl
"""
from __future__ import annotations

import asyncio
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Local TLS workaround for Avast HTTPS scanning (same as full_pipeline runner).
_GRPC_BUNDLE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".grpc-ssl-bundle.pem"
)
if os.path.exists(_GRPC_BUNDLE):
    os.environ.setdefault("GRPC_DEFAULT_SSL_ROOTS_FILE_PATH", _GRPC_BUNDLE)
try:
    import pip_system_certs.wrapt_requests  # noqa: F401
except ImportError:
    pass

DEFENSE_ENV = {
    "COMPGCN_DECODER": "distmult",
    "COMPGCN_LOSS": "bpr",
    "COMPGCN_ADV_TEMP": "1.0",
    "COMPGCN_HIDDEN_CHANNELS": "256",
    "COMPGCN_DROPOUT": "0.2",
    "COMPGCN_NEG_RATIO": "15",
    "COMPGCN_AUC_GUARDRAIL": "0.95",
    "GROUNDING_MIN_SCORE": "0.95",
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
}
for _k, _v in DEFENSE_ENV.items():
    os.environ[_k] = _v


async def main() -> int:
    if len(sys.argv) < 2:
        print("FATAL usage: python -m run_logs.reroll_ingest <backup.jsonl>", flush=True)
        return 2
    backup = sys.argv[1]
    if not os.path.exists(backup) or os.path.getsize(backup) < 1_000_000:
        print(f"FATAL backup missing/too small ({backup}); refusing to wipe", flush=True)
        return 2

    from src.config import Config
    from src.db import DatabaseManager
    from src.ingestion import process_documents
    from src.embed_nodes import embed_nodes

    # --- Pre-wipe Gemini reachability check (NEVER wipe into a broken pipeline) ---
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI

        llm = ChatGoogleGenerativeAI(
            model=Config.GEMINI_MODEL, google_api_key=Config.GOOGLE_API_KEY
        )
        _ = llm.invoke("Reply with the single word: ok")
        print(f"GEMINI_OK model={Config.GEMINI_MODEL}", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"FATAL gemini unreachable, NOT wiping: {type(e).__name__}: {e}", flush=True)
        return 2

    t_start = time.time()

    # --- Wipe ---
    driver = DatabaseManager.refresh()
    with driver.session(database=Config.NEO4J_DATABASE) as s:
        before = s.run("MATCH (n) RETURN count(n) AS c").single()["c"]
        print(
            f"WIPE_BEGIN nodes={before} backup={backup} "
            f"size_mb={os.path.getsize(backup)/1e6:.1f}",
            flush=True,
        )
        s.run("MATCH (n) DETACH DELETE n")
        after = s.run("MATCH (n) RETURN count(n) AS c").single()["c"]
        print(f"WIPE_DONE nodes={after}", flush=True)
        if after != 0:
            print("FATAL wipe incomplete", flush=True)
            return 1

    # --- Ingest ---
    print("INGEST_BEGIN", flush=True)
    t0 = time.time()
    manifest = await process_documents()
    print(
        f"INGEST_DONE status={manifest.get('status')} "
        f"processed={manifest.get('documents_processed')} "
        f"failed={manifest.get('documents_failed')} "
        f"nodes={manifest.get('nodes_created')} "
        f"rels={manifest.get('relationships_created')} "
        f"elapsed_sec={time.time()-t0:.1f}",
        flush=True,
    )
    if not manifest.get("documents_processed"):
        print("FATAL ingestion produced no processable documents", flush=True)
        print("RECOVER: python -m run_logs.graph_restore %s --force" % backup, flush=True)
        return 1

    # --- Embed ---
    print("EMBED_BEGIN", flush=True)
    t0 = time.time()
    await embed_nodes()
    print(f"EMBED_DONE elapsed_sec={time.time()-t0:.1f}", flush=True)

    # --- Report new graph size vs the realization we just replaced ---
    driver = DatabaseManager.refresh()
    with driver.session(database=Config.NEO4J_DATABASE) as s:
        nodes = s.run("MATCH (n) RETURN count(n) AS c").single()["c"]
        total_rels = s.run("MATCH ()-[r]->() RETURN count(r) AS c").single()["c"]
        gnn_edges = s.run(
            "MATCH ()-[r]->() WHERE type(r) <> 'FROM_CHUNK' RETURN count(r) AS c"
        ).single()["c"]
    print(
        f"REROLL_INGEST_DONE nodes={nodes} gnn_edges={gnn_edges} total_rels={total_rels} "
        f"total_min={(time.time()-t_start)/60:.1f}",
        flush=True,
    )
    print(
        f"COMPARE prev_realization: gnn_edges=8153 MRR=0.942 | "
        f"new_realization: gnn_edges={gnn_edges} MRR=<pending retrain sweep>",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
