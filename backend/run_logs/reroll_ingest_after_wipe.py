"""Reroll stage 1b: re-ingest + embed into an ALREADY-EMPTY graph.

Companion to reroll_ingest.py for when the wipe must be performed by the user
(the auto-permission classifier blocks programmatic mass-delete on the live DB).
Contains NO destructive statements. Refuses to ingest unless the graph is already
empty, so it can never duplicate onto an existing graph.

Modes (from backend/):
    python -m run_logs.reroll_ingest_after_wipe --check   # ping Gemini + report node count, no changes
    python -m run_logs.reroll_ingest_after_wipe           # ingest + embed (requires empty graph)
"""
from __future__ import annotations

import asyncio
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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
    check_only = "--check" in sys.argv[1:]

    from src.config import Config
    from src.db import DatabaseManager
    from src.ingestion import process_documents
    from src.embed_nodes import embed_nodes

    # Gemini reachability check.
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI

        llm = ChatGoogleGenerativeAI(
            model=Config.GEMINI_MODEL, google_api_key=Config.GOOGLE_API_KEY
        )
        _ = llm.invoke("Reply with the single word: ok")
        print(f"GEMINI_OK model={Config.GEMINI_MODEL}", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"FATAL gemini unreachable: {type(e).__name__}: {e}", flush=True)
        return 2

    driver = DatabaseManager.refresh()
    with driver.session(database=Config.NEO4J_DATABASE) as s:
        n = s.run("MATCH (n) RETURN count(n) AS c").single()["c"]
    print(f"GRAPH_STATE nodes={n}", flush=True)

    if check_only:
        print("CHECK_DONE gemini=ok (safe to wipe; re-run without --check after wiping)", flush=True)
        return 0

    if n != 0:
        print(
            f"FATAL graph not empty (nodes={n}); wipe it first. "
            "Refusing to ingest (would duplicate the corpus).",
            flush=True,
        )
        return 1
    print("GRAPH_EMPTY_CONFIRMED", flush=True)

    t_start = time.time()
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
        print(
            "RECOVER: python -m run_logs.graph_restore "
            "run_logs/graph_snapshot_20260621_202736.jsonl --force",
            flush=True,
        )
        return 1

    print("EMBED_BEGIN", flush=True)
    t0 = time.time()
    await embed_nodes()
    print(f"EMBED_DONE elapsed_sec={time.time()-t0:.1f}", flush=True)

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
        f"COMPARE prev_realization: gnn_edges=8153 MRR=0.942 | new_realization: gnn_edges={gnn_edges}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
