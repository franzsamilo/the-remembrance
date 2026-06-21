"""Full pipeline runner: ingest -> embed -> audit -> defense-preflight.

Used when Neo4j has been wiped and the entire corpus must be re-ingested
end-to-end. Reaches the same DEFENSE READY state that restore_defense_state.py
produces from an existing graph, but starts from zero data.

Usage (from backend/):
    python -m run_logs.full_pipeline_to_defense_ready

Each stage's progress is logged with structured BEGIN/DONE markers so an
external observer (a script tailing the log, or another LLM agent) can
parse progress without re-implementing the pipeline.
"""
from __future__ import annotations

import asyncio
import os
import sys
import time

# Make backend/ importable when run via `python -m run_logs.full_pipeline...`
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# --- Local-only TLS workaround for Avast HTTPS scanning ---
# If `backend/.grpc-ssl-bundle.pem` exists, point gRPC at it so Gemini calls
# survive the Avast MITM. This is opt-in: missing file = standard SSL behavior.
# Generated once via: cat <certifi-bundle> <avast-root> > .grpc-ssl-bundle.pem
_GRPC_BUNDLE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    ".grpc-ssl-bundle.pem",
)
if os.path.exists(_GRPC_BUNDLE):
    os.environ.setdefault("GRPC_DEFAULT_SSL_ROOTS_FILE_PATH", _GRPC_BUNDLE)
# Patch Python's ssl/requests to use the Windows certificate store (which
# already trusts Avast). No-op if not installed.
try:
    import pip_system_certs.wrapt_requests  # noqa: F401
except ImportError:
    pass

# Apply Run 8 defense env BEFORE importing src.config (which freezes them).
# Same overrides as restore_defense_state.py.
DEFENSE_ENV = {
    "COMPGCN_DECODER": "distmult",
    "COMPGCN_LOSS": "bpr",
    "COMPGCN_ADV_TEMP": "1.0",
    "COMPGCN_HIDDEN_CHANNELS": "256",
    "COMPGCN_DROPOUT": "0.2",
    "COMPGCN_NEG_RATIO": "15",
    "COMPGCN_SEED": "42",
    "COMPGCN_AUC_GUARDRAIL": "0.95",
    "GROUNDING_MIN_SCORE": "0.95",
    # Force HuggingFace into offline mode so SSL-MITM environments don't burn
    # 30+ seconds per model on failed HEAD requests. The DistilBERT model
    # used by the embedder + the SimpleKGPipeline is already cached under
    # ~/.cache/huggingface/hub/.
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
}
for k, v in DEFENSE_ENV.items():
    if os.environ.get(k) != v:
        os.environ[k] = v


async def main() -> int:
    from src.config import Config
    from src.ingestion import process_documents
    from src.embed_nodes import embed_nodes
    from src.gnn_module import run_audit
    from src.evaluation import run_grounding_evaluation, run_threshold_sweep
    from src.db import DatabaseManager

    t_start = time.time()

    # Stage 1: PDF Ingestion (Gemini extraction over each PDF in DOCS_DIR)
    print("STAGE_1_INGEST_BEGIN", flush=True)
    t0 = time.time()
    manifest = await process_documents()
    elapsed = time.time() - t0
    print(
        f"STAGE_1_INGEST_DONE status={manifest.get('status')} "
        f"processed={manifest.get('documents_processed')} "
        f"failed={manifest.get('documents_failed')} "
        f"nodes_created={manifest.get('nodes_created')} "
        f"rels_created={manifest.get('relationships_created')} "
        f"elapsed_sec={elapsed:.1f}",
        flush=True,
    )
    if not manifest.get("documents_processed"):
        print("FATAL ingestion produced no processable documents", flush=True)
        return 1

    # Stage 2: Embed nodes with DistilBERT
    print("STAGE_2_EMBED_BEGIN", flush=True)
    t0 = time.time()
    await embed_nodes()
    print(f"STAGE_2_EMBED_DONE elapsed_sec={time.time()-t0:.1f}", flush=True)

    # Stage 3: GNN audit (train CompGCN from scratch under Run 8 config)
    print("STAGE_3_AUDIT_BEGIN", flush=True)
    t0 = time.time()
    run_audit()
    print(f"STAGE_3_AUDIT_DONE elapsed_sec={time.time()-t0:.1f}", flush=True)

    # Stage 4: Confirm Neo4j has scored edges
    print("STAGE_4_NEO4J_VERIFY_BEGIN", flush=True)
    driver = DatabaseManager.refresh()
    with driver.session(database=Config.NEO4J_DATABASE) as session:
        s = session.run(
            """
            MATCH ()-[r]->()
            WHERE r.plausibility_score IS NOT NULL
            RETURN count(r) AS scored,
                   avg(r.plausibility_score) AS avg_score,
                   min(r.plausibility_score) AS min_score,
                   max(r.plausibility_score) AS max_score
            """
        ).single()
        print(
            f"STAGE_4_NEO4J_VERIFY scored={s['scored']} "
            f"avg={s['avg_score']:.4f} min={s['min_score']:.4f} max={s['max_score']:.4f}",
            flush=True,
        )

    # Stage 5: Ablation matrix (full_stack | graph_no_gnn | prompt_only)
    print("STAGE_5_EVAL_FULL_STACK_BEGIN", flush=True)
    t0 = time.time()
    full = await run_grounding_evaluation(mode="full_stack")
    print(
        f"STAGE_5_EVAL_FULL_STACK_DONE grounding={full.get('grounding_score')} "
        f"faithfulness={full.get('faithfulness_score')} "
        f"n={full.get('sample_count')} elapsed_sec={time.time()-t0:.1f}",
        flush=True,
    )

    print("STAGE_5_EVAL_GRAPH_NO_GNN_BEGIN", flush=True)
    t0 = time.time()
    gng = await run_grounding_evaluation(mode="graph_no_gnn")
    print(
        f"STAGE_5_EVAL_GRAPH_NO_GNN_DONE grounding={gng.get('grounding_score')} "
        f"faithfulness={gng.get('faithfulness_score')} "
        f"n={gng.get('sample_count')} elapsed_sec={time.time()-t0:.1f}",
        flush=True,
    )

    print("STAGE_5_EVAL_PROMPT_ONLY_BEGIN", flush=True)
    t0 = time.time()
    po = await run_grounding_evaluation(mode="prompt_only")
    print(
        f"STAGE_5_EVAL_PROMPT_ONLY_DONE grounding={po.get('grounding_score')} "
        f"faithfulness={po.get('faithfulness_score')} "
        f"n={po.get('sample_count')} elapsed_sec={time.time()-t0:.1f}",
        flush=True,
    )

    # Stage 6: Threshold sweep
    print("STAGE_6_THRESHOLD_SWEEP_BEGIN", flush=True)
    t0 = time.time()
    sweep = await run_threshold_sweep()
    for tau, m in sweep.items():
        print(
            f"STAGE_6_SWEEP tau={tau} grounding={m.get('grounding_score')} "
            f"faithfulness={m.get('faithfulness_score')}",
            flush=True,
        )
    print(f"STAGE_6_THRESHOLD_SWEEP_DONE elapsed_sec={time.time()-t0:.1f}", flush=True)

    # Final KPI gating (same logic as restore_defense_state.py)
    grounding = full.get("grounding_score")
    faithfulness = full.get("faithfulness_score")
    from src.gnn_module import get_training_history
    th = get_training_history()
    auc = th.get("final_auc_roc")
    mrr = th.get("final_mrr")

    print(
        f"PREFLIGHT_SUMMARY grounding={grounding} faithfulness={faithfulness} "
        f"auc={auc} mrr={mrr}",
        flush=True,
    )

    targets = [
        ("Grounding",    grounding,    0.98, "H3"),
        ("Faithfulness", faithfulness, 0.90, "H3"),
        ("AUC-ROC",      auc,          0.95, "H2"),
        ("MRR",          mrr,          0.95, "H2"),
    ]
    failed = []
    for name, value, target, h in targets:
        if value is None:
            print(f"KPI_GATE {name:<13} value=None target={target} {h} WARN (missing)", flush=True)
            failed.append((name, value, target))
            continue
        passed = value >= target
        status = "PASS" if passed else "FAIL"
        print(f"KPI_GATE {name:<13} value={value:.4f} target={target} {h} {status}", flush=True)
        if not passed:
            failed.append((name, value, target))

    print(f"TOTAL_ELAPSED_MIN={(time.time()-t_start)/60.0:.2f}", flush=True)

    if failed:
        print("DEFENSE NOT READY -- the following KPIs are below paper target:", flush=True)
        for name, value, target in failed:
            shown = f"{value:.4f}" if isinstance(value, (int, float)) else "missing"
            print(f"  - {name}: {shown} < {target}", flush=True)
        return 3

    print("DEFENSE READY -- all four paper KPIs pass.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
