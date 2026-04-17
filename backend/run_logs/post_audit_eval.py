"""Post-audit verification + evaluation chain.

Runs the moment `run_audit` finishes. Steps:
  1. Confirm plausibility scores are written to Neo4j (count + sample distribution).
  2. Run full-stack grounding/faithfulness evaluation.
  3. Run threshold sweep.
  4. Run prompt-only ablation (for comparison).

Writes everything to evaluation_results.json via the existing evaluation module.
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import time

# Running from backend/run_logs/; backend/ must be on sys.path for `src` imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


async def main():
    from src.config import Config
    from src.db import DatabaseManager
    from src.evaluation import (
        run_grounding_evaluation,
        run_threshold_sweep,
    )

    t0 = time.time()

    # Step 1: Verify Neo4j sync
    driver = DatabaseManager.get_driver()
    with driver.session(database=Config.NEO4J_DATABASE) as session:
        result = session.run(
            """
            MATCH ()-[r]->()
            WHERE r.plausibility_score IS NOT NULL
            RETURN count(r) AS rels_with_score,
                   avg(r.plausibility_score) AS avg_score,
                   min(r.plausibility_score) AS min_score,
                   max(r.plausibility_score) AS max_score
            """
        ).single()
        rels = result["rels_with_score"]
        avg = result["avg_score"]
        mn = result["min_score"]
        mx = result["max_score"]
        print(
            f"NEO4J_SCORES rels_with_score={rels} avg={avg:.4f} min={mn:.4f} max={mx:.4f}",
            flush=True,
        )

        # Distribution buckets
        dist = session.run(
            """
            MATCH ()-[r]->()
            WHERE r.plausibility_score IS NOT NULL
            WITH r.plausibility_score AS s
            RETURN
                sum(CASE WHEN s < 0.50 THEN 1 ELSE 0 END) AS below_50,
                sum(CASE WHEN s >= 0.50 AND s < 0.85 THEN 1 ELSE 0 END) AS b50_85,
                sum(CASE WHEN s >= 0.85 AND s < 0.95 THEN 1 ELSE 0 END) AS b85_95,
                sum(CASE WHEN s >= 0.95 AND s < 0.99 THEN 1 ELSE 0 END) AS b95_99,
                sum(CASE WHEN s >= 0.99 THEN 1 ELSE 0 END) AS above_99
            """
        ).single()
        print(
            f"SCORE_DIST <0.50={dist['below_50']} 0.50-0.85={dist['b50_85']} "
            f"0.85-0.95={dist['b85_95']} 0.95-0.99={dist['b95_99']} >=0.99={dist['above_99']}",
            flush=True,
        )

        # AuditRun metadata snapshot
        latest_run = session.run(
            f"""
            MATCH (run:{Config.AUDIT_RUN_LABEL})
            RETURN run.run_id AS run_id, run.auc_roc AS auc_roc, run.mrr AS mrr,
                   run.loss AS loss, run.completed_at AS completed_at
            ORDER BY run.completed_at DESC LIMIT 1
            """
        ).single()
        if latest_run:
            print(
                f"LATEST_AUDIT_RUN run_id={latest_run['run_id']} auc_roc={latest_run['auc_roc']} "
                f"mrr={latest_run['mrr']} loss={latest_run['loss']} completed_at={latest_run['completed_at']}",
                flush=True,
            )

    # Step 2: Full-stack grounding/faithfulness
    print("EVAL_FULL_STACK_BEGIN", flush=True)
    full = await run_grounding_evaluation(mode="full_stack")
    print(
        f"EVAL_FULL_STACK grounding={full.get('grounding_score')} "
        f"faithfulness={full.get('faithfulness_score')} n={full.get('sample_count')}",
        flush=True,
    )

    # Step 3: Threshold sweep
    print("THRESHOLD_SWEEP_BEGIN", flush=True)
    sweep = await run_threshold_sweep()
    for tau, metrics in sweep.items():
        print(
            f"SWEEP tau={tau} grounding={metrics.get('grounding_score')} "
            f"faithfulness={metrics.get('faithfulness_score')}",
            flush=True,
        )

    # Step 4: Prompt-only ablation (re-run to capture under new codebase)
    print("EVAL_PROMPT_ONLY_BEGIN", flush=True)
    po = await run_grounding_evaluation(mode="prompt_only")
    print(
        f"EVAL_PROMPT_ONLY grounding={po.get('grounding_score')} "
        f"faithfulness={po.get('faithfulness_score')} n={po.get('sample_count')}",
        flush=True,
    )

    print(f"EVAL_DONE elapsed_min={(time.time()-t0)/60.0:.2f}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
