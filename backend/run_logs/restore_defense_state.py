"""Defense-day state restore script.

Resets the live deployment back to the Run 8 (DistMult + BPR + self-adv α=1.0)
configuration the paper v6.4 reports as the recommended defense configuration.
Does NOT retrain — only re-syncs the existing Run 8 checkpoint's scores into
Neo4j and re-runs the grounding/faithfulness ablation matrix.

Run BEFORE the defense if:
- `evaluation_results.json` is in a non-Run-8 state (e.g. Run 9 RotatE numbers)
- Neo4j plausibility scores don't reflect Run 8 DistMult (avg should be ~0.97)
- The ablation comparison on the dashboard is stale or null

Usage:
    cd backend
    python -m run_logs.restore_defense_state

Side effects:
- Overwrites every `r.plausibility_score` in Neo4j with the Run 8 checkpoint's score.
- Overwrites `backend/evaluation_results.json` with fresh grounding/faithfulness numbers.
- Both are idempotent — re-running is safe.

What this script does NOT do:
- Re-train. The Run 8 checkpoint at `backend/run_logs/compgcn_best.pt` is used as-is.
- Touch the paper docx. The paper reports the multi-seed mean MRR (0.958);
  this script does NOT regenerate that — the multi-seed run is a separate
  procedure documented in `multi_seed_mrr_run8.log`.
"""
from __future__ import annotations

import asyncio
import os
import sys
import time

# Running from backend/run_logs/; backend/ must be on sys.path for `src` imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# Defense-day environment overrides. These match the Run 8 winning config
# documented in SESSION_LOG_2026-05-03.md and the paper v6.4 §3.3.
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
}


def apply_defense_env() -> None:
    """Apply Run 8 env overrides BEFORE importing src.config (which freezes them)."""
    for key, value in DEFENSE_ENV.items():
        current = os.environ.get(key)
        if current != value:
            print(f"ENV_OVERRIDE {key}: {current!r} -> {value!r}", flush=True)
            os.environ[key] = value


async def main() -> int:
    apply_defense_env()

    from src.config import Config
    from src.db import DatabaseManager
    from src.evaluation import (
        run_grounding_evaluation,
        run_threshold_sweep,
    )
    from src.gnn_module import recover_from_checkpoint

    t0 = time.time()

    # Step 1: Confirm checkpoint exists
    checkpoint_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "run_logs",
        "compgcn_best.pt",
    )
    if not os.path.exists(checkpoint_path):
        print(f"FATAL no checkpoint at {checkpoint_path}", flush=True)
        return 1
    print(f"CHECKPOINT_FOUND {checkpoint_path}", flush=True)

    # Step 2: Restore Run 8 scores into Neo4j
    print("RECOVER_BEGIN", flush=True)
    metrics = recover_from_checkpoint()
    if metrics is None:
        print("FATAL recover_from_checkpoint returned None", flush=True)
        return 2
    print(
        f"RECOVER_DONE auc_roc={metrics.get('final_auc_roc')} "
        f"mrr={metrics.get('final_mrr')} epoch={metrics.get('best_epoch')}",
        flush=True,
    )

    # Step 3: Confirm Neo4j state
    driver = DatabaseManager.get_driver()
    with driver.session(database=Config.NEO4J_DATABASE) as session:
        sync = session.run(
            """
            MATCH ()-[r]->()
            WHERE r.plausibility_score IS NOT NULL
            RETURN count(r) AS rels_with_score,
                   avg(r.plausibility_score) AS avg_score,
                   min(r.plausibility_score) AS min_score,
                   max(r.plausibility_score) AS max_score
            """
        ).single()
        avg = sync["avg_score"] or 0.0
        mx = sync["max_score"] or 0.0
        print(
            f"NEO4J_SYNC rels_with_score={sync['rels_with_score']} "
            f"avg={avg:.4f} min={sync['min_score']:.4f} max={mx:.4f}",
            flush=True,
        )
        if mx < 0.95:
            print(
                "WARN max plausibility < 0.95 — score range looks collapsed "
                "(RotatE-style). Investigate before defense.",
                flush=True,
            )

    # Step 4: Full ablation matrix — full_stack | graph_no_gnn | prompt_only
    print("EVAL_FULL_STACK_BEGIN", flush=True)
    full = await run_grounding_evaluation(mode="full_stack")
    print(
        f"EVAL_FULL_STACK grounding={full.get('grounding_score')} "
        f"faithfulness={full.get('faithfulness_score')} n={full.get('sample_count')}",
        flush=True,
    )

    print("EVAL_GRAPH_NO_GNN_BEGIN", flush=True)
    gng = await run_grounding_evaluation(mode="graph_no_gnn")
    print(
        f"EVAL_GRAPH_NO_GNN grounding={gng.get('grounding_score')} "
        f"faithfulness={gng.get('faithfulness_score')} n={gng.get('sample_count')}",
        flush=True,
    )

    print("EVAL_PROMPT_ONLY_BEGIN", flush=True)
    po = await run_grounding_evaluation(mode="prompt_only")
    print(
        f"EVAL_PROMPT_ONLY grounding={po.get('grounding_score')} "
        f"faithfulness={po.get('faithfulness_score')} n={po.get('sample_count')}",
        flush=True,
    )

    # Step 5: Threshold sweep — repopulates the paper's τ ∈ {0.30, 0.50, 0.85, 0.95}
    print("THRESHOLD_SWEEP_BEGIN", flush=True)
    sweep = await run_threshold_sweep()
    for tau, metrics_at_tau in sweep.items():
        print(
            f"SWEEP tau={tau} grounding={metrics_at_tau.get('grounding_score')} "
            f"faithfulness={metrics_at_tau.get('faithfulness_score')}",
            flush=True,
        )

    # Final preflight summary
    print(
        "PREFLIGHT_SUMMARY "
        f"full_stack_g={full.get('grounding_score')} "
        f"full_stack_f={full.get('faithfulness_score')} "
        f"graph_no_gnn_g={gng.get('grounding_score')} "
        f"prompt_only_g={po.get('grounding_score')} "
        f"recover_auc={metrics.get('final_auc_roc')} "
        f"recover_mrr={metrics.get('final_mrr')}",
        flush=True,
    )

    # Gate every KPI against its paper-defense target. These thresholds
    # mirror frontend/lib/constants.ts PAPER_KPIS and paper v6.4 §3.
    targets = [
        ("Grounding",    full.get("grounding_score"),    0.98, "H3"),
        ("Faithfulness", full.get("faithfulness_score"), 0.90, "H3"),
        ("AUC-ROC",      metrics.get("final_auc_roc"),   0.95, "H2"),
        ("MRR",          metrics.get("final_mrr"),       0.95, "H2"),
    ]
    failed: list[tuple[str, float | None, float]] = []
    for name, value, target, hypothesis in targets:
        if value is None:
            print(f"KPI_GATE {name:<13} value=None target={target} {hypothesis} WARN (missing)", flush=True)
            failed.append((name, value, target))
            continue
        passed = value >= target
        status = "PASS" if passed else "FAIL"
        print(
            f"KPI_GATE {name:<13} value={value:.4f} target={target} {hypothesis} {status}",
            flush=True,
        )
        if not passed:
            failed.append((name, value, target))

    if failed:
        print(
            "DEFENSE NOT READY — the following KPIs are below paper target:",
            flush=True,
        )
        for name, value, target in failed:
            shown = f"{value:.4f}" if isinstance(value, (int, float)) else "missing"
            print(f"  - {name}: {shown} < {target}", flush=True)
        print(
            "Investigate before defense. See docs/DEMO_RUNBOOK.md preflight section.",
            flush=True,
        )
        print(f"DONE elapsed_min={(time.time()-t0)/60.0:.2f}", flush=True)
        return 3

    print("DEFENSE READY — all four paper KPIs pass.", flush=True)
    print(f"DONE elapsed_min={(time.time()-t0)/60.0:.2f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
