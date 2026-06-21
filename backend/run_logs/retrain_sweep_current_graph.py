"""Characterize the achievable CompGCN MRR ceiling on the CURRENT graph.

Post-2026-06-20 incident: the rebuilt graph yields MRR ~0.877 (< 0.95 target)
under the Run-8 config. This sweep tests whether training-seed variance closes
the gap. For each training seed it trains from scratch (run_audit), saves that
checkpoint aside, then estimates clean MRR via seed-reset eval
(recover_from_checkpoint) over a few eval seeds. The overall best checkpoint
(including the Jun 20 control already on disk) is left as compgcn_best.pt and
re-synced to Neo4j, so the live system ends on its strongest model.

Safe to run: the graph is backed up (run_logs/graph_snapshot_*.jsonl) and the
Jun 20 checkpoint is preserved (run_logs/compgcn_best.jun20.pt).

Usage (from backend/):
    python -m run_logs.retrain_sweep_current_graph
"""
from __future__ import annotations

import os
import sys
import shutil
import statistics

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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

from src.config import Config  # noqa: E402
from src.gnn_module import (  # noqa: E402
    run_audit,
    recover_from_checkpoint,
    CHECKPOINT_PATH,
    CHECKPOINT_META_PATH,
)

HERE = os.path.dirname(os.path.abspath(__file__))
TRAIN_SEEDS = [0, 7, 13]
EVAL_SEEDS = [0, 42]
TARGET = 0.95


def eval_current(label: str) -> tuple[float, float]:
    """Mean AUC / MRR_uniform of whatever checkpoint is at CHECKPOINT_PATH."""
    aucs: list[float] = []
    mrrs: list[float] = []
    for es in EVAL_SEEDS:
        Config.COMPGCN_SEED = es
        r = recover_from_checkpoint()
        if r is None:
            print(f"EVAL {label} seed={es} FAILED no_result", flush=True)
            continue
        aucs.append(r["final_auc_roc"])
        mrrs.append(r["mrr_uniform"])
        print(
            f"EVAL {label} seed={es} AUC={r['final_auc_roc']:.4f} "
            f"MRR_uniform={r['mrr_uniform']:.4f}",
            flush=True,
        )
    return (
        statistics.mean(aucs) if aucs else 0.0,
        statistics.mean(mrrs) if mrrs else 0.0,
    )


def save_aside(tag: str) -> tuple[str, str]:
    pt = os.path.join(HERE, f"sweep_{tag}.pt")
    mj = os.path.join(HERE, f"sweep_{tag}_meta.json")
    shutil.copy(CHECKPOINT_PATH, pt)
    shutil.copy(CHECKPOINT_META_PATH, mj)
    return pt, mj


def main() -> int:
    candidates: dict[str, tuple[float, float, str, str]] = {}

    # Control: the checkpoint currently on disk is the Jun 20 model.
    print("SWEEP_CONTROL eval=jun20 (current compgcn_best.pt)", flush=True)
    pt, mj = save_aside("jun20")
    a, m = eval_current("jun20")
    candidates["jun20"] = (a, m, pt, mj)
    print(f"SWEEP_CONTROL_DONE meanAUC={a:.4f} meanMRR={m:.4f}", flush=True)

    for ts in TRAIN_SEEDS:
        print(f"SWEEP_TRAIN seed={ts} BEGIN", flush=True)
        Config.COMPGCN_SEED = ts
        run_audit()  # trains from scratch, saves best to compgcn_best.pt, syncs
        pt, mj = save_aside(f"train{ts}")
        a, m = eval_current(f"train{ts}")
        candidates[f"train{ts}"] = (a, m, pt, mj)
        print(f"SWEEP_TRAIN seed={ts} DONE meanAUC={a:.4f} meanMRR={m:.4f}", flush=True)

    best_tag = max(candidates, key=lambda k: candidates[k][1])

    print("", flush=True)
    print(f"SWEEP_SUMMARY (mean over eval seeds {EVAL_SEEDS}):", flush=True)
    for tag, (a, m, _, _) in sorted(candidates.items(), key=lambda kv: -kv[1][1]):
        flag = "  <-- BEST" if tag == best_tag else ""
        verdict = "PASS" if m >= TARGET else "FAIL"
        print(f"  {tag:<10} meanAUC={a:.4f} meanMRR={m:.4f} {verdict}{flag}", flush=True)

    # Leave the live system on the strongest checkpoint found.
    ba, bm, bpt, bmj = candidates[best_tag]
    shutil.copy(bpt, CHECKPOINT_PATH)
    shutil.copy(bmj, CHECKPOINT_META_PATH)
    Config.COMPGCN_SEED = 42
    recover_from_checkpoint()
    verdict = "PASS" if bm >= TARGET else "FAIL"
    print(
        f"SWEEP_DONE best={best_tag} meanMRR={bm:.4f} target={TARGET} {verdict} "
        f"(restored to compgcn_best.pt + re-synced to Neo4j)",
        flush=True,
    )
    return 0 if bm >= TARGET else 3


if __name__ == "__main__":
    raise SystemExit(main())
