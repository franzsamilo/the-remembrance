"""Multi-seed MRR confirmation on the CURRENT (live) checkpoint.

Context
-------
The full-pipeline rebuild on 2026-06-20 retrained CompGCN from scratch and
OVERWROTE backend/run_logs/compgcn_best.pt. The paper's reported multi-seed
MRR=0.958 was characterized in run_logs/multi_seed_mrr_run8.log against the
PREVIOUS checkpoint (best_epoch=158, AUC 0.985), which no longer exists on disk.
The new checkpoint is best_epoch=196, AUC 0.9779 (same Run 8 config:
DistMult + BPR + self-adversarial alpha=1.0).

The full pipeline's KPI gate FAILED on MRR=0.9135 because it uses the
single-seed *training-time* MRR, whose RNG state was contaminated by ~196
epochs of training consumption (see the headline finding in
multi_seed_mrr_run8.log). This script re-confirms MRR under the canonical
seed-reset methodology (Sun et al. 2019, Vashishth et al. 2020) for the LIVE
checkpoint, so the live system's number matches what the paper claims.

Method
------
recover_from_checkpoint() reads Config.COMPGCN_SEED and re-seeds torch+numpy at
its start, then re-fetches the graph, reloads the checkpoint, re-splits the
val set, and evaluates AUC / MRR_uniform / MRR_type_aware. Because it re-seeds
deterministically on entry, calling it repeatedly in one process after
reassigning Config.COMPGCN_SEED yields results identical to the documented
per-process loop -- just without paying the torch import cost 12 times.

Usage (from backend/):
    python -m run_logs.multi_seed_confirm_current
"""
from __future__ import annotations

import os
import sys
import statistics

# Make backend/ importable when run via `python -m run_logs.multi_seed_confirm_current`.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Apply Run 8 defense env BEFORE importing src.config (which freezes them).
# COMPGCN_SEED is intentionally omitted -- it is driven per-iteration below.
DEFENSE_ENV = {
    "COMPGCN_DECODER": "distmult",
    "COMPGCN_LOSS": "bpr",
    "COMPGCN_ADV_TEMP": "1.0",
    "COMPGCN_HIDDEN_CHANNELS": "256",
    "COMPGCN_DROPOUT": "0.2",
    "COMPGCN_NEG_RATIO": "15",
    "COMPGCN_AUC_GUARDRAIL": "0.95",
    "GROUNDING_MIN_SCORE": "0.95",
    # recover_from_checkpoint touches only Neo4j + the local checkpoint (no
    # HF download, no Gemini), but keep offline flags on for parity/safety.
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
}
for _k, _v in DEFENSE_ENV.items():
    os.environ[_k] = _v

from src.config import Config  # noqa: E402
from src.gnn_module import recover_from_checkpoint  # noqa: E402

# Same 12 seeds as run_logs/multi_seed_mrr_run8.log for direct comparability.
SEEDS = [0, 1, 2, 5, 7, 11, 13, 23, 31, 42, 99, 100]
TARGET = 0.95


def _summary(name: str, xs: list[float]) -> str:
    mean = statistics.mean(xs)
    std = statistics.pstdev(xs)
    n_pass = sum(1 for v in xs if v >= TARGET)
    verdict = "PASS" if mean >= TARGET else "FAIL"
    return (
        f"STAT {name:<16} mean={mean:.4f} std={std:.4f} "
        f"min={min(xs):.4f} max={max(xs):.4f} "
        f">= {TARGET}: {n_pass}/{len(xs)} {verdict}"
    )


def main() -> int:
    print("MULTISEED_BEGIN checkpoint=compgcn_best.pt seeds=%s" % SEEDS, flush=True)
    aucs: list[float] = []
    mrr_u: list[float] = []
    mrr_t: list[float] = []

    for seed in SEEDS:
        Config.COMPGCN_SEED = seed  # recover_from_checkpoint reads this + re-seeds
        r = recover_from_checkpoint()
        if r is None:
            print(f"SEED={seed} FAILED no_result", flush=True)
            continue
        a = r["final_auc_roc"]
        mu = r["mrr_uniform"]
        mt = r["mrr_type_aware"]
        aucs.append(a)
        mrr_u.append(mu)
        mrr_t.append(mt)
        print(
            f"SEED={seed} AUC={a:.4f} MRR_uniform={mu:.4f} MRR_type_aware={mt:.4f}",
            flush=True,
        )

    if not mrr_u:
        print("MULTISEED_FATAL no seeds succeeded", flush=True)
        return 2

    print("", flush=True)
    print(_summary("AUC", aucs), flush=True)
    print(_summary("MRR_uniform", mrr_u), flush=True)
    print(_summary("MRR_type_aware", mrr_t), flush=True)

    mean_mrr_u = statistics.mean(mrr_u)
    verdict = "PASS" if mean_mrr_u >= TARGET else "FAIL"
    print(
        f"MULTISEED_DONE MRR_uniform_mean={mean_mrr_u:.4f} target={TARGET} {verdict}",
        flush=True,
    )
    return 0 if mean_mrr_u >= TARGET else 3


if __name__ == "__main__":
    raise SystemExit(main())
