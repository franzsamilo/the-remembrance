"""Run 9 finer threshold sweep for RotatE-trained scores.

RotatE's edge_scores = sigmoid(-distance) where distance >= 0, so scores
are bounded above by 0.5. The standard post_audit_eval threshold sweep
(0.30, 0.50, 0.85, 0.95) may report 'no triplets pass' at the high end.
This finer sweep at (0.20, 0.30, 0.35, 0.40, 0.45, 0.49) calibrates the
canonical RotatE tau analogous to Run 5's tau=0.30 calibration for
BCE+label-smoothing.

Run after rotate_audit.py + post_audit_eval.py if the standard sweep
returns null at high tau.
"""
from __future__ import annotations

import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


async def main():
    from src.evaluation import run_threshold_sweep

    print("ROTATE_FINER_SWEEP_BEGIN", flush=True)
    # Empirically RotatE scores on this corpus span [0, ~0.001] (sigmoid(-distance)
    # collapses near 0 because positive-edge distances are large). Standard sweep
    # at tau >= 0.30 rejects all triplets. Sweep across the actual range.
    thresholds = [0.0001, 0.0003, 0.0005, 0.0007, 0.001, 0.005, 0.01]
    sweep = await run_threshold_sweep(thresholds=thresholds)
    for tau, metrics in sweep.get("results", {}).items():
        print(
            f"FINER_SWEEP tau={tau} "
            f"G={metrics.get('grounding_score')} "
            f"F={metrics.get('faithfulness_score')} "
            f"n={metrics.get('sample_count')}",
            flush=True,
        )
    print("ROTATE_FINER_SWEEP_DONE", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
