"""Multi-run grounding/faithfulness on legal queries — cut through LLM-judge noise.

Two runs at the SAME setting (tau=0.95) gave Grounding 0.912 and 0.960, i.e. the
n=5 LLM-as-judge has ~+/-0.05 run-to-run noise. Reporting a single point is
misleading. This runs the full-stack eval N times and reports mean +/- std —
the same rigor the paper applies to MRR via multi-seed evaluation.

Whatever the mean is, that is what gets reported.

Usage (from backend/):
    python -m run_logs.reroll_grounding_multirun
"""
from __future__ import annotations

import asyncio
import os
import statistics
import sys

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

for _k, _v in {
    "COMPGCN_DECODER": "distmult",
    "COMPGCN_LOSS": "bpr",
    "COMPGCN_ADV_TEMP": "1.0",
    "COMPGCN_HIDDEN_CHANNELS": "256",
    "COMPGCN_DROPOUT": "0.2",
    "COMPGCN_NEG_RATIO": "15",
    "GROUNDING_MIN_SCORE": "0.95",
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
}.items():
    os.environ[_k] = _v

RUNS = 3  # n=18 queries per run already stabilizes the within-run mean
TAU = 0.95


async def main() -> int:
    from src.evaluation import run_grounding_evaluation

    print(f"GROUNDING_MULTIRUN_BEGIN runs={RUNS} tau={TAU} queries=legal", flush=True)
    gs: list[float] = []
    fs: list[float] = []
    for i in range(RUNS):
        r = await run_grounding_evaluation(
            mode="full_stack", grounding_threshold=TAU, persist_to_ablation=False
        )
        g = r.get("grounding_score")
        f = r.get("faithfulness_score")
        n = r.get("sample_count")
        if g is None or f is None:
            print(f"RUN {i+1} FAILED (g={g} f={f} n={n})", flush=True)
            continue
        gs.append(g)
        fs.append(f)
        print(f"RUN {i+1} grounding={g:.4f} faithfulness={f:.4f} n={n}", flush=True)

    if not gs:
        print("GROUNDING_MULTIRUN_FATAL no successful runs", flush=True)
        return 2

    gm, gsd = statistics.mean(gs), statistics.pstdev(gs)
    fm, fsd = statistics.mean(fs), statistics.pstdev(fs)
    print(
        f"GROUNDING     mean={gm:.4f} std={gsd:.4f} min={min(gs):.4f} max={max(gs):.4f} n_runs={len(gs)}",
        flush=True,
    )
    print(
        f"FAITHFULNESS  mean={fm:.4f} std={fsd:.4f} min={min(fs):.4f} max={max(fs):.4f} n_runs={len(fs)}",
        flush=True,
    )
    print(
        f"VERDICT grounding_mean={gm:.4f} vs>=0.95:{'PASS' if gm >= 0.95 else 'FAIL'} "
        f"vs>=0.98:{'PASS' if gm >= 0.98 else 'FAIL'} | "
        f"faithfulness_mean={fm:.4f} vs>=0.90:{'PASS' if fm >= 0.90 else 'FAIL'}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
