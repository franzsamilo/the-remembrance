"""Grounding / Faithfulness eval on the current graph (new realization).

Completes the 4-KPI scorecard for the reroll: the GNN AUC/MRR come from the
12-seed eval; this measures the LLM-as-judge KPIs (Grounding > 0.98,
Faithfulness > 0.90) at the canonical tau=0.95, plus the prompt-only ablation
that shows the GNN's contribution.

Usage (from backend/):
    python -m run_logs.reroll_eval
"""
from __future__ import annotations

import asyncio
import os
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


async def main() -> int:
    from src.evaluation import run_grounding_evaluation

    full = await run_grounding_evaluation(mode="full_stack")
    g = full.get("grounding_score")
    f = full.get("faithfulness_score")
    print(
        f"FULL_STACK grounding={g} faithfulness={f} n={full.get('sample_count')}",
        flush=True,
    )

    po = await run_grounding_evaluation(mode="prompt_only")
    print(
        f"PROMPT_ONLY grounding={po.get('grounding_score')} "
        f"faithfulness={po.get('faithfulness_score')} n={po.get('sample_count')}",
        flush=True,
    )

    print(
        f"KPI_GROUNDING value={g} target=0.98 "
        f"{'PASS' if (g is not None and g > 0.98) else 'FAIL'}",
        flush=True,
    )
    print(
        f"KPI_FAITHFULNESS value={f} target=0.90 "
        f"{'PASS' if (f is not None and f > 0.90) else 'FAIL'}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
