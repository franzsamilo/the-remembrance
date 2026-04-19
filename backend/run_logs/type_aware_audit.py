"""BPR + type-aware negative sampling audit run.

Sets COMPGCN_LOSS=bpr and COMPGCN_NEG_SAMPLING=type_aware, runs the audit,
prints a single-line summary suitable for greppable TUNING_LOG entries.

Rollback: re-run `bpr_audit.py` to return to uniform-sampling BPR scores.
"""
from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ["COMPGCN_LOSS"] = "bpr"
os.environ["COMPGCN_BPR_MARGIN"] = os.environ.get("COMPGCN_BPR_MARGIN", "0.0")
os.environ["COMPGCN_NEG_SAMPLING"] = "type_aware"


def main():
    from src.gnn_module import run_audit, get_training_history
    from src.evaluation import persist_gnn_metrics

    t0 = time.time()
    print("TYPE_AWARE_AUDIT_BEGIN", flush=True)
    try:
        run_audit()
        th = get_training_history()
        persist_gnn_metrics(th)
        print(
            f"TYPE_AWARE_AUDIT_DONE elapsed_min={(time.time()-t0)/60.0:.2f} "
            f"best_epoch={th.get('best_epoch')} "
            f"auc={th.get('final_auc_roc')} "
            f"mrr_uniform={th.get('mrr_uniform')} "
            f"mrr_type_aware={th.get('mrr_type_aware')} "
            f"guardrail_tripped={th.get('guardrail_tripped')} "
            f"pool_sizes={th.get('label_pool_sizes')}",
            flush=True,
        )
    except Exception as e:
        import traceback
        print(f"TYPE_AWARE_AUDIT_FAILED {type(e).__name__} {e}", flush=True)
        traceback.print_exc()


if __name__ == "__main__":
    main()
