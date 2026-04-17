"""BPR ablation audit run.

Sets COMPGCN_LOSS=bpr and re-runs the audit. All other hyperparameters stay the
same as the BCE run. Writes scores to Neo4j (overwrites BCE scores) so the
grounding eval that follows reflects BPR-trained plausibility values.

Rollback: to go back to the BCE run's scores, re-run audit_bce.log's command.
"""
from __future__ import annotations

import os
import sys
import time

# Running from backend/run_logs/; backend/ must be on sys.path for `src` imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Must set before importing src.config (load_dotenv runs at import-time, but
# os.environ[...] takes precedence over .env when load_dotenv uses override=False)
os.environ["COMPGCN_LOSS"] = "bpr"
os.environ["COMPGCN_BPR_MARGIN"] = os.environ.get("COMPGCN_BPR_MARGIN", "0.0")


def main():
    from src.gnn_module import run_audit, get_training_history
    from src.evaluation import persist_gnn_metrics

    t0 = time.time()
    print("BPR_AUDIT_BEGIN", flush=True)
    try:
        run_audit()
        th = get_training_history()
        persist_gnn_metrics(th)
        print(
            f"BPR_AUDIT_DONE elapsed_min={(time.time()-t0)/60.0:.2f} "
            f"best_epoch={th.get('best_epoch')} auc={th.get('final_auc_roc')} "
            f"mrr={th.get('final_mrr')}",
            flush=True,
        )
    except Exception as e:
        import traceback
        print(f"BPR_AUDIT_FAILED {type(e).__name__} {e}", flush=True)
        traceback.print_exc()


if __name__ == "__main__":
    main()
