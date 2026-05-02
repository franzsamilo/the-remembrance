"""Run 8: BPR + self-adversarial negative weighting (alpha=1.0).

Sets COMPGCN_LOSS=bpr, COMPGCN_NEG_SAMPLING=uniform, COMPGCN_ADV_TEMP=1.0 and
runs the full audit + Neo4j sync. Targets MRR > 0.95 by reweighting BPR
gradients toward hard negatives (RotatE Sun+ 2019, eq. 5).

Rollback to Run 6 (BPR uniform-mean): re-run backend/run_logs/bpr_audit.py.
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
os.environ["COMPGCN_NEG_SAMPLING"] = os.environ.get("COMPGCN_NEG_SAMPLING", "uniform")
os.environ["COMPGCN_ADV_TEMP"] = os.environ.get("COMPGCN_ADV_TEMP", "1.0")
os.environ["COMPGCN_AUC_GUARDRAIL"] = os.environ.get("COMPGCN_AUC_GUARDRAIL", "0.95")


def main():
    from src.gnn_module import run_audit, get_training_history
    from src.evaluation import persist_gnn_metrics

    t0 = time.time()
    print("SELF_ADV_AUDIT_BEGIN", flush=True)
    print(
        f"  loss=bpr neg_sampling={os.environ['COMPGCN_NEG_SAMPLING']} "
        f"adv_temp={os.environ['COMPGCN_ADV_TEMP']} "
        f"auc_guardrail={os.environ['COMPGCN_AUC_GUARDRAIL']}",
        flush=True,
    )
    try:
        run_audit()
        th = get_training_history()
        persist_gnn_metrics(th)
        print(
            f"SELF_ADV_AUDIT_DONE elapsed_min={(time.time()-t0)/60.0:.2f} "
            f"best_epoch={th.get('best_epoch')} "
            f"auc={th.get('final_auc_roc')} "
            f"mrr_uniform={th.get('mrr_uniform')} "
            f"mrr_type_aware={th.get('mrr_type_aware')}",
            flush=True,
        )
    except Exception as e:
        import traceback
        print(f"SELF_ADV_AUDIT_FAILED {type(e).__name__} {e}", flush=True)
        traceback.print_exc()


if __name__ == "__main__":
    main()
