"""Run 9: BPR + self-adversarial alpha=1.0 + RotatE decoder.

Sets COMPGCN_LOSS=bpr, COMPGCN_NEG_SAMPLING=uniform, COMPGCN_ADV_TEMP=1.0,
COMPGCN_DECODER=rotate and runs the full audit + Neo4j sync. Targets
MRR > 0.95 by replacing DistMult's symmetric scoring with RotatE's
relational rotation in complex space (Sun et al. 2019).

Encoder is unchanged from Run 8 (3-layer CompGCN + LayerNorm). Loss is
unchanged from Run 8 (BPR + self-adversarial alpha=1.0). Only the decoder
swaps DistMult -> RotatE.

Rollback to Run 8 (DistMult decoder): re-run backend/run_logs/self_adversarial_audit.py.
"""
from __future__ import annotations

import os
import sys
import time

# Running from backend/run_logs/; backend/ must be on sys.path for `src` imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Must set before importing src.config
os.environ["COMPGCN_LOSS"] = "bpr"
os.environ["COMPGCN_BPR_MARGIN"] = os.environ.get("COMPGCN_BPR_MARGIN", "0.0")
os.environ["COMPGCN_NEG_SAMPLING"] = os.environ.get("COMPGCN_NEG_SAMPLING", "uniform")
os.environ["COMPGCN_ADV_TEMP"] = os.environ.get("COMPGCN_ADV_TEMP", "1.0")
os.environ["COMPGCN_DECODER"] = os.environ.get("COMPGCN_DECODER", "rotate")
os.environ["COMPGCN_AUC_GUARDRAIL"] = os.environ.get("COMPGCN_AUC_GUARDRAIL", "0.95")


def main():
    from src.gnn_module import run_audit, get_training_history
    from src.evaluation import persist_gnn_metrics

    t0 = time.time()
    print("ROTATE_AUDIT_BEGIN", flush=True)
    print(
        f"  loss=bpr neg_sampling={os.environ['COMPGCN_NEG_SAMPLING']} "
        f"adv_temp={os.environ['COMPGCN_ADV_TEMP']} "
        f"decoder={os.environ['COMPGCN_DECODER']} "
        f"auc_guardrail={os.environ['COMPGCN_AUC_GUARDRAIL']}",
        flush=True,
    )
    try:
        run_audit()
        th = get_training_history()
        persist_gnn_metrics(th)
        print(
            f"ROTATE_AUDIT_DONE elapsed_min={(time.time()-t0)/60.0:.2f} "
            f"best_epoch={th.get('best_epoch')} "
            f"auc={th.get('final_auc_roc')} "
            f"mrr_uniform={th.get('mrr_uniform')} "
            f"mrr_type_aware={th.get('mrr_type_aware')}",
            flush=True,
        )
    except Exception as e:
        import traceback
        print(f"ROTATE_AUDIT_FAILED {type(e).__name__} {e}", flush=True)
        traceback.print_exc()


if __name__ == "__main__":
    main()
