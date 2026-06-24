"""Reconcile the hand-cleaned docx (figures/formatting) with the verified content.

The cleaned docx already integrated most of v6.11 (its own §5.8.1 'Post-Rebuild',
recovered grounding 0.984, the §3.3.3 expansion note, the campaign-graph Table 5.10
disambiguation). Only a small, bounded content delta remains. This applies it
IDEMPOTENTLY (each edit replaces if its exact target is present, else skips — so it
cannot corrupt the hand-edited structure) and works on a COPY, leaving the original
untouched.

Decision: Option A (fits the cleaned docx's campaign-Table-5.10 structure) — §3.3.3 is
rewritten to honestly describe the 5 general campaign-probe queries; Table 5.10 stays
as the campaign result (already labelled campaign-graph in §5.8.1); the live 18-query
legal eval remains prominent.

Input : Project_Study_Report__The_Remembrance_6_11_cleaned.docx
Output: Project_Study_Report__The_Remembrance_6_12_cleaned.docx
"""
from __future__ import annotations

import os
from docx import Document

ROOT = r"C:/Users/Franz Samilo/Desktop/the-remembrance"
INPUT_DOCX = os.path.join(ROOT, "Project_Study_Report__The_Remembrance_6_11_cleaned.docx")
OUTPUT_DOCX = os.path.join(ROOT, "Project_Study_Report__The_Remembrance_6_12_cleaned.docx")

EDITS = [
    # (1) §3.3.3 (P273) — Option A: describe the REAL general campaign queries, honestly.
    ("Third, the queries are deliberately corpus-aligned rather than gotcha-aligned. The five "
     "queries cover: (a) constitutional rights across decisions, (b) majority-opinion authorship "
     "attribution, (c) precedent citation and modification, (d) procedural standards for motions "
     "for reconsideration, and (e) intellectual property disputes. These represent open-ended "
     "legal-research questions a practitioner would pose, not adversarial probes constructed to "
     "favor any particular architectural outcome.",
     "Third, the queries are open-ended rather than gotcha-aligned. The five campaign queries are "
     "general, domain-agnostic corpus probes — the principal findings, the main contributors, the "
     "methods applied, the principal results, and the datasets and concepts discussed — chosen to "
     "exercise retrieval and synthesis end-to-end without being constructed to favour any "
     "particular architectural outcome. The final live-system evaluation (§5.8.1) replaces these "
     "with eighteen corpus-aligned intellectual-property questions specific to this corpus (the "
     "dominancy and holistic tests, unfair competition, copyright scope, collective-rights "
     "enforcement, and IPOPHL administrative procedure)."),
    # (2) abstract uplift -> live (P14)
    ("baseline is +45% Grounding and +204% Faithfulness",
     "baseline is +75% Grounding and +423% Faithfulness"),
    # (3) conclusion uplift -> live (P460); leaves campaign-progression +45% spots intact
    ("+45% Grounding and +204% Faithfulness uplift of the full-stack",
     "+75% Grounding and +423% Faithfulness uplift of the full-stack"),
    # (4) §5.8.1 (P395) — append 9-seed KPIs + the committed second grounding run
    ("and the structural KPIs (AUC-ROC, MRR) hold on the rebuilt graph.",
     "and the structural KPIs reproduce on the rebuilt graph (nine-seed mean: AUC-ROC 0.986, "
     "MRR 0.959, 12/12 PASS). Against the prompt-only chunk-RAG baseline on the same eighteen-query "
     "set, the integrity layer's uplift is +75% Grounding and +423% Faithfulness (baseline "
     "0.561 / 0.188); a second committed five-run evaluation reproduced grounding at 0.993 ± 0.011 "
     "(four of five runs ≥ 0.98), confirming the result is not a single-run artifact. The passing "
     "artifacts are committed to the repository (multi_seed_confirm_current.json, "
     "reroll_grounding_confirm.json, reroll_eval_commit.json)."),
]


def replace_once(par, old, new):
    runs = par.runs
    full = "".join(r.text for r in runs)
    idx = full.find(old)
    if idx == -1:
        return 0
    end = idx + len(old)
    pos = 0; ranges = []
    for r in runs:
        ranges.append((pos, pos + len(r.text), r)); pos += len(r.text)
    inserted = False
    for rs, re_, r in ranges:
        if re_ <= idx or rs >= end:
            continue
        ls, le = max(idx, rs) - rs, min(end, re_) - rs
        seg = r.text
        r.text = (seg[:ls] + new + seg[le:]) if not inserted else (seg[:ls] + seg[le:])
        inserted = True
    return 1


def main():
    if not os.path.exists(INPUT_DOCX):
        print(f"FATAL no input: {INPUT_DOCX}"); return 1
    print(f"OPEN {os.path.basename(INPUT_DOCX)}", flush=True)
    doc = Document(INPUT_DOCX)
    paras = list(doc.paragraphs)
    for t in doc.tables:
        for row in t.rows:
            for c in row.cells:
                paras.extend(c.paragraphs)

    applied = 0
    for i, (old, new) in enumerate(EDITS, 1):
        hit = sum(replace_once(p, old, new) for p in paras)
        label = ["§3.3.3 query desc", "abstract uplift", "conclusion uplift", "§5.8.1 KPI append"][i-1]
        print(f"  EDIT {i} [{label}]: {'APPLIED' if hit else 'SKIPPED (target not found)'}", flush=True)
        applied += hit

    # verify
    joined = "\n".join(p.text for p in doc.paragraphs)
    checks = {
        "general, domain-agnostic corpus probes": "§3.3.3 rewritten",
        "constitutional rights across decisions": "OLD §3.3.3 gone (want absent)",
        "+75% Grounding and +423% Faithfulness uplift of the full-stack": "conclusion uplift live",
        "nine-seed mean: AUC-ROC 0.986": "§5.8.1 KPIs appended",
        "reproduced grounding at 0.993": "second-run note",
    }
    print("VERIFY:", flush=True)
    for tok, desc in checks.items():
        present = tok in joined
        print(f"  {'YES' if present else 'no ':3} {desc}", flush=True)

    doc.save(OUTPUT_DOCX)
    print(f"SAVED {os.path.basename(OUTPUT_DOCX)}  (edits applied: {applied}/4)", flush=True)
    print(f"tables={len(doc.tables)} paras={len(doc.paragraphs)} "
          f"images={sum(1 for r in doc.part.rels.values() if 'image' in r.reltype)} (figures preserved)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
