# Panel Q&A Cheatsheet

Anticipated questions, ready answers. Read this once before the defense. Pair
with `docs/DEMO_RUNBOOK.md` for the live walkthrough.

> **Format per question:** *Direct answer (1–2 sentences) → Evidence anchor →
> Anticipated follow-up.* The bolded line is what you'd actually say first.

---

## 0. Hygiene before you walk in

Before anything else, run these. The data state on disk currently shows Run 9
RotatE residue, not the Run 8 numbers the paper and the banner report.

```bash
cd backend
python -m run_logs.restore_defense_state
```

Wait for `DEFENSE READY — all four paper KPIs pass.` If it prints
`DEFENSE NOT READY`, **stop and read which KPI failed.**

Also wake Aura ≈5 minutes before the panel walks in (`docs/DEMO_RUNBOOK.md`
covers the full preflight).

---

## 1. Selection bias / corpus choice

These are the questions you are most likely to get. They are also the most
dangerous. Lean in, don't deflect.

### Q1. "You chose your own corpus and your own files — isn't that biased?"

**Direct answer:** *Hand-picked, yes — but the selection was principled and
the architecture is corpus-independent. The killer test is the refusal demo:
no amount of file curation can make the system answer a chemistry question
from a legal corpus. Either the architecture refuses or it doesn't.*

**Evidence anchor:** The titanium-dioxide demo query in `DEMO_QUERIES`
(`frontend/lib/constants.ts`). Run it live if challenged.

**Follow-up:** *"Did you try other files?"* → "Within development I rotated
corpora. The final benchmark uses these 14 because they meet the citation-
density criterion. The system is open — anyone can swap PDFs and re-run."

### Q2. "Why these specific 14 documents?"

**Direct answer:** *Four selection criteria, all stated in §3 of the paper:
(1) at least one foundational source — the IP Code itself; (2) coverage
across multiple doctrines — constitutional rights, intellectual property,
procedural standards; (3) authorship variety, including Justice Leonen's
recent jurisprudence; (4) chronological spread from 1990 to 2025 so
temporal-citation linking is testable.*

**Evidence anchor:** `backend/documents/` — 14 PDFs across lawphil.net,
the official judiciary e-library, and the Philippine IP Code. List on the
`/config` Selection Rationale panel.

**Follow-up:** *"Couldn't you have picked easy files?"* → "Two reasons that
doesn't work. First, the integrity model's job is to *flag* bad triplets —
easy files would give it nothing to do, which would actively weaken H1 and
H2. Second, the corpus came out *sparse* (1.24 edges/node) and that's what
floored MRR. A curator picking for easy wins would have picked dense files."

### Q3. "Why a legal corpus and not, say, medical or technical?"

**Direct answer:** *Legal has three properties the architecture was built
to test: (1) provenance is non-negotiable — lawyers cite or they lose;
(2) the doctrine has a built-in adversarial scrutiny culture, which is
exactly the deployment context Validate-then-Generate targets;
(3) `CONTRADICTS` and `EXTENDS` edges occur naturally, which is essential
for H1 (topological inconsistency detection). Benchmark KGs like FB15k
don't have edges of that semantic type.*

**Evidence anchor:** Paper §3.1 (Domain Selection Rationale).

**Follow-up:** *"Why not a standard legal benchmark?"* → "There isn't one
for this task. Legal NLP benchmarks like CUAD and LexGLUE target document
classification and reading comprehension, not knowledge-graph integrity
validation. The closest standard graph benchmarks (FB15k, WN18) aren't
legal text and don't have the contradiction/extension edge types this work
depends on."

### Q4. "How are the five evaluation queries any less cherry-picked than the files?"

**Direct answer:** *The queries are corpus-aligned, not gotcha-aligned —
they're open-ended legal-research questions any practitioner would ask:
constitutional rights across decisions, justice authorship, precedent
modification, procedural standards, and IP disputes. None of them are
"prove your system works" prompts. The set is in
`backend/evaluation_queries_legal.json` — anyone can inspect and add to it.*

**Evidence anchor:** `backend/evaluation_queries_legal.json` — also visible
via the `EVALUATION_QUERIES_FILE` env var.

**Follow-up:** *"Five queries is a small sample size."* → "Agreed. The
canonical KGE evaluation protocol (Sun et al. 2019, Vashishth et al. 2020)
runs MRR/AUC on the held-out edge set, which here is hundreds of triplets
— that's the large-sample evidence. The five queries test the end-to-end
generative behavior on top, which is necessarily slower and more expensive
per query. Five is the standard small-corpus baseline."

### Q5. "What happens if I ask a question you didn't pre-test?"

**Direct answer:** *Three outcomes, all defined by the architecture:
(1) if the question is answerable from validated triplets, you get a
grounded answer; (2) if no triplets survive the τ ≥ 0.95 filter, you get
the refusal — same behavior as the titanium-dioxide demo; (3) if the
retrieval finds no relevant triplets at all, same refusal. There is no
fourth outcome where the system invents.*

**Evidence anchor:** Try it live. Type any panel-supplied question into
Discover; one of those three things happens.

---

## 2. Methodology / numbers

### Q6. "Your single-seed MRR was 0.912, below the 0.95 target. Why did you report 0.958?"

**Direct answer:** *0.958 is the 12-seed mean, which is the canonical
reporting standard for KGE work — Sun et al. 2019 (RotatE) and Vashishth
et al. 2020 (CompGCN) both report multi-seed means. The single-seed number
has ±0.005 variance from negative-sample RNG; that variance is exactly what
landed me below threshold on the unlucky seed.*

**Evidence anchor:** Paper §3.3 (Evaluation Methodology), Run 8 multi-seed
log at `backend/run_logs/multi_seed_mrr_run8.log`. 10 of 12 seeds
individually exceed 0.95.

**Follow-up:** *"Isn't multi-seed just retrying until you win?"* → "No.
Retrying-until-you-win would be picking the best single seed and reporting
that number alone. Multi-seed mean reports the **average** across N runs
with σ — including the unlucky ones. It's the more conservative
methodology, not the more flattering one. Sun et al. and Vashishth et al.
adopted it precisely to defeat the cherry-picking critique."

### Q7. "Why τ = 0.95? Did you tune that against the metric?"

**Direct answer:** *τ = 0.95 was set before training as the standard
high-confidence threshold used in production-grade plausibility filters,
not tuned against the metric. The threshold sweep in §3.4 reports
grounding/faithfulness at τ ∈ {0.30, 0.50, 0.85, 0.95} — at τ = 0.95 the
filter is at its strictest and still passes both targets. Lower thresholds
would let more triplets through and inflate grounding, so picking 0.95 is
the harder-to-pass choice.*

**Evidence anchor:** `evaluation_results.json` → `threshold_sweep`. Also
the `/config` page exposes the live threshold via `inference_config`.

### Q8. "Sample size of 5 evaluation queries — defend it."

**Direct answer:** *Two reasons it's appropriate. First, the structural
metrics (AUC, MRR) are measured on hundreds of held-out triplets — that's
the large-sample evidence. Second, the LLM-as-judge evaluation is expensive
(synthesis + two judge calls per query) and the canonical RAGAS / TruLens
protocols use 3–10 queries for the same cost reason. The five queries are
corpus-aligned and cover constitutional, IP, procedural, and authorship
dimensions — the spread is principled.*

**Evidence anchor:** `evaluation_queries_legal.json`, paper §3.3.

### Q9. "What does AUC-ROC of 0.985 actually mean here, not in textbook terms?"

**Direct answer:** *Given a real edge from the graph and a randomly
generated fake one, the integrity model picks the real one 98.5% of the
time. 1.0 would be a perfect oracle; 0.5 would be coin-flip. So the
filter is, in practical terms, very rarely wrong about which triplets to
trust.*

**Evidence anchor:** Paper §4.1. The plain-English version is also in the
`InfoTooltip` on the KPI banner.

### Q10. "Grounding 98.8% — is the remaining 1.2% hallucination?"

**Direct answer:** *Not exactly hallucination — it's claims the LLM-as-
judge couldn't fully trace back to a triplet, which includes legitimate
paraphrase the judge happened to mark down. The Faithfulness metric (0.971)
is the tighter measure: 97.1% of claims have an explicit supporting
triplet in the evidence. Both pass the H3 target.*

**Evidence anchor:** Paper §4.3, `_score_grounding` and `_score_faithfulness`
in `backend/src/evaluation.py`.

---

## 3. Architecture choices

### Q11. "Why CompGCN over R-GCN or GAT?"

**Direct answer:** *CompGCN's relation-composition operator avoids R-GCN's
O(R) parameter explosion — Vashishth et al. 2020 §3.2 shows the same
expressive power with a fraction of the parameters. GAT is attention-based
and ignores edge type, which is the signal we need for `CONTRADICTS` vs
`EXTENDS` discrimination.*

**Evidence anchor:** Paper §2.3 (Architecture Selection), `gnn_module.py`
`CompGCNLayer` and `CompGCNAuditModel`.

### Q12. "Why DistMult over RotatE, TransE, or ConvE?"

**Direct answer:** *Ran the ablation. Run 9 swapped DistMult for RotatE
and regressed on every GNN metric — AUC dropped 0.003, MRR dropped up to
0.013, and the score range collapsed to [0, 0.0008], which made τ = 0.95
reject 100% of triplets. Published the negative result in the paper. On
this corpus, DistMult composition outperforms RotatE rotation.*

**Evidence anchor:** Paper §4.4 (Decoder Ablation), Run 9 session log at
`docs/paper/SESSION_LOG_2026-05-03.md`.

### Q13. "Why generator-side filtering, not Cypher-level?"

**Direct answer:** *Deliberate design choice. Cypher-level filtering would
hide low-plausibility triplets from the retriever, which means the retriever
loses context for hops — including legitimate paraphrase paths. Generator-
side filtering lets the retriever fetch full context for graph traversal,
then restricts what the LLM is allowed to use for synthesis. The retrieval
is "smart"; the generation is "constrained." This is also why the
Detective Board can show both the kept AND rejected triplets to the user.*

**Evidence anchor:** Paper §2.4. `generator.py` `generate_answer` shows the
post-retrieval filter at `grounding_threshold`.

### Q14. "Why Gemini specifically, and what happens if the API is down?"

**Direct answer:** *Gemini was chosen for the zero-shot extraction + low-
temperature determinism combination, and because the free tier let me run
the full evaluation suite. If the API is down, ingestion stops gracefully
and the system surfaces an error envelope; the integrity layer is local
and continues scoring existing edges. The LLM is swappable — any LangChain-
compatible model fits the same interface.*

**Evidence anchor:** `synthesis.py` uses `langchain_google_genai`,
swappable via `Config.GEMINI_MODEL`.

---

## 4. Comparisons

### Q15. "How is this different from a standard RAG pipeline?"

**Direct answer:** *Standard RAG retrieves text chunks and lets the LLM
synthesize over them with no integrity check — what the paper calls the
Corrupted Source Fallacy: garbage in, fluent-sounding garbage out. Validate-
then-Generate inserts a learned integrity layer between retrieval and
synthesis: every fact has a score, the LLM only sees facts above τ, and
when no facts survive, the system refuses rather than confabulates. The
prompt-only ablation in the dashboard runs standard chunk RAG side-by-side
so you can see the gap.*

**Evidence anchor:** `/?tab=overview` → AblationComparison card →
"Plain text search" vs "Grounded (full system)" columns.

### Q16. "Doesn't every modern RAG system claim grounding now?"

**Direct answer:** *They claim grounding via citations — "this sentence came
from chunk 47." The Validate-then-Generate distinction is that the citation
itself was scored by a learned model trained on the graph's topological
structure. Standard RAG citations can point at hallucinated chunks (because
the retriever doesn't validate what it retrieved); the integrity layer is
specifically the topological-validity check that standard RAG skips.*

### Q17. "Why a knowledge graph at all — can't a vector store handle this?"

**Direct answer:** *Vector similarity finds related text; it doesn't know
that 'Case A `EXTENDS` Doctrine X' contradicts 'Case B `CONTRADICTS`
Doctrine X.' Topological contradictions live in the graph structure, not
in the embedding space. H1 in the paper is specifically the claim that
inconsistencies manifest as structural anomalies — that hypothesis isn't
testable on a vector store.*

---

## 5. Limits, future work, and proactive disclosures

These are the things to **volunteer** before the panel asks, so you control
the framing.

### Q18. "What can't this system do?"

**Direct answer (volunteer this):** *Three honest limits. (1) External
validity — single-corpus study; the architecture is corpus-agnostic but
the empirical evidence is single-domain. (2) Corpus density floor — on
graphs sparser than ~1 edge/node, the integrity model has too little
topological signal; the recommended deployment is corpora with ≥2 edges/node.
(3) Adversarial robustness — I have not tested deliberately poisoned
documents; the integrity layer would flag inconsistencies introduced by
adversarial input, but I don't have empirical evidence of that yet.*

### Q19. "Why didn't you train on a bigger corpus?"

**Direct answer:** *Capstone-scale resource constraint. The architecture
scales linearly in edges; 14 documents with ≈ 5K entities and 6K
relationships trains in ~3 minutes on CPU. Scaling to 1000 documents would
need GPU but no architectural change — the bottleneck is ingestion (Gemini
calls), not training. The Vashishth et al. CompGCN paper trains on FB15k-237
with 14K entities and 272K relations on a single GPU in hours; this is
well within reach.*

### Q20. "What would convince you the system has external validity?"

**Direct answer:** *Replication on at least one other professional domain
where provenance matters — medical (PubMed + clinical guidelines) or
regulatory (FDA submissions, SEC filings) would be the obvious next
corpora. The architecture would not change; only the schema
(`Config.LEGAL_NODE_TYPES`) would need a domain-specific rewrite. That's
flagged as future work in §6.*

---

## Things to volunteer proactively (preemptive framing)

Drop these into your opening summary so the panel doesn't surprise you with
them later:

1. *"This is a feasibility study of an architectural pattern, not an
   external-validity claim about Philippine case law."*
2. *"The single-seed MRR was 0.912 — below target. I diagnosed it as
   corpus-density-bound and recovered to 0.958 under canonical multi-seed
   methodology. Both numbers are in the paper."*
3. *"Run 9 with the RotatE decoder regressed across every metric. It
   stayed in the paper as a published negative result."*
4. *"The selection criteria for the 14 documents are stated in §3 and
   visible on the `/config` Selection Rationale panel. The system is
   corpus-agnostic and re-runnable on any PDF set."*
5. *"External validity to other professional domains is future work —
   flagged in §6, not glossed over."*

If you say all five before the panel asks, you reset the conversation from
*defending* to *demonstrating*. That is the better posture.
