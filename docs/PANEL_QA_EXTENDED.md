# Panel Q&A — Extended Drilldown

Tier-2 prep doc. `PANEL_QA.md` covers the 20 most likely questions; this
covers the harder follow-ups, the rapid-fire round, and the meta-skills
(rehearsal script, questions to ask the panel, depth-3 followups).

Use this **after** you can recite `PANEL_QA.md` cold.

---

## 0. The 90-second opening (rehearse verbatim)

Memorize this. It preempts five of the six most likely questions and resets
the panel's posture from skeptical to curious.

> *Thank you. Before I demo, three framing points so we use our time well.*
>
> *First, this is a **feasibility study of an architectural pattern**, not an
> external-validity claim about Philippine case law. The contribution is the
> Validate-then-Generate pattern — inserting a learned integrity layer
> between retrieval and synthesis — and the empirical question is whether
> that pattern works at all on a high-stakes professional corpus.*
>
> *Second, all four paper KPIs pass under the recommended configuration:
> Grounding 0.988, Faithfulness 0.971, AUC-ROC 0.985, MRR 0.958 — the last
> two as 12-seed means under canonical KGE methodology, Sun et al. 2019 and
> Vashishth et al. 2020.*
>
> *Third, I want to flag two embarrassments up front. Single-seed MRR was
> 0.912 — below the 0.95 target — and I diagnosed it as corpus-density-bound.
> The multi-seed recovery is in the paper, but so is the original
> shortfall. Separately, Run 9 with a RotatE decoder regressed across every
> metric, and that's also in the paper as a published negative result.*
>
> *With that in mind, I'd like to walk through three demo queries — one
> showing cross-document precedent, one showing relational reasoning, and
> one showing the refusal behavior that I think is the strongest evidence
> that the architecture is not just curve-fitting.*

That is your 90 seconds. Speak it deliberately, not fast. When you finish,
**the panel has heard your worst numbers from your mouth first** — which
neutralizes their power as ambush material.

---

## 1. Statistical rigor — the questions you haven't been asked yet

### EQ1. "Where are your confidence intervals?"

**Direct:** *Reported. AUC-ROC: 0.985 ± 0.001 (12-seed σ); MRR: 0.958 ±
0.005 (12-seed σ). Grounding and Faithfulness are point estimates because
the LLM-as-judge protocol is itself stochastic; the variance across repeated
runs of the same query is approximately ±0.05 (paper §6.5), which is why
the threshold for "passes" is set generously above target.*

**Followup-trap:** *"σ = 0.001 across 12 seeds is suspiciously tight."* →
"It reflects that AUC is a ranking metric over a large held-out edge set
(hundreds of edges per evaluation), while MRR is per-positive-edge with
K=15 negatives and so has more sampling noise. Both behaviors are
consistent with KGE-literature findings — see Sun et al. §4.2 for the same
pattern on FB15k-237."

### EQ2. "What's your effect size, not just your p-value?"

**Direct:** *No p-values in the paper because this isn't a null-hypothesis
significance test — it's an ablation study. The effect sizes I report are
relative-uplift percentages over the prompt-only baseline: +45% Grounding,
+204% Faithfulness. The architectural baseline (prompt-only chunk RAG)
provides the comparator, not a randomized control.*

**Followup-trap:** *"+204% sounds huge; is the denominator small?"* →
"Yes — prompt-only Faithfulness is 0.32; full-stack is 0.97. The relative
percentage looks dramatic because the denominator is low, but the absolute
delta (0.65) is what matters for the architectural argument: the integrity
filter is responsible for two-thirds of the achievable Faithfulness."

### EQ3. "How would you respond to a Bonferroni correction?"

**Direct:** *Not applicable in the conventional sense — the four KPIs aren't
independent multiple-hypothesis tests against a null. They're the four
target metrics defined in §3.3 a priori from the hypothesis structure. The
risk Bonferroni addresses (inflated Type-I error from many tests) doesn't
apply when the metrics are pre-registered as the success criteria.*

**Followup-trap:** *"Pre-registered?"* → "Targets are stated in §3.3
Table 3.3 before any results appear in Chapter 5. The four targets — AUC
0.95, MRR 0.95, Grounding 0.95, Faithfulness >baseline — are the same four
in the abstract, methodology, and discussion. The order of definition
preceded the experiments."

### EQ4. "Your sample size in evaluation queries is five. How is that defensible?"

Covered in `PANEL_QA.md` Q8 and now in the paper at §3.3.3. The new framing
beat: *"It's the convention. RAGAS uses 3–10; TruLens uses 5–10. Larger
n is in future work."* No more elaboration unless they push.

### EQ5. "What's your inter-rater reliability for the LLM-as-judge?"

**Direct:** *Single-judge protocol — Gemini 2.5 Flash scoring at T=0 for
determinism. Inter-judge reliability is what RAGAS and TruLens both skip
for the same compute-cost reason; the canonical approach is the prompt
template, not multi-judge consensus. Adding human-judge ground truth on a
held-out subset is future work in §6.5.*

**Followup-trap:** *"Determinism at T=0 isn't true reliability."* → "Agreed.
Determinism rules out within-judge variance, not between-judge variance.
The paper does not claim the latter. The honest framing is: this work
measures whether the Validate-then-Generate filter improves the metric
under a fixed judge, not whether the metric itself is the ground truth."

---

## 2. Comparison to prior art — the "have you read X" questions

### EQ6. "How does this compare to Microsoft GraphRAG?"

**Direct:** *Different problem framing. GraphRAG (Edge et al. 2024) targets
**global sensemaking** — questions whose answers span an entire corpus —
using community summarization. The Validate-then-Generate pattern targets
**local accuracy** — every triplet in the answer should be verifiable. The
two systems compose: GraphRAG could provide the community structure;
CompGCN could provide the per-edge integrity score. The paper cites GraphRAG
in §2.6 (Hybrid Retrieval and Global Sensemaking) as the architectural
relative; it is not a benchmark target.*

**Followup-trap:** *"So GraphRAG with an integrity layer is your real
contribution?"* → "That's a fair framing of what the architecture composes
to. The thesis-level contribution is the integrity-layer pattern itself —
specifically, that a CompGCN trained on the graph's topology can produce
plausibility scores tight enough to gate generation without destroying
recall."

### EQ7. "How does this compare to RAGAS?"

**Direct:** *RAGAS is the **evaluation protocol** I use, not a competing
system. RAGAS scores RAG outputs on Faithfulness, Answer Relevance, Context
Precision, and Context Recall. My evaluation pipeline reproduces the RAGAS
Faithfulness protocol (LLM-as-judge against retrieved triplets) and adds
Grounding (claim traceability) as a tighter measure. The numbers in §5.8
are RAGAS-style scores against my system, not against the RAGAS reference
implementation.*

### EQ8. "How does this compare to Barry et al. 2025 (GraphRAG finance)?"

**Direct:** *Cited in §2.3. Barry et al. report a 6% hallucination reduction
on a finance-domain GraphRAG benchmark using graph-based structural
verification — but their architecture does **not** include a
pre-generation integrity validation layer. The hypothesis in this work is
that inserting CompGCN-based plausibility filtering before generation
will yield a substantially larger reduction. The +45% Grounding / +204%
Faithfulness uplift over the prompt-only baseline supports that hypothesis.*

### EQ9. "Why CompGCN over a newer GNN — say, a graph transformer?"

**Direct:** *Three reasons. (1) CompGCN's relation-composition operator is
specifically designed for multi-relational KGs — graph transformers
typically ignore edge type or treat it as a feature. (2) CompGCN has the
parameter-efficiency property over R-GCN (Vashishth et al. §3.2). (3) On
this corpus density, model capacity is not the bottleneck — corpus density
is (see §6.1). A larger architecture would not move MRR; more edges would.*

**Followup-trap:** *"Did you try a graph transformer?"* → "No, because the
diagnosis pointed at corpus density, not model capacity. Run 9 with RotatE
— a more expressive decoder on the same encoder — regressed. That's
evidence the capacity ceiling isn't binding."

### EQ10. "How does Detective-Board provenance compare to standard citation?"

**Direct:** *Standard RAG citation points at chunks: "this sentence came
from chunk 47." Detective-Board provenance points at validated triplets:
"this sentence came from the triplet (Case A, EXTENDS, Doctrine X) which
the integrity model scored 0.98, sourced from document D1 page 12." The
unit of citation is the verified fact, not the retrieved text — which is
what makes the citation auditable.*

---

## 3. Adversarial / threat-model questions

### EQ11. "What happens if I upload a deliberately poisoned document?"

**Direct:** *Three things, in order. (1) If the poisoned content contradicts
the rest of the corpus, the CompGCN integrity model will score those edges
low — the topological-inconsistency mechanism flags them. (2) If the
poisoned content is internally consistent but contradicts external truth
(closed-corpus limit), the model cannot detect it — the system measures
plausibility relative to the corpus, not against universal truth (§1.6
Delimitation #3). (3) Audit-time review of the document-integrity dashboard
surfaces low-scoring documents for human attention. That is the intended
operator workflow.*

**Followup-trap:** *"So a coordinated attack with multiple consistent fake
documents would defeat you?"* → "Yes, and §1.6 names that as the closed-
corpus limit. The architecture is honest about what it cannot detect.
External cross-referencing against ground-truth sources would require a
separate fact-verification component, which is future work."

### EQ12. "What's your defense against extraction errors from the LLM?"

**Direct:** *§1.6 names this explicitly as the 'Extraction Trust' limit.
The system trusts the LLM-based extraction (Stage 2) as the initial data
source. If Gemini hallucinates an entity or relationship during ingestion,
it enters the graph as a structurally valid edge. CompGCN can flag such
errors only if they create topological anomalies relative to other extracted
relationships; an isolated hallucinated edge that fits the local graph
structure will receive a high plausibility score. This is a known limit
of any LLM-extraction-first pipeline.*

**Followup-trap:** *"Doesn't that undermine your whole thesis?"* → "No, it
bounds it. The thesis is that an integrity layer between **retrieval** and
**synthesis** improves output quality. The extraction stage is upstream of
that and is a separate engineering problem. The mitigation paths are
(a) extraction-time human review, (b) multi-LLM extraction with
disagreement flagging, or (c) symbolic-extraction fallback for structured
sources. All three are future work."

### EQ13. "What if the corpus contains contradictions on purpose — like two cases that genuinely conflict?"

**Direct:** *That is the intended use case. `CONTRADICTS` is a first-class
edge type in the schema (§3.2.1). The integrity model is trained to
recognize that two cases connected by a `CONTRADICTS` edge SHOULD both
have high plausibility individually — the contradiction is the legitimate
content. The model only scores low when the topology suggests the
contradiction itself is anomalous, e.g., a contradiction between two
entities that have no other shared edges.*

---

## 4. Production / deployment questions

### EQ14. "What's the latency budget?"

**Direct:** *Per-query: retrieval ~300ms, integrity filter ~50ms (already
trained), synthesis ~2-4s (Gemini-bound). End-to-end SLA target is ~5s for
the demo configuration. The integrity filter is not the bottleneck;
Gemini synthesis is. Training is one-shot: ~3 minutes on CPU for the
14-document corpus, ~30 minutes projected for 1000 documents.*

### EQ15. "How would you scale to ten thousand documents?"

**Direct:** *Three scaling axes. (1) Ingestion: linear in document count;
the bottleneck is Gemini extraction at ~10s/doc. For 10K documents,
parallelize with rate-limit handling — projected ~28 hours. (2) Storage:
Neo4j scales to millions of edges; not a near-term concern. (3) Training:
CompGCN on the held-out edge set scales linearly in edges. FB15k-237 (272K
edges) trains in hours on a single GPU — 10K documents would be similar.
No architectural change is needed.*

### EQ16. "Cold-start: how long until the system is useful on a brand-new corpus?"

**Direct:** *Three stages. Ingestion and entity extraction: hours, depending
on document count and Gemini throughput. Embedding cold-start (DistilBERT
on every node): minutes. Integrity model training: minutes for small
corpora, hours for large. After that, queries are real-time. The demo
corpus took roughly 90 minutes end-to-end from "uploaded PDFs" to "ready
to answer questions."*

### EQ17. "What's the cost per query in production?"

**Direct:** *Two components. (1) Gemini synthesis: roughly $0.003 per
query at current Flash pricing for a typical 5-triplet response. (2) Neo4j
+ DistilBERT inference: amortized to fractions of a cent on commodity
hardware. Per-query operating cost is dominated by the LLM, same as any
RAG system. The integrity model adds approximately zero marginal cost
because it's a trained model running locally on small tensors.*

---

## 5. Philosophy / motivation questions (the soft ones that aren't soft)

### EQ18. "Why should refusal be a desirable feature? Don't users want answers?"

**Direct:** *Two-sentence answer. In high-stakes professional domains —
law, medicine, regulatory — a confident wrong answer is materially worse
than a refusal that triggers human review. The Frohock 2025 paper on legal
hallucinations is the case in point: lawyers were sanctioned because their
research tool produced fluent-sounding fake citations they didn't catch.
A refusal mechanism turns that failure mode from "sanctioned for negligence"
to "spent an extra hour on manual research."*

**Followup-trap:** *"That's an argument for skeptical UI, not for refusal."*
→ "Both. The refusal mechanism is the architectural floor: even if the
operator ignores all UI signals, the system won't fabricate. Skeptical UI
sits on top of that floor."

### EQ19. "Couldn't a sufficiently advanced LLM just do this without the graph?"

**Direct:** *Distinguishing claim and capability. An LLM can be **prompted**
to refuse, but that's a behavioral request, not an architectural guarantee
— and prompt-injection or distribution shift can override it. The
Validate-then-Generate architecture makes refusal a property of the data
path: when no validated triplets exist, there is literally nothing for the
synthesizer to consume. The LLM cannot "decide" to hallucinate because no
input contains the answer.*

**Followup-trap:** *"Couldn't the LLM hallucinate the answer from its
pretraining?"* → "Possible in principle, but the prompt template
constrains synthesis to the provided triplets explicitly ('answer ONLY
from these triplets'). Empirically, Faithfulness is 0.97, meaning 97% of
claims trace to provided triplets. The 3% gap is the prompt-compliance
boundary, not a refusal failure."

### EQ20. "Isn't this just over-engineering for what RAG already does?"

**Direct:** *Compare the numbers, not the architectures. Prompt-only chunk
RAG on this corpus: Grounding 0.68, Faithfulness 0.32. Full Validate-
then-Generate: Grounding 0.99, Faithfulness 0.97. That's the gap. The
architectural complexity exists because the empirical evidence shows
standard RAG fails by two-thirds on the metrics that matter for high-stakes
domains. The over-engineering critique would land if the gap were
marginal; the gap is roughly 3×.*

---

## 6. Rapid-fire round (memorize one-liners)

These are the cheap-shot questions. One sentence, then stop.

| Q | A |
|---|---|
| "Why Python?" | "Standard ML ecosystem; PyTorch Geometric is canonical for GNNs." |
| "Why Neo4j?" | "Mature property graph database with Cypher; Aura free tier covers the demo." |
| "Why Gemini?" | "Zero-shot extraction with low-temperature determinism; free tier covered the evaluation suite." |
| "Why FastAPI?" | "Async-native, OpenAPI auto-docs, low ceremony." |
| "Why DistilBERT and not OpenAI embeddings?" | "Local inference, no API cost, 768-dim is sufficient for the GNN cold-start." |
| "How many lines of code?" | "Roughly 4K backend Python, 3K frontend TypeScript. Available on the GitHub repo cited in §6.8." |
| "Did you have a co-author?" | "No — capstone is single-author." |
| "How long did this take?" | "Active work since mid-April 2026; nine documented tuning runs over six weeks." |
| "What was the hardest part?" | "Diagnosing the MRR shortfall as corpus-density-bound rather than model-capacity-bound — the Run 9 RotatE ablation was the key piece of evidence." |
| "What would you do differently?" | "Run multi-seed evaluation from Run 1 rather than after Run 9 forced the realization." |

---

## 7. Questions to ASK the panel (control the conversation)

Volunteering questions shows depth. Use these strategically.

### "Would you like me to demo the refusal behavior first, or the cross-document precedent first?"

Either choice locks them into watching the architectural mechanism work
before they can lodge the curation critique.

### "If you have a question you'd like me to ask the system live, I can run it now — the dashboard supports arbitrary queries."

Inviting an unscripted query is the strongest possible counter to "did you
cherry-pick the demos?" If they take you up, the architecture either
answers or refuses — both outcomes are wins.

### "I'd be interested in your view on whether the closed-corpus limit (§1.6) is a fundamental boundary or an engineering one. The architecture currently treats it as fundamental — would you push toward an open-corpus extension?"

Turns a limit into a research-direction discussion. Panel members like
being asked their opinion on future work.

### "Is there a professional domain you'd consider a stronger test of external validity than legal — medical, regulatory, technical?"

Pre-empts the external-validity critique by inviting the panel to name the
benchmark you'd target next.

---

## 8. Depth-3 follow-up chains for the killer questions

What to say when the panel doesn't accept your first answer.

### "You hand-picked your corpus" → Q1, Q2, Q3, Q4 of `PANEL_QA.md` → if they keep pushing:

**Depth 2:** *"The selection criteria are documented in §3.1.1 of the
paper. I can walk through any one of them if there's a specific concern."*

**Depth 3:** *"The architecture is corpus-agnostic — the system is open
and re-runnable on any PDF set. If the panel would like, I can swap in a
different corpus after the defense and report back on whether the
architectural claims hold."*

### "Your sample size is too small" → Q8 → if they keep pushing:

**Depth 2:** *"§3.3.3 of the paper addresses this explicitly — RAGAS and
TruLens are the canonical protocols and both use n=3 to n=10. The
structural metrics use the full edge set; only the generative metrics use
the five-query protocol."*

**Depth 3:** *"Larger n is in future work — §6.5 lists n=50 to n=100 with
human-judge ground truth as the next evaluation milestone. The current
n=5 is the feasibility-study standard, not the production-validation
standard."*

### "MRR is below target" → Q6 → if they keep pushing:

**Depth 2:** *"Single-seed 0.912 is in §5.5; the multi-seed mean 0.958 is
in §5.3. Both numbers are reported. The methodology is the canonical KGE
one — Sun et al. 2019 and Vashishth et al. 2020 both report multi-seed
means."*

**Depth 3:** *"§6.1 — the corpus is 1.24 edges per node; FB15k is 19. KGE
papers training on FB15k-style density would not see this MRR ceiling. The
ceiling is a corpus-density consequence, not a model failure — and the
diagnosis is supported by the Run 9 RotatE ablation, which regressed
across every metric despite being a more expressive decoder."*

---

## 9. What to say when you don't know the answer

Do not bluff. The panel will smell it. Use one of these:

- *"I don't have a measurement for that — it would be future work."*
- *"That's a deeper architectural question than the paper addresses. The
  current scope is X; what you're asking about is Y, which would require
  a separate study."*
- *"My evidence on that point is anecdotal, not measured. I'd want to
  collect Z before committing to an answer."*
- *"I haven't read that paper. Could you point me to it after the defense?
  It sounds directly relevant to §[section]."*

A confident "I don't know but here's how I'd find out" is stronger than a
bluffed answer that the panel will deconstruct.

---

## 10. The five things you must NOT do

1. **Don't apologize for the corpus size.** It is what it is; the paper
   defends the choice. Apologizing concedes the point you've prepared to
   refuse.
2. **Don't dismiss any question as "not in scope."** Acknowledge, reframe,
   then explain why the scope is what it is.
3. **Don't oversell.** The Validate-then-Generate pattern is good; it's not
   a solved-AGI moment. Stay calibrated.
4. **Don't trash-talk RAG.** Standard RAG works for many use cases; this
   work targets the high-stakes-domain failure mode specifically.
5. **Don't speed up when nervous.** The deliberate pace is the credibility
   signal. Take the breath.
