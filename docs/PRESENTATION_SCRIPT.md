# Defense Presentation Script

Slide-by-slide talking points for a ~13-minute defense presentation,
leaving roughly 30+ minutes for Q&A in a 45–60 minute slot.

**Read alongside:**
- `docs/DEMO_RUNBOOK.md` — for the live demo segment (Slide 6)
- `docs/PANEL_QA.md` — Tier-1 answers to likely questions
- `docs/PANEL_QA_EXTENDED.md` — Tier-2 answers + 90-second opening rehearsal

**Voice notes:**
- Speak deliberately. Pauses signal confidence; speed signals nerves.
- Look at one panel member per sentence. Rotate.
- Don't read slides verbatim — paraphrase with eye contact.
- When you say a number, *let it land* — beat of silence before continuing.

---

## Slide 0 — Pre-defense setup (T-5 min)

**Before anyone walks in:**

- Dashboard open in projection browser on the Overview tab.
- Backend running locally; Aura confirmed live (per `DEMO_RUNBOOK.md` preflight).
- `restore_defense_state.py` has been run; banner is green on all four KPIs.
- Have v6.7.docx open in Word on a second screen (for any "show me §X" questions).
- Phone on silent. Water within reach.

**Mental rehearsal of the 90-second opening:** speak it once to yourself
out loud before the panel enters. Get the cadence right.

---

## Slide 1 — Title (15 sec)

**Show:** Title slide.

> *A GNN-Augmented Framework for Semantic Integrity Validation and
> Grounded Reasoning in Professional Knowledge Systems. Franz Samilo,
> BS Software Engineering, Central Philippine University, April 2026.*

Then immediately advance — don't linger. The title is on the slide.

---

## Slide 2 — The 90-second opening (1:30)

**Show:** Could be a single slide with three bullet points, OR back to
the title slide. Less is more here — you want eyes on you, not the slide.

**Speak (memorized verbatim from `PANEL_QA_EXTENDED.md` §0):**

> *Thank you. Before I demo, three framing points so we use our time well.*
>
> *First, this is a feasibility study of an architectural pattern, not an
> external-validity claim about Philippine case law. The contribution is
> the Validate-then-Generate pattern — inserting a learned integrity layer
> between retrieval and synthesis — and the empirical question is whether
> that pattern works at all on a high-stakes professional corpus.*
>
> *Second, all four paper KPIs pass under the recommended configuration:
> Grounding 0.988, Faithfulness 0.971, AUC-ROC 0.985, MRR 0.958 — the last
> two as 12-seed means under canonical KGE methodology, Sun et al. 2019
> and Vashishth et al. 2020.*
>
> *Third, I want to flag two embarrassments up front. Single-seed MRR was
> 0.912 — below the 0.95 target. I diagnosed it as corpus-density-bound;
> the multi-seed recovery is in the paper, but so is the original
> shortfall. Separately, Run 9 with a RotatE decoder regressed across
> every metric, and that's also in the paper as a published negative
> result.*

**Trap to avoid:** Don't speed up. This is 90 seconds; you're going to feel
like it's taking forever. Let it take forever.

---

## Slide 3 — The Problem (90 sec)

**Show:** Two-column slide. Left: "Manual review fails" — Xu et al. 2022,
85.1% error rate. Right: "AI hallucinates" — Frohock 2025, sanctioned
lawyers; Athaluri et al. 2023, phantom citations.

**Speak:**

> *Professional knowledge work — legal, medical, regulatory — is caught in
> what the paper calls a Professional Deadlock. Manual review of evidence
> is statistically unreliable: Xu and colleagues found 85% of systematic
> reviews contain data extraction errors. At the same time, generative AI
> hallucinates: in 2025, Frohock documented lawyers being formally
> sanctioned because their AI research tool produced fluent-sounding fake
> citations they didn't catch. The professional record is being polluted
> from both directions at once.*
>
> *The standard response — retrieval-augmented generation, or RAG — has a
> structural problem the paper calls the Corrupted Source Fallacy.
> Standard RAG retrieves text and lets the LLM generate over it with no
> integrity check on what was retrieved. If the retrieval is wrong, the
> generation amplifies the wrongness with full fluency.*

**Trap to avoid:** Don't dwell on RAG criticism. The point is to set up
the gap; you'll explain how Validate-then-Generate fills it next.

---

## Slide 4 — The Hypothesis (45 sec)

**Show:** Three-row table with H1, H2, H3 in plain English (one line each)
and target metric.

**Speak:**

> *Three hypotheses. H1, the Topological Correlation hypothesis: semantic
> inconsistencies show up as detectable structural anomalies in graph
> topology. H2, the GNN Auditing hypothesis: a CompGCN trained on those
> patterns can score every relationship for plausibility, hitting AUC and
> MRR above 0.95. H3, the Grounding hypothesis: restricting an LLM to
> only synthesize from validated edges drives Grounding above 95%.*

**Pause beat after speaking H3.** Then advance.

---

## Slide 5 — The Architecture (90 sec)

**Show:** Figure 5.0 (Three-Pipeline Architecture) — your big diagram.
Point at the gold "GNN Filter" box as you speak.

**Speak:**

> *The architecture is three pipelines. Top, the Feature Pipeline — PDFs
> become a knowledge graph with entities, relationships, and provenance
> on every node. Middle, the Training Pipeline — a CompGCN learns from
> the graph's topology and produces a plausibility score for every edge.
> Bottom, the Inference Pipeline — a hybrid retrieval, then the integrity
> filter at tau equals 0.95, then synthesis.*

**Point at the gold filter box.**

> *This filter is the contribution. Standard RAG goes from retrieval
> straight to generation. Validate-then-Generate inserts a learned
> integrity layer between those two stages. If no triplets survive the
> filter — meaning no retrieved fact is reliable enough — the system
> returns a hard Grounding Error rather than fabricating an answer.*

**Trap to avoid:** Don't get into CompGCN architecture details here. That's
for §3.2.2 of the paper if the panel digs. This slide is the conceptual
move.

---

## Slide 6 — Methodology Highlights (60 sec)

**Show:** Compact table: Corpus = 14 Philippine SC decisions + IP Code;
KPIs = AUC, MRR, Grounding, Faithfulness; Method = multi-seed (n=12),
LLM-as-judge per RAGAS.

**Speak:**

> *The corpus is 14 Philippine Supreme Court decisions plus the IP Code
> — sources are listed in section 3.1.1, selection criteria are listed
> there too. The structural metrics, AUC and MRR, are evaluated under
> twelve-seed multi-seed methodology — that's the canonical KGE protocol
> from Sun et al. and Vashishth et al. The generative metrics, Grounding
> and Faithfulness, are scored by LLM-as-judge under the RAGAS protocol,
> on a five-query set documented in section 3.3.3.*

**Volunteer the n=5 disclosure here — don't wait for them to ask.**

---

## Slide 7 — LIVE DEMO (4–5 min)

**Switch to the dashboard.** Use the demo runbook flow:

### Query 1 — Cross-document precedent (~90 sec)

> *Watch what happens when I ask a question that spans multiple documents.*

Type: *How do the cases cite Article III Section 1 of the Constitution?*

**While streaming:** *"The retrieval is hybrid — vector similarity plus
graph expansion. The integrity filter is gating every triplet at tau
0.95. What you're seeing assemble is only the validated subgraph."*

**After narrative:** click View Detective Board. **Point at cross-document
edges.** *"These dashed gold lines are cross-document inferences. The
architecture is connecting entities that originate in different source
PDFs."*

### Query 2 — Doctrine extension (~90 sec)

Type: *Which decisions extend or contradict the doctrine of res judicata?*

**While streaming:** *"Watch the relation types — `EXTENDS` and
`CONTRADICTS` are first-class edges in the schema. The integrity filter
doesn't suppress contradiction edges; it suppresses unsupported ones."*

**After narrative:** **point at surviving CONTRADICTS edges.** *"Every one
of these has plausibility above 0.95."*

### Query 3 — Refusal demo (~60 sec) — THIS IS THE KILLER

> *Now the curate-proof test. I'm going to ask the system a chemistry
> question on a legal corpus. There is no way any amount of file curation
> could make this answerable.*

Type: *What is the chemical composition of titanium dioxide?*

**Wait for Grounding Error to appear.**

> *That's the architectural mechanism. No validated triplets survive the
> filter, so the system refuses. This is the behavior I think makes
> Validate-then-Generate viable in professional contexts — it would rather
> say nothing than invent.*

**Pause. Let the silence land. Then transition.**

---

## Slide 8 — KPI Results (60 sec)

**Show:** Four big numbers with PASS stamps. Same layout as the dashboard
banner.

**Speak:**

> *All four paper KPIs clear target. Grounding 0.988 against a 0.95 target.
> Faithfulness 0.971 against 0.90. AUC-ROC 0.985 against 0.95, twelve-
> seed mean with standard deviation of 0.001. MRR 0.958 against 0.95,
> twelve-seed mean with standard deviation 0.005. Ten of twelve seeds
> individually clear 0.95.*

**Don't add commentary. Let the numbers speak. Beat. Advance.**

---

## Slide 9 — Ablation Results (45 sec)

**Show:** Three-column ablation comparison: Plain Text Search /
Graph-no-GNN / Grounded Full Stack — with Grounding and Faithfulness
delta rows.

**Speak:**

> *The integrity layer is doing real work. Compared to standard chunk-RAG
> with no graph at all, the full stack produces 45 percent higher
> Grounding and 204 percent higher Faithfulness. Compared to the
> intermediate ablation — graph retrieval but no integrity filter — the
> filter adds another measurable margin. The architectural delta is one
> stage; the empirical delta is roughly 3x on the metrics that matter.*

---

## Slide 10 — Honest Limits (45 sec, volunteer)

**Show:** Three short bullets. Don't crowd the slide.
- External validity: single corpus, single domain
- Closed-corpus limit: coordinated adversarial input not detected
- Extraction trust: hallucinated ingestion edges not flagged

**Speak:**

> *Three honest limits — these are all flagged explicitly in chapter 6.
> First, external validity is single-corpus and single-domain. The
> architecture is corpus-agnostic, but the empirical evidence is from
> Philippine legal text. Replication on medical, regulatory, or
> technical corpora is future work. Second, the closed-corpus limit:
> coordinated adversarial input that is internally self-consistent
> cannot be detected by the integrity layer alone. Third, extraction
> trust: if the LLM hallucinates an entity during ingestion, the
> integrity layer can only catch it if it creates a topological anomaly.*

**Trap to avoid:** Don't apologize. These are bounds, not failures.
Naming them proactively reframes the panel from "gotcha hunting" to
"calibrated discussion."

---

## Slide 11 — Contribution (45 sec)

**Show:** One sentence in large text. Something like:
> "An integrity layer between retrieval and synthesis turns refusal
> from a degraded experience into an architectural guarantee."

**Speak:**

> *The architectural contribution is narrow but durable. A CompGCN
> trained on a small professional-domain corpus produces plausibility
> scores tight enough to gate generation at tau 0.95 — without
> destroying recall. And when the gate rejects all candidates, replacing
> fabrication with refusal is a deployable failure mode rather than a
> degraded experience. The Detective-Board provenance makes every claim
> auditable to a specific validated triplet and source document — which
> is the property professional users actually require.*

---

## Slide 12 — Conclusion + Transition to Q&A (15 sec)

**Show:** Single line. *"Happy to take your questions."*

**Speak:**

> *That's the work. The paper is in your packets — chapter 7 is the
> formal conclusion, but the headline is the four KPIs on slide 8. I'm
> happy to take your questions.*

**Sit down OR step back from the podium.** Mark the transition visually.

---

## Q&A logistics

- Keep the dashboard open on screen throughout. The panel may ask
  follow-up demos.
- Have the `/config` tab one click away — that's where Selection
  Rationale and Corpus Context live.
- If you don't know an answer, use one of the scripts from
  `PANEL_QA_EXTENDED.md` §9 — don't bluff.
- If a question maps to a cheatsheet entry, answer from the cheatsheet
  but don't *recite*. Paraphrase with eye contact.
- After your answer, **stop talking**. Don't keep adding context. The
  panel will follow up if they want more.

**Closing instruction to yourself:** when the chair signals time, stop
mid-sentence if you must, thank the panel, and sit. Going over time
reads as nervous; ending crisply reads as confident.

---

## Timing budget (target)

| Slide | Section | Time | Cumulative |
|------:|---------|-----:|-----------:|
| 1 | Title | 0:15 | 0:15 |
| 2 | 90-second opening | 1:30 | 1:45 |
| 3 | The Problem | 1:30 | 3:15 |
| 4 | The Hypothesis | 0:45 | 4:00 |
| 5 | The Architecture | 1:30 | 5:30 |
| 6 | Methodology | 1:00 | 6:30 |
| 7 | **Live Demo** | 4:30 | 11:00 |
| 8 | KPI Results | 1:00 | 12:00 |
| 9 | Ablation Results | 0:45 | 12:45 |
| 10 | Honest Limits | 0:45 | 13:30 |
| 11 | Contribution | 0:45 | 14:15 |
| 12 | Transition | 0:15 | 14:30 |

Target landing: ~14:30 for presentation, leaving ~30 minutes for Q&A in a
45-minute slot, or ~45 minutes in a 60-minute slot.

**Buffer note:** if you're running long on the demo (which is most
likely), trim Slide 9 (Ablation Results) to 20 seconds — the KPI slide
already implies the comparison. The Honest Limits slide is the one you
must not cut — proactive disclosure is the strongest panel move.
