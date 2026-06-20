# Lectern Cheat Card — Single Page

Print this. Tape it to the lectern. Glance only when you blank.

```
═══════════════════════════════════════════════════════════════════
  THE FOUR NUMBERS (memorize cold)
═══════════════════════════════════════════════════════════════════
  Grounding        0.988   target > 0.98   PASS    (H3)
  Faithfulness     0.971   target > 0.90   PASS    (H3)
  AUC-ROC          0.985 ± 0.001           PASS    (H2, n=12 seeds)
  MRR              0.958 ± 0.005           PASS    (H2, n=12 seeds)

  Single-seed MRR was 0.912 (below 0.95) — VOLUNTEER this.
  Multi-seed methodology = Sun+ 2019, Vashishth+ 2020.
═══════════════════════════════════════════════════════════════════
```

## 90-SECOND OPENING (3 beats)

1. **Frame:** *"This is a feasibility study of an architectural pattern,
   not an external-validity claim."*
2. **Numbers:** *"All four KPIs pass — 0.988, 0.971, 0.985, 0.958."*
3. **Volunteer:** *"Two embarrassments up front: single-seed MRR 0.912 was
   below target; Run 9 RotatE regressed. Both are in the paper."*

## 5 THINGS TO VOLUNTEER (before they ask)

1. *"Feasibility study, not external-validity claim."*
2. *"Single-seed MRR was below target; multi-seed methodology is canonical."*
3. *"Run 9 RotatE regressed — published negative result."*
4. *"14 documents, selection criteria in §3.1.1, system is corpus-agnostic."*
5. *"External validity to other domains is future work, §6.5."*

## 5 THINGS NOT TO DO

1. Don't apologize for corpus size — it's defended.
2. Don't dismiss any question as "not in scope" — reframe instead.
3. Don't oversell — it's a pattern, not solved-AGI.
4. Don't trash-talk RAG — it works for many use cases.
5. Don't speed up when nervous — pause is the credibility signal.

## KILLER ANSWERS (3 most likely questions)

### Q. "Didn't you choose your own files?"
> *Hand-picked, yes — but selection criteria are in §3.1.1, AND the
> refusal demo is curate-proof: no amount of legal-corpus curation makes
> the system answer a chemistry question. Either the τ-filter rejects
> all triplets or it doesn't.*

### Q. "Why is MRR a multi-seed mean?"
> *Canonical KGE methodology — Sun et al. 2019 (RotatE) and Vashishth et
> al. 2020 (CompGCN) both report multi-seed means. Single-seed 0.912
> shows ±0.005 variance from RNG; 10 of 12 seeds individually clear
> 0.95. Multi-seed is the more conservative report, not the more
> flattering one.*

### Q. "How is this different from standard RAG?"
> *Three numbers. Plain-text RAG: Grounding 0.68, Faithfulness 0.32.
> Validate-then-Generate: 0.99 / 0.97. The integrity layer is doing
> measurable work — +45% Grounding, +204% Faithfulness uplift. The
> architectural delta is one stage; the empirical delta is ~3×.*

## DEMO FLOW (4-5 min)

1. **Cross-document precedent** — *"How do the cases cite Article III
   Section 1 of the Constitution?"* — point at dashed-gold cross-doc edges
2. **Doctrine extension** — *"Which decisions extend or contradict res
   judicata?"* — point at surviving CONTRADICTS edges (>0.95)
3. **Refusal demo (KILLER)** — *"What is the chemical composition of
   titanium dioxide?"* — wait for Grounding Error, let silence land

## PREFLIGHT (T-5 min before panel enters)

- [ ] Run `python -m run_logs.restore_defense_state` — wait for DEFENSE READY
- [ ] Wake Aura (free tier auto-pauses)
- [ ] Backend + frontend running locally
- [ ] Dashboard open on Overview tab; banner green on all 4
- [ ] Refusal demo confirmed working privately
- [ ] Phone silent, water poured

## IF YOU DON'T KNOW

Pick one. Don't bluff:
- *"I don't have a measurement for that — it would be future work."*
- *"Out of scope for this study. The current scope is X; what you're
  asking about is Y, which would require a separate study."*
- *"Anecdotal, not measured. I'd want to collect Z before committing."*
- *"I haven't read that paper. Could you point me to it after?"*

## IF THE DEMO BREAKS

- **Aura cold-start:** *"Free-tier database auto-paused; let me wake it."*
- **Gemini quota:** *"Free-tier per-minute quota tripped. The error
  envelope is architectural — let me swap keys."*
- **Both fail:** Pivot to /config tab to walk through architecture
  statically. The PDF (printed) is the backup.

## REMEMBER

**You wrote 4K lines of Python, 3K of TypeScript, ran 9 systematic tuning
runs, and built a working system that refuses to lie. The panel is
reviewing your work, not testing whether you deserve to be there. You do.**
