# MELM Roadmap (historical progress record)

> **Status update:** This roadmap is a historical progress record. The current
> authoritative architecture is in `docs/assistant_os_architecture.md`, the active
> execution plan is `docs/superpowers/plans/2026-06-19-mvp3-implementation.md`,
> and the next milestone in design is causal reasoning.
> `docs/archive/local_assistant_os_mvp_plan_v2.md` and `docs/archive/assistant_os_spec.md` are
> superseded by those documents.

## Current Product MVP Direction

The active product direction is the Local Assistant OS kernel and v0.3 / MVP3 path,
documented in `docs/assistant_os_architecture.md` and executed through
`docs/superpowers/plans/2026-06-19-mvp3-implementation.md`. The causal-reasoning
milestone is the next design target.

```text
membrane policy
  + homeostatic state
  + autobiographical memory
  + user/self model
  + opportunity planner
  + local inventories
  + budgeted evidence runtime
  + local/tool/action/cloud triage
  + typed knowledge (world_fact) and atom persistence
  + causal reasoning (design in progress)
```

Avoid drift: tokenizer, small-model, memory, and dialogue experiments should be
promoted only when they strengthen that assistant OS substrate.

### What exists today

- **~1,886 tests pass** across all suites — 0 regressions, 1 pre-existing bundle failure.
- **110+ registered contracts** in `registry.v1.json`. M0–M3 entity store, lexicon, classifiers — 9/9 migrated to frame linker.
- **M4 decoder scaffold** (TemplateBackend + LlguidanceBackend) — 37 decoder+atom tests.
- **C1–C4 (UOL)**: reads from `lexical_senses`, 156 causal predicates, 263 states, E3 reranker, T2 experience writer.
- **G1–G16 mood/affect/store**: decay, ring buffer, 5 CRUD bugfixes. **T4+T5 moral cognition**: pure function + ring buffer + 5 router patch sites.
- **Phase 5 creative behaviors** — 5 behaviors, gated off by default.
- **V0.4.1**: open-domain templates, causal NLG, 47-entry world_relations, contact enrichment.
- **Verb atoms + 5 batch knowledge extractions** — all priority knowledge extracted to contracts.
- **All v01 CLI commands** implement the blocker-evidence / pi-bundle / target-report / acceptance pipeline.
- **Remaining work**: Pi benchmark + BitNet backend (gated on hardware); content-rich story generation; live inventory soak across longer cycles.

## Six-Month Validation Track (Historical)

| Month | Milestone | Status |
|---|---|---|
| 1 | BabyLM reproduction, tokenizer harness, event-memory prototype | Implemented |
| 2 | Tokenizer ablations and episodic benchmark draft | Implemented |
| 3 | 125M-370M baselines and event memory vs RAG | Implemented |
| 4 | Best 370M integration if gates pass | Implemented |
| 5 | Persistent dialogue demo | Implemented |
| 6 | Validation report and release artifacts | Deferred — active focus is MVP3 kernel |

## Gate Summary

- Morphology must beat BPE/Unigram on a meaningful metric or become auxiliary.
- Event memory must beat ordinary RAG by at least 15% on controlled episodic recall.
- The best 370M model must beat the same-size BPE baseline or clearly win episodic tasks without language-quality regression.

## Phase 1 Snapshot (Historical)

The tokenizer stage gate advanced `tiered_morph_unigram` to scaled neural ablation;
BabyLM local stage completed with tiered hybrid beating HF BPE by 2.38% bits/byte.
State-memory integration achieved 100% evidence-gated accuracy on all test fixtures.
Persistent dialogue demo achieved 100% accuracy across regular, paraphrased/noisy,
and transcript-derived fixtures with 4 distractor events. See `reports/phase1_report.md`
for the full record. Sound-symbolism was removed from the active gate; the current
validation target is higher-confidence morpheme/root/usage inference.
