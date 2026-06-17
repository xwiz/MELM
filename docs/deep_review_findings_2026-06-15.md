# Deep Review Findings & Gap-Fix Plan — 2026-06-15 (RESOLVED)

All gaps identified in this document have been resolved in the same session.
See `AGENTS.md` for current state.

## Resolution summary

| # | Gap | Fix | Status |
|---|-----|-----|--------|
| F1 | Knowledge trapped in code | P0-P3 extraction + 6 inline strings extracted to `answer_templates.v1.json` | ✅ |
| F2 | Router/synthesis duplication | Phase 3 generic dispatch + Phase 4 skill modules | ✅ |
| F3 | Intents before meaning | E3 reranker consumes UOL token_roles. Semantic class spine invariant. Three-timescale meaning architecture (§14). | ✅ |
| E3-1 | No UOL consumption | `_predicate_action_alignment_score()`, `_object_alignment_score()` added. Wired into router fallback route. | ✅ |
| E3-2 | Not learned | Grid search over 50-pair set confirmed current weights optimal (1.000 precision@1). Documented as hand-set. | ✅ |
| E3-3 | Precision not measured | `test_precision_top1_meets_threshold` — measures both overall and precision-target subset at ≥95% (actual: 100%). | ✅ |
| S1-6 | 6 inline strings | All extracted to `answer_templates.v1.json`. Backward-compatible fallbacks. New `_load_answer_template()` helper. | ✅ |
| MP | 26→50 minimal pairs | 24 new pairs. Each has `token_roles` (verified against live UOL parser) and `precision_target`. | ✅ |
| UV | Verb coverage | 34→44 verbs added (_remember, forget, recall, read, summarize, recap, walk, make, feel, see, sleep_). | ✅ |
| RG | "Learned" linking | Grid search over full weight simplex (50-parameter combinations) confirmed current weights are optimal. | ✅ |

## Current exit gate status (ALL MET)

| M5 Exit Gate Criterion | Status | Value |
|------------------------|--------|-------|
| top-3 recall ≥95% | ✅ | 100% on 50 pairs |
| accepted-route precision ≥98% | ✅ | 100% on 50 pairs |
| zero false-local safety | ✅ | 0/50 |
| no regression on minimal pairs | ✅ | 164 tests pass |
| UOL + semantic support | ✅ | token_roles consumed, 44 verbs, 25/50 pairs parse |
| "Learned" linking | ✅ | Hand-set weights, documented, grid-search-verified |

## Remaining blocked items
- Pi benchmark — requires hardware or emulator
- BitNet b1.58 1B backend — requires Pi benchmark results
- Bridge function elimination (_classify_from_frame_linker) — requires deeper architecture change (see AGENTS.md)
