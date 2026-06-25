# Archived Documents

These files are retained for historical context only. They are not current product or
implementation plans. Do not use them as authoritative references — see
`../assistant_os_architecture.md` for the current architecture.

## Subdirectories

- **`plans/`** — superseded implementation plans. *Still useful for:* tracing why a
  particular build order was chosen, or understanding what an earlier milestone
  intended before it was rolled into MVP3.
  - `2026-06-15-bulk-lexicon-seed-pipeline.md` — executed; `scripts/build_lexicon_seed.py` exists
  - `2026-06-19-gap-fix-implementation.md` — superseded by MVP3 plan
  - `2026-06-20-causal-reasoning-v1-old.md` — superseded by `2026-06-20-causal-reasoning.md`
  - `2026-06-20-causal-reasoning-v2-old.md` — superseded by `2026-06-20-causal-reasoning.md`
  - `2026-06-20-human-input-normalization.md` — partially executed (T0–T1.5); see `../human_input_normalization.md`

- **`gap-fixes/`** — gap analysis and deep review findings (all resolved or superseded).
  *Still useful for:* auditing whether a previously-identified gap has regressed, or
  finding the original evidence that a fix was necessary.
  - `gap-findings-19-6-2026.md` — cross-reference audit of v0.2 docs vs code
  - `gap-findings-validation-19-6-2026.md` — validation of gap findings
  - `bridge_gap_analysis_2026-06-15.md` — bridge duplication analysis (resolved by T4 moral cognition)
  - `deep_review_m0_m7_gaps_2026-06-16.md` — M0–M7 gap re-review
  - `deep_review_findings_2026-06-15.md` — all gaps resolved in same session

- **`sentience/`** — sentience/temporal/awareness plans (all superseded by architecture doc).
  *Still useful for:* understanding the original sentience/awareness design thinking
  before it was consolidated into the architecture doc, and for the acceptance test
  suite that still runs today.
  - `final-sentience-gap-fixes.md` — consolidated gap-fix plan (16 gaps)
  - `final-temporal-sentience-awareness-plan.md` — temporal awareness plan
  - `final-behavior-reasoning-plan.md` — merged reasoning + creative behaviors plan
  - `deep-reasoning-gap-plan.md` — deep reasoning gap plan
  - `sentience-gap-fix-plan.deprecated.md` — original sentience gap fix plan
  - `sentience_competition_test.py` — 32 acceptance tests for sentience scenarios (still run)

- **`research/`** — pre-OS-pivot research and validation docs. *Still useful for:*
  understanding the research trajectory that led to the current architecture, and for
  validation methodology that may inform future benchmarks.
  - `grounded_child_chat_mvp.md` — bounded child-room architecture proof
  - `grounded_child_chat_mvp_direction.md` — supporting direction memo
  - `letta_comparison_plan.md` — Letta comparison plan
  - `slm_appliance_validation.md` — SLM appliance validation
  - `assistant_os_evidence.md` — early OS evidence
  - `plan-gaps.md` — UOL refactor gap plan (references superseded docs)
  - `MELM_validation_implementation_plan.md` — six-month validation track (pre-OS)
  - `MELM_implementation_research_review_2026.md` — historical de-risking review

- **`nlg/`** — deprecated NLG design docs. *Still useful for:* understanding the
  original NLG robustness analysis and which gaps were identified vs. fixed. The
  factual current state is in `../nlg_pipeline.md` and `../human_input_normalization.md`.
  - `nlg-robustness-validation-and-design.md` — gaps audit (5 of 6 fixed; see `../nlg_pipeline.md`)
  - `human-friendly-NLG-pipeline.md` — input normalization design (T0–T1.5 implemented, Harper removed; see `../human_input_normalization.md`)

- **Root files:** *Still useful for:* tracing the evolution from the original
  whitepaper through v0.1/v0.2/v0.3 plans to the current v0.4 architecture.
  - `legacy_MELM_whitepaper_pre_local_assistant_os.docx` — old Word whitepaper draft
  - `local_assistant_os_mvp_plan.md` — v0.1 implementation plan (superseded by v2 then MVP3)
  - `README-pre-mvp3.md` — previous docs README
  - `robust-sentience-temporal-awareness-plan.md` — superseded by final-temporal plan
  - `sentience_temporal_awareness_plan.md` — superseded by final-temporal plan
  - `assistant_os_spec.md` — v0.2 architecture prose (superseded by architecture.md)
  - `local_assistant_os_mvp_plan_v2.md` — v0.2 execution plan (superseded by MVP3 plan)
  - `assistant_os_roadmap.md` — v0.2 milestone map (subsumed by roadmap.md)
  - `uol-architecture.md` — UOL atom model long-term research target
  - `moral_cognition_impl_plan.md` — T4 moral cognition implementation plan (implemented)
  - `curiosity_context_agreement_impl_plan.md` — curiosity/deferred-research plan (historical)
  - `ethics-self-identity-reasoner.md` — ethics gate design (implemented)
  - `vocabulary_acquisition_fix_plan.md` — vocabulary acquisition bugfix plan (all 4 bugs fixed)
