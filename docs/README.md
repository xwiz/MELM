# MELM Docs

This directory holds project planning, architecture, and validation documents
for the MELM build.

## Authoritative documents

- `assistant_os_architecture_v0_4.md`: **authoritative V0.4 architecture and
  sequencing specification.**
  Grounded against the current tree on 2026-06-20. Defines the next implementation
  order across NLG/normalization, curiosity/context hardening, and causal V4.
- `assistant_os_architecture.md`: **authoritative architecture specification (v0.3 / MVP3).**
  Documents the target typed-expert-cascade architecture, the skill/knowledge
  boundary, the knowledge-first (auto-research) design pattern, the UOL atom model,
  input normalization, response generation (NLG), self-identity derivation,
  foundational rules, and the anti-regression checklist.
- `superpowers/plans/2026-06-19-mvp3-implementation.md`: **authoritative execution plan.**
  MVP3 build order, gates, and contracts.
- `superpowers/plans/2026-06-20-causal-reasoning.md`: **next milestone design (work in progress).**
  Causal reasoning extension to the UOL/contract architecture.
- `human_input_normalization.md`: **factual input normalization pipeline.**
  Current implemented state of T0-T1.5 (contract expansion, SymSpell, agreement fix,
  NER mask). Harper removed; T2-T4 are design-only.
- `nlg_pipeline.md`: **factual response generation pipeline.**
  Current implemented state of mood pools, atom templates, emoji safety, gibberish
  handling. Absurdity state machine and curiosity NLG are design-only.

## Superseded documents (historical context only — moved to `archive/`)

- `archive/assistant_os_spec.md`: v0.2-target architecture background.
- `archive/local_assistant_os_mvp_plan_v2.md`: v0.2 execution-plan background.
- `archive/assistant_os_roadmap.md`: v0.2 milestone background.
- `archive/uol-architecture.md`: UOL atom model long-term research target (core model
  implemented; storage/parser targets are aspirational).
- `archive/vocabulary_acquisition_fix_plan.md`: vocabulary acquisition bugfix plan (all 4 bugs fixed).

## Supporting documents

- `../README.md`: root landing guide and drift rule.
- `../MELM_whitepaper.md`: revised validation-first whitepaper; supporting
  thesis only, not current product direction.
- `abstention_strategy.md`: current evidence-admission/abstention stance and open risk.
- `adtc_competition_plan_2026.md`: ADTC competition plan.
- `babylm_evaluation_2026.md`: official BabyLM 2026 evaluation adapter.
- `babylm_reproduction.md`: local commands for corpus/tokenizer/checkpoint/eval reproduction.
- `corpus_selection_2026.md`: BabyLM corpus choice and tokenizer/training results.
- `tokenizer_strategy_2026.md`: hybrid morphology-plus-Unigram tokenizer strategy.
- `sound_symbolism_deferred.md`: rationale for deferring active sound-symbolism claims.
- `support_refunds_dataset_authoring.md`: support/refunds dataset authoring.
- `support_refunds_external_blind_handoff.md`: preregistered handoff and freeze workflow.
- `validation_report_template.md`: final report skeleton.
- `archive/moral_cognition_impl_plan.md`: T4 moral cognition implementation plan (implemented; historical).
- `archive/curiosity_context_agreement_impl_plan.md`: curiosity/deferred-research plan (implemented; historical).
- `archive/ethics-self-identity-reasoner.md`: ethics gate design (implemented; historical).
- `knowledge-typing-fact-negation-design.md`: knowledge typing design (partially implemented).
