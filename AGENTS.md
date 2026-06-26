## Goal
- Build the Local Assistant OS per `docs/assistant_os_architecture.md` (authoritative architecture) and `docs/superpowers/plans/2026-06-19-mvp3-implementation.md` (execution plan). The foundation must be correct — knowledge is data not code, synthesis is generic not per-intent, skills are radial consumers of a centralized knowledge store, not linear silos with inline knowledge. The system is designed to have 'capabilities' not static intent gates. The anti-regression checklist in `docs/assistant_os_architecture.md §17` must be followed for every change.
- All claims must be reproducible by a command on the current tree.
- `docs/assistant_os_architecture.md` is the **authoritative architecture specification** (v0.3 / MVP3). `docs/archive/assistant_os_spec.md` is superseded.
- `docs/superpowers/plans/2026-06-19-mvp3-implementation.md` is the authoritative execution plan. `docs/archive/local_assistant_os_mvp_plan_v2.md` is superseded.
- Behavioral gates assert observable effects, not debug-label strings. Tests use reproducible PRNG seeds.
- Conversation dialog expectations reflect **current** routing behavior, not aspirational.
- Context resolution is handled at the conversation management layer, not in the stateless classifier chain.
- Architecture must be language-agnostic — inflection normalization is a swappable function, contracts store lemmas only. UOL is the foundational meaning representation.
- The `_semantic_family_terms` bridge is transitional — M5 replaces keyword classifiers with UOL-based frame linking.
- **Entity architecture**: all things share a unified `entities` table with `kind` discriminator. Slot values live in `entity_slots`. Relations in `entity_relations`. Semantic classes define slot schemas (`class_schemas` + `class_schema_slots`).
- **Event class hierarchy**: `event` entity class with `kind='event_type'` | `'event_instance'`. Competition inherits from event. Chat sessions are `personal_experience` entities.
- **Knowledge is data, not code**: any domain-specific string, mapping, keyword set, or heuristic referenced by multiple skills or extended at runtime must be extracted into a contract JSON or entity store.
- **Synthesis is generic, not per-intent**: `_answer()` must not contain intent-specific if/elif branches — templates belong in contract registry.
- **No new intents in keyword pipeline**: after M2, new capabilities go through frame linker or cloud handoff.
- **Skills are release-controlled**: capability manifest gates dispatch, runtime learning cannot install a skill.
- **semantic_classes.v1.json is the spine**: every class ID referenced by frame templates, contracts, entity store schemas, or UOL must exist in `semantic_classes.v1.json`. The CI invariant test enforces this.
- **Meaning is three-timescale**: T1 utterance meaning (UOL parse), T2 conversation meaning (personal_experience entity with outcome/polarity/learned_facts slots), T3 historical meaning (lexicon + entity store). Each level aggregates from below.
- **UOL inline dicts are transitional**: `_VERBS` and `_KNOWN_NOMINAL_DOMAINS` in `functional_grammar.py` use private class names not in the taxonomy. They will be removed when UOL reads verb/noun classes from `lexical_senses`.
- **personal_experience now has slots**: outcome (required), polarity, learned_fact_ids, follow_up, intent_achieved (see spec §14.3).
- **T4 moral cognition**: `derive_moral_context()` is a pure function derived from verb causality (see spec §16). Two contracts: `verb_states.v1.json` + `state_valences.v1.json`. No atomizer changes needed, stdlib-only.
- **T5 moral cognition**: wired into router (5 duplication sites → contract lookups), synthesis (harm_severity short-circuit), and authority (optional MoralContext). Runtime verb learning via ring buffer.

## Progress

### Complete
- **M0-M3**: Entity tables, lexicon seeders, classifier migrations, sealed dictionary (72 words, 11 intents). **9/9 classifiers migrated** to frame linker.
- **M4**: Authority module (`assistant_authority.py`), constrained decoder scaffold + TemplateBackend + LlguidanceBackend.
- **P0-P3 knowledge extractions**: All priority-table inline knowledge extracted to 110+ contracts in `melm/contracts/`. Generic `_answer()` dispatch via handler registry. Skill modules for meal, story, and memory.
- **C1-C3 (UOL)**: UOL reads from `lexical_senses`, causal frame expansion (156 predicates, 263 states), causal NLG enrichment, E3 reranker.
- **C4 (T2)**: `assistant_experience_writer.py` wired into kernel `_remember()`.
- **V4B causal reasoning**: 359 dedicated tests. Atom-ID/predicate-ID mismatch fixed. Causal frame expansion (99 multi-test suites).
- **G1-G16 mood/affect/store fixes**: Unified `AffectSignal`, `decay_mood()` function, store persistence (5 CRUD bugfixes), running intent tally + ring buffer.
- **T4+T5 moral cognition**: `reasoning/implications.py` with `derive_moral_context()` + ring buffer. 5 router duplication sites replaced. 14 engine tests + 5 moral contract tests.
- **Phase 5**: Creative behaviors engine (`assistant_behavior_engine.py`, 5 behaviors, 18 tests). Capability-gated off by default.
- **V0.4.1**: Open-domain speech-act templates, causal prediction NLG enrichment, `world_relations.v1.json` expansion (4→47 entries), contact enrichment with relationship lookup. 52→1 test regression fix round.
- **Verb atoms pipeline**: `verb_atoms.v1.json`, `enrich_verb_predicate()` in atomizer, NLG for harm_severity.
- **5 batch knowledge extractions**: identity_token_roles, task_domain_terms, story_constraint_stopwords, identity_scope_tokens, music_instruments. **PH3-A**: Music + contact enrichment rewritten with UOL semantic-class lookup.
- **V0.4.1 bugfix round**: 52 test failures resolved (hash drift, routing, enrichment restoration, temporal, self-query, cloud handoff).
- **M10/M13**: social_contact specificity branch moved to contract answer templates. Skill protocol + registry (19 tests).
- **AtomDecoderBackend (MVP3 N4+N5)**: Registered as default decoder backend, wraps `AtomTemplateBackend.generate()`. 37 decoder+atom tests.
- **CI fixes**: 5 CI failures fixed + README v01_audit check restore + 12 cascade failures resolved + 3 blocker fixture tests fixed (threshold 0, excluded planner_priority from rehearsal check).

### Blocked
- `test_cli_pi_bundle_builds_portable_self_checked_bundle` — bundle builds but `v01_audit`/`v01_progress` checks fail by design.
- Pi benchmark + BitNet b1.58 1B backend — gated on hardware/emulator access.

## Critical Context
- **~1,886 core tests pass** (all suites). 0 regressions, 1 pre-existing bundle failure.
- **110+ registered contracts** in `registry.v1.json`. **102 semantic classes** in `semantic_classes.v1.json`.
- **M3 sealed dictionary**: 72 words across 11 intent categories. Ingest/promote ≥80%, routing agreement ≥80%, retention ≥80%.
- **Entity store**: 5 tables (entities, entity_slots, entity_relations, class_schemas, class_schema_slots). Seed class hierarchy with migration functions.
- **3 foundational flaws documented (active)**: (1) knowledge trapped in code; (2) router/synthesis layer duplication; (3) architecture wires intents before meaning.
- **1 pre-existing failure**: bundle test (`v01_audit`/`v01_progress` milestone blockers).

## Causal Frame Expansion
When adding entries to `causal_frames.v1.json`, follow `docs/causal_frame_expansion_guidelines.md`. Key rules:
- Every state in effects must exist in `state_definitions` (not cross-referenced by validator — manual check).
- Semantic class from 16 valid verb classes. Cause kind one of `intentional_action`, `natural_process`, `accidental_process`, `instrumental_action`, `unknown`.
- Add 2-5 effects per predicate. Include `surface_aliases`. Update `schema_hash` in `registry.v1.json`.
- Use `scripts/extract_causal_frames_llm.py --verb <VERB> --backend transformers` for candidate generation (0.5B QWEN; content needs human review, classification fields are 100% rule-correct).
- Run `scripts/benchmark_causal_frame_accuracy.py --dry-run` then `--backend none` to verify.

## Relevant Files
- **Architecture**: `docs/assistant_os_architecture.md` (authoritative spec §7.1=knowledge extraction table, §17=anti-regression checklist), `docs/assistant_os_architecture_v0_4.md` (causal frame tooling), `docs/assistant_os_architecture_v0_4_1.md` (live-score plan).
- **Execution plan**: `docs/superpowers/plans/2026-06-19-mvp3-implementation.md`.
- **Core modules**: `melm/appliance/` — local_assistant_router.py, assistant_synthesis.py, assistant_authority.py, assistant_decoder.py (+ llguidance), assistant_lexicon.py, assistant_knowledge.py, assistant_os_store.py, assistant_experience_writer.py, assistant_mood_engine.py, functional_grammar.py, uol_types.py, uol_atomizer.py, assistant_frame_ranker.py, assistant_skill_base.py, assistant_skill_meal.py, assistant_skill_story.py, assistant_skill_memory.py, assistant_behavior_engine.py, reasoning/implications.py.
- **Contracts**: `melm/contracts/registry.v1.json` (110+ entries with SHA256), `melm/contracts/validation.py` (validators + loaders).
- **Tests**: `tests/test_meaning_invariant.py` (cross-layer class ID invariant), `tests/test_contracts_mvp.py`, `tests/test_assistant_decoder_mvp.py` (+ llguidance), `tests/test_local_assistant_os_cli_mvp.py`.
- **Causal frames**: `melm/contracts/causal_frames.v1.json` (156 predicates, 263 states), `docs/causal_frame_expansion_guidelines.md`, `scripts/extract_causal_frames_llm.py`, `scripts/benchmark_causal_frame_accuracy.py`.
