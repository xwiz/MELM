## Goal
- Build the Local Assistant OS per `docs/assistant_os_spec.md` (authoritative architecture) and `docs/local_assistant_os_mvp_plan_v2.md` (execution plan). The foundation must be correct — knowledge is data not code, synthesis is generic not per-intent, skills are radial consumers of a centralized knowledge store, not linear silos with inline knowledge. The anti-regression checklist in `docs/assistant_os_spec.md §13` must be followed for every change.

## Constraints & Preferences
- All claims must be reproducible by a command on the current tree.
- `docs/assistant_os_spec.md` is the **authoritative architecture specification**. It documents the target architecture, the skill/knowledge boundary, the knowledge-first (auto-research) design pattern, foundational rules, and the anti-regression checklist.
- `docs/local_assistant_os_mvp_plan_v2.md` is the authoritative execution plan (milestones, gates, timeline).
- Behavioral gates assert observable effects, not debug-label strings.
- Tests use reproducible PRNG seeds for deterministic output.
- Conversation dialog expectations reflect **current** routing behavior, not aspirational.
- Context resolution is handled at the conversation management layer, not in the stateless classifier chain.
- Architecture must be language-agnostic — inflection normalization is a swappable function, contracts store lemmas only. UOL is the foundational meaning representation; everything else is built on it.
- The `_semantic_family_terms` bridge is transitional — M5 replaces keyword classifiers with UOL-based frame linking. Do not treat the bridge as permanent architecture.
- **Entity architecture**: all things (persons, events, places, objects) share a unified `entities` table with `kind` discriminator. Slot values live in `entity_slots`. Relations between entities live in `entity_relations`. Semantic classes define what slots entities of that class can have (`class_schemas` + `class_schema_slots`).
- **Event class hierarchy**: `event` is an entity class with `kind='event_type'` or `kind='event_instance'`. Competition inherits from event. Chat sessions are personal-experience events (`kind='personal_experience'`).
- **Knowledge is data, not code**: any domain-specific string, mapping, keyword set, or heuristic that could be referenced by multiple skills or extended at runtime must be extracted into a contract JSON or entity store.
- **Synthesis is generic, not per-intent**: `_answer()` must not contain intent-specific if/elif branches — templates belong in contract registry.
- **No new intents in keyword pipeline**: after M2, new capabilities go through frame linker or cloud handoff.
- **Skills are release-controlled**: capability manifest gates dispatch, runtime learning cannot install a skill.
- **semantic_classes.v1.json is the spine**: every class ID referenced by frame templates, contracts, entity store schemas, or UOL must exist in `semantic_classes.v1.json`. The CI invariant test enforces this — no new class ID without a spine entry.
- **Meaning is three-timescale**: T1 utterance meaning (UOL parse), T2 conversation meaning (personal_experience entity with outcome/polarity/learned_facts slots), T3 historical meaning (lexicon + entity store). Each level aggregates from below.
- **UOL inline dicts are transitional**: The `_VERBS` and `_KNOWN_NOMINAL_DOMAINS` dicts in `functional_grammar.py` use private class names not in the taxonomy. They will be removed when UOL reads verb/noun classes from the lexical_senses table.
- **personal_experience now has slots**: outcome (required), polarity, learned_fact_ids, follow_up, intent_achieved (see spec §14.3)
- **T4 moral cognition**: action meaning is a fourth timescale derived from verb causality (see spec §16). `derive_moral_context()` is a pure function, not a pipeline stage. Two contracts: `verb_states.v1.json` (~50 verb entries) and `state_valences.v1.json` (~40 state→score mappings). No atomizer changes needed — engine reads existing UOL atoms. Replaces the 5 hardcoded duplication sites in the router. Total: ~205 new Python lines + ~18 KB contract data. Pi-compatible: pure dict lookups, no ML, stdlib-only.

## Progress
### Done
- **M4 authority module built** — `assistant_authority.py` with `AuthorityEvidenceItem`, `AuthorityEvidencePacket`, `AnswerPlan`, `VerificationResult`, `DecoderResult`, `AuthorityInfo` dataclasses. `build_evidence_packet`, `build_answer_plan`, `verify_answer` functions. Negation-aware constraint checking (e.g., "not a diagnosis" does not trigger diagnosis forbids).
- **M4 concrete decoder scaffold + template fallback** — `assistant_decoder.py` with `ConstrainedDecoder` registry/dispatch, `TemplateBackend` (zero-dep), `DecodingGrammar` data class, `build_decoding_grammar()` helper. Wired into synthesis `_decode_verified()`. 18 tests.
- **M4 LlguidanceBackend** — `assistant_decoder_llguidance.py` with lazy-loaded HF CausalLM + llguidance `LLInterpreter` token-masked greedy decoding, `HFCompatTokenizer` adapter, `build_llguidance_grammar`, `build_regex_pattern`, `build_llm_prompt`. 29 tests.
- **Comprehensive architecture rewrite** — `docs/assistant_os_spec.md` rewritten as authoritative architecture doc: honest assessment (3 foundational flaws), skill/knowledge boundary, knowledge-first radial design, P0-P3 extraction priority table, anti-regression checklist, migration path.
- **P0 knowledge extraction: `_food_tags` → `food_tags.v1.json`** — 13-entry food→tag mapping extracted from `local_assistant_router.py:1077` into `contracts/food_tags.v1.json`. Registered in `registry.v1.json`, validated by `validate_food_tags()`, loaded via `load_food_tags()`.
- **P1 knowledge extraction: health_disclaimers.v1.json** — 5 urgent medical disclaimer responses extracted from `assistant_synthesis.py:996` into `contracts/health_disclaimers.v1.json`.
- **P1 knowledge extraction: safety_policies.v1.json** — public clothing safety template + destination mapping extracted from `assistant_synthesis.py:1028` into `contracts/safety_policies.v1.json`.
- **P2 knowledge extraction: story_components.v1.json** — 3 story generation heuristic groups (image: 8 keyword→image pairs across title/full-text, challenge: 5 topic→challenge, lesson: 4 topic→lesson) extracted from `assistant_synthesis.py:707-751` into `contracts/story_components.v1.json`.
- **P2 knowledge extraction: weather_concepts.v1.json** — 3 weather domain terms extracted from `local_assistant_router.py:1608` into `contracts/weather_concepts.v1.json`.
- **P3 knowledge extraction: meal_scopes.v1.json** — 5 scope tokens (breakfast/lunch/dinner/cook/cooking → scope) extracted from `local_assistant_router.py:1029` into `contracts/meal_scopes.v1.json`.
- **P3 knowledge extraction: assistant_identity.v1.json** — 4 identity/status response templates (introduction, status_unavailable, status_next_steps, status_default) extracted from `assistant_synthesis.py:763-816` into `contracts/assistant_identity.v1.json`.
- **Phase 3: Generic _answer() dispatch** — replaced the 100-line `_answer()` if/elif chain with a handler registry (`_ANSWER_HANDLERS`) + contract template renderer (`_render_contract_answer`). Complex intents (story, meal, identity, status, etc.) have explicit handler functions registered in the dispatch dict. Template-suitable intents (weather) use `answer_templates.v1.json` with reason-gated template selection. No intent-specific branching in `_answer()` itself.
- **Phase 4: Skill module pattern — assistant_skill_meal.py** — extracted all meal suggestion logic (`suggest_meal()`, `format_meal_answer()`, `MealSuggestion` dataclass, and 12 helper functions) from `local_assistant_router.py` and `assistant_synthesis.py` into `assistant_skill_meal.py`. The skill module is a radial consumer of knowledge contracts (`food_tags.v1.json`, `meal_scopes.v1.json`). Synthesis `_handle_meal_suggestion` delegates to `format_meal_answer()`. Router `choose_local_meal` delegates to `suggest_meal()` for backward compatibility.
- **Phase 4: Skill module pattern — assistant_skill_story.py** — extracted story answer formatting (`format_story_answer()`, `format_story_frame()`, `_story_image()`, `_story_challenge()`, `_story_lesson()`) from `assistant_synthesis.py`. Radial consumer of `story_components.v1.json` contract. Keyword iteration now reads keys from contract instead of hardcoded literals.
- **Phase 4: Skill module pattern — assistant_skill_memory.py** — extracted all autobiographical and personal memory formatting (`personal_memory_summary()`, `autobiographical_memory_summary()`, `autobiographical_session_summary()`, `autobiographical_digest_summary()`, `_event_memory_parts()`, `_event_label()`, `_event_memory_insight_text()`) from `assistant_synthesis.py` into `assistant_skill_memory.py`.
- **Phase 4: Extract memory_insights.v1.json** — hardcoded intent+reason→text mappings from `_event_memory_insight_text` (10 rules + 1 consented-stored pattern) extracted into `contracts/memory_insights.v1.json`. Validator `validate_memory_insights`, loader `load_memory_insights` in `validation.py`. Registered in `registry.v1.json`.
- **Phase 4: Add trigger keywords to health_disclaimers.v1.json and safety_policies.v1.json** — `health_disclaimers.v1.json` responses now carry `triggers` arrays (9 keywords across 4 entries), `safety_policies.v1.json` destinations carry `triggers` arrays and renamed `text`→`phrase`. Validators updated. Synthesis `_urgent_health_answer()` and `_public_clothing_safety_answer()` now iterate contract triggers instead of hardcoded keyword checks.
- **(earlier M0-M3 items persist — entity tables, lexicon seeders, classifier migrations, etc.)**
- **M10: Intent-specific social_contact branch removed from `_answer_specificity()`** — moved to `answer_templates.v1.json` as `answer_specificity_bonuses`.
- **M13: Formal Skill protocol** — `assistant_skill_base.py` with `SkillManifest`, `Skill`, `SkillRegistry` structural protocol + global registry. 19 tests.
- **Context gates fixed** — `require_health_terms` (bare-domain blocker + concern verb detection) and `require_meal_frame` (imperative allowance + second-person block). Fixed 8 test failures.
- **C1+C2+C3: UOL reads from lexical_senses** — `_UOL_LEXICON` ref + `set_uol_lexicon()` in `functional_grammar.py`. `_lemma()` and `_semantic_class()` fall back to lexicon for verbs not in `_VERBS`. Wired into `local_assistant_router.py` at module init. 16 tests.
- **C4: T2 personal_experience entity writer** — `assistant_experience_writer.py` with `record_conversation_experience()` wired into kernel `_remember()`. Writes `outcome`/`polarity`/`intent_achieved`/`learned_fact_ids`/`follow_up` slots from synthesis result. 23 tests.
- **Fixed `seed_class_schemas()` bug** — `class_schema_slots` INSERT for personal_experience referenced nonexistent `updated_at` column. Removed column ref to match table DDL. Unblocks 4 entity architecture tests.
- **G4: Unified `AffectSignal`** — deleted engine's duplicate, single `uol_types.AffectSignal` with all 11 fields (valence, arousal, confidence, source, recovery_signals, is_complaint, mood_id, dominant_tags, identity_claim, identity_probe). Updated all consumers.
- **G5: Fixed `load_mood_regions()` key** (`"regions"` → `"moods"`). Contract now consumable.
- **G12: Fixed `_filter_recovery()` and `_is_complaint()`** — check semantic tags (`"recovery_signal"`, `"complaint"`) not lemma-strings.
- **G16: Extracted `_BE_FORMS` frozenset** to module-level constant (4 inline constructs removed).
- **G2: Router calls `compute_utterance_affect(lemmas, uol_act, lexicon)`** instead of `infer_affect()`. Added `lemmas` to `_ParseBundle`. All three tiers (lexicon, UOL, perception) now run.
- **G1: Added `decay_mood()` pure function** (6h valence / 1.5h arousal half-lives, configurable baselines). Added `last_updated` to `MoodState`. Applied decay in `update_session_mood()` (between-turn) and `initial_mood_from_baseline()` (T3 summaries). 12h annoyed → near-neutral without positive input.
- **G13: Fixed `MAX(event_id)` lexicographic bug** → `ORDER BY rowid DESC LIMIT 1` in both `count_intent_occurrences_in_session` and `count_utterance_occurrences_in_session`.
- **G6/G7/G9/G14/G15: Fixed 5 store bugs** — entity_id collision (uuid), SQL WHERE filters (user_id/session_id), non-deterministic hash (sha256), missing commit (update_lexical_sense).
- **G3: Wired store persistence** — kernel `_remember()` calls `set_mood_state()`; `handle()` calls `record_session_summary()` + `set_ambient_mood()`. Router `_load_or_init_mood()` passes real store.
- **G8/G11: Added running tally + ring buffer** — `_intent_tallies` (per-session per-intent) and `_event_ring_buffer` (bounded 50 entries) to store. O(1) per turn instead of O(history) COUNT.
- **32-test competition test** — `docs/sentience_competition_test.py` covers all acceptance scenarios (temporal decay, affect detection, EMA integration, cross-session, store persistence, running tally, T3 baseline, identity probes, O(L) efficiency, unified class, contract loading). 32/32 pass.
- **106 regression tests pass** — store, synthesis, kernel, context gates, meaning invariant. Zero regressions.
- **T5 moral cognition: contracts + engine + router patches (×5) + synthesis + authority** — `verb_states.v1.json` (59 verbs), `state_valences.v1.json` (98 valences), `reasoning/implications.py` with `derive_moral_context()` pure function + `record_verb_candidate()` ring buffer. Patched 5 router duplication sites: Sites 1-3 (urgent health → moral engine), Site 4 `_safety()` (verb from parse bundle + `_resolve_patient_type()` mapping + contract triggers), Site 5 `_health()` (removed hardcoded text). Synthesis `_answer()` short-circuits on high harm_severity. Authority `build_answer_plan()` accepts optional `MoralContext`. 14 engine tests + 5 moral contract tests. **223 total tests, 0 failures.**
- **T5 bugfix: fixed dead code in synthesis `_answer()`** — `decision.tokens` field didn't exist on `AssistantDecision` dataclass, making the entire moral cognition check unreachable. Replaced with `_simple_tokenize(decision.utterance)` + contract caching via module-level globals (never re-reads from disk).
- **T5 bugfix: `_has_urgent_health_frame()` contract caching** — was loading `health_disclaimers.v1.json` from disk on every call (4+ call sites per utterance). Now cached via `_URGENT_HEALTH_CACHE` module-level dict.
- **T5 bugfix: `_safety()` verb extraction from UOL parse** — was using `tokens[0]` (first word of utterance, e.g. "i" or "can") instead of the actual verb. Now accepts `parse_bundle` and uses `_extract_verb()`.
- **T5 bugfix: patient type → semantic class mapping** — `_extract_patient_type()` returned raw surface text (e.g. "john", "wall") which never matched `verb_states.v1.json` semantic class labels (e.g. "person", "physical_object"). Added `_PATIENT_TYPE_CLASS_MAP` dict and `_resolve_patient_type()` function. Default fallback is "person".
- **T5 bugfix: missing valence entries** — "bruised" (-0.4) and "in_pain" (-0.5) scores were missing from `state_valences.v1.json`, silently scoring 0.0 for hit/hurt/assault/batter/injure/wound/torture verbs. Added both entries.
- **T5 bugfix: `record_verb_candidate()` wired into router** — the runtime learning pathway (ring buffer → session entity) was defined but never called from production code. Wired at Sites 1-3 (classification) and Site 4 (`_safety()`) with lazy imports.
- **T5 bugfix: contract loaders not exported from `melm/contracts/__init__.py`** — `load_verb_states`, `load_state_valences`, `validate_verb_states`, `validate_state_valences` added to imports and `__all__`.
- **T5 bugfix: registry hash drift resolved** — `assistant_identity.v1` hash was MD5 but `check_compatibility()` uses SHA256 (12 pre-existing failures). `state_valences.v1` hash updated after adding entries. Both now match SHA256.
- **Curiosity/background plan grounded** — `docs/curiosity_context_agreement_impl_plan.md` updated with Part I (Ground-Truth Assessment): codebase reality vs plan assumptions, 20 findings, P0-P3 recommendations, passive-over-threaded architecture recommendation.
- **Self-identity test file** — `tests/test_assistant_self_identity_mvp.py` with 28 tests across 5 test classes (skill pure functions, router name patterns, kernel integration, store persistence, edge cases). Covers higher-mean-polarity wins, per-user isolation, min_data_points gate, name awareness/origin routing, full pipeline integration.
- **Router name-awareness pattern expansion** — `_identity_composition()` in `local_assistant_router.py` now matches `"your"` OR `"you"` (without `"my"`) OR `"yourself"` alongside `"name"` for name_awareness/name_origin frames. Fixes "Do you have a name?" and "Did you name yourself?" routing to `assistant_identity`.
- **Fixed `datetime.utcnow()` deprecation** — `assistant_skill_self_identity.py:123` replaced with `datetime.now(timezone.utc)`. Zero deprecation warnings.

### In Progress
- **(none)**

### Blocked
- `test_cli_pi_bundle_builds_portable_self_checked_bundle` — bundle builds but `v01_audit`/`v01_progress` checks fail by design.
- **Pi benchmark** — requires hardware or emulator access for tok/s/TTFT/RSS.
- **Knowledge is data, not code** — formal rule. Any domain knowledge in code is a defect until extracted to a contract.
- **Synthesis is generic, not per-intent** — `_answer()` must be replaced with a generic AnswerPlan renderer. Each intent's template belongs in the contract registry.
- **No new intents in keyword pipeline** — after M2, frame linker or cloud handoff only.
- **Skills are release-controlled** — capability manifest gates dispatch. Runtime learning can extend knowledge, not install skills.
- **Contracts before code** — define contract schema + validator + registry entry before writing domain logic.
- **Architecture is radial, not linear** — knowledge at center (contracts + lexicon + entities), skills as consumers. Auto-research pattern: teach once, serve all skills.
- **Entity architecture**: unified `entities` table with `kind` discriminator. Entity relations CRUD on store.
- **Pluggable SLM architecture**: M4 decoder uses a `ConstrainedDecoder` registry supporting llguidance (HuggingFace + CFG grammar) and BitNet b1.58 1B (TQ2_0/TQ1_0 + LoRA). Template fallback is the zero-dep baseline.
- Legacy hardcoded data (e.g. health disclaimer texts, safety policies) must be migrated to contracts per the priority table in `docs/assistant_os_spec.md` §11.1.

## Next Steps
1. **Pi benchmark** — install on Pi 4/5, measure tok/s/TTFT/RSS with llguidanceBackend + small CausalLM.
2. **Phase 5 (optional): Creative behaviors** — mood_narrative, curiosity_follow_up, fatigue_pacing, distress_callback, rhythm_observation. Capability-gated off by default.
3. **curiosity_context_agreement_impl_plan** — execute Phase 1 (schema + contracts), then Phase 2-6 per grounded plan at `docs/curiosity_context_agreement_impl_plan.md §I`.
4. **BitNet b1.58 1B backend** — when Pi benchmark validates need for smaller/quantized decoder.


## Critical Context
- **~1,359 core tests pass** (key suites: 57 router, 70 lexicon, 27 frame_linker, 75 entity, 24 authority, 38 reranker, 18 decoder, 29 llguidance, 40 contracts, 17 meaning_invariant, 16 uol_lexicon, 23 experience_writer, 32 sentience_competition, 19 skill_base, 16 context_gates, 14 store_mvp, 14 kernel_mvp, 14 synthesis_mvp, 14 implications, 26 story_planning, 10 story_prompt, 9 story_cache, 7 story_integration, 28 self_identity, 28 moral_negation). 0 regressions.
- **54 registered contracts** in `registry.v1.json` (includes sense_candidate, semantic_classes, wn_supersense_map, verbnet_map, reserved_lexemes, router_lexicon_families, frame_templates, food_tags, health_disclaimers, safety_policies, story_components, weather_concepts, meal_scopes, assistant_identity, answer_templates, capability_manifest, memory_insights, router_semantic_aliases, verb_states, state_valences, story_plan_schema, and 33 others).
- **M3 sealed dictionary**: 72 words across 11 intent categories. Ingest/promote ≥80%, routing agreement ≥80%, retention ≥80%.
- **9/9 classifiers migrated** to frame linker.
- **`_FRAME_LINKER_MIGRATED_INTENTS`**: 8 intents — weather, story, media_playback, autobiographical_memory, meal_suggestion, common_sense_safety, social_contact, health_advice.
- **89 semantic classes** in `semantic_classes.v1.json`.
- **Entity store**: 5 tables (entities, entity_slots, entity_relations, class_schemas, class_schema_slots). Seed class hierarchy (entity→person/event/place/object, competition→event, personal_experience). Migration functions for contacts→persons, user_facts→self entity.
- **1 pre-existing failure**: bundle test (`v01_audit`/`v01_progress` milestone blockers).
- **3 foundational flaws documented**: (1) knowledge trapped in code, not referenceable as data; (2) router/synthesis layer duplicates each other; (3) architecture wires intents before meaning.
- **P0-P3 extraction complete** (all priority-table items in contracts). **Phase 3 generic dispatch** complete. **Phase 4 skill modules** (meal, story, memory) complete.
- **Router semantic aliases** (`_semantic_object_role_tokens` + `_secondary_meaning_hint_groups`, ~140 token→intent mappings) extracted to `router_semantic_aliases.v1.json`. Both functions now read from contract.
- **Health disclaimers** contract extended with `urgent_terms` and `urgent_pairs` — `_has_urgent_health_frame` reads from contract instead of hardcoded sets.
- **Answer templates** contract extended with `evidence_count_targets` (7 intent→int mappings) and `answer_specificity_phrases` (11 intent→phrase patterns). `_target_evidence_count` and `_answer_specificity` read from contract.
- **M5 E3 reranker built** — `assistant_frame_ranker.py` with predicate + object alignment using UOL token_roles. Wired into router fallback route. 38 tests.
- **Meaning architecture documented** — spec §14: three-timescale model (T1 UOL → T2 personal_experience → T3 lexicon/entity store), semantic_classes.v1.json as the spine invariant, personal_experience slots (outcome, polarity, learned_fact_ids, follow_up, intent_achieved).
- **T2 personal_experience writer built** — `assistant_experience_writer.py` implements `record_conversation_experience()`, wired into kernel `_remember()` after each turn.
- **Cross-layer class ID invariant test** — `test_meaning_invariant.py`: 17 tests verifying every class ID in frame templates, contracts, entity store schemas, and UOL is defined in the spine. Documents 20 transitional UOL-only classes and 3 overlapping classes.

## Relevant Files
- **`docs/assistant_os_spec.md`** — Comprehensive authoritative architecture specification. Knowledge extraction priority table, anti-regression checklist, migration path.
- **`docs/local_assistant_os_mvp_plan_v2.md`** — Authoritative execution plan.
- **`melm/contracts/food_tags.v1.json`** — P0 contract: 12 food→tag marker mappings (extracted from router).
- **`melm/contracts/health_disclaimers.v1.json`** — P1 contract: 5 urgent medical disclaimer responses (extracted from synthesis).
- **`melm/contracts/safety_policies.v1.json`** — P1 contract: public clothing safety template + 3 destination phrases (extracted from synthesis).
- **`melm/contracts/story_components.v1.json`** — P2 contract: 3 story generation heuristic groups (image: 8 keyword→image pairs across title/full-text, challenge: 5 topic→challenge, lesson: 4 topic→lesson).
- **`melm/contracts/weather_concepts.v1.json`** — P2 contract: 3 weather domain terms (extracted from router).
- **`melm/contracts/meal_scopes.v1.json`** — P3 contract: 5 scope tokens with default (extracted from router).
- **`melm/contracts/assistant_identity.v1.json`** — P3 contract: 4 identity/status templates (extracted from synthesis).
- **`melm/contracts/answer_templates.v1.json`** — Phase 3 contract: 2 intent templates with reason gates and evidence checks.
- **`melm/contracts/registry.v1.json`** — 18 registered contracts with validators.
- **`melm/contracts/validation.py`** — Validators: `validate_food_tags`, `validate_health_disclaimers`, `validate_safety_policies`, `validate_story_components`, `validate_weather_concepts`, `validate_meal_scopes`, `validate_assistant_identity`, `validate_answer_templates`, `validate_memory_insights`. Loaders: `load_food_tags`, `load_health_disclaimers`, `load_safety_policies`, `load_story_components`, `load_weather_concepts`, `load_meal_scopes`, `load_assistant_identity`, `load_answer_templates`, `load_memory_insights`.
- **`melm/appliance/assistant_decoder.py`** — `ConstrainedDecoder` with registry/dispatch, `TemplateBackend` fallback. 18 tests.
- **`melm/appliance/assistant_decoder_llguidance.py`** — `LlguidanceBackend` (lazy HF CausalLM + llguidance LLInterpreter), `HFCompatTokenizer`, grammar/prompt builders. 29 tests.
- **`melm/appliance/local_assistant_router.py`** — `_food_tags()` now reads from contract. `_is_family_installed` manifest enforcement. 17-classifier cascade. `set_uol_lexicon()` wired at module init.
- **`melm/appliance/assistant_synthesis.py`** — `_answer()` uses generic dispatch via `_ANSWER_HANDLERS` registry + `_render_contract_answer()`. All story/health/safety/identity functions read from contracts. Handlers delegate to skill modules for meal, story, and memory.
- **`melm/appliance/assistant_authority.py`** — Authority: evidence packets, answer plans, verification, negation-aware forbids.
- **`melm/appliance/assistant_lexicon.py`** — Factored store-backed vocabulary with acquisition lifecycle.
- **`melm/appliance/assistant_frame_linker.py`** — Correct pattern: meaning-first dispatch. 9 classifiers migrated.
- **`melm/appliance/assistant_os_store.py`** — Entity CRUD, class schema seeding, migration functions.
- **`melm/appliance/assistant_experience_writer.py`** — T2 personal_experience entity writer, wired into kernel `_remember()`.
- **`tests/test_assistant_decoder_mvp.py`** — 18 decoder scaffold tests.
- **`tests/test_assistant_decoder_llguidance_mvp.py`** — 29 llguidance tests.
- **`tests/test_contracts_mvp.py`** — 6 contract validation tests.
- **`tests/test_meaning_invariant.py`** — 17 tests: cross-layer semantic class spine invariant. Documents UOL transition state. 7 realistic conversation tests (tell story, play music, feel sick, eat pasta, call mom, weather, UOL+reranker agreement).
- **`docs/assistant_os_spec.md §14`** — Meaning representation architecture: three-timescale model, semantic class spine invariant, personal_experience slot schema (outcome/polarity/learned_fact_ids/follow_up/intent_achieved), extensibility contract, anti-regression checklist update.
- **`melm/appliance/functional_grammar.py`** — UOL lexicon integration: `_UOL_LEXICON` ref, `set_uol_lexicon()`, lexicon fallback in `_lemma()` and `_semantic_class()`. 16 tests.
- **`melm/appliance/assistant_skill_base.py`** — Skill protocol: `SkillManifest`, `Skill` structural protocol, `SkillRegistry`. 19 tests.
- **`melm/appliance/assistant_frame_ranker.py`** — E3 reranker with UOL token_roles consumption (_predicate_action_alignment_score, _object_alignment_score). 38 tests.
- **`melm/appliance/uol_types.py`** — Unified `AffectSignal` with all 11 fields, updated `UolAct.to_dict()` serialization.
- **`melm/appliance/assistant_mood_engine.py`** — `decay_mood()`, fixed `load_mood_regions()` key, fixed `_filter_recovery()`/`_is_complaint()`, `_BE_FORMS` constant, `last_updated` on `MoodState`, decay in `update_session_mood()` and `initial_mood_from_baseline()`.
- **`melm/appliance/local_assistant_router.py`** — `compute_utterance_affect(lemmas, uol_act, lexicon)` replaces `infer_affect()`, `_ParseBundle.lemmas` field, `_load_or_init_mood` passes real store.
- **`melm/appliance/assistant_os_store.py`** — `_intent_tallies` + `_event_ring_buffer`, `get_intent_tally()`, `get_recent_events()`, fixed `set_mood_state` (uuid), `current_mood_state` (WHERE clause), `query_session_summaries` (SQL filter), `write_anonymous_fact` (hashlib), `update_lexical_sense` (commit), `MAX(event_id)` fix.
- **`melm/appliance/assistant_os_kernel.py`** — store passed to router, `_remember()` calls `set_mood_state()`, `handle()` calls `record_session_summary()` + `set_ambient_mood()`.
- **`docs/final-sentience-gap-fixes.md`** — consolidated gap-fix plan (16 gaps, performance benchmarks, DeepSeek lessons, anti-regression invariants, sequencing).
- **`docs/sentience-competition-test.py`** — 32 acceptance tests for all sentience scenarios. Run: `python docs/sentience_competition_test.py`.
- **NAMELESS verb_state exploration** — analyzed 1,795 verb causality entries across 353 files at `C:\dev\nameless_vector\verb_state\`. Schema: verb → goals, mechanisms, pre/post states (physical/emotional/mental/positional) for subject and object. 115 verbs carry `_if_sentient` qualifiers on final object states — the natural bridge to moral cognition. Entity type overlap with our system is partial (193 subject types, 645 object types; `biological_body`, `object`, `concept`, `place` overlap).
- **Moral cognition design** — `docs/assistant_os_spec.md §16`: defines T4 action meaning timescale, `derive_moral_context()` pure function, `verb_states.v1.json` + `state_valences.v1.json` contracts, integration plan to replace 5 router duplication sites. Total cost: ~205 Python lines + ~18 KB JSON. No atomizer changes, no new pipeline stage, pure stdlib.
- **Moral cognition implementation plan** — `docs/moral_cognition_impl_plan.md`: 6-phase implementation plan with exact file lines, before/after patches, contract schemas, engine code, test plan, and documentation cleanup for 5 stale docs. Covers 3 learning pathways: NAMELESS seed extraction (`scripts/extract_nameless_verb_states.py`), VerbNet/Wikipedia offline extraction (`scripts/extract_verbnet_verb_states.py`), and runtime chat learning (bounded ring buffer → session entity → offline consolidation script). All offline extraction runs on dev machine only; runtime is bounded ring buffer + flush to personal_experience slot.
