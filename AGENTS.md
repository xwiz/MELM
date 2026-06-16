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
1. **Phase 4: Build skill module pattern** — `assistant_skill_meal.py`, `assistant_skill_story.py`, `assistant_skill_memory.py` complete.
2. **Pi benchmark** — install on Pi 4/5, measure tok/s/TTFT/RSS with llguidanceBackend + small CausalLM.
3. **BitNet b1.58 1B backend** — when Pi benchmark validates need for smaller/quantized decoder.


## Critical Context
- **391 core tests pass** (57 router, 70 lexicon, 27 frame_linker, 75 entity, 24 authority, 38 reranker, 18 decoder, 29 llguidance, 6 contracts, 4 eval, 2 lifecycle, rest jobs/cli, 17 meaning_invariant, 16 uol_lexicon, 23 experience_writer). 0 regressions.
- **18 registered contracts** in `registry.v1.json` (sense_candidate, semantic_classes, wn_supersense_map, verbnet_map, reserved_lexemes, router_lexicon_families, frame_templates, food_tags, health_disclaimers, safety_policies, story_components, weather_concepts, meal_scopes, assistant_identity, answer_templates, capability_manifest, memory_insights, router_semantic_aliases).
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
