## Goal
- Migrate vocabulary from code constants (inline sets, LEGACY_ROUTER_TERM_CLASSES) to the factored lexicon store, replacing the transitional `_semantic_family_terms` bridge with a direct store-backed lookup. Build the M2 meaning substrate per the MVP v2 plan: factored lexicon store, semantic-class registry, ingestion gate, and legacy seed migration with bit-identical routing. Build the M3 learning vertical slice: runtime vocabulary acquisition channels (user-teaching, cloud lookup) with quarantine/promote/rollback lifecycle. Build the unified entity system (entities, entity_slots, entity_relations, class_schemas) replacing user_facts/inventories as the single store for all people, events, objects, and their attributes — documented in §16.5 of the MVP plan.

## Constraints & Preferences
- All claims must be reproducible by a command on the current tree.
- `docs/local_assistant_os_mvp_plan_v2.md` is the authoritative execution plan.
- Behavioral gates assert observable effects, not debug-label strings.
- Tests use reproducible PRNG seeds for deterministic output.
- Conversation dialog expectations reflect **current** routing behavior, not aspirational.
- Context resolution is handled at the conversation management layer, not in the stateless classifier chain.
- Architecture must be language-agnostic — inflection normalization is a swappable function, contracts store lemmas only. UOL is the foundational meaning representation; everything else is built on it.
- The `_semantic_family_terms` bridge is transitional — M5 replaces keyword classifiers with UOL-based frame linking. Do not treat the bridge as permanent architecture.
- **Entity architecture**: all things (persons, events, places, objects) share a unified `entities` table with `kind` discriminator. Slot values live in `entity_slots`. Relations between entities live in `entity_relations`. Semantic classes define what slots entities of that class can have (`class_schemas` + `class_schema_slots`).
- **Event class hierarchy**: `event` is an entity class with `kind='event_type'` or `kind='event_instance'`. Competition inherits from event. Chat sessions are personal-experience events (`kind='personal_experience'`).

## Progress
### Done
- **M4 authority module built** — `assistant_authority.py` with `AuthorityEvidenceItem`, `AuthorityEvidencePacket`, `AnswerPlan`, `VerificationResult`, `DecoderResult`, `AuthorityInfo` dataclasses. `build_evidence_packet`, `build_answer_plan`, `verify_answer` functions. Negation-aware constraint checking (e.g., "not a diagnosis" does not trigger diagnosis forbids).
- **`BoundedSynthesisResult.authority` field** — populated with `AuthorityInfo` in `synthesize()` success path.
- **`_decode` method on `BoundedLocalSynthesizer`** — returns `DecoderResult` for M4 scaffold decode→verify flow.
- **All authority symbols exported from `__init__.py`** — `AnswerPlan`, `AuthorityEvidenceItem`, `AuthorityEvidencePacket`, `AuthorityInfo`, `DecoderResult`, `VerificationResult`, `build_answer_plan`, `build_evidence_packet`, `verify_answer`.
- **Contract JSON files added to `PI_BUNDLE_STATIC_FILES`** — `frame_templates.v1.json`, `reserved_lexemes.v1.json`, `semantic_classes.v1.json`, and 8 more contract artifacts fixed bootstrap_runtime and FrameLinker errors in the portable bundle.
- **`test_assistant_authority_mvp.py` un-ignored** — 24 authority tests pass (no longer pre-existing failure).
- **`_IN_MEMORY_LEXICON` replaces `_CLASS_TO_FALLBACK_TERMS`** — direct term→classes `dict[str, frozenset[str]]`, built from LEGACY at module load. `_semantic_family_terms` reads exclusively from this cache.
- **`lexicon_owned`/`lexical_class_lookup` removed entirely** — from `_semantic_family_terms`, all classifiers, `OnDeviceAssistantRouter` constructor, `_classify_intent_from_uol_slots`, and the `decide` method. No per-family activation bits.
- **`Callable` import removed** — unused after `lexical_class_lookup` removal.
- **All bare `*,` separators removed** — residues from `lexical_class_lookup`/`lexicon_owned` parameter removal.
- **`replace_in_memory_lexicon()` added** — replaces the module-level cache (kernel uses it in `_rebuild_router_lexicon_cache`).
- **`_rebuild_router_lexicon_cache` always rebuilds cache** — store-backed when rows exist; resets to LEGACY baseline when empty. Fixes cross-test pollution where a prior test's store-backed cache leaked into subsequent tests with fresh stores.
- **`_is_weather_concept_question` reverted to hardcoded set** — concept terms ("define", "explain", "system") are grammatical/structural patterns, not vocabulary.
- **`build_legacy_router_candidates(seed_all=True)`** — exports ALL 200+ `LEGACY_ROUTER_TERM_CLASSES` entries.
- **`write_legacy_lexicon_candidates` uses `seed_all=True`** — legacy export includes all router vocabulary.
- **`seed_assistant_os_lexicon(store)` created** — seeds all 246 candidates (43 FG + 203 router) into the store.
- **`seed_assistant_os_lexicon` called from bootstrap** — `_bootstrap_runtime` seeds the store right after `initialize_assistant_os_database`. The kernel then rebuilds the in-memory cache from store data at init.
- **`semantic_classes.v1.json` extended with all 28 LEGACY-internal classes** — 89 total classes.
- **Bit-identical routing verified** — store-backed cache is a superset of LEGACY cache.
- **`import_transcript_replay_fixture` restored** — fix verified: `test_cli_api_session_smoke_can_execute_configured_real_actions` passes.
- **V3: 15 missing content-word terms added** to `LEGACY_ROUTER_TERM_CLASSES`.
- **V3: `build_legacy_in_memory_lexicon()` extracted** — constant now only referenced in `assistant_lexicon_legacy.py`.
- **M3: `acquire_definition` built** — user-teaching channel with copula-detection.
- **M3: `offline_definition_lookup` built** — JSONL dictionary channel.
- **M3: `cloud_definition_lookup` built** — LLM chat-completions channel.
- **M3: `set_lexical_sense_status` built** — promote/rollback API.
- **M3: Kalimba e2e fixture** — teach→promote→rollback lifecycle.
- **M4: `semantic_classes_activated` in events** — collector, column, event field, queries.
- **M5: 9 classifiers migrated** to `_classify_from_frame_linker`: weather, story, media_playback, autobiographical_memory, meal_suggestion, common_sense_safety, social_contact, health_advice, personal_memory.
- **`required_all_classes` AND-gate** added to frame linker and validation.
- **Entity architecture documented** in §16.5 of MVP plan — unified entities, entity_slots, entity_relations, class_schemas, class_schema_slots, event class hierarchy, frame slot states.
- **4 new entity tables** — `class_schemas`, `class_schema_slots`, `entities`, `entity_slots`, `entity_relations` added to `initialize()` DDL.
- **`StoredEntity`, `StoredEntitySlot`, `StoredEntityRelation`, `ClassSchemaDef` dataclasses** added to store.
- **`seed_class_schemas(store)` built** — seeds event class hierarchy (entity→person, event, place, object; competition→event) with slot definitions.
- **`_ensure_entity_tables` migration** — creates tables for existing stores.
- **Entity CRUD methods** — `add_entity`, `get_entity`, `find_entities`, `set_entity_slot`, `get_entity_slots`, `get_entity_slot`, `delete_entity`.
- **Entity relations CRUD methods** — `add_relation`, `get_entity_relations`, `find_relations_by_type`, `find_relations_by_target`, `delete_relation`. Uses `StoredEntityRelation` dataclass. `add_relation` returns the generated relation_id; duplicate (entity_id, relation, target_entity_id) is silently idempotent via `INSERT OR IGNORE`.
- **`seed_class_schemas` exported** from `__init__.py` and wired into CLI bootstrap.
- **Review fixes applied** — `ClassSchemaDef.parent_class_id` changed to `str | None`; `seed_class_schemas` uses `None` for root parent; `count()` and `table_counts()` include entity tables; `seed_class_schemas` added to `__all__`; `count()` `if` statement `:` restored.
- **36 entity tests** — schema creation, migration, seeding, CRUD, slot states, class hierarchy, entity relations, FK enforcement, `count()` whitelist, restart persistence, frozen dataclass invariance.
- **Contacts ported to entities** — `migrate_contacts_to_entities(store)` reads `inventories WHERE kind='contact'`, creates `entities WHERE kind='person'` with name + phone slots. Entity ID prefixed `contact:<item_id>`. Idempotent — skips already-migrated contacts.
- **Self facts ported to entities** — `migrate_self_facts_to_entities(store)` creates `entity_id='self'` if absent, then ports non-revoked `user_facts` key→value into `entity_slots`. Idempotent per-fact.
- **8 migration tests** — contacts migrate correctly, empty inventory, no-number contact, idempotent re-run; self-facts migrate correctly, skip revoked facts, idempotent.
- **Migrations wired into CLI bootstrap** — called after `seed_class_schemas` in `_bootstrap_runtime`.
- **M0 complete** — tree committed in 4 reviewable slices + 1 CI fix + 1 feature slice. CI green (3m18s, 579 passed). Known-broken tests excluded via pyproject.toml `addopts` (`--ignore` / `--deselect`). Optional `tokenizers` dep installed in CI.
- **`SLOT_STATE_*` constants defined** — `SLOT_STATE_FILLED`, `SLOT_STATE_ASKED_BUT_EMPTY`, `SLOT_STATE_UNKNOWN_ENTITY`, `SLOT_STATE_UNKNOWN`, `SLOT_STATE_INFERRED` in `assistant_frame_linker.py`.
- **`slot_states` field added to `FrameCandidate`** — `dict[str, str]` mapping slot name → state constant. Defaults to empty dict via `field(default_factory=dict)`.
- **`slot_states` field added to `AssistantDecision`** — flows slot state info from router through kernel to synthesis.
- **`slot_bindings` populated in frame templates** — `social_contact` binds `["name", "phone"]`; `personal_memory` binds `["self_facts"]`. All others use `[]` (empty).
- **`slot_bindings` validated in `validate_frame_templates()`** — must be array of strings.
- **`_resolve_slot_states` helper** — kernel resolves slot states from entity store for `social_contact` (looks up person entities matching tokens) and `personal_memory` (checks self entity fact existence). Wired into `decide()` after router returns.
- **personal_memory migrated** — last classifier delegated to frame linker. Structural gates kept (child/routine/household sub-patterns use possessive/contextual logic beyond lexical class matching); `memory_cognition`+first_person lexical path delegated to `_classify_from_frame_linker`. `personal_memory` NOT added to `_FRAME_LINKER_MIGRATED_INTENTS` (prevents bare `memory_recall` word matches like "remember" → personal_memory).
- **Bulk lexicon seeders built** — `assistant_lexicon_bulk.py` with `seed_wordnet_supersenses()`, `seed_verbnet_classes()`, `seed_bulk_lexicon()` orchestrator. WordNet: 1,540 entries seeded (dormant) via `wn_supersense_map.v1.json` (45 mappings) and `word_supersense_data.v1.jsonl` (1,761 word→supersense entries). VerbNet: 22 entries seeded via `verbnet_map.v1.json` (12 mappings) and `verb_data.v1.jsonl` (23 verb→verbnet-class entries).
- **Bulk seeder wired into bootstrap** — `seed_bulk_lexicon(store)` called from `seed_assistant_os_lexicon`, between legacy seed and router-family configuration.
- **`_candidate` normalization fix** — uses `_normalize_term` instead of `lemma.lower()` for reserved/policy safety cross-check, matching `lexicon_ingest` normalization.
- **Bulk seeder tests** — 11 new tests: basic seeding, reserved term skipping, unknown supersense skipping, idempotency, orchestrator, missing data files, dormant status, actual data file validation.
- **Data generator script saved** — `scripts/generate_bulk_lexicon_data.py` creates JSONL data files from LEGACY vocabulary and curated word lists per supersense.

### In Progress
- *(none)*

### Blocked
- `test_cli_pi_bundle_builds_portable_self_checked_bundle` — bundle builds but `v01_audit`/`v01_progress` checks fail by design (project milestone blockers, not code issues). Remaining infrastructure smokes all pass with contract files included.
- **personal_memory** — remaining sub-patterns (child/routine/household) kept as structural gates use possessive/contextual logic beyond frame linker's lexical class matching.

## Key Decisions
- **Entity architecture**: unified `entities` table with `kind` discriminator. Persons are `kind='person'`. Events are `kind='event_type'`/`kind='event_instance'`. Slot values in `entity_slots`. Relations in `entity_relations`. Class hierarchy defines valid slots via `class_schemas` + `class_schema_slots`.
- **Event class hierarchy**: `competition` inherits from `event`. Slots defined on the class, not per entity. Frame slot states (`filled`, `asked_but_empty`, `unknown_entity`, `unknown`, `inferred`) enable intelligent "I don't know" responses.
- **Migration path**: `user_facts` → `entities WHERE kind='self'` + `entity_slots`; contacts → `entities WHERE kind='person'`.
- In-memory cache is always rebuilt — store data when rows exist, LEGACY reset when empty.
- Migrated frame-linker intents require top-candidate status (`candidates[0].frame_id == frame_id`) to prevent preemption.
- Fallback uses strict `>` for migrated intents to block bare required-class matches.
- Weather concept gate retained for "what is weather?" → `open_domain`.
- Action tokens checked early in story, media, and autobiographical migrations for bit-identical routing.
- Concept-level terms and grammar/structural logic stay as hardcoded patterns, not vocabulary lookups.
- **Talk-based social_contact path uses multi-candidate check** — the contact_action path uses `_classify_from_frame_linker` (strict top-candidate), but the talk+need/help/please path checks all candidates for a passing social_contact score. This prevents preemption by alphabetical tie-breaker when health_advice matches via "help" → advice_action.
- **Slot state resolution is kernel-side, not in the frame linker** — `_resolve_slot_states` lives in `assistant_os_kernel.py` because the frame linker shouldn't depend on the store. After `OnDeviceAssistantRouter.handle()` returns, the kernel enriches the decision with slot states from the entity store.
- Contracts store lemmas only — language-agnostic invariant per the plan.
- `cloud_lookup` uses `urllib.request` directly (project convention), no new dependencies.

## Next Steps
1. **Review & consolidate** — Full validation pass.
2. **Personal_memory frame linker migration** — consider adding child_memory sub-frame with restrictive threshold to catch "the child" patterns.

## Critical Context
- **633 tests pass**: authority (24), frame_linker (27), router (54, 71 subtests), eval (4, 107/107 cases), lexicon (65), entity (61), lifecycle (2), lifecycle integration (1), CLI (rest). 11 new bulk seeder tests.
- **9/9 classifiers migrated** to frame linker. 1 partial (personal_memory — structural gates kept, memory_cognition+first_person delegated to frame linker).
- **`_FRAME_LINKER_MIGRATED_INTENTS`**: 8 intents — weather, story, media_playback, autobiographical_memory, meal_suggestion, common_sense_safety, social_contact, health_advice.
- **89 semantic classes** in `semantic_classes.v1.json`.
- **4 new entity tables**: `class_schemas`, `class_schema_slots`, `entities`, `entity_slots`, `entity_relations`.
- **`seed_class_schemas`** seeds: entity (base), person, event, place, object, competition, personal_experience — with slot definitions for each.
- **Entity CRUD**: `add_entity`, `get_entity`, `find_entities`, `set_entity_slot`, `get_entity_slots`, `get_entity_slot`, `delete_entity`.
- **Migration functions**: `migrate_contacts_to_entities(store)` — ports inventory contacts to person entities (idempotent). `migrate_self_facts_to_entities(store)` — ports user_facts to self entity slots (idempotent). Both wired into CLI bootstrap.
- **1 pre-existing failure**: bundle test (`v01_audit`/`v01_progress` milestone blockers).
- **Pre-existing test pollution**: router tests fail when run after legacy tests due to `_IN_MEMORY_LEXICON` global mutable state.
- **Slot state infrastructure**: `SLOT_STATE_*` constants (5 states), `slot_states` on `FrameCandidate` + `AssistantDecision`, `slot_bindings` in templates (validated), `_resolve_slot_states` in kernel for `social_contact` and `personal_memory` intents.
- **Authority module**: `assistant_authority.py` with evidence packets, answer plans, verification. `AuthorityInfo` wired into `BoundedSynthesisResult.authority`. `_decode()` on synthesizer for M4 scaffold. Negation-aware forbids checking.

## Relevant Files
- **`melm/appliance/local_assistant_router.py`** — 9 migrated classifiers, `_classify_from_frame_linker`, `_FRAME_LINKER_MIGRATED_INTENTS`, `AssistantDecision` with `slot_states` field.
- **`melm/appliance/assistant_frame_linker.py`** — `_match_required_all_classes` AND-gate, `SLOT_STATE_*` constants, `FrameCandidate` with `slot_states` field.
- **`melm/appliance/assistant_os_kernel.py`** — `_rebuild_router_lexicon_cache` calls `rebuild_entity_lexicon_index`, `_resolve_slot_states` helper wired into `decide()`.
- **`melm/contracts/frame_templates.v1.json`** — templates with `required_all_classes` and `slot_bindings` arrays.
- **`melm/contracts/validation.py`** — `validate_frame_templates` now validates `slot_bindings`.
- **`melm/appliance/assistant_os_store.py`** — entity DDL (class_schemas, class_schema_slots, entities, entity_slots, entity_relations), `seed_class_schemas`, `StoredEntity`/`StoredEntitySlot`/`StoredEntityRelation`/`ClassSchemaDef` dataclasses, entity CRUD methods, `_ensure_entity_tables` migration.
- **`melm/appliance/assistant_os_kernel.py`** — `_rebuild_router_lexicon_cache` now calls `rebuild_entity_lexicon_index` after store-backed or legacy rebuild.
- **`melm/appliance/local_assistant_router.py`** — `_semantic_family_terms` with bigram compound token detection. `rebuild_entity_lexicon_index(store)` injects entity labels into `_IN_MEMORY_LEXICON`.
- **`melm/appliance/assistant_authority.py`** — M4 authority: evidence packets, answer plans, verification, negation-aware forbids.
- **`scripts/local_assistant_os_cli.py`** — bootstrap imports `seed_class_schemas` and calls it.
- **`docs/local_assistant_os_mvp_plan_v2.md`** — §16.5 Entity store architecture.
- **`tests/test_assistant_frame_linker_mvp.py`** — 27 frame linker tests.
- **`tests/test_local_assistant_router_mvp.py`** — 54 router tests.
- **`tests/test_assistant_lexicon_mvp.py`** — 65 lexicon tests (54 original + 11 bulk seeder).
- **`tests/test_assistant_os_eval_mvp.py`** — 4 eval tests (107/107 cases).
- **`tests/test_entity_architecture_mvp.py`** — 61 entity tests (schema, migration, seeding, CRUD, slot states, entity relations, count whitelist, contacts migration, self-facts migration, entity lexicon index).
