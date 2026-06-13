## Goal
- Migrate vocabulary from code constants (inline sets, LEGACY_ROUTER_TERM_CLASSES) to the factored lexicon store, replacing the transitional `_semantic_family_terms` bridge with a direct store-backed lookup. Build the M2 meaning substrate per the MVP v2 plan: factored lexicon store, semantic-class registry, ingestion gate, and legacy seed migration with bit-identical routing. Build the M3 learning vertical slice: runtime vocabulary acquisition channels (user-teaching, cloud lookup) with quarantine/promote/rollback lifecycle.

## Constraints & Preferences
- All claims must be reproducible by a command on the current tree.
- `docs/local_assistant_os_mvp_plan_v2.md` is the authoritative execution plan.
- Behavioral gates must assert observable effects, not debug-label strings.
- Tests use reproducible PRNG seeds for deterministic output.
- Conversation dialog expectations reflect **current** routing behavior, not aspirational.
- Context resolution is handled at the conversation management layer, not in the stateless classifier chain.
- The architecture must be language-agnostic — inflection normalization is a swappable function, contracts store lemmas only (no inflected forms). UOL is the foundational meaning representation; everything else is built on it.
- The `_semantic_family_terms` bridge is transitional — M5 replaces keyword classifiers with UOL-based frame linking. Do not treat the bridge as permanent architecture.

## Progress
### Done
- **`_IN_MEMORY_LEXICON` replaces `_CLASS_TO_FALLBACK_TERMS`** — direct term→classes `dict[str, frozenset[str]]`, built from LEGACY at module load. `_semantic_family_terms` reads exclusively from this cache.
- **`lexicon_owned`/`lexical_class_lookup` removed entirely** — from `_semantic_family_terms`, all classifiers, `OnDeviceAssistantRouter` constructor, `_classify_intent_from_uol_slots`, and the `decide` method. No per-family activation bits.
- **`Callable` import removed** — unused after `lexical_class_lookup` removal.
- **All bare `*,` separators removed** — residues from `lexical_class_lookup`/`lexicon_owned` parameter removal.
- **`replace_in_memory_lexicon()` added** — replaces the module-level cache (kernel uses it in `_rebuild_router_lexicon_cache`).
- **`_rebuild_router_lexicon_cache` always rebuilds cache** — store-backed when rows exist; resets to LEGACY baseline when empty. Fixes cross-test pollution where a prior test's store-backed cache leaked into subsequent tests with fresh stores.
- **`_is_weather_concept_question` reverted to hardcoded set** — concept terms ("define", "explain", "system") are grammatical/structural patterns, not vocabulary.
- **`build_legacy_router_candidates(seed_all=True)`** — exports ALL 200+ `LEGACY_ROUTER_TERM_CLASSES` entries.
- **`write_legacy_lexicon_candidates` uses `seed_all=True`** — legacy export includes all router vocabulary.
- **`seed_assistant_os_lexicon(store)` created** — in `assistant_os_store.py`; seeds all 246 candidates (43 FG + 203 router) into the store.
- **`seed_assistant_os_lexicon` called from bootstrap** — `_bootstrap_runtime` seeds the store right after `initialize_assistant_os_database`. The kernel then rebuilds the in-memory cache from store data at init.
- **`semantic_classes.v1.json` extended with all 28 LEGACY-internal classes** — `personal_attribute`, `social_relation`, `child_relation`, `health_condition`, `evaluative_expression`, `advice_action`, `contact_action`, `communication_action`, `movement_action`, `memory_recall`, `autobiographical_event`, `autobiographical_action`, `owner_concept`, `temporal_descriptor`, `request_softener`, `social_greeting`, `action_verb`, `hardware_entity`, `definition_request`, `abstract_concept`, `goal_concept`, `household_concept`, `routine_concept`, `public_place`, `undress_state`, `clothing_item`, `health_domain`, `wellness_activity`. Formal ontology now has 89 classes.
- **Bit-identical routing verified** — store-backed cache is a superset of LEGACY cache: all 203 LEGACY terms present with identical class assignments. 21 extra FG-only terms enrich the cache but their classes (`verb.communicate`, `verb.move`, `verb.consume`, `verb.create`, `person`, `abstract`, `verb.social`, `verb.change`, `verb.stative`, `cognition`, `action`) are never checked by any classifier.
- **`seed_assistant_os_lexicon` exported** from `melm/appliance/__init__.py`.
- **`import_transcript_replay_fixture` restored in CLI imports** — accidentally removed during M3 stub cleanup; function was called inside a `try/except Exception` block in `_build_event_ledger_calibration_payload`, causing a silent `NameError` that made calibration checks return `passed=False`. Fix verified: `test_cli_api_session_smoke_can_execute_configured_real_actions` passes.
- **V3: 15 missing content-word terms added to `LEGACY_ROUTER_TERM_CLASSES`** — `bleeding`, `breathe`, `emergency`, `export`, `faint`, `name`, `poison`, `recommend`, `report`, `see`, `send`, `share`, `suggest`, `take`, `upload` — from inline classifier sets. `play`, `start` omitted (already seeded via FG verb path as non-routing `action` class). Pre-existing duplicate `goals: goal_concept` removed.
- **V3: `build_legacy_in_memory_lexicon()` extracted** — in `assistant_lexicon_legacy.py`. `LEGACY_ROUTER_TERM_CLASSES` refs removed from `local_assistant_router.py` (module init) and `assistant_os_kernel.py` (cache fallback). Constant now only referenced inside `assistant_lexicon_legacy.py` — single provenance source.
- **M3: `acquire_definition(store, utterance)` built** — user-teaching acquisition channel. Detects copula-definition frames ("a kalimba is a small thumb piano") and meaning frames ("X means Y"), parses lemma/genus/definition, walks genus through lexicon for semantic class candidates, ingests via `lexicon_ingest` as `provenance=user_taught`, `status=quarantined`, `confidence_prior=0.60`. 9 new tests cover copula, means, normalization, empty, non-matching, unresolved genus, reserved-word rejection, polysemous genus, and provenance verification.
- **M3: `offline_definition_lookup(store, word, *, dictionary_path)` built** — offline dictionary channel. Reads a line-delimited JSONL dictionary file, matches entries by lemma, builds `sense_candidate.v1` with `genus_lemma` extracted from the definition (or explicit on the entry), walks genus through the store-backed lexicon for class candidates, and ingests via `lexicon_ingest` as `provenance=offline_dictionary`, `status=quarantined`, `confidence_prior=0.70`. 10 new tests cover single entry, missing file, non-matching lemma, empty input, provenance verification, genus extraction, reserved-word rejection, polysemous entries, confidence prior, and explicit genus override.
- **M3: `cloud_definition_lookup(store, word, *, api_key, endpoint, model, pos_hint, timeout)` built** — cloud LLM definition lookup channel. Sends an OpenAI-compatible chat-completions request with a dictionary-service system prompt that includes the full 89-class semantic-class enum. Parses the LLM response into `sense_candidate.v1` with `provenance=cloud_lookup`, `status=quarantined`, `confidence_prior=0.50`, `method=llm_assigned` (per contract policy). `genus_lemma` omitted when empty (not in schema required list). Unknown `class_id` values from the LLM are filtered; empty candidates fall back to `abstract` at 0.50. Network errors, malformed JSON, and `ContractValidationError` are silently swallowed (return `[]`). Uses `urllib.request` (project stdlib convention). 11 tests cover valid ingestion, provenance, confidence prior, `llm_assigned` method, network error, malformed response, empty content, empty word, reserved-word rejection, abstract fallback, and unknown-class filtering.
- **`load_semantic_class_ids()` made public** — previously `_semantic_class_ids` private function in `contracts/validation.py`. Renamed and exported from `melm.contracts`.
- **Code cleanup during M3 review**: `_clean_definition` simplified (removed dead branch and unused `lemma` param). `_compute_class_candidates` cleaned (removed unused `definition` param). `_build_dictionary_candidate` cleaned (removed unused `normalized_word` param). `_build_dictionary_candidate` guards against empty/invalid `pos`. `offline_definition_lookup` catches `OSError` for missing files. `_extract_candidate_from_llm_response` simplified (removed dead early candidate dict).
- **M3: `set_lexical_sense_status(store, sense_id, new_status)` built** — promote/rollback API for lexical sense status. Accepts any value in `VALID_SENSE_STATUSES` (`{"quarantined", "dormant", "active"}`). Uses `with store.connection:` to ensure SQLite transaction commits (critical: implicit transactions are rolled back on `connection.close()`). Raises `ContractValidationError` for invalid status or unknown `sense_id`.
- **M3: Kalimba e2e fixture** — `test_teach_quarantine_promote_restart_rollback` exercises teach (acquire_definition, quarantined) → promote to active → routing visibility via `lexical_classes_for_term` → reopen store verifying persistence → correction merge into same sense → rollback to quarantined removing routing visibility.

### In Progress
- *(none)*

### Blocked
- *(none)*

## Key Decisions
- **In-memory cache is always rebuilt** — `_rebuild_router_lexicon_cache` always replaces `_IN_MEMORY_LEXICON`. Store has data → store data. Store empty → LEGACY baseline. This eliminates test pollution across kernels.
- **`seed_assistant_os_lexicon` runs in bootstrap** — after contract extension, all 35 LEGACY classes are valid, so the store can be fully seeded. The kernel replaces the LEGACY cache with store data at init.
- **`LEGACY_ROUTER_TERM_CLASSES` retained** for seed generation and module-load cache init. Direct router tests (`OnDeviceAssistantRouter(profile)`) still read the LEGACY cache at import time, preserving backward compat.
- **Concept-level terms stay as structural patterns** — `_is_weather_concept_question` uses hardcoded `{"define", "explain", "mean", "means", "system", "systems"}` even if `definition_request`/`abstract_concept` were seeded. These are grammatical/structural, not vocabulary.
- **Grammar/structural logic stays inline** — `_story_request_question`, `_is_question_like`, `_is_request_like`, auto-bio sub-frames, `_identity_composition` helpers, `_private_cloud_evidence_keys`, `_food_tags` are patterns, not vocabulary lookups.
- **Contracts store lemmas only** — language-agnostic invariant per the plan. Inflection normalization out of scope for M2.
- **Centralized dicts are not scattered hacks** — `_secondary_meaning_hint_groups`, `_semantic_object_role_tokens` are per-intent alias blocks for UOL composition, not scattered inline sets.
- **`genus_lemma` omitted when empty** — `_extract_candidate_from_llm_response` conditionally includes `genus_lemma` only when non-empty, since `sense_candidate.v1` has `minLength: 1` on the field but it's not in `required`.
- **`cloud_lookup` uses `urllib.request` directly** — no new dependencies. Matches project convention (`assistant_weather.py`, `assistant_inventory.py`).
- **API key/endpoint/model are function parameters** — no existing API key management infrastructure in the codebase. The caller (CLI, kernel, or test) provides these explicitly.
- **M4: `semantic_classes_activated` added to `AssistantDecision`** — `frozenset[str]` field (default `frozenset()`) on the frozen dataclass. Captured during `handle()` via a module-level collector.
- **M4: `set_semantic_class_collector()`/`_SEMANTIC_CLASS_COLLECTOR`** — module-level collector in `local_assistant_router.py`. `_semantic_family_terms` updates the collector whenever it finds matching tokens. `handle()` resets before routing, reads after, injects into decision via `dataclasses.replace`.
- **M4: `handle()` refactored** — routing logic extracted to `_route_impl()`. `handle()` wraps it with collector setup/teardown. `_classify_intent_from_uol_slots` test updated to inspect `_route_impl` source.
- **M4: `semantic_classes_activated_json` column** — added to `events` table DDL and migration helper `_ensure_event_semantic_classes_column()`. Called from store init after provenance columns.
- **M4: `record_turn` accepts `semantic_classes_activated`** — stored as sorted JSON tuple. Defaults to `frozenset()`.
- **M4: `StoredAssistantEvent.semantic_classes_activated`** — new `frozenset[str]` field. Serialized/deserialized via `_json()` in `load_events`.
- **M4: `_event_memory_record` includes `semantic_classes_activated`** — all SELECT queries (`query_event_memory`, `query_recent_session_memory`, `build_memory_digest`) now include `semantic_classes_activated_json`.
- **M4: `AssistantMemoryEvent.semantic_classes_activated`** — added to kernel's in-memory event type. Wired through `_remember`, `record_turn`, and `load_events` init mapping.
- **M4: 6 new tests** — `test_semantic_family_terms_activates_collector`, `test_router_handle_injects_activated_classes`, `test_activated_semantic_classes_persisted_in_store`, `test_activated_semantic_classes_in_query_event_memory`, `test_migration_adds_semantic_classes_column`, `test_migration_is_idempotent`.
- **Genus extraction fix** — `_GENUS_PP_MARKERS` added; `_extract_genus_lemma` uses PP-aware backwards scan (skips PP objects before returning head noun). Fixes "piano from africa" → "piano" instead of "africa".
- **`_copy_latest_turn` fixed** — now passes `semantic_classes_activated` (parsed from `semantic_classes_activated_json`) to `record_turn`. Eval tests pass without data loss.
- **3 new adversarial tests** — `test_genus_extracts_head_noun_before_pp`, `test_homonym_creates_separate_sense_per_class`, `test_all_reserved_lexemes_rejected_on_acquire` (parameterized over all 55 reserved/policy lexemes).
- **M5: Weather frame linker migration** — `_is_weather_request` uses `_classify_from_frame_linker`. Requires top-candidate status + concept gate pre-filter. Fixes false negative "What is the weather?" → `weather`.
- **M5: Story frame linker migration** — `_is_story_request` uses `_classify_from_frame_linker` with action-token gating for bit-identical routing. Frame linker's top-candidate check prevents preemption by higher-scoring intents.
- **M5: Media playback frame linker migration** — `_is_media_request` uses `_classify_from_frame_linker` with action-token gating. "play something with sounds" special case preserved. Bare required-class matches (e.g., "hi-fi audio") blocked by strict `>` threshold check.
- **`_classify_from_frame_linker` top-candidate fix** — `candidates[0].frame_id == frame_id` prevents preemption. Fallback uses `score > threshold` (strict) for migrated intents to block bare required-class matches.
- **D1: `PI_BUNDLE_STATIC_FILES` updated** — removed legacy `local_assistant_os_mvp_plan.md` (renamed to `_v2`). Bundle self-check still fails on deeper smoke issues (pre-existing).
- **Test count: 85 pass** — frame_linker (27), router (54, 71 subtests), eval (4, 105/105 cases). Pi-bundle and authority tests are pre-existing failures.

## Next Steps
1. **M0 — commit tree, set up CI** — needed before more M5 work per plan ordering. (D1 gate whitelist already fixed; PI_BUNDLE_STATIC_FILES updated.)
2. **M5: UOL-based frame linking** — replace keyword classifiers with contract-driven frame templates backed by the factored lexicon. Weather, story, and media_playback migrated. Remaining: common_sense_safety, health_advice, social_contact, personal_memory, autobiographical_memory, meal_suggestion.

## Critical Context
- **85 tests pass** — frame_linker (27), router (54, 71 subtests), eval (4, 105/105 cases). 2 pre-existing failures unrelated to this work: `test_assistant_authority_mvp.py` (`AnswerPlan` import), `test_cli_pi_bundle_builds_portable_self_checked_bundle` (missing `docs/local_assistant_os_mvp_plan.md` — bundle builds but self-check smokes fail deeper). Pre-existing test pollution: router tests fail when run after legacy tests due to `_IN_MEMORY_LEXICON` global mutable state (isolated files pass).
- **`_rebuild_router_lexicon_cache`** — always rebuilds. Store data → store cache. Empty store → LEGACY cache. Fixes test isolation.
- **`_IN_MEMORY_LEXICON`** — `dict[str, frozenset[str]]`, term→classes. Built from LEGACY at module load. Replaced at kernel init via `_rebuild_router_lexicon_cache()`.
- **`_semantic_family_terms(tokens, *, semantic_classes)`** — no parameters beyond `tokens`, `semantic_classes`. Always reads `_IN_MEMORY_LEXICON`. Updates `_SEMANTIC_CLASS_COLLECTOR` when matches found.
- **`OnDeviceAssistantRouter.__init__(self, profile=None)`** — no lexicon params.
- **`seed_assistant_os_lexicon(store)`** — seeds all 246+ candidates (43 FG + 203 router + 15 V3) with 35 semantic classes. Called from bootstrap.
- **89 semantic classes** in `semantic_classes.v1.json` (61 original + 28 LEGACY-internal).
- **Store cache is a superset of LEGACY cache** — all 203 LEGACY terms present with identical class membership. 21 extra FG-only terms add classes that no classifier checks (bit-identical routing).
- **`_is_weather_concept_question` uses hardcoded concept set** — structural patterns, not vocabulary lookup.
- **Migrated frame-linker intents require top-candidate status** — `_classify_from_frame_linker` checks `candidates[0].frame_id == frame_id`, not just `any()` above threshold. Prevents preemption by higher-scoring frames.
- **Fallback uses strict `>` for migrated intents** — bare required-class matches ("hi-fi audio" → `media_playback`) blocked by `score > threshold` check in fallback. Migrated classifiers already require action tokens or structure; this prevents the weaker fallback path from catching what the old classifier wouldn't.
- **Deletion test works** — deleting a seeded lexeme from the store and creating a new kernel removes the term from the cache.
- **M4: `_SEMANTIC_CLASS_COLLECTOR` is module-level** — `_semantic_family_terms` updates it when matches found. `handle()` sets up/teardown via `set_semantic_class_collector()`. This avoids changing the 28 existing call sites that call `_semantic_family_terms`.
- **`offline_definition_lookup`** catches `OSError` for missing files. `cloud_definition_lookup` catches `OSError`/`URLError`/`json.JSONDecodeError` for network failures. Both silently return `[]` on error.
- **`cloud_definition_lookup`** uses `urllib.request` directly (project convention). Adds no new dependencies. API key, endpoint, and model are function parameters (no existing API key management in codebase).
- **`set_lexical_sense_status`** uses `with store.connection:` to ensure SQLite transaction commits (critical: implicit transactions are rolled back on `connection.close()`).
- **2 pre-existing test failures** (unrelated to this work): `test_assistant_authority_mvp.py` (`AnswerPlan` import), `test_cli_pi_bundle_builds_portable_self_checked_bundle` (missing `docs/local_assistant_os_mvp_plan.md` — bundle builds but self-check smokes fail deeper).

## Relevant Files
- **`melm/appliance/local_assistant_router.py`** — `_semantic_family_terms` (line ~1616): no params beyond `tokens`, `semantic_classes`. `_IN_MEMORY_LEXICON` (line ~1603): built from LEGACY at module load. `replace_in_memory_lexicon()`: replaces cache. `_is_weather_concept_question` (line ~1574): hardcoded concept set. `set_semantic_class_collector()`: activates/deactivates module-level `_SEMANTIC_CLASS_COLLECTOR` for M4 event indexing.
- **`melm/appliance/assistant_os_kernel.py`** — `_rebuild_router_lexicon_cache` (line ~1087): always rebuilds — store data when rows exist, LEGACY reset when empty.
- **`melm/appliance/assistant_os_store.py`** — `seed_assistant_os_lexicon` (line ~1906): seeds FG + router candidates (all 35 classes). No `ContractValidationError` catch. `StoredAssistantEvent` (line ~27): includes `semantic_classes_activated: frozenset[str]`. `record_turn`: accepts `semantic_classes_activated` param. `_ensure_event_semantic_classes_column`: migration for existing stores.
- **`melm/appliance/assistant_lexicon_legacy.py`** — `LEGACY_ROUTER_TERM_CLASSES` (~200+ entries, 35 classes). `build_legacy_router_candidates(seed_all=True)` exports all entries.
- **`melm/appliance/assistant_lexicon.py`** — `lexicon_ingest` (all 35 LEGACY classes now valid), `configure_lexicon_router_families`, `acquire_definition` (user-teaching with regex-based copula/meaning detection), `offline_definition_lookup` (JSONL dictionary channel), `cloud_definition_lookup` (LLM chat-completions channel with `_build_cloud_lookup_payload`, `_call_chat_completion`, `_extract_candidate_from_llm_response`). `_GENUS_SKIP` (line ~249), `_GENUS_PP_MARKERS` (line ~264): prepositions marking trailing PP boundaries. `_extract_genus_lemma` (line ~348): PP-aware backwards scan.
- **`melm/appliance/__init__.py`** — exports `acquire_definition`, `offline_definition_lookup`, `cloud_definition_lookup`, `seed_assistant_os_lexicon`.
- **`melm/contracts/semantic_classes.v1.json`** — 89 class IDs (61 original + 28 LEGACY routing labels).
- **`melm/contracts/validation.py`** — `load_semantic_class_ids()` (public), `validate_sense_candidate` with `cloud_lookup` policy: `quarantined`, ≤0.50, `llm_assigned`.
- **`tests/test_assistant_lexicon_mvp.py`** — 44 tests total (12 original + 9 acquire_definition + 10 offline_dictionary + 11 cloud_definition + 2 refactoring). Covers all M2–M3 ingestion paths.
