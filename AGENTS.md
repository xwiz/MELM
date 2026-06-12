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
- **All 126 related tests pass (86 subtests)** — lexicon, seed, legacy, store, kernel, router, and inventory importers. Pre-existing failures: `test_assistant_authority_mvp.py` (`AnswerPlan` import), `test_cli_pi_bundle` (missing `docs/local_assistant_os_mvp_plan.md`).
- **M3: `acquire_definition(store, utterance)` built** — user-teaching acquisition channel. Detects copula-definition frames ("a kalimba is a small thumb piano") and meaning frames ("X means Y"), parses lemma/genus/definition, walks genus through lexicon for semantic class candidates, ingests via `lexicon_ingest` as `provenance=user_taught`, `status=quarantined`, `confidence_prior=0.60`. 9 new tests cover copula, means, normalization, empty, non-matching, unresolved genus, reserved-word rejection, polysemous genus, and provenance verification.

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

## Next Steps
1. **Build ingestion gate** for runtime vocabulary acquisition (M3 scope) — DONE: `acquire_definition` built with 9 tests.
2. **Clean up CLI script** — remove stale imports (`acquire_definition`, `build_definition_sense_candidate`) that were added in a prior M3 stub session but never deployed. ✓
3. **Remove `LEGACY_ROUTER_TERM_CLASSES`** as module-load cache builder — DONE: extracted `build_legacy_in_memory_lexicon()`; constant is now only referenced inside `assistant_lexicon_legacy.py`.
4. **Expand `_to_lemma`** for language-agnostic inflection normalization (post-M2).
5. **Build UOL-based frame linking** to replace keyword classifiers (M5 scope).
6. **Build offline dictionary lookup** — bundled Wiktextract/WordNet-derived JSONL lookup for words not acquired via user teaching (M3 scope).
7. **Build membrane-gated cloud lookup** — strict-JSON LLM prompt for word definitions, emitting sense_candidate.v1 (M3 scope).
8. **Build kalimba end-to-end fixture** — teach → quarantine → minimal pairs → promote → reuse after restart → correct → rollback (M3 gate).

## Critical Context
- **132/132 related tests pass** (86 subtests) — contracts, lexicon, seed, legacy, store, kernel, router, and inventory importers.
- **`_rebuild_router_lexicon_cache`** — always rebuilds. Store data → store cache. Empty store → LEGACY cache. Fixes test isolation.
- **`_IN_MEMORY_LEXICON`** — `dict[str, frozenset[str]]`, term→classes. Built from LEGACY at module load. Replaced at kernel init via `_rebuild_router_lexicon_cache()`.
- **`_semantic_family_terms(tokens, *, semantic_classes)`** — no other parameters. Always reads `_IN_MEMORY_LEXICON`.
- **`OnDeviceAssistantRouter.__init__(self, profile=None)`** — no lexicon params.
- **`seed_assistant_os_lexicon(store)`** — seeds all 246 candidates (43 FG + 203 router) with 35 semantic classes. Called from bootstrap. Removed `ContractValidationError` catch since all classes are now valid.
- **89 semantic classes** in `semantic_classes.v1.json` (61 original + 28 LEGACY-internal).
- **Store cache is a superset of LEGACY cache** — all 203 LEGACY terms present with identical class membership. 21 extra FG-only terms add classes that no classifier checks (bit-identical routing).
- **`_is_weather_concept_question` uses hardcoded concept set** — structural patterns, not vocabulary lookup.
- **Deletion test works** — deleting a seeded lexeme from the store and creating a new kernel removes the term from the cache.
- **2 pre-existing test failures** (unrelated to this work): `test_assistant_authority_mvp.py` (`AnswerPlan` import), `test_cli_pi_bundle_builds_portable_self_checked_bundle` (missing `docs/local_assistant_os_mvp_plan.md`).

## Relevant Files
- **`melm/appliance/local_assistant_router.py`** — `_semantic_family_terms` (line ~1616): no params beyond `tokens`, `semantic_classes`. `_IN_MEMORY_LEXICON` (line ~1603): built from LEGACY at module load. `replace_in_memory_lexicon()`: replaces cache. `_is_weather_concept_question` (line ~1574): hardcoded concept set.
- **`melm/appliance/assistant_os_kernel.py`** — `_rebuild_router_lexicon_cache` (line ~1087): always rebuilds — store data when rows exist, LEGACY reset when empty.
- **`melm/appliance/assistant_os_store.py`** — `seed_assistant_os_lexicon` (line ~1906): seeds FG + router candidates (all 35 classes). No `ContractValidationError` catch.
- **`melm/appliance/assistant_lexicon_legacy.py`** — `LEGACY_ROUTER_TERM_CLASSES` (~200+ entries, 35 classes). `build_legacy_router_candidates(seed_all=True)` exports all entries.
- **`melm/appliance/assistant_lexicon.py`** — `lexicon_ingest` (all 35 LEGACY classes now valid), `configure_lexicon_router_families`, `acquire_definition` (user-teaching acquisition channel with regex-based definition frame detection, genus_walk class candidates, quarantine-by-default).
- **`melm/contracts/semantic_classes.v1.json`** — 89 class IDs (61 original + 28 LEGACY routing labels).
- **`scripts/local_assistant_os_cli.py`** — bootstrap calls `seed_assistant_os_lexicon(store)` after database init.
