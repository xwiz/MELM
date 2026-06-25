# Causal Reasoning V4 Design

**Date:** 2026-06-20
**Status:** Reviewed and narrowed before implementation planning
**Author:** AssistantOS design session
**V0.4 sequencing:** `docs/assistant_os_architecture_v0_4.md` owns the ordering:
F1 (NLG foundation) → F2 (curiosity hardening) → C1 (V4A hardening) → C2 (V4B
graph, gated behind F1 multi-atom NLG). This doc is the authoritative V4 design.
**Relationship to v3:** This refines the next design direction after
`docs/superpowers/plans/2026-06-20-causal-reasoning.md`. It does not invalidate
the v3 MVP path, which matches the current production code more closely.

## 1. Goal

Upgrade causal reasoning from the current bounded `detect_reasoning_task()` +
`solve()` shortcut into a UOL-grounded semantic capability without violating the
Assistant OS architecture:

- knowledge is contract/entity data, not inline code;
- reasoning remains behind the release-controlled `reasoning` capability family;
- synthesis keeps using the generic `_render_reasoning()` path;
- moral cognition keeps using `derive_moral_context()` and its existing
  contracts without depending on the causal graph.

The design is intentionally split into a near-term **V4A hardening milestone**
and a later **V4B graph milestone**. The original draft mixed both into one
implementation tranche, which created bloat and several code-alignment risks.

## 2. Current Tree Findings

These findings are reproducible on the current tree:

- `rg -n "def atomize_syntax_graph|content=\\(atom,\\)" melm/appliance/uol_atomizer.py`
  shows the production syntax-graph atomizer still emits one atom.
- `rg -n "_detect_causal|_causal_explanation|_causal_prediction" melm/appliance/reasoning`
  shows v3 causal reasoning already lives in `task_router.py` and `solvers.py`.
- `rg -n "causal" melm/contracts/semantic_classes.v1.json melm/contracts/frame_templates.v1.json`
  returns no causal spine classes or causal frame templates.
- `rg -n "_VALID_ROLES|_VALID_SUBROLES" melm/contracts/validation.py`
  shows `function_words.v1.json` cannot accept a new `causal_cue` role today.
- `rg -n "default_capability_manifest|_is_family_installed\\(\"reasoning\"\\)" melm/appliance/local_assistant_router.py`
  shows dispatch is gated by `default_capability_manifest.v1.json`, while
  `capability_manifest.v1.json` is the schema contract.

## 3. Scope

### 3.1 V4A in scope

- Add a `causal_cues.v1.json` contract for causal cue lemmas, relation type, and
  direction. Do not encode causal cue knowledge as inline sets in the router.
- Extend the causal detector to use `causal_cues.v1.json`, existing UOL roles,
  and existing lemma normalization. It may still run inside
  `detect_reasoning_task()` because reasoning tasks already outrank closed
  intents and are capability-gated.
- Harden `causal_effects.v1.json` metadata and validation so generated rules can
  carry provenance and review status without changing the runtime store schema.
- Keep the solver as a pure, deterministic contract consumer. If the logic grows
  past small helpers, extract `melm/appliance/reasoning/causal_engine.py` and
  have `solvers.py` call it.
- Add `causal_contrast` only when the detector can bind both sides from the
  utterance with observable tests, for example "Why did A happen but B did not?"
- Improve tests for precedence and false positives, especially assistant
  identity questions that currently risk being intercepted by causal prediction.

### 3.2 V4B deferred

These are valid future architecture goals but should not be bundled into the
next implementation plan:

- Add `causes` and `caused_by` fields to `AtomLinks`.
- Split causal clauses into multiple linked UOL atoms.
- Add `causal_link`, `causal_rule`, and `causal_cue` classes to
  `semantic_classes.v1.json`.
- Add a `causal_reasoning` frame template.
- Add runtime `causal_rule` entities and a contract/entity merge layer.
- Add background or on-demand cloud research jobs.
- Learn causal rules directly from `personal_experience` entities.

Each deferred item needs its own prerequisite checks and tests. The current
`SyntaxGraph` dependency builder is lightweight and does not emit `advcl`,
`ccomp`, or marker edges, so a plan that assumes those labels already exist is
not implementation-ready.

### 3.3 Out of scope

- No new closed-intent keyword branch in the router.
- No new intent-specific synthesis branch.
- No runtime process that writes directly into the curated contract without
  review.
- No LLM/cloud-generated fact treated as ground truth.
- No changes to `derive_moral_context()` or the `verb_states.v1.json` /
  `state_valences.v1.json` moral-cognition path.

## 4. Architecture Overview

### 4.1 V4A production-aligned flow

```text
utterance
  -> parse_bundle (tokens, lemmas, UOL act)
  -> _try_reasoning() gated by default_capability_manifest.v1.json
  -> detect_reasoning_task() consumes causal_cues.v1.json + UOL fallback
  -> solve() or causal_engine reads causal_effects.v1.json
  -> AssistantDecision(intent="reasoning:causal_*", reasoning_result=...)
  -> synthesize() -> _render_reasoning()
```

### 4.2 V4B target flow

```text
utterance
  -> UOL atomizer emits linked causal atoms
  -> causal graph resolver derives explanation/prediction/contrast task
  -> solver reads contract rules plus approved causal_rule entities
  -> AssistantDecision(reasoning_result=...)
  -> synthesize() -> _render_reasoning()
```

The V4B flow is the architectural target, not the next unchecked patch set.

## 5. Causal Cue Contract

Create `melm/contracts/causal_cues.v1.json` before adding new causal cue logic.
The contract should be language-agnostic and lemma-based:

```json
{
  "schema_id": "melm.causal_cues.v1",
  "entries": [
    {
      "lemma": "because",
      "language": "en",
      "relation": "explanation",
      "direction": "effect_to_cause",
      "pattern": "subordinator"
    },
    {
      "lemma": "if",
      "language": "en",
      "relation": "prediction",
      "direction": "cause_to_effect",
      "pattern": "condition"
    }
  ]
}
```

Validator requirements:

- `schema_id == "melm.causal_cues.v1"`;
- every entry has non-empty `lemma`, `language`, `relation`, `direction`, and
  `pattern`;
- `relation` is one of `explanation`, `prediction`, `result`, `contrast`;
- `direction` is one of `effect_to_cause`, `cause_to_effect`,
  `subject_to_object`;
- lemmas are lowercase and unique per language.

Do not add `role: "causal_cue"` to `function_words.v1.json` unless
`validate_function_words()` is deliberately extended first. The current
function-word contract is for universal grammar roles, not domain reasoning
semantics.

## 6. Detection and Routing

The next implementation should improve the existing reasoning detector rather
than add a separate router branch.

Required behavior:

- `detect_reasoning_task()` remains the entry point.
- It reads causal cue knowledge from `causal_cues.v1.json`.
- It uses existing lemmas from the parse bundle when available.
- It refuses to classify second-person identity/status probes as causal tasks.
- It returns structured task dictionaries:
  - `{"task": "causal_explanation", "effect": "..."}`
  - `{"task": "causal_prediction", "cause": "..."}`
  - `{"task": "causal_contrast", "effect": "...", "contrast_effect": "..."}`

The route remains `reasoning:<task>`. Do not add `causal_reasoning` as a new
closed intent. If a future frame template is introduced, it should feed the same
reasoning task shape and stay behind the `reasoning` capability family.

## 7. Solver and Knowledge Merge

### 7.1 V4A solver

The V4A solver reads only curated contracts:

- `causal_effects.v1.json` for cause -> effect rules;
- `causal_cues.v1.json` for cue semantics if needed by helper code.

It may return tentative language when confidence is below a configured
threshold, but it must not invent missing causes or effects.

### 7.2 V4B merge layer

Only after `causal_rule` entities exist should the solver merge contract and
runtime rules.

Merge rules for that later milestone:

- Curated contract rules override entity-store rules for the same normalized
  `(cause_lemma, effect_state, patient_type)` key.
- Runtime rules must carry `provenance`, `review_status`, `confidence`, and
  `scope`.
- `review_status != "approved"` rules may be surfaced as candidates, not as
  factual answers.
- Sensitive domains (health, legal, finance) require manual review before they
  can affect answers.

## 8. Entity Store: Deferred Design

When V4B is planned, add the `causal_rule` class through the existing entity
architecture:

- add `causal_rule` to `semantic_classes.v1.json`;
- seed `class_schemas` and `class_schema_slots` in `seed_class_schemas()`;
- store values in `entity_slots`, not in a separate causal table;
- use `entity_relations` for links to source `personal_experience` or
  `uol_parse` entities.

Suggested slots:

- `cause_lemma` (required text);
- `effect_state` (required text);
- `effect_domain` (text);
- `patient_types` (json);
- `confidence` (real);
- `provenance` (text: `manual_curated`, `offline_extractor`,
  `user_stated`, `cloud_candidate`);
- `review_status` (text: `pending`, `approved`, `rejected`);
- `scope` (text: `global`, `user_local`, `session_local`);
- `source_entity_id` (text, optional).

Do not teach the experience writer to emit causal rules directly. A separate
validated extractor may read T1/T2 records and create pending candidates.

## 9. Auto-Research Pipeline

The draft's "background job mutates the contract" direction is too risky for
the current architecture. Use this safer split:

### 9.1 Release-time enrichment

`scripts/extract_causal_effects.py` can continue to generate or refresh
`causal_effects.v1.json` from `nameless_extracted_verbs.json`, but the result is
a reviewed release artifact:

- contract changes go through validator + registry hash update;
- generated entries carry provenance/review metadata when the schema supports
  it;
- tests verify deterministic output from fixed inputs.

### 9.2 Runtime candidate collection

Runtime or cloud-assisted research may create candidate data only:

- write pending `causal_rule` entities after V4B entity support exists; or
- write a local candidate report for manual review before V4B.

Cloud/LLM output is a candidate generator, never ground truth. It cannot install
skills, flip capability flags, or write directly to curated contracts.

## 10. UOL Atom Graph: Deferred V4B Requirements

When implementing causal atom links, make these prerequisites explicit:

- Add `causes` and `caused_by` to `AtomLinks` and `UolAtom.to_dict()`.
- Update tests that serialize UOL atoms.
- Build clause splitting against the actual `SyntaxGraph` available in this
  repo. The current simple dependency builder does not emit `advcl`, `ccomp`,
  or `mark`.
- Start with `atomize_syntax_graph()` because it is the production path.
  `atomize()` can follow only if legacy flat parses still need parity.
- Use `causal_cues.v1.json` for cue detection.
- Prove link direction with examples:
  - "The ground is wet because it rained" -> rain causes wet.
  - "If it rains, the ground gets wet" -> rain causes wet.
  - "It rained, so the ground is wet" -> rain causes wet.

Frame-linker routing should not depend on causal links until these atomizer
tests are passing.

## 11. Testing and Anti-Regression

### 11.1 V4A tests

- Contract tests for `causal_cues.v1.json`.
- Detector tests for explanation, prediction, and false positives.
- Solver tests for known cause/effect, unknown cause/effect, and low-confidence
  wording.
- Kernel tests proving `reasoning:causal_*` still routes through `_try_reasoning()`.
- Synthesis tests proving reasoning results still use `_render_reasoning()`.
- Precedence tests for assistant identity/status questions.
- Deterministic extractor tests with temporary input files and fixed expected
  output.

### 11.2 V4B tests

- Atomizer tests for multi-atom causal clauses and link direction.
- Serialization tests for `causes` and `caused_by`.
- Semantic spine invariant tests for `causal_link`, `causal_rule`, and
  `causal_cue`.
- Frame-linker tests only after causal links exist.
- Entity merge tests only after `causal_rule` entities exist.

### 11.3 Required regression commands

Run the focused causal suite after V4A changes:

```powershell
python -m pytest tests/test_causal_effects_contract.py tests/test_causal_enrichment.py tests/test_causal_reasoning.py -v
```

Run architecture-sensitive suites before claiming completion:

```powershell
python -m pytest tests/test_meaning_invariant.py tests/test_reasoning_plumbing.py tests/test_moral_negation.py tests/test_implications_mvp.py -v
```

### 11.4 Architecture checklist

- No new class ID without a `semantic_classes.v1.json` entry.
- No new contract without validator, loader, registry entry, and SHA-256 hash.
- No inline causal cue or effect keyword sets in router/synthesis code.
- No new closed-intent keyword branch.
- No new skill/capability dispatch without `default_capability_manifest.v1.json`
  coverage.
- Runtime learning can extend candidate knowledge only; it cannot install a
  skill or write reviewed contract data.
- Synthesis remains generic; no per-intent causal answer branch.
- Dialog tests assert observable routing/answer behavior, not debug labels.
- Contracts store lemmas only; inflection normalization remains swappable.
- `derive_moral_context()` remains independent of the causal graph.

## 12. Phased Implementation Order

### Phase 0: Baseline verification

- Run the current causal, reasoning, moral, and meaning-invariant suites.
- Document any pre-existing failure before editing.

### Phase 1: Causal cue contract

- Add `causal_cues.v1.json`.
- Add validator, loader, exports, registry entry, and hash.
- Add contract tests.

### Phase 2: Detector hardening

- Replace inline causal cue checks in `task_router.py` with contract-backed
  helpers.
- Use parse-bundle lemmas/UOL fallback for cause/effect extraction.
- Add false-positive tests for assistant identity/status queries.

### Phase 3: Solver hardening

- Normalize cause/effect keys consistently.
- Add confidence/provenance handling that matches the current
  `causal_effects.v1.json` schema or the planned schema migration.
- Keep answer rendering in `_render_reasoning()`.

### Phase 4: Optional contrast task

- Add `causal_contrast` only with tests that bind both compared effects.
- Keep unsupported contrast as a typed refusal, not a guessed answer.

### Phase 5: Release-time enrichment

- Update `scripts/extract_causal_effects.py` only if the contract schema changes.
- Produce deterministic candidate/contract output from fixed fixture inputs.

### Phase 6: V4B planning gate

Before starting graph/entity work, write a new implementation plan that proves:

- causal cue classes exist in the semantic spine;
- the atomizer can build multi-atom causal clauses on the actual `SyntaxGraph`;
- frame-linker integration will feed the existing reasoning task shape;
- runtime candidate rules have review semantics and cannot become authoritative
  without approval.

## 13. Success Criteria

V4A is successful when:

- existing v3 causal tests still pass;
- causal cue knowledge is contract-backed;
- identity/status probes are not intercepted by causal reasoning;
- causal explanation and prediction still route through `reasoning:*`;
- synthesis still uses `_render_reasoning()`;
- extractor output is deterministic;
- no moral-cognition regression appears in focused tests.

V4B is successful only after a separate plan delivers linked UOL causal atoms and
approved runtime-rule merging without bypassing the architecture checklist.
