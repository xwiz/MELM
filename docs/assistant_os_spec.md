# Local Assistant OS — Architecture Specification

Version: v0.2-target | Contract registry: `melm/contracts/`
Supersedes: architecture prose in `docs/local_assistant_os_mvp_plan_v2.md`,
            all prior architecture descriptions in `docs/archive/`

## Table of Contents

1. [Honest assessment: what exists today](#1-honest-assessment-what-exists-today)
2. [Target architecture: typed expert cascade](#2-target-architecture-typed-expert-cascade)
3. [The skill/knowledge boundary](#3-the-skillknowledge-boundary)
4. [Knowledge-first design (auto-research pattern)](#4-knowledge-first-design-auto-research-pattern)
5. [Contract registry](#5-contract-registry)
6. [Pipeline stages](#6-pipeline-stages)
7. [Data model](#7-data-model)
8. [Membrane policy](#8-membrane-policy)
9. [Capability manifest](#9-capability-manifest)
10. [Synthesis boundary](#10-synthesis-boundary)
11. [Foundational rules](#11-foundational-rules)
12. [Migration path](#12-migration-path)
13. [Anti-regression checklist](#13-anti-regression-checklist)

---

## 1. Honest assessment: what exists today

The current codebase (`melm/appliance/`) implements a **linear classification pipeline** that differs significantly from the typed expert cascade described below. This section documents what actually exists so architectural decisions are made against reality, not aspiration.

### 1.1 Current pipeline

```
utterance
  → normalize + tokenize
  → 17-classifier intent cascade (keyword matching + frame linker)
  → capability manifest gate
  → intent dispatch table (14 handlers)
  → handler-specific data resolution (profile, store, weather cache, etc.)
  → template answer generation (in handler)
  → synthesis layer re-generates the same templates
  → authority verification
```

### 1.2 What is actually happening

**14 parallel mini-applications in one router.** Each intent (story, weather, meal, health, safety, media, contact, personal_memory, autobiographical_memory, assistant_identity, assistant_status, greeting, behavior, open_domain) has its own:

| Component | Location | Per-intent pattern |
|---|---|---|
| Classifier | `local_assistant_router.py` | `_is_{intent}_request()` with inline keyword sets |
| Handler | `local_assistant_router.py` | `_{intent}()` with profile/store data access |
| Template | `assistant_synthesis.py:_answer()` | `if intent == X:` branch with hardcoded strings |
| Evidence | `assistant_synthesis.py:_resolve_evidence()` | `if key.startswith("X"):` branch |
| Quality | `assistant_synthesis.py:_answer_specificity()` | `if decision.intent == X:` branch |
| Domain data | Mixed into all of the above | Hardcoded dicts, strings, keyword sets |

### 1.3 Foundational flaws

**Flaw 1 — Knowledge is trapped in code, not referenceable as data.**

Every piece of domain knowledge lives in exactly one code path and cannot be queried by any other part of the system:

| Knowledge domain | Location (line) | Only used by |
|---|---|---|
| Food→tag mapping | `local_assistant_router.py:1077` (`_food_tags`) | `_meal()` handler |
| Urgent health triage texts | `assistant_synthesis.py:996` (`_urgent_health_answer`) | `health_advice` synthesizer |
| Safety policy texts | `assistant_synthesis.py:1028` (`_public_clothing_safety_answer`) | `common_sense_safety` synthesizer |
| Story image heuristics | `assistant_synthesis.py:705` (`_story_image`) | `story` synthesizer |
| Story challenge heuristics | `assistant_synthesis.py:727` (`_story_challenge`) | `story` synthesizer |
| Story lesson heuristics | `assistant_synthesis.py:740` (`_story_lesson`) | `story` synthesizer |
| Weather concept terms | `local_assistant_router.py:1619` (inline in `_is_weather_concept_question`) | `weather` classifier |
| Urgent medical terms | `local_assistant_router.py:4598` (inline in `_is_health_advice_request`) | `health_advice` classifier |
| Clothing policy text | `assistant_synthesis.py:489-490` (inline in `_resolve_evidence`) | `common_sense_safety` evidence |
| Meal scope tokens | `local_assistant_router.py:1029` (inline) | `meal` classifier |
| Identity/status fallback text | `assistant_synthesis.py:774-827` (inline) | identity + status synthesizers |
| Health disclaimer text | `assistant_synthesis.py:238-243` (inline) | `health_advice` synthesizer |
| Weather answer text | `assistant_synthesis.py:310-317` (inline) | `weather` synthesizer |

**Rule:** A piece of knowledge that exists in only one code path and cannot be referenced by ID is an architectural defect. It must be extracted into a contract or entity store with a schema, version, and validator.

**Flaw 2 — The router and synthesis layer duplicate each other.**

The router's `handle()` method produces `decision.answer` (a template string). Then synthesis's `_answer()` has a complete if/elif chain that re-generates the exact same templates. The router's answer is overwritten ~95% of the time.

```
Router handler:   _meal() → "You could eat rice."              → decision.answer
Synthesis:        _answer() → "You could eat rice..."           → result.answer
                                                                    ↑ overwrites
```

The two copies share no code. They *do* share bugs: if the meal template is fixed in one place but not the other, the fix disappears because synthesis always wins.

**Rule:** The synthesis layer must be a generic AnswerPlan renderer, not a per-intent template engine. Intent-specific answer logic lives in the skill's handler or its template contract, never in both places.

**Flaw 3 — The architecture wires intents before meaning.**

The current pipeline classifies intent *first* using keyword matching, then routes to a handler that knows what to do. The plan requires: parse the utterance into UOL (meaning representation), match frames against the meaning, then dispatch to the skill that handles that frame.

Current: `classify_intent → route → render`
Target: `UOL parse → frame selection → skill dispatch → evidence → render`

The consequence of the current ordering: every new intent requires a new classifier (+ keyword sets), a new handler (+ dispatch entry in two places), a new template (+ evidence keys + quality heuristics). Knowledge is per-intent rather than shareable. The system scales O(n) in complexity with each intent.

**Rule:** No new intents may be added to the keyword classification pipeline after M2. New capabilities go through the frame linker or cloud handoff. The intent→handler mapping is a transitional artifact.

### 1.4 What exists that is correct

The following pieces are architecturally sound and should be preserved:

- **Frame linker** (`assistant_frame_linker.py`): semantic-class-based frame matching with `required_all_classes` AND-gate. This is the correct pattern — it matches meaning, not keywords.
- **Lexicon** (`assistant_lexicon.py`): store-backed vocabulary with ingestion gate, promotion/rollback lifecycle, provenance tracking. This is the correct pattern — knowledge is data, not code.
- **Entity store** (`assistant_os_store.py`): unified entities/entity_slots/entity_relations with class schemas. This is the correct pattern — all things (persons, events, objects) share one store.
- **Authority module** (`assistant_authority.py`): AnswerPlan, verification, evidence packets. This is the correct pattern — synthesis is constrained and verifiable.
- **ConstrainedDecoder** (`assistant_decoder.py`): pluggable decoder registry with template fallback. This is the correct pattern — generation is bounded and fallible.
- **Capability manifest** (`contracts/default_capability_manifest.v1.json`): explicit installed capability control. This is the correct pattern — skills are release-gated, not discovery-gated.
- **Membrane policy** (in kernel): per-fact consent/scope cloud boundary. This is the correct pattern — privacy is structural, not advisory.

---

## 2. Target architecture: typed expert cascade

A deterministic kernel gates stage-specialized experts. The kernel owns policy,
membrane, routing, and fallbacks. Experts propose typed artifacts; they never
choose an action, disclose evidence, or declare output safe.

```
InputEnvelope
  -> UOLParse
  -> FrameCandidate[]
  -> RouteDecision
  -> EvidencePacket
  -> AnswerPlan
  -> rendered answer
  -> VerificationResult
```

Every arrow is a versioned contract. Every stage may abstain. A failed or
missing expert falls back to rules, templates, clarify, or cloud per the
already-typed route.

### 2.1 Performance envelope (target)

| Metric | Target |
|--------|--------|
| Steady-state generation | >30 tok/s |
| Integrated p95 first token | <1.5 s for local-answer routes |
| Peak process RSS | <1.2 GB |
| Model + lexical assets | <600 MB |

### 2.2 Expert roster and authority boundary

| Stage | Responsibility | Initial implementation | Authority / failure |
|---|---|---|---|
| **K0 typed kernel** | Contract validation, event ledger, memory digest, membrane, capability policy, routing, action confirmation, fallbacks | Existing Python + SQLite kernel | Sole route/action authority; invalid input fails closed |
| **E1 lexical retriever** | Sense candidates, similarity, OOV suggestions | Shared small ONNX encoder (optional) | Proposes candidates only; cannot promote a sense or route |
| **E2 UOL parser** | Tokens/spans → clauses, roles, morphology, unknowns | Rules first; small tagger by construction family | Emits UOLParse; shadow output never routes |
| **E3 frame linker** | Rank frame candidates from UOL + semantic support | Rules first; shared-encoder reranker on top-k | Emits scores only; K0 applies thresholds and policy |
| **E4 verbalizer** | Render an AnswerPlan into bounded language | Template baseline, then small GGUF decoder | Cannot see raw history or blocked evidence; verifier failure discards output |
| **E5 definition parser** | Definition → candidate lexical sense | E2 + deterministic genus/differentia rules | Writes quarantine only; neural specialization deferred |

---

## 3. The skill/knowledge boundary

### 3.1 Definitions

A **skill** is a behavioral capability implemented in the kernel or an expert.
Skills are registered in the capability manifest, have a frame family classifier,
an evidence resolver, and optional template/verifier overrides.

**Knowledge** is declarative data referenced by skills. Knowledge lives in
contract files (JSON), the lexicon store, or the entity store. Knowledge is
skill-agnostic: any skill may query any knowledge by ID.

| | Skill | Knowledge |
|---|---|---|
| **What** | Behavioral capability | Declarative facts |
| **Where** | Code + capability manifest | Contracts + lexicon + entities |
| **How added** | Release-controlled manifest | Runtime acquisition pipeline |
| **Versioned by** | Release | Contract version |
| **Referenced by** | N/A (the skill is the reference point) | ID: contract key, lemma, entity_id |
| **Example** | meal_suggestion classifier + evidence + template | food_tags.v1.json entries |

### 3.2 The boundary rule

**A skill module may reference knowledge contracts by key. It may never contain inline data that should be referenceable by another skill or learnable at runtime.**

Concretely:
- `_food_tags` must become `contracts/food_tags.v1.json` — food knowledge that any skill can query
- `_urgent_health_answer` texts must become `contracts/health_disclaimers.v1.json` — health knowledge that any skill can reference
- `_story_image`/`_story_challenge`/`_story_lesson` must become `contracts/story_components.v1.json` — story component heuristics that can be extended without code changes
- `_public_clothing_safety_answer` text must become `contracts/safety_policies.v1.json` — policy text that can be versioned and localized

### 3.3 What a skill module looks like

```
assistant_skill_meal.py
├── class MealSuggestionSkill:
│   ├── family = "meal_suggestion"
│   ├── frames = ["meal_suggestion"]           # which frame templates match
│   ├── evidence = ["food_inventory",           # what evidence to gather
│   │              "weekly_weather.*"]
│   ├── knowledge_refs = ["food_tags.v1"]      # contracts consumed
│   ├── template = "meal_answer.v1"            # optional template contract
│   ├── policy = {privacy: "local_only"}        # privacy/action policy
│   └── synthesize(plan, evidence) -> str      # optional override
```

The synthesizer is generic: it reads `template` from the contract registry,
fills slots from evidence, and falls back to `decision.answer` if no template
exists. A skill only provides a custom `synthesize()` if its answer logic
requires computation (scoring, ranking, filtering) beyond template filling.

---

## 4. Knowledge-first design (auto-research pattern)

### 4.1 The insight

Karpathy's auto-research teaches: **teach the system about a subject once, let
that knowledge serve all skills.** Knowledge is first-class — it has an ID, a
schema, a version, provenance, and a query interface. Skills are consumers that
reference knowledge by ID.

### 4.2 Radial vs linear architecture

**Current (linear):**
```
utterance → classify intent → route to skill → skill has inline knowledge
```

Each skill is a self-contained silo. Knowledge cannot cross skill boundaries.
Adding a new skill requires duplicating or re-extracting knowledge.

**Target (radial):**
```
                    ┌─────────────────┐
                    │  Knowledge Store │
                    │  (contracts +    │
                    │   lexicon +      │
                    │   entities)      │
                    └────────┬────────┘
                             │
              ┌──────────────┼──────────────┐
              │              │              │
        ┌─────▼─────┐  ┌────▼─────┐  ┌─────▼─────┐
        │ meal skill │  │health    │  │story skill│ ...
        │ queries    │  │skill     │  │queries    │
        │ food_tags  │  │queries   │  │story_     │
        │ inventory  │  │disclaimers│  │components │
        └───────────┘  └──────────┘  └───────────┘
```

Knowledge radiates from the center. Every skill queries the same store. Adding
knowledge benefits all skills. Adding a skill consumes existing knowledge.

### 4.3 What this changes

| Concern | Linear (current) | Radial (target) |
|---|---|---|
| Food knowledge | `_food_tags` in router code, only `_meal()` can read it | `food_tags.v1.json` in contracts, any skill can query |
| Health knowledge | Hardcoded strings in `_answer()` and `_urgent_health_answer()` | `health_disclaimers.v1.json`, any skill can reference by ID |
| Story components | `_story_image()`/`_story_challenge()`/`_story_lesson()` heuristic functions | `story_components.v1.json` with updatable keyword→text mappings |
| Policy texts | `_resolve_evidence()` inline strings, `_public_clothing_safety_answer()` | `safety_policies.v1.json`, versioned and localizable |
| New knowledge | Requires code change in exactly one skill | Add entry to contract JSON, all skills see it |
| New skill | Requires duplicating or re-extracting knowledge | Declares which knowledge contracts it consumes |
| Runtime learning | Cannot affect skill behavior (knowledge is in code) | Can promote learned terms that skills reference |

### 4.4 The acquisition pipeline

The lexicon acquisition pipeline (§21.6 of the v2 plan) is the mechanism by
which knowledge grows at runtime. All vocabulary enters through the same
`sense_candidate.v1` gate — batch seeders (WordNet, VerbNet, legacy migration)
and runtime channels (user teaching, offline dictionary, cloud lookup).

A learned word's semantic class determines which skills can use it:
- `food_item` → meal skill can match and rank it
- `health_domain` → health skill can recognize it
- `physical_object.instrument` → story/media skills can use it

The key constraint from §20.3: **learned semantics must never mutate policy,
action types, handlers, or capability availability.** Skills are release-gated;
knowledge is runtime-grown. The two are orthogonal.

---

## 5. Contract registry

All cross-stage contracts live in `melm/contracts/`:

| Contract | File | Purpose |
|----------|------|---------|
| Registry v1 | `registry.v1.json` | Contract version registry |
| Semantic classes v1 | `semantic_classes.v1.json` | Semantic class taxonomy (89 classes) |
| Router lexicon families v1 | `router_lexicon_families.v1.json` | Lexicon family definitions |
| VerbNet map v1 | `verbnet_map.v1.json` | Verb class mapping |
| WN supersense map v1 | `wn_supersense_map.v1.json` | WordNet supersense mapping |
| Reserved lexemes v1 | `reserved_lexemes.v1.json` | Reserved namespace lexemes |
| Sense candidate v1 | `sense_candidate.v1.json` | Acquisition pipeline contract |
| Frame templates v1 | `frame_templates.v1.json` | Frame template definitions |
| Capability manifest v1 | `default_capability_manifest.v1.json` | Installed capability manifest |
| Food tags v1 | *(planned)* `food_tags.v1.json` | Food→tag mapping (extracted from `_food_tags`) |
| Health disclaimers v1 | *(planned)* `health_disclaimers.v1.json` | Health disclaimer texts (extracted from `_urgent_health_answer`) |
| Safety policies v1 | *(planned)* `safety_policies.v1.json` | Safety policy texts (extracted from `_public_clothing_safety_answer`) |
| Story components v1 | *(planned)* `story_components.v1.json` | Story image/challenge/lesson heuristics |
| Validation | `validation.py` | Contract validation helpers |

---

## 6. Pipeline stages

### 6.1 K0 — Typed kernel

Sole route/action authority. Invalid input fails closed.

Responsibilities:
- Contract validation for all pipeline stages
- Membrane policy enforcement (per-fact consent/scope tags)
- Capability manifest checks
- Routing with thresholded frame candidates
- Action confirmation (dry-run default, typed confirmed actions)
- Fallback routing (clarify / cloud_handoff / template)

### 6.2 E1 — Lexical retriever

| Property | Value |
|----------|-------|
| Responsibility | Sense candidates, similarity, OOV suggestions |
| Implementation | Shared small ONNX encoder (optional) |
| Authority | Proposes candidates only; cannot promote a sense or route |
| Failure | Omitted entirely; kernel continues with rules |

### 6.3 E2 — UOL parser

| Property | Value |
|----------|-------|
| Responsibility | Tokens/spans → clauses, roles, morphology, unknowns |
| Implementation | Rules first; small tagger by construction family |
| Authority | Emits `UOLParse`; shadow output never routes |
| Failure | Falls back to basic NLP, kernel owns route |

### 6.4 E3 — Frame linker

| Property | Value |
|----------|-------|
| Responsibility | Rank frame candidates from UOL + semantic support |
| Implementation | Rules first; shared-encoder reranker on top-k |
| Authority | Emits scores only; K0 applies thresholds and policy |
| Failure | Falls back to rule-generated candidates |

### 6.5 E4 — Verbalizer

| Property | Value |
|----------|-------|
| Responsibility | Render an `AnswerPlan` into bounded language |
| Implementation | Template baseline, then small GGUF decoder |
| Authority | Cannot see raw history or blocked evidence; verifier failure discards output |
| Failure | Template fallback when model absent or rejected |

### 6.6 E5 — Definition parser

| Property | Value |
|----------|-------|
| Responsibility | Definition → candidate lexical sense |
| Implementation | E2 + deterministic genus/differentia rules |
| Authority | Writes quarantine only; neural specialization deferred |
| Failure | Falls back to offline dictionary or cloud lookup |

---

## 7. Data model

### 7.1 Core tables

The SQLite schema has the following table groups:

**Lexicon** (words and meanings):
- `lexemes`, `word_forms`, `lexical_senses`, `semantic_classes`
- `lexicon_entries`, `lexicon_families`, `promotions`

**Entity** (unified person/event/place/object store):
- `class_schemas`, `class_schema_slots`
- `entities`, `entity_slots`, `entity_relations`

**Event ledger** (conversation history):
- `events`, `synthesis_traces`, `response_integrity`
- `membrane_decisions`, `homeostatic_snapshots`

**Learning** (acquisition and self-improvement):
- `session_improvement_consent`, `improvement_candidates`
- `opportunities`, `inventories`, `pending_actions`, `jobs`

**Legacy** (being migrated to entity store):
- `user_facts` → entities WHERE kind='self'
- `inventories` (contacts) → entities WHERE kind='person'

### 7.2 Entity architecture

All things (persons, events, places, objects) share a unified `entities` table
with `kind` discriminator. Slot values live in `entity_slots`. Relations between
entities live in `entity_relations`. Class hierarchy defines valid slots via
`class_schemas` + `class_schema_slots`.

Entity kinds:
- `person` — a known person (contacts, family, friends, user-self)
- `event_type` — a class of recurring events (competition, holiday)
- `event_instance` — a specific occurrence (World Cup 2026)
- `place` — a known location
- `object` — a physical or digital object (device, appliance, document)
- `personal_experience` — a chat session or past interaction
- `self` — the user (for facts about the user)

Frame slot states:
- `filled` — value is known and populated
- `asked_but_empty` — system asked but user didn't know
- `unknown_entity` — entity exists but this slot is unknown
- `unknown` — neither entity nor slot resolved
- `inferred` — value derived from context, needs confirmation

---

## 8. Membrane policy

- Every fact carries consent scope and provenance
- Cloud payloads are filtered by membrane policy before crossing boundary
- Actions require typed confirmation (dry-run default)
- Tombstones prevent re-admission of revoked facts

---

## 9. Capability manifest

- Installed capability is owned by release-controlled manifest
- Frame acceptance may consult promoted semantic/experience support
- An atlas write cannot enable a handler or action
- Learned vocabulary cannot grant a skill
- The manifest is checked at route-dispatch time, not at module-load time

---

## 10. Synthesis boundary

- `AnswerPlan` + admitted evidence + stance → answer
- Refusal, stale evidence, missing constraints, and verifier-failure cases are first-class outputs
- Template fallback is the permanent floor
- `AnswerPlan` limits what claims are authoritative; verifier discards unsupported output
- The synthesis layer is a **generic AnswerPlan renderer**, not a per-intent template engine

---

## 11. Foundational rules

### 11.1 Knowledge is data, not code

**Rule:** Any domain-specific string, mapping, keyword set, or heuristic that
could be referenced by multiple skills or extended at runtime must be extracted
into a contract JSON or the entity store.

**Violations to fix (with priority):**

| Priority | Artifact | Current location | Target |
|---|---|---|---|
| P0 | `_food_tags` | `local_assistant_router.py:1077` | `contracts/food_tags.v1.json` |
| P1 | Health disclaimer texts | `assistant_synthesis.py:996-1025` | `contracts/health_disclaimers.v1.json` |
| P1 | Safety policy texts | `assistant_synthesis.py:1028-1040` | `contracts/safety_policies.v1.json` |
| P2 | Story image/challenge/lesson heuristics | `assistant_synthesis.py:705-749` | `contracts/story_components.v1.json` |
| P2 | Weather concept terms | `local_assistant_router.py:1619-1620` | `contracts/weather_concepts.v1.json` |
| P2 | Urgent medical terms | `local_assistant_router.py:4598` (inline) | `contracts/medical_urgency.v1.json` |
| P3 | Meal scope tokens | `local_assistant_router.py:1029` | `contracts/meal_scopes.v1.json` |
| P3 | Identity/status fallback text | `assistant_synthesis.py:774-827` | `contracts/assistant_identity.v1.json` |

**Violation detection:** A grep for `\"[A-Z][^\"]{10,}\"` inside the `_answer()`
method or classifier functions that isn't matched by a contract schema is a
candidate violation. CI should flag new hardcoded answer strings in synthesis
code.

### 11.2 Synthesis is generic, not per-intent

**Rule:** The `_answer()` method in `assistant_synthesis.py` must not contain
intent-specific if/elif branches. Each branch is a skill-specific template that
belongs in the contract registry, not in Python code.

**Violations to fix:**

The entire `_answer()` method (lines 217-318 of assistant_synthesis.py) with
its 15 intent-specific branches must be replaced with a generic renderer that:
1. Reads the skill's template from the contract registry
2. Fills template slots from evidence
3. Falls back to `decision.answer` (from the router handler) if no contract template exists

**Violation detection:** Any `if decision.intent ==` in synthesis code is a
violation. The only exception is the `_decode_verified` path which is
intent-agnostic by design.

### 11.3 No new intents in the keyword pipeline

**Rule:** After M2, no new intents may be added to the keyword classification
cascade (`_classify_intent_from_uol_slots` in `local_assistant_router.py`).
New capabilities must go through:
1. The frame linker (E3) with a new frame template
2. Cloud handoff (for capabilities not yet installed locally)

**Rationale:** Every new intent in the keyword pipeline:
- Requires a classifier function + keyword sets (perpetuates Flaw 3)
- Requires a handler + dispatch entry (perpetuates Flaw 2)
- Requires a template branch in synthesis (perpetuates Flaw 2)
- Requires evidence resolution entries (perpetuates Flaw 1)
- Hardens the wrong architecture

### 11.4 Skills are release-controlled

**Rule:** A skill must appear in the capability manifest before it can be
dispatched. Runtime learning can extend knowledge that skills reference, but
cannot install a new skill.

**Enforcement:** The `_is_family_installed()` gate in `_route_impl` (line 285)
checks the manifest before dispatching any intent. An intent whose family is
not installed must be re-routed to `open_domain` → cloud handoff.

### 11.5 Contracts before code

**Rule:** Before writing any new domain logic, define the contract schema
first. A contract is:
1. A JSON Schema file in `melm/contracts/`
2. A validator function in `melm/contracts/validation.py`
3. An entry in `melm/contracts/registry.v1.json`

The contract defines the interface between knowledge (data) and skill (code).
Code reads contracts; it does not duplicate them.

---

## 12. Migration path

### Phase 1 — Freeze (immediate)

1. No new intent branches in `_answer()` — new answer types use a
   contract-referencing generic path
2. No new inline data in router or synthesis — all new domain knowledge goes
   into contract JSONs immediately
3. No new classifier functions — new capabilities use the frame linker or
   cloud handoff

### Phase 2 — Extract P0 knowledge (days)

Extract the highest-priority inline knowledge into contracts:

| Contract | Source | Impact |
|---|---|---|
| `food_tags.v1.json` | `_food_tags` dict at router:1077 | Meal skill uses contract lookup instead of module constant |
| `health_disclaimers.v1.json` | `_urgent_health_answer()` at synthesis:996 | Health skill uses contract text instead of inline function |
| `safety_policies.v1.json` | `_public_clothing_safety_answer()` at synthesis:1028 | Safety skill uses contract text instead of inline function |

Each extraction: create contract JSON → update skill to read from contract →
remove inline source → verify zero behavioral change.

### Phase 3 — Collapse router→synthesis duplicate (days)

Make synthesis generic:
1. Move each intent's template to a contract file (e.g., `meal_answer.v1.json`)
2. Replace `_answer()` with a loop over `decision.evidence_keys → contract lookup → template fill`
3. Remove the if/elif chain
4. Verify zero behavioral change via existing tests

### Phase 4 — Build skill module pattern (weeks)

Formalize the skill interface and migrate one skill as the template:

1. Define `Skill` protocol (family, frames, evidence, knowledge_refs, template)
2. Extract `meal_suggestion` into `assistant_skill_meal.py` with all three
   knowledge→contract extractions from Phase 2
3. Register skill in capability manifest
4. Verify 100% route agreement

---

## 14. Meaning representation architecture

### 14.1 The three-timescale model

Meaning operates at three different granularities. One layer cannot serve all three.

```
T1: Utterance meaning (per-turn)
    ┌─────────────────────────────────────────────┐
    │ token roles, polarity, speech act, relations │
    │ produced by: FunctionalParse (UOL)           │
    │ lives in: token_roles (per-parse)            │
    └─────────────────────────────────────────────┘

T2: Conversation meaning (per-session)
    ┌─────────────────────────────────────────────┐
    │ outcome, polarity, intent_achieved,          │
    │ learned_fact_ids                             │
    │ produced by: entity store                    │
    │ lives in: entities WHERE kind='personal_experience'│
    └─────────────────────────────────────────────┘

T3: Historical meaning (cross-session)
    ┌─────────────────────────────────────────────┐
    │ acquired lexemes, created entities,           │
    │ changed entity_relations                      │
    │ produced by: lexicon pipeline + entity store  │
    │ lives in: lexical_senses + entities tables    │
    └─────────────────────────────────────────────┘
```

**Rule:** T1 feeds T2 feeds T3. Each level aggregates from the one below. UOL
handles T1. The entity store handles T2 (personal_experience rows) and T3
(entities + lexical_senses). The bridge between them is the **semantic class
taxonomy** — all three layers reference the same class IDs from
`semantic_classes.v1.json`.

### 14.2 The spine: semantic class taxonomy

`semantic_classes.v1.json` is the **single source of truth** for all class IDs
used anywhere in the system. No layer may invent its own class IDs.

**Invariant:** Every class ID referenced by any of these layers must exist in
`semantic_classes.v1.json`:
- Frame template activation sets (required_classes, optional_classes,
  exclude_classes, required_all_classes, action_tokens)
- Entity store class schemas (class_schemas.semantic_class_id)
- UOL verb/noun assignments (transitional: the inline `_VERBS` and
  `_KNOWN_NOMINAL_DOMAINS` dicts; permanent: the lexicon `lexical_senses`
  table)
- Contract entries that reference class IDs
- Router semantic aliases that map tokens to class-based intent families

**Enforcement:** `test_semantic_class_spine_invariant` in the cross-layer test
suite reads every class ID from every layer and asserts all appear in the spine.

```
Spine (semantic_classes.v1.json)
  ├── frame_templates.v1.json activation sets
  ├── class_schemas table (entity store seeder)
  ├── UOL _VERBS dict (transitional) / lexical_senses table (permanent)
  └── contracts that reference class IDs
```

### 14.3 personal_experience entity schema

Every conversation session becomes an entity of kind `personal_experience`. Its
slots capture the meaning of the interaction:

| Slot | Type | Required | Description |
|------|------|----------|-------------|
| outcome | text | yes | Resolution: resolved / unresolved / escalated / abandoned |
| polarity | real | no | Aggregate sentiment: -1.0 to +1.0 |
| learned_fact_ids | json | no | Entity IDs of facts created during this experience |
| follow_up | text | no | Follow-up needed: check_tomorrow / monitor / null |
| intent_achieved | text | no | Whether primary intent was achieved: yes / partial / no |

These slots are defined in the entity store seeder (`seed_class_schemas`) and
read from the `class_schema_slots` table at runtime. Adding a new slot requires
only a database migration — no code change.

### 14.4 The extensibility contract

| You want to... | How | Code change? |
|---|---|---|
| Add a new verb/noun class | Add to `semantic_classes.v1.json` + add to `wn_supersense_map.v1.json` or `verbnet_map.v1.json` → ingested into `lexical_senses` table → UOL reads from lexicon | Zero |
| Add a new frame template | Add to `frame_templates.v1.json` using existing class IDs | Zero (CI validates) |
| Add a new personal_experience slot | Add to `class_schema_slots` table via seeder or migration | Zero |
| Add a new sentiment dimension | Add `polarity` property to `verb.emotion` class in taxonomy | Zero |
| Add a new domain contract | Add JSON to `contracts/` + register in `registry.v1.json` | Zero |
| Add a skill | Define as module consuming contracts + frame templates | One-time per skill |

**The rule:** If you need a code change to introduce new meaning to the system,
the extensibility contract is broken and must be re-examined.

### 14.5 Why past regressions happened

Every past regression followed the same pattern: a new layer invented its own
class IDs that didn't exist in the spine.

| Regression | Layer | Inline class | Spine says |
|---|---|---|---|
| UOL `_VERBS` | functional_grammar.py | "preference", "guidance", "function" | no such classes |
| Frame templates | frame_templates.v1.json | "verb.emotion", "verb.stative" | exist ✅ (was coincidental) |
| personal_experience slots | seed_class_schemas() | (none defined) | zero slots ⚠️ |

No cross-layer invariant existed to catch these. The spine invariant test
(`test_semantic_class_spine_invariant`) now prevents every class of this bug.

### 14.6 Transition: UOL from inline dicts to lexicon reads

The inline `_VERBS` dict (34 verbs), `_KNOWN_NOMINAL_DOMAINS` dict (9 nouns),
and closed-class sets in `functional_grammar.py` are **transitional**. They
exist because the lexicon pipeline was built after UOL. The permanent
architecture is:

1. UOL reads verb→class mappings from the `lexical_senses` table at startup
   (cached in memory for performance)
2. UOL reads noun→domain mappings from the `lexical_senses` table (same
   mechanism)
3. The closed-class sets (greetings, wh-words, modals, etc.) remain as
   performance shortcuts because they are stable closed classes, not domain
   knowledge

**Migration condition:** When the `lexical_senses` table contains entries for
all 34 verbs currently in the inline dict, the inline dict can be removed.

---

## 15. Anti-regression checklist

Use this checklist before merging any change to the kernel, router, or synthesis:

- [ ] Does this change add a new intent to the keyword classification cascade?
      If yes: blocked. Use frame linker or cloud handoff instead.
- [ ] Does this change add a new branch to `_answer()` in synthesis?
      If yes: blocked. Use contract-referencing generic path instead.
- [ ] Does this change add inline domain knowledge (strings, mappings, keyword
      sets) to router or synthesis code? If yes: blocked. Add to a contract instead.
- [ ] Does this change modify knowledge that exists in both the router handler
      and the synthesis `_answer()`? If yes: blocked. They must not duplicate.
- [ ] Does this change add a new contract? If yes: it must have a schema,
      validator, and registry entry before the code references it.
- [ ] Does this change add behavior that could be achieved by extending an
      existing knowledge contract? If yes: prefer contract extension over new code.
- [ ] Does this change reference a semantic class ID that does not exist in
      `semantic_classes.v1.json`? If yes: blocked. Add the class to the spine first.
- [ ] Does the existing test suite pass with the same route decisions, evidence
      keys, and answer texts? If no: regression, must be corrected.
