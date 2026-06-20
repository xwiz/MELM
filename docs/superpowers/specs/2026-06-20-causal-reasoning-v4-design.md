# Causal Reasoning V4 Design

**Date:** 2026-06-20
**Status:** Draft — awaiting implementation plan
**Author:** AssistantOS design session
**Supersedes:** `docs/superpowers/plans/2026-06-20-causal-reasoning.md` (v3 implementation)

## 1. Goal

Upgrade causal reasoning from a surface-keyword reasoning shortcut into a first-class UOL semantic layer, while adding an auto-research pipeline that continuously expands the causal knowledge base.

## 2. Scope

### 2.1 In scope

- Extend `UolAtom` and `AtomLinks` with explicit causal relations.
- Make the atomizer split causal clauses into multiple linked atoms.
- Add causal-cue semantic classes to the lexicon / function-word contracts.
- Introduce a `causal_reasoning` frame template for frame-linker routing.
- Extend the reasoning solver to traverse the causal graph and the merged knowledge base.
- Add `causal_contrast` as a new reasoning task.
- Build an auto-research pipeline that grows both the curated contract and the runtime entity store.
- Add background and on-demand auto-research triggers.
- Add validation gates, provenance tracking, and a review queue for learned rules.

### 2.2 Out of scope

- No new closed-intent keyword branches in the router.
- No intent-specific synthesis branches.
- No experience-writer causal learning without validation.
- No LLM-generated facts treated as ground truth.
- No new UOL atom link fields beyond `causes`/`caused_by`.
- No broad entity-store schema changes outside the `causal_rule` entity class.

## 3. Background and rationale

The v3 implementation intentionally avoided atomizer changes because the production `atomize_syntax_graph()` path emits a single atom. V3 used a keyword detector (`why`, `what happens if`) and a contract lookup (`causal_effects.v1.json`) to deliver a bounded MVP. V4 removes that shortcut by:

1. Teaching the atomizer to emit multiple atoms for causal clauses.
2. Adding explicit causal links between those atoms.
3. Teaching the frame linker and solver to consume those links.
4. Adding auto-research so the knowledge base grows from offline corpora, user conversations, and approved cloud candidates.

## 4. Architecture overview

```
Utterance
  ↓
UOL atomizer (multi-atom causal clauses + causal links)
  ↓
UolAct with content = [main_atom, causal_atom, ...]
  ↓
Frame linker: causal_reasoning template
  ↓
Reasoning task: causal_explanation | causal_prediction | causal_contrast
  ↓
Solver: contract + entity-store merge layer
  ↓
Answer (via generic _render_reasoning)
```

## 5. UOL atomizer changes

### 5.1 Extend `AtomLinks`

Add two new fields to `melm/appliance/uol_types.py::AtomLinks`:

```python
@dataclass(frozen=True)
class AtomLinks:
    subordinate_atoms: tuple[str, ...] = ()
    coreferent_atoms: tuple[str, ...] = ()
    entity_refs: tuple[str, ...] = ()
    temporal_anchor: str = ""
    causes: tuple[str, ...] = ()        # atom IDs this atom causes
    caused_by: tuple[str, ...] = ()     # atom IDs that cause this atom
```

### 5.2 Multi-atom causal clause splitting

Both `atomize()` and `atomize_syntax_graph()` must detect causal subordinates and emit separate atoms.

Detected patterns:

- `advcl` with markers: `because`, `if`, `since`, `when`, `so`, `therefore`.
- `ccomp` with causal verbs: `make`, `cause`, `lead to`, `result in`.
- Coordinate causal conjunctions: `so`, `therefore`, `thus`.

Link direction rules:

| Pattern | Example | Link direction |
|---|---|---|
| Effect because cause | "The ground is wet because it rained" | `effect.caused_by = (cause.id,)`; `cause.causes = (effect.id,)` |
| If cause then effect | "If it rains, the ground gets wet" | `effect.caused_by = (cause.id,)` |
| Cause makes effect | "Rain makes the ground wet" | `effect.caused_by = (cause.id,)` |
| Cause so effect | "It rained, so the ground is wet" | `effect.caused_by = (cause.id,)`; `cause.causes = (effect.id,)` |

### 5.3 Causal cue semantic classes

Add causal cue entries to the lexicon and function-word contracts:

- `because`, `since`, `as` → `causal_reason:effect_to_cause`
- `if` → `causal_condition:cause_to_effect`
- `so`, `therefore`, `thus` → `causal_result:cause_to_effect`
- `make`, `cause`, `lead to`, `result in` → `causal_verb:subject_to_object`

Each entry references an existing semantic class in `semantic_classes.v1.json`.

## 6. Causal frame template

Add a `causal_reasoning` frame template to the frame registry.

- **Trigger:** chat frame contains an atom with `causes` or `caused_by` links, or an interrogative targets a causal role.
- **Slots:**
  - `causal_effect`: the atom being explained.
  - `causal_cause`: the linked cause atom.
  - `causal_relation`: one of `explanation`, `prediction`, `contrast`.
- **Route:** `reasoning:<causal_relation>`.

This replaces the v3 keyword detector in `task_router.py`.

## 7. Reasoning solver updates

The solver consumes the merged causal knowledge base and the UOL atom graph.

### 7.1 Solver dispatch

- `causal_explanation`: user asks why an effect happened.
- `causal_prediction`: user asks what would happen given a cause.
- `causal_contrast`: user asks why A happened but B did not.

### 7.2 Resolution strategy

1. If the `UolAct` contains causal atom links, use the linked atoms to answer.
2. If the query is generic (e.g., "Why is the ground wet?"), look up the effect in the merged knowledge base.
3. If no rule exists, return `reasoning_result` with `auto_research_needed` and a helpful refusal message.

### 7.3 Merge layer

The solver reads from two sources:

| Source | Format | Contents | Trust |
|---|---|---|---|
| Curated contract | `causal_effects.v1.json` | Seed + auto-research enriched rules | High |
| Runtime store | `causal_rule` entities | Learned rules, user-specific rules | Medium |

Merge rules:

- Contract rules override entity-store rules for the same `cause_verb` + `effect_state` pair.
- Entity-store rules are tagged with `provenance` and `review_status`.
- Rules with confidence below the threshold are surfaced as tentative or rejected.

## 8. Entity-store schema

Add a new entity class `causal_rule` in `semantic_classes.v1.json` and seed its schema.

Required slots:

- `cause_verb`: lemma of the causing verb/action.
- `effect_state`: state that results.
- `effect_domain`: domain of the effect (physical, emotional, mental, social).
- `confidence`: float 0–1.
- `provenance`: `auto_research`, `user_stated`, `manual_curated`, `cloud_candidate`.
- `source_utterance_id`: link to the originating experience entity (optional).
- `review_status`: `pending`, `approved`, `rejected`.

## 9. Auto-research pipeline

### 9.1 Triggers

- **Background**: scheduled job scans offline corpora and user experience entities.
- **On-demand**: when a solver query fails and the user consents, run a research job.

### 9.2 Sources

1. **Offline corpora**: extend the `nameless_extracted_verbs.json` extractor; add additional curated text sources.
2. **User conversations**: extract causal statements from `personal_experience` entities.
3. **Cloud/LLM**: used only as a candidate generator, never as ground truth.

### 9.3 Pipeline stages

```
source_text
  ↓
extractor (pattern-based / dependency-based)
  ↓
candidate rule (cause_verb, effect_state, domain, confidence, source)
  ↓
validator (semantic class, contradiction, safety filter)
  ↓
review queue
  ↓
contract (if seed-worthy) or entity store (if user-specific)
```

### 9.4 Validation gates

- `cause_verb` must map to a semantic class in `semantic_classes.v1.json`.
- `effect_state` must exist in `verb_states.v1.json` or pass manual review.
- No contradictions with existing high-confidence rules.
- No sensitive-domain rules (health, legal, finance) unless explicitly allowed.

## 10. Testing and anti-regression

### 10.1 New tests

- Atomizer tests: verify causal clause splitting and link direction.
- Frame-linker tests: verify `causal_reasoning` template matches.
- Solver tests: verify explanation, prediction, and contrast tasks.
- Merge-layer tests: verify contract overrides entity store and provenance handling.
- Auto-research tests: verify extraction, validation, and review queue.
- Registry tests: verify `causal_rule` class exists in `semantic_classes.v1.json`.

### 10.2 Anti-regression checklist

- `derive_moral_context()` continues to consume `verb_states.v1.json` and `state_valences.v1.json`; no dependency on the causal graph.
- Existing v3 causal reasoning tests continue to pass or are explicitly upgraded.
- No new keyword-based closed intents are added.
- Synthesis remains generic; no per-intent branches.

## 11. Phased implementation order

1. **Phase 0**: Extend `AtomLinks` and add causal cue semantic classes.
2. **Phase 1**: Implement clause splitting in `atomize_syntax_graph()`.
3. **Phase 2**: Add the `causal_reasoning` frame template and update the frame linker.
4. **Phase 3**: Update `task_router.py` and `solvers.py` to use the causal graph.
5. **Phase 4**: Add the `causal_rule` entity class and merge layer.
6. **Phase 5**: Build the auto-research pipeline (extractor, validator, review queue).
7. **Phase 6**: Add background and on-demand triggers.
8. **Phase 7**: Full regression and anti-regression verification.

## 12. Success criteria

- "The ground is wet because it rained" produces two linked atoms.
- "Why is the ground wet?" routes via `reasoning:causal_explanation` and answers using the linked atoms.
- "What happens if it rains?" routes via `reasoning:causal_prediction` and answers from the merged knowledge base.
- Auto-research enriches `causal_effects.v1.json` by at least 10% from new sources without manual editing.
- All existing moral-cognition and v3 causal tests pass.
- No new closed intents or per-intent synthesis branches.
