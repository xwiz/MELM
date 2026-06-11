# Local Assistant OS — Architecture Specification

Version: v0.2-target | Contract registry: `melm/contracts/`
Supersedes: architecture prose in `docs/local_assistant_os_mvp_plan_v2.md`

## 1. Architecture: typed expert cascade

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

### 1.1 Performance envelope (target)

| Metric | Target |
|--------|--------|
| Steady-state generation | >30 tok/s |
| Integrated p95 first token | <1.5 s for local-answer routes |
| Peak process RSS | <1.2 GB |
| Model + lexical assets | <600 MB |

## 2. Contract registry

All cross-stage contracts live in `melm/contracts/`:

| Contract | File | Purpose |
|----------|------|---------|
| Registry v1 | `registry.v1.json` | Contract version registry |
| Semantic classes v1 | `semantic_classes.v1.json` | Semantic class taxonomy |
| Router lexicon families v1 | `router_lexicon_families.v1.json` | Lexicon family definitions |
| VerbNet map v1 | `verbnet_map.v1.json` | Verb class mapping |
| WN supersense map v1 | `wn_supersense_map.v1.json` | WordNet supersense mapping |
| Reserved lexemes v1 | `reserved_lexemes.v1.json` | Reserved namespace lexemes |
| Sense candidate v1 | `sense_candidate.v1.json` | Acquisition pipeline contract |
| Validation | `validation.py` | Contract validation helpers |

## 3. Kernel (K0)

Sole route/action authority. Invalid input fails closed.

Responsibilities:
- Contract validation for all pipeline stages
- Membrane policy enforcement (per-fact consent/scope tags)
- Capability manifest checks
- Routing with thresholded frame candidates
- Action confirmation (dry-run default, typed confirmed actions)
- Fallback routing (clarify / cloud_handoff / template)

## 4. Pipeline stages

### 4.1 E1 — Lexical retriever

| Property | Value |
|----------|-------|
| Responsibility | Sense candidates, similarity, OOV suggestions |
| Implementation | Shared small ONNX encoder (optional) |
| Authority | Proposes candidates only; cannot promote a sense or route |
| Failure | Omitted entirely; kernel continues with rules |

### 4.2 E2 — UOL parser

| Property | Value |
|----------|-------|
| Responsibility | Tokens/spans → clauses, roles, morphology, unknowns |
| Implementation | Rules first; small tagger by construction family |
| Authority | Emits `UOLParse`; shadow output never routes |
| Failure | Falls back to basic NLP, kernel owns route |

### 4.3 E3 — Frame linker

| Property | Value |
|----------|-------|
| Responsibility | Rank frame candidates from UOL + semantic support |
| Implementation | Rules first; shared-encoder reranker on top-k |
| Authority | Emits scores only; K0 applies thresholds and policy |
| Failure | Falls back to rule-generated candidates |

### 4.4 E4 — Verbalizer

| Property | Value |
|----------|-------|
| Responsibility | Render an `AnswerPlan` into bounded language |
| Implementation | Template baseline, then small GGUF decoder |
| Authority | Cannot see raw history or blocked evidence; verifier failure discards output |
| Failure | Template fallback when model absent or rejected |

### 4.5 E5 — Definition parser

| Property | Value |
|----------|-------|
| Responsibility | Definition → candidate lexical sense |
| Implementation | E2 + deterministic genus/differentia rules |
| Authority | Writes quarantine only; neural specialization deferred |
| Failure | Falls back to offline dictionary or cloud lookup |

## 5. Data model

14 tables in SQLite: metadata, user_facts, self_state, events,
membrane_decisions, homeostatic_snapshots, synthesis_traces,
response_integrity, session_improvement_consent, improvement_candidates,
opportunities, inventories, pending_actions, jobs.

## 6. Membrane policy

- Every fact carries consent scope and provenance
- Cloud payloads are filtered by membrane policy before crossing boundary
- Actions require typed confirmation (dry-run default)
- Tombstones prevent re-admission of revoked facts

## 7. Capability manifest

- Installed capability is owned by release-controlled manifest
- Frame acceptance may consult promoted semantic/experience support
- An atlas write cannot enable a handler or action
- Learned vocabulary cannot grant a skill

## 8. Synthesis boundary

- `AnswerPlan` + admitted evidence + stance → answer
- Refusal, stale evidence, missing constraints, and verifier-failure cases
  are first-class outputs
- Template fallback is the permanent floor
- `AnswerPlan` limits what claims are authoritative; verifier
  discards unsupported output
