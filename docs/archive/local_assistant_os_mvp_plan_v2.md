# Local Assistant OS MVP Plan v2

> **DEPRECATED.** This document is superseded by
> `docs/superpowers/plans/2026-06-19-mvp3-implementation.md` and
> `docs/assistant_os_architecture.md`. It is retained as historical v0.2 execution
> context. Do not use it as the active plan for current work.
>

Date: 2026-06-10

Status: authoritative execution plan for the Local Assistant OS v0.2 path.
`docs/local_assistant_os_mvp_plan.md` (v1) is retained as historical v0.1
implementation context only.

v1 grew to 2,676 lines of mixed specification, evidence claims, and build
changelog, and its headline status claims became contradicted by the code it
described. This plan separates those concerns, records what was independently
re-verified on 2026-06-10, and defines the path to a v0.2 that is honest,
externally checkable, and actually useful.

Rule carried forward from v1 and made stricter: **a claim in this document is
either reproduced by a command on the current tree, or it is labeled a goal.**
This document never stores pass counts or rates as prose; it names the command
that produces them.

---

## 1. Verified reality baseline (2026-06-10)

Independent re-run of the v1 evidence on the current working tree.

### 1.1 What reproduces

| Claim | Command | Result |
|---|---|---|
| 105-case eval passes | `eval --json` | Reproduced (105 passed) |
| 17-step lifecycle, 0.706 local resolution | `run-lifecycle --reset --json` | Reproduced |
| Capability probe | `capability-probe --json` | Reproduced — but it is **18** cases, not the "cases=50" still claimed at v1 line ~449 |
| Transcript replay local resolution 17/25 (0.68) | `run-transcript-replay --reset --json` | Rate reproduced |
| Membrane / privacy / confirmation discipline | unit tests | 118 of 120 assistant-OS tests pass |
| Inventory learning loop (story cloud→local after import) | lifecycle + soak gates | Reproduced |

### 1.2 What is broken right now (v1 claims green)

| v1 claim | Actual |
|---|---|
| "Reference development runs pass all 22/22 checks" (`pi-smoke`) | **20/22.** `open_trace_debug_gate_passed` and `transcript_replay_gate_passed` fail. |
| `run-transcript-replay` gate passes | **Fails**: check `primary_uol_chatframe_not_secondary_phrase_route`. |
| `run-open-traces` gate passes | **Fails**: same check. |
| `target-report`, `pi-bundle`, `v01-acceptance` green | All include the two failing gates → red by cascade. |

Root cause (verified): the functional-grammar path emits a fourth primary
evidence label, `"source": "weighted_functional_relation"`
(`local_assistant_router.py:1822`). Router unit tests assert it and
`assistant_integrity.py:185` scores it 0.92 — it is an *intended* primary
source. But the gate whitelist in
`assistant_open_traces.py:1290-1311` still only accepts
`{token_role_relation, slot_role_relation, no_local_composition}`. One
subsystem's vocabulary moved; the verifier's did not. Failing turns:
"explain relativity like a scientist" (replay) and "Tell me the latest news
about Mars" (open traces).

Two lessons, both structural:

1. **No CI.** The contradiction sat undetected; the authoritative doc kept
   claiming green.
2. **Gates verify labels, not behavior.** The routes on the failing turns are
   *correct* (cloud_handoff / clarify); only the self-reported string changed.
   A gate that breaks when behavior is right — and would pass if the code
   emitted the old string while behaving differently — is attestation, not
   verification.

### 1.3 What exists in code

- ~18,200 lines, stdlib-only Python + SQLite. 14 tables: metadata, user_facts,
  self_state, events, membrane_decisions, homeostatic_snapshots,
  synthesis_traces, response_integrity, session_improvement_consent,
  improvement_candidates, opportunities, inventories, pending_actions, jobs.
- Real and tested: membrane policy, homeostatic snapshots, typed confirmed
  actions (dry-run default), consent/tombstones, event ledger with session
  links, memory digests, opportunity→job→inventory loop, Gutenberg/IA
  importers with retry/backoff/quality floors, weather cache, localhost
  API/browser UI, portable bundle with hash manifest, 62 CLI commands,
  ~232 tests.

### 1.4 What does NOT exist despite v1 prose

| v1 architecture element | Reality |
|---|---|
| World atlas (relation edges, strengths, provenance, negative edges) | **No module, no tables.** Only `ChildWorldAtlas`, a deterministic stub in the micro-MVP. v1's own evidence map admits "fragments rather than a first-class module", but its Goal/Evidence sections read as if frames already cite atlas support. They cannot cite what is not stored. |
| Learning ledger (quarantine, corrections, contradictions, promotion, rollback) | No tables. `improvement_candidates` + consent rows are the only fragments. |
| Grounding/research adapter | Prose only. |
| Any learned component | **None anywhere.** `functional_grammar.py` "weights" are hand-set constants (0.58–1.0). No training, no embeddings, no model. The MELM thesis — *meaning learned from use and observation* — is not implemented in any code path. |
| Guided SLM / stance rendering | Prose only. Synthesis is f-string templates over evidence fields. |

### 1.5 Honest characterization of routing

v1 repeatedly claims primary routes come from "token-role analysis plus
UOL/ChatFrame composition, never phrase tables." Verified reality: the intent
classifiers that *select* the frame are ~14 functions over ~60–70 hardcoded
token-set comparisons containing ~200+ keyword literals across 11 of them;
11 families now use `_semantic_family_terms` with lexicon-backed semantic class
lookup — all contracted, seeded, and activated by default in
`seed_assistant_os_lexicon`. The UOL composition is still built *after* keyword
intent selection; the 2-gate+1-membrane architecture (K0→Membrane→Ejector)
remains unchanged. What v1 actually enforces is narrower than what it says: the
*secondary hints table* may not route, and matching is token-bounded rather
than substring. That is a real discipline worth keeping — but a hand-authored
token-set is still a vocabulary table wearing a role costume. The shortcut-audit
polices a definition of "shortcut" that the primary path satisfies by
construction.

This is acceptable for a v0.x deterministic kernel. It is not acceptable to
describe it as something else.

### 1.6 Evaluation validity

- Every fixture (105-case eval, 25-turn replay, 29-turn traces, lifecycle
  scripts) is authored in-repo by the same author as the router.
- Baselines are deliberately crippled by construction ("locked to secondary
  lexical hints", "must not borrow UOL/ChatFrame"). The headline "+0.40
  local-resolution gain over best static baseline" measures the system against
  a weakened sibling on the system's own fixture. The numbers reproduce; their
  external meaning is near zero.
- Quality metrics miss the most visible failure: asked for "a story about a
  dragon and a robot," the assistant returns the same canned "Moon Drum Walk"
  template and silently drops the requested topic. By the project's own
  standards this is a wrong-local-answer class, and `wrong_local_answers=0` in
  every report because no fixture tests constraint adherence.
- v1 does honestly list user-derived evidence as a blocker and keeps
  `architecture_complete=false`. That candor is preserved and extended here.

### 1.7 Process state

The entire kernel, tests, fixtures, and the v1 plan are **untracked or
uncommitted in git** (`??` / modified). One bad `reset --hard` loses the MVP.
No CI exists.

---

## 2. Defect register (fix before any new feature)

| # | Defect | Fix | Effort |
|---|---|---|---|
| D1 | Gate whitelist desync (`assistant_open_traces.py:1291`) | Decide the label taxonomy once: `weighted_functional_relation` is an intended primary source (tests + integrity say so) → add it to the gate whitelist *and* to the calibration gate in the CLI; or rename labels to one canonical set. Re-run `pi-smoke` → must be 22/22. | hours |
| D2 | Stale prose claims (22/22, cases=50 vs 18, "Met" rows resting on red gates) | This document supersedes; v1 gets a banner pointing here. Numbers leave prose permanently (see §6). | hours |
| D3 | Work uncommitted, no CI | Commit in reviewable slices. Add CI (GitHub Actions or local pre-push runner): `pytest -q` + `pi-smoke --reset --json` + `shortcut-audit --json`; artifact-diff the JSON reports. | 1 day |
| D4 | Story constraint dropped silently (dragon/robot → canned template) | Story frames must extract requested topic/entity modifiers from the already-parsed UOL slots; if inventory has no match above threshold: say so and offer what exists, or route to clarify/cloud. Add eval cases asserting constraint adherence and honest decline. | days |
| D5 | 2,676-line monolithic plan doc | Split per §6: spec / auto-generated evidence ledger / roadmap. | with D2 |

---

## 3. Product spine decision

What this system actually is today: a **deterministic personal kernel** —
router + memory + policy + inventory + bookkeeping — with a stub verbalizer.
That is a legitimate product seed (a privacy membrane and memory OS that a
host model or apps sit behind), but v1 oscillates between three identities
without choosing:

1. **Immune-system kernel for a host model** — MELM owns boundary, memory,
   triage, actions; a local SLM or cloud model owns language.
2. **Self-contained rule appliance** — what exists today; usefulness ceiling
   is templates over inventories.
3. **Learning organism (MELM thesis)** — meaning/capability accrete from
   experience via atlas + learning ledger.

v2 commits to: **kernel (1) as the product, learning substrate (3) as the
research differentiator built on it, and explicitly retires (2) as an end
state** — templates remain only as the fallback verbalizer. This resolves the
"SLM-efficiency work stays secondary" deferral in v1, which had quietly turned
the headline use cases (stories, advice) into permanent stubs.

---

## 4. Workstreams

### T1 — Truth & infrastructure (blocking, week 1)

- D1–D5 fixes.
- **Behavioral gate rewrite**: gates assert *observable effects* — route
  chosen, SQLite rows written, facts excluded from cloud payloads, actions
  pending vs executed, answer text honoring constraints — not debug-label
  strings emitted by the code under test. Debug labels may be *logged* in
  evidence, never *asserted* as the pass condition. Acceptance: flipping any
  debug string constant in the router fails zero behavioral gates; changing a
  route or leaking a fact fails at least one.
- CI green required before any feature merge.

### T2 — Evaluation integrity (weeks 1–3, parallel)

- **Blind preregistered eval**: reuse the existing
  `support_refunds_external_blind_preregistration.json` pattern for the
  assistant OS: preregister case counts, annotator protocol, and thresholds;
  cases authored by someone (or some process) that has not read the router.
  Until then, every fixture-derived rate is labeled `internal-fixture` in
  reports.
- **Baseline fairness rule**: a baseline may be simpler, but not artificially
  forbidden from techniques the kernel uses. Add at least one honest baseline
  (e.g. the full keyword intent classifier *without* memory/lifecycle) so the
  measured delta isolates what memory+lifecycle actually buy — that is the
  claim that matters.
- **Constraint-adherence metric** (from D4) joins privacy/action metrics as a
  first-class safety column.
- User-derived transcripts through the existing `import-transcript-replay` /
  `calibrate-transcript-replay` path — the v1 blocker list already names
  this; it becomes the only accepted source for routing-rate claims in v0.2.

### T3 — Learning substrate (weeks 2–6)

The MELM thesis gets implemented or the claims get cut. Implementation:

- New factored lexicon tables, `atlas_edges`, `learning_candidates`,
  `corrections`, and `promotions`, all using the contracts in Part V.
- Frame acceptance may consult promoted semantic/experience support, but an
  atlas write cannot enable a handler or action. Installed capability remains
  owned by the release-controlled capability manifest.
- Corrections lower edge strength, write negative edges, and require
  re-promotion — wired to the existing `improvement_candidates` consent flow.
- Intent vocabulary migrates from code constants to seeded lexemes, senses,
  semantic classes, and frame definitions with provenance `seed.v2`. The
  classifier reads stores; code stops being the vocabulary.

### T4 — Bounded generation (weeks 3–8, decision-gated)

- **Decision gate first**: benchmark small decoder candidates in the measured
  Pi envelope using `AnswerPlan` + membrane-admitted evidence. Record exact
  model/runtime hashes, tokens/s, TTFT, RSS, constraint retention, verifier
  rejection rate, and fallback rate before fine-tuning.
- If go: `GuidedLocalSLM` adapter behind the existing synthesis boundary —
  same typed claims and evidence requirements, same refusal on blocked/cloud/
  action routes, and template fallback when the model is absent or rejected.
- If no-go: the product claim changes honestly — stories/advice are
  cloud-handoff capabilities, local answers are status/memory/cache/action
  only, and the README stops implying otherwise.

### T5 — Hardening (continuous)

Keep v1's genuinely good remaining items: live inventory soak escalation,
configured host-app probes on target devices, autoimmune suite growth,
digest calibration on real transcripts. These stay verbatim from v1's blocker
list; they were never the problem.

---

## 5. Workstream-to-milestone map

The workstreams above state enduring responsibilities. They do not define a
second schedule. Part IV §18 is the authoritative phase order:

| Workstream | Authoritative milestones |
|---|---|
| T1 truth/infrastructure | M0-M1 |
| T2 evaluation integrity | begins M0; gates M3-M7 |
| T3 learning substrate | M2-M3, then supports M5-M6 |
| T4 bounded generation | M4 in parallel after M1 |
| T5 hardening | continuous; release gate at M7 |

---

## 6. Document governance

- **`docs/assistant_os_spec.md`** — architecture contracts only (membrane,
  frames, atlas, action gate, synthesis boundary). No numbers, no history.
  Target ≤ 600 lines.
- **`docs/assistant_os_evidence.md`** — *generated*, never hand-edited: a CI
  job renders it from the JSON artifacts of `pi-smoke`, `setup-integration-smoke`,
  `eval`, `run-transcript-replay`, etc., with timestamps and git SHA. Stale-by-
  construction claims become impossible.
- **`docs/assistant_os_roadmap.md`** — Part IV §18 plus Part V §23, pruned as
  phases close. Completed work moves to git history, not to an ever-growing
  "Completed slices" list (v1's items 1–65).
- v1 (`local_assistant_os_mvp_plan.md`) is archived under `docs/archive/`
  with a banner; it remains valuable as the design-rationale record.

---

## 7. Risks

| Risk | Mitigation |
|---|---|
| Meaning-store migration destabilizes routing | Migrate one frame family at a time behind existing minimal-pair tests; seeded rows reproduce current behavior exactly before learned senses or edges may differ. |
| Local SLM uses unsupported latent knowledge | `AnswerPlan` limits authoritative claims; mode-specific verification discards invalid output; hard metric is zero unsupported applied claims. Prompt-injection probes join the autoimmune suite. |
| Pi-class hardware can't run the SLM | M4 records a no-go and keeps templates/cloud handoff; the learning and routing thesis still proceeds. |
| Behavioral gates slower than label checks | They already run full kernel turns; asserting on DB rows instead of strings costs nothing material. |
| Solo-author blind eval is awkward | The prereg pattern in-repo already solves this for support_refunds: freeze cases + thresholds by hash *before* running the kernel on them; recruit one outside annotator for the v0.2 gate. |

---

## 8. Immediate next actions (ordered)

Execute Part V §23. In short: recover truth, freeze contracts, put the
evidence/answer authority boundary around today's templates, migrate one
meaning family, then prove the `kalimba` acquisition loop. Run the M4 decoder
feasibility lane in parallel only after the M1 contracts exist.

---

# Part II — Language-core assessment (2026-06-10 addendum)

This part evaluates the project against its own deepest thesis: a system that
*structurally understands language* and therefore **continuously expands its
vocabulary at runtime** — encountering a new word, acquiring its meaning, and
using it correctly next time, the way a person reads a dictionary entry once
("fast mapping"). Where Part I fixed truth and evidence problems, Part II
fixes the gap between the thesis and the language core, and amends the
workstreams accordingly.

## 9. The dictionary test (target capability, made falsifiable)

The thesis compiles to one concrete loop:

```text
1. DETECT   an unknown word in a parsed utterance
2. ACQUIRE  a candidate meaning: user explanation in chat, bundled offline
            dictionary, or membrane-gated cloud lookup (word only, no context)
3. REPRESENT it as a first-class lexical entry with provenance and confidence,
            quarantined until promoted
4. GENERALIZE: inflect it morphologically, inherit its class's argument
            structure and routing behavior
5. REUSE    it correctly next turn AND next session: right slot, right route,
            right inflection, honest refusal when no local capability backs it
```

A system that does 1 but not 2–5 does not understand language structurally;
it detects ignorance without curing it.

## 10. What exists today (verified against code, 2026-06-10)

**Stage 1 (detect) exists.** `functional_grammar.py` emits
`semantic_unknown_tokens`; `assistant_integrity.py` scores unknown-token
coverage into understanding scores; `improvement_candidates` queues
low-understanding turns behind consent. This is genuinely good scaffolding.

**Stages 2–5 do not exist, and the current structure actively prevents them:**

| Fact (verified) | Consequence |
|---|---|
| Entire active vocabulary: **34 verbs** (`_VERBS`) + **9 nominal domains** (`_KNOWN_NOMINAL_DOMAINS`) + closed function-word classes, all module-level Python constants in `functional_grammar.py:69-114` | A learned meaning has nowhere to live. Runtime cannot add an entry to a frozen code constant. |
| Lemmatizer (`_lemma`, `functional_grammar.py:642-677`): 15 irregular forms + suffix rules that only resolve when the stem is **already in `_VERBS`** | A new verb can never even lemmatize. Teach the system "zorp" and "zorping" still parses as an unknown nominal. Morphological generalization is structurally impossible for acquired words. |
| Routing vocabulary: ~200+ keyword literals across 14 intent classifiers in `local_assistant_router.py`; 11 families now use `_semantic_family_terms` with lexicon-backed semantic class lookup via `lexicon_owned`/`lexical_class_lookup` params — all contracted, seeded, and store-backed. 3 families (assistant_identity, assistant_status, assistant_greeting) don't use content-word vocabulary and remain code-constant. | A new word can never change routing. Even a perfectly acquired noun routes to unknown/cloud forever, because capability gating reads code constants, not a store. |
| Unknown content words get role `content_nominal`, class `semantic_class_unknown`, weight 0.58 | The parser slots unknown words generically — good — but nothing downstream can ever upgrade that slot. |
| No fixture in any eval/trace/lifecycle contains an unseen-word acquisition scenario | The 105-case eval, 25-turn replay, and all smokes measure a **closed-world** system. Vocabulary growth is untested because it is impossible. |
| Whitepaper Layers 4–5 (morphology-aware input, compositional lexical representation) explicitly deferred; v1 plan defers SLM and atlas | Every layer that could implement the thesis is deferred in every document. The thesis exists only as prose. |

**The three-layer conflation.** The deepest structural problem: three
distinct mappings are fused into code constants —

```text
lexical layer     word/inflection -> lemma, POS, sense        (_VERBS, _lemma)
conceptual layer  sense -> semantic class                      (_VERBS values, _KNOWN_NOMINAL_DOMAINS)
capability layer  semantic class -> frame/intent/route         (intent classifier keyword sets)
```

Because all three are code, every new word × every capability = a Python
edit, a template edit, a fixture edit, and a gate edit. v1's own 65-item
completed-slices log is the empirical proof: each capability was a multi-file
code change. That is O(developer) scaling, the exact "lexical acquisition
bottleneck" that killed hand-built symbolic NLU systems in the 1980s–90s.
The project is currently walking into a well-documented wall.

**The good bones.** The typed-frame discipline, membrane policy,
provenance/quarantine/promotion design (T3), unknown-token honesty, and
experience-gated capability instinct are *exactly* the safety scaffold that
runtime vocabulary growth needs and that ML-only systems lack. The skeleton
is right; the organ is missing. This is a missing-subsystem problem, not a
wrong-architecture problem.

**UOL bridges the lexical and conceptual layers.** The UOL schema (§21.2)
records lemmas (lexical layer) and `semantic_class_id` values (conceptual
layer). Frame candidates (§21.4) consume UOL to derive capability (routing,
evidence, answer planning). This makes UOL the architectural invariant: the
lexical layer (language, tokenizer, inflection normalizer) and the capability
layer (routing policy, capability manifest) can vary independently as long as
UOL remains the stable intermediate representation. The M2 factored lexicon
builds the store that feeds UOL; M5 replaces keyword classifiers with
UOL-based frame linking. The bridge pattern in M2 (`_semantic_family_terms`
with `lexicon_owned` switch) preserves backward compatibility while the store
matures — but keyword-first intent classification terminates at M5, not
before.

## 11. Growable lexicon architecture (L1–L6)

This is the word-level part of T3, made precise. Lexicon and atlas are
distinct stores: the **lexicon** maps words to senses/classes; the **atlas**
maps concepts to concepts and capability-relevant experience; the
**capability manifest** controls installed behavior; **frames** bind requests.
Words are learnable cheaply; capabilities remain conservative.

### L1 — Lexicon as data

Use a factored store (lexeme + forms + senses + reusable templates) so a new
word costs data rows, never a routing rule. The definitive schema is §21.3;
the conceptual fields are:

```text
lemma, surface_variants, pos, morph_class,
semantic_class, argument_frame (template ref), concept_id,
provenance   (seed.v2 | wordnet | wiktionary | verbnet | user_taught |
              cloud_lookup | inferred),
confidence, status (active | quarantined | defeated),
source_event_id, created_at, last_used_at, use_count
```

Migration: `_VERBS`, `_KNOWN_NOMINAL_DOMAINS`, and the router's intent
keyword sets become rows with `provenance=seed.v2`, loaded into an in-memory
cache at startup. The parser and intent classifiers read the store; **code
stops being the vocabulary.** Behavior must be bit-identical after migration
(existing minimal-pair tests are the guard) before any learned entry is
allowed to differ.

**Bulk seeding (the decisive scale move, per §14.4):** the lexicon's breadth
must be *imported, not authored*. Adopt, license-checked:

- **Open English WordNet** — ~117k synsets / 155k words, ~12 MB, BSD-style →
  noun/adjective senses, genus hierarchy for L2 definition mapping.
- **Wiktextract (kaikki.org) English subset** — CC BY-SA, monthly dumps →
  definitions, inflections, surface variants for the offline dictionary.
- **VerbNet 3.4** — 273 verb classes covering ~8,500 verb entries → the
  `semantic_class`/`argument_frame` inventory (replaces the hand-invented
  34-verb table; "VerbNet-lite" means importing the subset of classes the
  device domain touches, not re-authoring classes).
- **PropBank frame files 3.4** — CC BY-SA → argument-structure cross-check.
- **Avoid:** UDPipe models (CC BY-NC-SA) and pre-2016 FrameNet data
  (CC BY-NC-SA) — non-commercial licenses, unusable in a shipped appliance.

How each source becomes rows is not source-specific code scattered through
importers: every source (including the runtime channels in L2 and the cloud
prompt) emits the same `sense_candidate.v1` contract and passes one ingestion
gate — see §21.10. The end-to-end bootstrap is the focused mini-plan in §24.

### L2 — Acquisition pipeline (the fast-mapping loop)

Trigger: `semantic_unknown_tokens` non-empty on a turn whose frame fails or
degrades. Acquisition sources in privacy order:

1. **User explanation in chat** — "a kalimba is a small thumb piano" parses
   as a definition frame (copula + genus + differentia, structure the UOL
   layer already approximates). Consent is inherent; provenance `user_taught`.
2. **Bundled offline dictionary** — ship a compact Wiktionary/WordNet-derived
   JSONL (open-licensed; WordNet ~13 MB covers ~150k entries). Lookup is
   local, free, instant; provenance `offline_dictionary`.
3. **Membrane-gated cloud lookup** — the bare word only, never utterance
   context, behind the existing membrane policy; provenance `cloud_lookup`,
   quarantined by default.

Definition → candidate entry: extract POS; map the **genus term** through the
existing lexicon to a semantic class ("a kalimba is a *piano*…" → genus
`piano` → media/instrument class); for verbs, "to X means to Y…" inherits Y's
argument frame. Candidate lands `quarantined`; promotion requires the T3
gates (auto-generated minimal pair passes, no conflict with active entries,
policy check). Corrections defeat entries through the same ledger.

All three channels emit `sense_candidate.v1` (§21.10) and enter through the
same ingestion gate as the batch seeders — one validator, two tempos.

### L3 — Generalization machinery

- **Morphology:** regular English inflection rules keyed by `morph_class`
  apply to *all* entries, including learned ones — the lemmatizer consults
  the lexicon store, not `_VERBS`. (This is where the whitepaper's morpheme
  thesis finally earns implementation, in its simplest viable form.)
- **Verb classes:** map the existing `semantic_class` values onto imported
  VerbNet classes (273 exist; the device domain needs a few dozen), each
  carrying default argument slots, so a new verb inherits syntax from its
  class instead of needing hand-wired slots.
- **Similarity fallback (optional, flagged):** a tiny embedding model
  (MiniLM-class, ~22 M params) proposing nearest-known-word when no
  dictionary hit exists; provenance `inferred`, lowest confidence, never
  promotable without user confirmation. This is the one place a learned
  component enters before the T4 SLM decision, and it is severable.

### L4 — Routing integration (capability stays honest)

**Partial progress.** Intent classification for 3 families (story, weather,
media) now optionally consults the lexicon: token → semantic class →
class-level rules (any `media_content` noun + `play`-class verb → media frame).
A learned word **inherits routing through its class** via
`_semantic_family_terms`/`lexical_class_lookup` — keyword sets are replaced by
lexicon queries when `lexicon_owned=True`. All 11 families that use
`_semantic_family_terms` are now contracted, seeded via
`build_legacy_router_candidates`, and activated by default in
`seed_assistant_os_lexicon`. The infrastructure pattern
(`lexicon_owned`/`lexical_class_lookup` params on every classifier) is in
place and all families are store-backed.

Full L4 requires all 14 families to be lexicon-owned (contract + seed data),
with intent keyword sets as code ceasing to exist. Capability remains
manifest-bound and evidence-complete: "tell me a story about kalimbas" may
enter the story frame, but local routing still requires an installed story
handler and matching inventory. It honestly misses (per D4) — acquisition
removes false *unknowns*; it never fabricates capability.

### L5 — The dictionary benchmark (new first-class gate)

A fixture of N unseen words (nouns/verbs/modifiers; real rare words + nonce
words), exercised through all three acquisition channels, scored on:

```text
acquisition accuracy   correct POS/class/argument frame from definition
next-turn use          correct slot + route + inflection on first reuse
retention              same, next session, after process restart (SQLite)
interference           zero regressions on the existing minimal-pair suite
honest refusal         nonce word with no acquisition stays unknown;
                       contradictory redefinition quarantines, not overwrites
```

This is the falsifiable test the whitepaper promises and no current fixture
attempts. It joins `pi-smoke` once green and becomes the project's headline
metric, displacing local-resolution rate on self-authored fixtures.

### L6 — Parser strategy decision gate

The hand-weighted grammar is acceptable for the bounded device domain but has
a known ceiling: no coordination ("play a song and set a timer"), no particle
verbs ("turn off" — `turn` is not even in `_VERBS`), no subordination,
ellipsis, typo robustness, or multiword expressions. M6 evaluates candidate
parsers/taggers on user-derived and sealed construction-family sets for role
accuracy, latency, RAM, and contract compliance. Off-the-shelf dependency
parsers are teachers/baselines; the shipped owner may be rules or a distilled
E2 tagger. The stdlib-only rule is a deployment preference, not a thesis
requirement, and UOL remains the stable interface regardless of backend.

## 12. Bloat assessment

Measured against the thesis, effort allocation is inverted:

- Language core: **650 of ~18,200 lines (3.6%)**. Vocabulary: 43 content
  words. Acquisition mechanism: none.
- Packaging/evidence/attestation: 62 CLI subcommands, seven inventory smokes,
  bundle/manifest/hash/attestation/blocker-evidence machinery — thousands of
  lines proving that a **static** system is reproducibly packaged.
- v1 plan: 2,676 lines; the vocabulary-growth mechanism appears only as the
  prohibition of phrase tables, never as a design.

Rules going forward:

1. **Freeze packaging surface.** No new smoke/bundle/attestation commands
   until the dictionary benchmark (L5) exists and is green. Gate count may
   grow only with new *behavior*, not new packaging.
2. **Collapse where cheap:** the seven inventory smokes become one
   parameterized gate; attestation machinery is maintenance-frozen until
   there are external users to attest.
3. Effort ratio target for the next quarter: ≥50% of merged lines touch the
   language core, lexicon, learning substrate, or their tests.

## 13. Mapping into the authoritative milestones

- L1 and the semantic-class/frame split ship in M2.
- L2, L3 morphology, and L5 form the M3 learning vertical slice.
- L6 no longer creates a separate "Phase 1.5." Rules remain the initial UOL
  owner; E2 is evaluated and promoted by construction family in M6.
- Similarity fallback is optional E1 work in M3. It may share E3's encoder,
  but neither expert is required for the first dictionary acquisition proof.
- E5 remains E2 + deterministic definition rules until evidence justifies a
  separate model.

---

# Part III — External research validation (2026-06-10)

Web-research pass (multi-angle search with primary sources) on the
architecture's novelty and feasibility. Sources inline; full agent reports in
session transcripts.

## 14. Novelty verdicts per component

| MELM component | Verdict | Prior art |
|---|---|---|
| Small local model answers common requests; cloud LLM handles complex (the headline pitch) | **Well-established, productionized** | Apple Intelligence: ~3B on-device foundation model + orchestration escalating to a server model in Private Cloud Compute, shipped 2024 (machinelearning.apple.com). Google Gemini Nano via Android AICore, same split. Academic: FrugalGPT 2023 cascades (~98% cost cut at matched accuracy, arXiv 2305.05176); RouteLLM 2024 (85%+ cost cut keeping ~95% GPT-4 quality, arXiv 2406.18665); NVIDIA/Belcak position paper "SLMs are the Future of Agentic AI" (arXiv 2506.02153). |
| Privacy boundary gating what crosses to cloud | **Established concept; MELM's granularity is the interesting part** | Apple PCC is the strongest form (stateless, attested trusted compute). MELM's *per-fact, consent/scope-tagged data-minimization membrane* on commodity hardware is a different mechanism and is rare at that granularity — defensible as engineering novelty, not as concept novelty. |
| Typed intent/slot frames for device commands, on-device, private | **Established 2018; hand-authored variant abandoned ~2000s** | SNIPS voice platform (arXiv 1805.10190): embedded private-by-design intent+slot NLU on IoT-class hardware, 98.88% intent accuracy, 97.07 slot F1 — *ML-trained*. Hand-authored frame/slot grammars = CMU Phoenix lineage (1990s); the field migrated to statistical parsing precisely because rule grammars were brittle and labor-intensive. The HWU64 benchmark (arXiv 1903.05566) gives the bar commercial NLU reached: intent F1 ≈ 0.88 across 21 domains. Modern bar: Octopus v2, a 2B on-device function-calling model at 98–100% accuracy beating GPT-4 on accuracy and latency (arXiv 2404.01744); TinyAgent-1.1B exceeding GPT-4-Turbo on its tool domain (Berkeley BAIR 2024). |
| SQLite event ledger + fact store + digests | **Known pattern** | MemGPT/Letta tiered memory (core/recall/archival + sleep-time consolidation ≈ digests; arXiv 2310.08560) persisted in ordinary relational DBs; Zep/Graphiti's episode ledger → extracted facts → community summaries triad (arXiv 2501.13956); `sqliteai/sqlite-memory` is a shipping open-source SQLite-backed local-first agent memory. MELM's schema is a competent local-first re-implementation, not an invention. |
| Runtime vocabulary growth with provenance/quarantine/promotion | **Known pieces, rare combination — MELM's best candidate claim** | Hypothesize-then-promote lexicon induction is GENLEX-shaped: generate candidate word→meaning entries, keep those that survive parsing. Factored lexeme+template lexicons, definition-based learning, one-shot word binding, and runtime lexicon induction all have prior art. The product-level claim that this remains uncommon in device assistants is a survey hypothesis, not yet a proved absence claim. A database lexicon still offers concrete provenance and rollback advantages over weight editing. |
| Opportunity planner: repeated gaps → local capability building | **Rare combination** | The NVIDIA SLM-agentic paper describes an LLM-to-SLM *conversion algorithm* for migrating recurring requests to local models — same instinct at the model level. MELM's device-level loop (gap → job → inventory → route flip, already implemented and tested) is an unusual and demonstrable concretization. |
| Immune/membrane/homeostasis framing | Framing novelty only | Cognitive-architecture lineage; valuable as design discipline, not citable novelty. |

**Net verdict:** the pitch "SLM handles device-capability chat locally, cloud
LLM handles complex" is not a novel thesis. MELM's candidate differentiator
is a **provenance-first, quarantine-gated, continuously growing lexicon +
experience atlas on commodity hardware, with manifest-controlled capability
and repeated gaps feeding explicit preparation work**. The implementation can
prove the mechanism; a separate documented survey must establish any market
novelty claim.

## 14.1 The history lesson (directly applicable)

The hand-authored lexicon path MELM is currently on is the most-documented
dead end in NLU history: Briscoe's 1991 "lexical bottleneck", the CACM 1996
lexicon survey, and CYC's four-decade grind all say per-project hand
authoring caps at a few thousand entries while English needs 10⁵–10⁷ senses.
What survived: (a) shared lexicographer-built resources (WordNet 1990,
COMLEX 1994, VerbNet, PropBank) and (b) induction from data. Part II's L1
bulk-seeding directive is therefore not optional polish; it is the difference
between repeating the 1990s and skipping them.

## 14.2 Feasibility numbers (target-hardware reality)

| Fact | Source |
|---|---|
| Raspberry Pi 5: ≤1.5B-param models sustain 5–15 tok/s; 3B ≈ 2–5 tok/s; sub-360M >20 tok/s; Q4_K_M recommended | SBC inference survey, arXiv 2511.07425 |
| Llama 3.2 3B Q4_K_M ≈ 4–6 tok/s with OpenBLAS; GGUF file size ≈ required RAM; 8 GB Pi advised for 3B | tinyweights.dev Pi 5 benchmarks |
| llamafile up to 4× throughput and 30–40% lower power vs Ollama on SBCs | arXiv 2511.07425 |
| 1B-instruct models = best speed/resource ratio for interactive use on Pi 5 | Stratosphere Lab 2025 |
| MiniLM-L6-v2 embeddings: 22.7M params, 384-dim, tens of MB quantized — negligible next to an SLM | HF model card |
| spaCy `en_core_web_sm`: ~12 MB, MIT, LAS ≈ 89.9 | HF/spaCy model card |

Implication (sizing advice retired by Part IV — numbers stand): decoder
candidate selection and the actual parameter/quantization ceiling are M4
artifacts, not prose commitments. The durable points: MiniLM-class
embeddings + spaCy-small deliver open-vocabulary robustness for ~25 MB
total without any generative model, and Octopus v2 (2B, function-calling)
is the direct competitor benchmark for the router itself.

## 14.3 Consequences folded into this plan

1. **Accuracy bars are published; adopt them.** The router currently reports
   only self-authored fixture rates. Add an external-benchmark gate: evaluate
   intent/slot accuracy on a SNIPS/HWU64-style subset mapped to MELM's
   domains. Published bars: SNIPS 98.9% intent / 97.1 slot F1 (embedded!),
   commercial NLU ≈ 0.88 intent F1. If the hand parser can't approach the
   embedded-1.9-MB-SNIPS bar on its own bounded domain, that decides L6's
   bake-off (and likely T4) on evidence.
2. **Reframe README/whitepaper novelty claims** per §14's table — claim the
   lexicon-growth loop and experience-gated capability, drop any implication
   that local/cloud routing or SQLite memory are novel.
3. **Historical T4 recommendation (retired by Part IV):** the initial review
   proposed 1B and 3B Q4 candidates. M4 now selects candidates from measured
   target constraints and evaluates on a new rendering set; transcript replay
   remains regression evidence, not a held-out quality set.
4. **L2 acquisition channels confirmed viable:** Wiktextract English JSONL
   (kaikki.org, monthly) for offline definitions; CPAE/dict2vec literature
   says definition-only learning works; GENLEX-style promote-on-survival is
   the proven gate design the learning ledger should copy.

> **Superseded in part by Part IV (2026-06-10, same day):** §14's table binned
> UOL/ChatFrame with hand-authored 1990s frame grammars, and §14.2's
> implication recommended a 1–3B monolithic SLM tier. Both are corrected in
> Part IV: the architecture under evaluation is a *typed staged expert
> cascade* (deterministic kernel gating + stage-specialized sub-1B neural
> experts + externalized knowledge), for which the Phoenix comparison is
> wrong and the monolith sizing is unnecessary. §14's verdicts on memory,
> hybrid routing, and lexicon resources stand.

---

### T5 — Moral cognition (weeks 4–6, parallel to T3/T4)

Goal: Replace 5 hardcoded moral/safety duplication sites in the router with
verb-driven implication detection (spec §16).

Deliverables:
- `verb_states.v1.json` (~60 verb entries) and `state_valences.v1.json` (~40
  state→score mappings) contracts
- `derive_moral_context()` pure function in `reasoning/implications.py`
- 5 router duplication sites patched (3 urgent-health token sets, 1 safety
  token check, 1 health fallback)
- `MoralContext` wired into `build_answer_plan()` authority verification
- `_urgent_harm_answer()` short-circuit in synthesis
- 3 learning pathways: NAMELESS seed extraction, VerbNet offline extraction,
  runtime chat candidate ring buffer

Exit gate:
- `pytest tests/test_implications_mvp.py tests/test_contracts_mvp.py -k moral`
  passes (≥19 tests)
- Zero hardcoded `urgent_health_terms` sets remain in `local_assistant_router.py`
- Router syntax valid (`python -c "import ast; ast.parse(...)"` green)

---

# Part IV — Compact expert architecture: the NAMELESS lineage build plan (2026-06-10)

## 15. The corrected novelty position

Two artifacts re-anchor the thesis:

**NAMELESS I (original concept deck).** The 2014-era design already contains
the full loop MELM is rebuilding: UOL as a language-neutral object
representation of sentences (subject/action/object + qualifier trees); word
analysis via *association strengths that grow with use* (the atlas, before it
had a name); sentential context analysis that resolves metaphor/absurdity
("Obi laid an egg" → "created great value") or raises **curiosity flags**
(the clarify route); knowledge stored *with source attribution* enabling
contextual answers ("what is my favorite food") and relational inference
("Obi is five years older than Ada" answers an age question with no stored
age). The project's continuity claim is real: MELM is NAMELESS with an
immune-system boundary and a device substrate.

**db-claw / SemanticSQL (sibling repo, same author, working alpha).** This is
the *already-hybrid* instance of the pattern in the SQL domain, and the
existence proof my §14 verdict missed:

```text
normalize → deterministic vocab pre-resolve (fallthrough on ambiguity)
→ intent-pattern bias (additive only, never route owner)
→ ONNX cross-encoder schema linker (recall@5 ≥95% target)
→ T5-class skeleton generator under llguidance CFG-constrained decoding
→ slot filler (top-1 ≥90% target)
→ typed decision gate: execute | ask_user | ask_llm | reject
→ dialect renderer + second-pass validation
```

Properties that matter: neural models are **stage-specialized and tiny**;
gating between stages is **deterministic and typed**, not learned softmax;
vocabulary/knowledge lives in a **provenance-layered SQLite store** (8 source
layers, user-approved memory at layer 7), not in weights; generation is
**grammar-constrained** so the model physically cannot emit out-of-schema
output; each stage has **independent metrics** (failure attribution a
monolith cannot give); governed routes report **0 wrong accepted SQL**.

**The architecture, named precisely:** a *typed staged expert cascade* — the
kernel is the gating network, the experts are stage-specialized sub-models
(understand / bind / decide / generate), world knowledge is externalized to
provenance stores, and decoding is constrained by the frame. "Small MoE for
NLU/NLP/NLG" is the informal name; unlike a learned-gating MoE, the gate is
the typed kernel itself, which is what makes the system auditable and the
membrane enforceable.

**Why this is not the abandoned 1990s pattern (correcting §14):** Phoenix-era
systems were rules end-to-end — brittle exactly where rules are weak
(open vocabulary, paraphrase, noisy syntax). The cascade puts tiny learned
models at precisely those points (embedding/linking, role tagging,
verbalization) while keeping rules where rules are strong (typing, policy,
provenance, capability gating). SNIPS proved embedded ML intent+slot at
98.9% in 1.9 MB in 2018 — but stopped at intent+slot. Apple ships a 3B
monolith + LoRA task adapters with knowledge in weights. Octopus v2 is a 2B
monolith. The working novelty hypothesis is the **full
understand→remember→decide→generate loop as a compact typed expert cascade
with a growable provenance lexicon on commodity CPU hardware**. That claim
still requires a documented product/prior-art survey before publication.
Part II's lexicon (L1–L6) plus Part I's membrane/learning-ledger are its
substrate; what changes is the neural plan around them.

## 16. Architecture: a typed expert cascade

This is **MoE-like**, not a conventional learned-gating MoE. The deterministic
kernel owns expert selection, policy, and fallbacks. Learned experts propose
typed artifacts; they never directly choose an action, disclose evidence, or
declare their own output safe.

```text
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
missing expert falls back to rules, templates, clarify, or cloud according to
the already-typed route; it does not trigger an improvised alternate path.

**UOL is the foundational meaning representation.** Every token in every
utterance produces a `UOLParse`. Frame selection, routing, evidence admission,
and answer planning consume UOL — not raw tokens. Any stage that reads raw
tokens (current keyword intent classifiers in K0, §10) is a v0.1 transitional
artifact. M5 replaces keyword classification with UOL-based frame linking; the
vocabulary migration in M2 (§18) is the prerequisite, not the destination.

### 16.1 Performance envelope

The Pi 5 target remains:

```text
steady-state generation      >30 tok/s target
integrated p95 first token   <1.5 s for local-answer routes
peak process RSS             <1.2 GB
model + lexical assets       <600 MB
```

These are release goals, not architectural facts. M1 measures candidate
runtimes and fixes the actual parameter/quantization ceiling. No model name or
parameter count is promoted into the architecture before that artifact exists.
Reports separate cold start from warm turns and include prompt length, output
length, runtime build, quantization, temperature, CPU governor, and model hash.

Encoder work is sequential and non-generative, but it is not assumed free:
every milestone reports integrated p50/p95 latency and RSS. E1 and E3 share one
encoder in the first implementation unless evidence shows that separate models
earn their memory cost.

### 16.2 Expert roster and authority boundary

| Stage | Responsibility | Initial implementation | Authority / failure behavior |
|---|---|---|---|
| **K0 typed kernel** | Contract validation, event ledger, memory digest, session context, membrane, capability policy, routing, action confirmation, experience capture, fallbacks | Existing Python + SQLite kernel | Sole route/action authority; invalid input fails closed |
| **E1 lexical retriever** | Sense candidates, similarity, OOV suggestions | Shared small ONNX encoder; optional | Proposes candidates only; cannot promote a sense or route |
| **E2 UOL parser** | Tokens/spans → clauses, roles, morphology, unknowns | Rules first; small tagger by construction family | Emits `UOLParse`; shadow output never routes |
| **E3 frame linker** | Rank frame candidates from UOL + semantic support | Rules first; shared-encoder reranker on top-k | Emits scores only; K0 applies thresholds and policy |
| **E4 verbalizer** | Render an `AnswerPlan` into bounded language | Template baseline, then small GGUF decoder | Cannot see raw history or blocked evidence; verifier failure discards output |
| **E5 definition parser** | Definition → candidate lexical sense | E2 + deterministic genus/differentia rules | Writes quarantine only; neural specialization is deferred |

The compactness claim is specialization plus externalized state, not that the
experts collectively contain no knowledge. A pretrained E4 **does contain
latent world knowledge**. The enforceable promise is narrower and stronger:
latent knowledge is not authoritative. Factual, policy, memory, and action
claims are accepted only when licensed by `AnswerPlan` and admitted evidence.

### 16.3 Event/chatframe experience capture — UOL relationship

The event ledger and memory system are separate from but fed by UOL:

```text
Current utterance
     │
     ▼
UOLParse (linguistic meaning: clauses, roles, lemmas, semantic classes)
     │
     ▼
FrameCandidate → RouteDecision → EvidencePacket → AnswerPlan → ChatFrame
     │                                                              │
     ├─ UOL refs (which clauses/roles anchored the frame)            │
     ├─ intent, route, reason, evidence keys                         │
     └─ membrane decision, action state, privacy scope               │
                                                                     │
     ┌───────────────────────────────────────────────────────────────┘
     ▼
Event ledger — persists ChatFrame + utterance + timestamp + session_id
     │
     ├─ memory digest: aggregates events → session context
     │                   (used for deixis resolution, pronoun binding)
     │
     ├─ improvement_candidates: queues low-UOL-coverage turns
     │                   → triggers acquisition or teaching
     │
     └─ atlas_edges: records observed relations across UOL parses
                       (concept A used-with concept B, etc.)
                       → quarantined until promoted via learning ledger
```

**Key relationships:**

- **UOL is ephemeral** — it exists for one utterance parse and is not stored.
- **ChatFrame is the durable record** — it captures the system's full interpretation
  (including the UOL refs that drove routing) and is persisted in the event ledger.
- **Memory digest reads events, not UOL directly** — it reconstructs cross-utterance
  context from stored ChatFrame data (previous intents, slots, evidence).
- **Atlas edges derive FROM multiple UOL parses** — observed co-occurrences,
  user corrections, and pattern detection across utterances produces edges
  that may influence future frame binding. They are quarantined by default and
  promoted only through the learning ledger.
- **Experience capture cannot modify UOL or routing** — it feeds the atlas and
  the learning ledger. Capability granting (§21.3) is release-controlled.

### 16.4 Thesis-to-mechanism map

| Thesis property | Implemented mechanism |
|---|---|
| Structural language representation | Versioned UOL with explicit spans, roles, senses, uncertainty, and unparsed coverage |
| Runtime vocabulary growth | New lexical sense as a provenance-bearing database row; no weight edit required |
| Meaning grounded in use | Atlas support and learning ledger record observed relations, corrections, and promotion history |
| Honest local capability | Release-controlled capability manifest + deterministic evidence completeness; language learning cannot grant a skill |
| Cross-utterance context | Event ledger stores ChatFrames; memory digest aggregates past intents/slots/evidence for deixis resolution |
| Experience capture | `improvement_candidates` queue feeds atlas edges; corrections lower edge strength and require re-promotion |
| Compact local generation | E4 receives only `AnswerPlan` + admitted evidence + stance, with templates as permanent fallback |
| Auditable routing | Candidate scores, threshold version, policy checks, evidence checks, and final reason persist in `RouteDecision` |

### 16.5 Entity store — unified person/object/event model

User facts, inventories, contacts, and event instances all describe the same
thing: **entities with typed slots**. A single store replaces the fragmented
`user_facts` + `inventories` + `membrane_decisions` pattern:

```
class_schemas                   entities                     entity_slots
┌──────────────────────┐       ┌──────────────────────┐     ┌──────────────────────┐
│ semantic_class_id PK │──┐    │ entity_id PK         │     │ slot_id PK           │
│ parent_class_id      │  │    │ kind (person|event    │──┬──│ entity_id FK         │
│ label                │  ├────│   |place|object|self) │  │  │ slot_name            │
│ base_entity_kind     │  │    │ label                 │  │  │ value_json           │
│ description          │  │    │ semantic_class_id FK  │  │  │ slot_state (filled|  │
└──────────────────────┘  │    │ canonical_lemma       │  │  │   asked_but_empty|   │
                          │    │ updated_at            │  │  │   unknown_entity|    │
class_schema_slots        │    └──────────────────────┘  │  │   unknown|inferred)  │
┌──────────────────────┐  │    entity_relations         │  │  │ provenance           │
│ slot_id PK           │  │    ┌──────────────────────┐ │  │  └──────────────────────┘
│ semantic_class_id FK ├──┘    │ relation_id PK       │ │  │
│ slot_name            │       │ entity_id FK         ├──┘  │
│ value_type           │       │ relation (member_of|  │     │
│ required             │       │   located_at|created_by│    │
│ description          │       │ target_entity_id FK   │     │
└──────────────────────┘       │ provenance           │     │
                               │ strength             │     │
                               └──────────────────────┘     │
```

**Entity kinds:**
- `person` — a known person (contacts, family, friends, user-self)
- `event_type` — a class of recurring events (competition, holiday, appointment)
- `event_instance` — a specific occurrence (World Cup 2026, tomorrow's meeting)
- `place` — a known location
- `object` — a physical or digital object (device, appliance, document)
- `personal_experience` — a chat session or past interaction (for memory retrieval)
- `self` — the user (for facts about the user, migrated from `user_facts`)

**Event class hierarchy:**
- `event` — base class. Slots: `start_time`, `end_time`, `periodicity`, `location`
- `competition` — inherits from `event`. Additional slots: `winner`, `participants`,
  `score`, `ranking`

Class hierarchy is stored in `class_schemas.parent_class_id`. Slot definitions
for each class are in `class_schema_slots`, enabling template validation and
schema-driven entity creation.

**Frame slot states apply to `entity_slots.slot_state`:**
- `filled` — value is known and populated
- `asked_but_empty` — the system asked but the user didn't know
- `unknown_entity` — the entity exists but this slot is unknown
- `unknown` — neither entity nor slot resolved
- `inferred` — value derived from context, needs confirmation

When the frame linker matches a frame with unfilled slots, it reports the slot
state so synthesis can generate frame-aware "I don't know" responses instead of
generic refusal.

**Entity relations enable cross-entity queries:**
- `member_of` — entity belongs to a group (person → family, event → championship)
- `located_at` — event or person at a place
- `created_by` — event or object created by a person
- `participated_in` — person participated in an event
- `owns` — person owns an object

**Migration path:**
1. `user_facts` rows → `entities WHERE kind='self'` + `entity_slots`
2. Contact inventory rows → `entities WHERE kind='person'`
3. User preferences → `entity_slots ON entities.kind='self'`
4. Event instances → `entities WHERE kind='event_instance'`

This unifies the query path: every entity question ("what's your mom's name",
"when is the competition", "what do I like for breakfast") resolves through the
same store interface.

## 17. Data, training, and artifact discipline

1. **Freeze contracts before datasets.** Every row records the UOL, semantic
   class, frame, evidence, and answer-plan schema versions used to create it.
   A model is compatible only with the contract versions in its manifest.
2. **Existing fixtures are regression sets, not held-outs.** They influenced
   the implementation and cannot support blind quality claims. New sealed
   dictionary, parser, frame-linking, and rendering sets are hashed before
   training or prompt development.
3. **No pretraining from scratch.** Start from license-approved encoder and
   decoder checkpoints. Record source, license, revision, tokenizer, and hash.
4. **E2 data:** UD/dependency sources + explicit UOL projection + independent
   annotation/adjudication. The hand parser may generate comparison output but
   is not the sole teacher.
5. **E3 data:** positive frames plus hard negatives that share vocabulary but
   differ in speech act, role binding, privacy, evidence state, or capability.
   Optimize accepted-route precision, not ranking accuracy alone.
6. **E4 data:** generate `AnswerPlan -> answer`, never raw-history -> answer.
   Values are randomized; unsupported target claims are rejected before
   training. Refusal, stale evidence, missing constraints, and verifier-failure
   cases are first-class.
7. **Artifact manifest:** every model ships with base-model hash, data hashes,
   split hashes, contract compatibility, training command, metrics,
   quantization command, runtime version, license, and rollback predecessor.

## 18. Authoritative implementation milestones

This table is the single execution schedule for v0.2. Part I workstreams and
Part II L1-L6 remain requirements, but they do not define a second ordering.
Pi measurements run on milestone release candidates; ordinary CI runs the same
functional gates without pretending shared CI hardware is a Pi.

| Phase | Scope and deliverable | Exit gate |
|---|---|---|
| **M0 — Recover truth** (days) | Fix D1-D5, preserve current behavior, commit the tree, add CI, label internal fixtures honestly | `pytest`, `pi-smoke`, and `shortcut-audit` green in CI; v1 superseded; changing a debug label does not fail a behavioral gate |
| **M1 — Contract kernel** (week 1) | Contract registry; schemas for UOL, semantic classes, frames, route decisions, evidence, answer plans, stance, and model manifests; adapters around current rules/templates; integrated perf harness | Current regression suite passes through validators; initial 60-case normative UOL set passes; incompatible versions fail closed; dev + Pi benchmark JSON produced |
| **M2 — Meaning substrate** (weeks 2-3) | Factored lexicon, semantic-class registry, frame registry, capability manifest, atlas/learning ledgers; migrate media, weather, and story vocabulary from code to seeded data | 100% route agreement on existing regression cases for migrated families; deleting a seed row changes behavior predictably; no user/atlas write can enable a capability or action |

**Router bridge during M2.** The intent classifiers use a transitional bridge:
`_semantic_family_terms` with a `lexicon_owned` switch. When the family is
store-enabled, vocabulary reads from lexical senses via `lexical_class_lookup`;
when not activated, it falls back to a deterministic inline dict. This permits
gradual family-by-family migration with bit-identical behavior at each step.
The bridge is **not permanent** — M5 replaces it with UOL-based frame linking
that selects frames from UOL, not from keyword matching against raw tokens.
| **M3 — Learning vertical slice** (weeks 3-5) | Detect → define → quarantine → test → promote → reuse → correct for nouns, modifiers, and one verb class; restart persistence | Sealed ≥60-word dictionary set: ≥80% correct next-turn use and retention; zero reserved-namespace promotions; zero capability grants; correction/rollback trace queryable end to end |
| **M4 — Bounded generation** (parallel after M1; weeks 2-5) | Benchmark small decoder candidates; wire `AnswerPlan`, constrained decoding, verifier, and template fallback; fine-tune only after zero-shot decision artifact | On Pi: report tok/s/TTFT/RSS; 0 unsafe **applied** outputs; 100% fallback on verifier failure; model accepted on ≥70% of eligible rendering cases and retains ≥95% of required constraints; go/no-go recorded before training spend |
| **M5 — Learned frame linking** (weeks 5-7) | E3 reranks rule-generated candidates using UOL + semantic support; K0 remains gate owner | On sealed set: top-3 frame recall ≥95%, accepted-route precision ≥98%, zero false-local safety cases; no regression on supported minimal pairs |
| **M6 — Learned UOL families** (weeks 6-9) | E2 enters shadow mode, then takes ownership only of qualified construction families such as particles and coordination | Per promoted family on ≥50 sealed examples: slot/role F1 beats rule owner, no safety regression, p95 parse latency within budget; ownership flip and rollback are data-only |
| **M7 — v0.2 integration** (weeks 9-10) | Default-on qualified experts, fallbacks, blind/user-derived evaluation, Pi package and hardware report | All hard safety invariants green; dictionary and external NLU bars reported; p95 TTFT <1.5 s and RSS <1.2 GB; if E4 ships, >30 tok/s, otherwise local-generation claims are explicitly reduced |

**Critical path:** M0 → M1 → M2 → M3 → M5 → M6 → M7.

**Parallel falsifier:** M4 starts after M1 because generation feasibility can
be measured without waiting for learned NLU. A no-go removes E4 from v0.2; it
does not block the lexicon, atlas, UOL, or routing thesis.

### 18.1 Milestone rules

- A hard safety gate is binary: privacy leak, unconfirmed action, unsupported
  applied factual claim, incompatible contract, or capability escalation must
  be zero.
- Quality targets report denominator, confidence interval, dataset hash, and
  whether the set is authored, sealed, user-derived, or external.
- No phase exits on component metrics alone. Its end-to-end path and fallback
  must also pass.
- A model improvement may increase accepted coverage; it may not weaken the
  verifier, membrane, capability manifest, or action gate.
- Each milestone produces one reviewable artifact directory and one rollback
  point. New packaging commands remain frozen until M3 is green.

## 19. Supersessions within this document

- §18 is the authoritative execution order. Part I §4-§5, §8, and Part II
  §13 are retained as rationale/backlog only where they do not conflict.
- The 1-3B monolith spike is retired. Models above the measured local envelope
  may be baselines or cloud routes, not silent target changes.
- E2 does not "replace/augment" the hand parser globally. Ownership moves one
  construction family at a time under §21.7.
- Atlas support does not grant capability. Only the capability manifest can
  enable a local handler or action.
- The claim "E4 has no world knowledge" is retired. The contract is that no
  unsupported model knowledge becomes an applied factual/policy/action claim.

---

# Part V — Hard parts and foundational contracts (2026-06-10)

The recurring failure to prevent is parallel, informal vocabularies: one
subsystem changes a label, another keeps the old meaning, and tests check the
label rather than behavior. The remedy is broader than a UOL schema:
**every cross-stage artifact has one owner, one version, one validator, and one
documented failure path.**

## 20. The five hardest builds

### 20.1 Stable meaning contracts

UOL, semantic classes, frames, atlas relations, evidence keys, and answer plans
will be embedded in databases, datasets, models, and reports. Casual additive
fields are therefore migrations, not harmless refactors.

Guard: the contract registry in §21.1, a small normative UOL set at M1, explicit
compatibility declarations, and startup refusal for incompatible artifacts.
The treebank grows to ~300 before M6, but M1 starts with 60 high-value examples
so contract work does not become a month-long annotation project.

### 20.2 Rendering without granting the model authority

Grammar constraints can enforce shape and copied tokens; they cannot prove the
meaning of unrestricted prose. A lexical verifier that checks only numbers,
dates, and proper nouns will miss claims such as "it is safe" or "you should
go outside."

Guard: factual/advice outputs are generated from typed claim slots in
`AnswerPlan`; unrestricted prose is allowed only in explicitly creative spans.
The verifier checks the plan, evidence, citations, required constraints, and
forbidden span classes. Failure discards the candidate and applies a template.
The safety metric is **unsupported applied claims**, not whether the raw model
ever attempted one.

### 20.3 Acquisition without poisoning or capability escalation

Definitions may be wrong, joking, ambiguous, or adversarial. Homonyms must
create senses rather than overwrite rows. Learned semantics must never mutate
policy, action types, handlers, or capability availability.

Guard: the state machine in §21.6, immutable reserved namespaces, closed and
versioned semantic classes, generated minimal pairs, conflict checks, explicit
provenance, and a separate release-controlled capability manifest.

### 20.4 Migrating from rules to learned experts without blending

Running two parsers or linkers is useful for comparison; combining their
outputs opportunistically creates behavior with no owner and no reproducible
threshold.

Guard: shadow outputs are telemetry only. A versioned ownership registry names
one active producer per construction/frame family. Promotion and rollback
change registry data, not scattered conditionals.

### 20.5 Calibrating local acceptance with scarce honest data

Top-1 frame accuracy is not the product metric. The costly error is accepting a
local route that lacks evidence, permission, or capability. Self-authored
fixtures cannot establish real-world thresholds.

Guard: E3 supplies scores, but K0 checks deterministic policy and completeness.
Thresholds are per family, versioned, calibrated on sealed/user-derived sets,
and optimized with asymmetric cost. A family remains rule-owned or non-local
until its acceptance precision is demonstrated.

## 21. Foundational contracts

### 21.1 Contract registry and compatibility

`melm/contracts/registry.v1.json` is the index and single owner map:

| Artifact | Canonical owner | Failure behavior |
|---|---|---|
| `UOLParse` | language-core | reject parse; clarify/fallback |
| semantic classes + relations | meaning-core | quarantine unknown IDs |
| lexical sense records | lexicon store | quarantine invalid write |
| frame definitions + candidates | frame registry | no accepted frame |
| family thresholds | evaluation/router | family stays rule-owned or non-local |
| capability manifest | kernel/policy | local handler disabled |
| `RouteDecision` | kernel/router | fail closed |
| `EvidencePacket` | membrane/evidence boundary | omit blocked item or reject route |
| `AnswerPlan` + `VerificationResult` | synthesis boundary | discard and template fallback |
| parser ownership | language-core | rules/default owner |
| model manifest | model runtime | model not loaded |

Every contract declares `schema_id`, semantic version, owner, producers,
consumers, validator, compatible predecessor versions, and migration command.
Unknown fields are rejected for safety-critical contracts. Additive changes
that alter interpretation require a new version. CI validates every fixture,
seed file, model manifest, and persisted example against the registry.

### 21.2 UOL v1

The schema records spans and uncertainty, not just convenient strings:

```text
UOLParse {
  schema_id, utterance_hash, tokenizer_id,
  producer: {id, version},
  clauses: [Clause],
  deixis: [{span, referent_id, confidence}],
  diagnostics: {coverage, unparsed_spans, warnings},
  confidence
}
Clause {
  clause_id, speech_act, polarity, tense,
  predicate: {span, lemma, sense_id?, semantic_class_id?,
              particles:[span], confidence},
  arguments: [{role, span, lemma, sense_id?, semantic_class_id?,
               lexicon_status, confidence}],
  modifiers: [{target_ref, span, lemma, modifier_type, confidence}],
  unknowns: [{span, normalized, candidate_class_ids, reason}]
}
```

Invariants:

- Spans use one tokenizer contract; repeated words are never map keys.
- `target_ref` names a clause, predicate, or argument ID; it is not an
  ambiguous label such as `"object"`.
- Multi-clause output is legal from v1. Unsupported material appears in
  `unparsed_spans`; no out-of-schema `unparsed_remainder` field is invented.
- Confidence is calibrated telemetry, not authority. Low confidence cannot be
  rescued by a downstream expert without a recorded decision.
- An unknown may constrain a candidate or trigger acquisition; it cannot by
  itself anchor a frame or grant a route.

The initial normative set covers simple commands/questions/statements,
negation, deixis, definitions, private-memory export, particle verbs,
coordination, constraint misses, and OOV words. Parser tests validate UOL;
frame tests validate UOL→candidate; a smaller end-to-end set validates the
whole path. Not every treebank sentence is expected to have a local answer.

**Language-agnostic UOL invariant.** UOL records lemmas (not inflected forms)
and `semantic_class_id` values (not surface strings). Inflection normalization
(`_to_lemma`) is a swappable function — the framework is language-independent;
only the normalizer is language-specific. Contracts store lemmas only, never
inflected forms. All runtime term resolution normalizes inflected tokens to
lemmas before UOL construction. This ensures the pipeline is language-agnostic:
changing the normalizer changes the source language without changing frames,
routing, or capability policy.

### 21.3 Meaning stores: lexicon, atlas, and capability

The earlier `lexicon_entries` sketch is split to preserve senses and forms:

```text
lexemes(lexeme_id, lemma, pos, language, reserved, frequency_rank, created_at)
word_forms(form_id, lexeme_id, surface, morph_features, provenance)
lexical_senses(sense_id, lexeme_id, semantic_class_id, concept_id,
               argument_template_id, confidence, status, provenance,
               source_event_id, created_at, last_used_at, use_count)
semantic_classes(class_id, parent_id, schema_version, policy_flags)
```

`atlas_edges` relate concepts and observations:

```text
edge_id, subject_concept_id, relation_id, object_concept_id,
polarity, strength, status, provenance, source_ref,
policy_scope, created_at, last_used_at, superseded_by
```

Atlas edges may support meaning, readiness, or evidence selection. They do
**not** enable handlers or actions. `capability_manifest.v1.json` alone maps a
frame family to an installed handler, permitted route types, evidence policy,
action type, and confirmation policy. It is release-controlled and
hash-attested; user teaching and ordinary learning-ledger promotion cannot
modify it.

### 21.4 Frame candidate and route decision

E3 emits candidates; K0 emits the decision:

```text
FrameCandidate {
  frame_id, frame_version, family,
  uol_refs, slot_bindings,
  score, score_components, producer
}

RouteDecision {
  decision_id, accepted_frame?,
  route: local_answer|local_tool|pending_action|cloud_handoff|clarify|reject,
  threshold_version, capability_manifest_version,
  checks: {frame_confident, capability_installed, evidence_complete,
           membrane_allowed, confirmation_satisfied},
  reason_codes, required_evidence, fallback
}
```

The local predicate is:

```text
local_eligible :=
    frame.score >= threshold[frame.family]
    AND capability_manifest.allows(frame.family, requested_route)
    AND evidence_policy[frame.family].is_complete(evidence)
    AND membrane.allows(evidence)
    AND action_policy.is_satisfied(frame)
```

Decision precedence is fixed and tested:

```text
policy reject
  > pending action / confirmation
  > clarify malformed or ambiguous meaning
  > local when every local check passes
  > typed cloud/fetch handoff when allowed
  > honest unsupported response
```

For a constrained story request, a local inventory item that ignores supplied
constraints is incomplete evidence. For weather, stale data is incomplete.
No debug label participates in this predicate.

### 21.5 Evidence, answer plan, and verification

`EvidencePacket` is produced after membrane admission:

```text
EvidenceItem {
  key, type, value, provenance, freshness,
  privacy_scope, content_hash, allowed_claim_types
}
EvidencePacket {
  packet_id, route_decision_id, admitted_items,
  blocked_item_keys, membrane_decision_id, packet_hash
}
```

E4 never receives blocked items, raw event history, policy internals, or
database access. The verifier/audit path retains the complete packet; the
renderer receives a derived view containing only `admitted_items` and no
blocked keys or blocked-item metadata. It also receives:

```text
AnswerPlan {
  mode: factual|procedural|creative_grounded|refusal,
  required_claims: [{claim_id, claim_type, evidence_key, copy_or_transform}],
  required_constraints: [typed constraint],
  allowed_free_spans: [opening|transition|creative_body],
  forbidden_claim_types,
  citation_policy, max_tokens, stance
}
```

Rules by mode:

- **Factual/procedural:** all substantive claims are typed slots backed by
  evidence or an approved static policy text. Free spans are bounded discourse
  glue, not open advice. Transforms come from a deterministic allowlist.
- **Creative grounded:** open generation is allowed, but requested characters,
  themes, length, and mechanically checkable safety constraints are verified.
  A required constraint that cannot be checked is out of scope or forces
  fallback. Generated details are not presented as retrieved facts.
- **Refusal:** reasons and alternatives come from route reason codes and
  capability/evidence state.

`VerificationResult` checks schema validity, packet/decision hashes, required
claim coverage, citations, constraint retention, blocked-token leakage, and
mode-specific forbidden claims. It does not pretend unrestricted semantic
truth can be proven cheaply. Therefore unrestricted factual/advice prose is
outside the v0.2 acceptance surface.

On failure:

```text
discard candidate
write synthesis trace with attempted-output hash and failure codes
render deterministic fallback from the same AnswerPlan
increment model_rejection_count
do not increment unsupported_applied_claims
```

Training targets must pass the same plan verifier before entering the corpus.
Empty-evidence knowledge probes must produce refusal or be rejected 100%.

### 21.6 Lexicon acquisition state machine

```text
observed_unknown
  -> candidate
  -> quarantined
  -> active
  -> defeated

candidate|quarantined -> rejected
defeated -> new candidate sense (never overwrite history)
```

Promotion requires:

1. Valid definition/source provenance and a known semantic class.
2. Reserved-namespace and policy checks.
3. No destructive collision with an active sense; homonyms create new
   `sense_id` values.
4. Generated positive/negative minimal pairs pass.
5. Explicit confirmation or the configured low-risk repeated-use rule.
6. A rollback record and source-event link.

Example: teaching `kalimba` may activate an instrument sense and improve media
frame binding, but it cannot install media playback, bypass inventory
completeness, or create a cloud/action permission. Redefining `weather`,
function words, frame anchors, action verbs, policy vocabulary, or semantic
class IDs is rejected and logged.

### 21.7 Parser and linker ownership

```text
construction_ownership.v1:
  imperative_simple:          rules@uol.v1
  question_copular:           rules@uol.v1
  definition_statements:      rules@uol.v1
  particle_verbs:             e2:model_hash@uol.v1
  coordination_multi_clause:  e2:model_hash@uol.v1
  default:                    rules@uol.v1
```

Both implementations may run, but only the owner output reaches E3/K0.
Disagreements write `parser_triage` telemetry. Promotion requires an
adjudicated family dataset, metric win, safety non-regression, compatible model
manifest, and rollback owner. There is no merge function and no time-based
"quiet for one week" gate.

The same rule applies to E3: learned ranking may own a frame family only after
its threshold artifact is promoted. Rule candidates remain fallback, not
unlabeled blended evidence.

### 21.8 Stance packet

```json
{"schema_id":"melm.stance.v1","verbosity":"short","caution":0.7,
 "warmth":0.4,"protectiveness":true,"confirmation_required":false,
 "derived_from":{"caution":["homeostatic.uncertainty","action_risk"],
                 "protectiveness":["privacy_risk"],
                 "warmth":["local_success_streak"]}}
```

Stance may change wording, length, and whether uncertainty is foregrounded. It
may not change route, evidence admission, claim set, action confirmation, or
verifier outcome. Frozen-state tests assert both sensitivity to cited inputs
and non-interference with those protected decisions.

### 21.9 Model and dataset manifest

Each model artifact declares:

```text
artifact_id, expert_stage, base_model, base_revision, tokenizer_hash,
weight_hash, quantization, runtime,
compatible_contracts, training_data_hashes, heldout_hashes,
license, training_command, evaluation_artifact,
predecessor, rollback_reason?
```

Startup refuses a model whose contract or tokenizer hashes do not match. CI
rejects train/held-out overlap by stable example IDs and content hashes.
Hardware reports identify the exact manifest; "same model family" is not
reproducible evidence.

### 21.10 Lexical ingestion contract — `sense_candidate.v1`

Every vocabulary source — batch seeders (WordNet, Wiktextract, VerbNet,
legacy code constants) and runtime channels (user teaching, offline
dictionary, cloud lookup) — emits the **same candidate JSON** and passes the
**same ingestion gate**. Source-specific logic lives only in thin adapters
that produce candidates; nothing else may write to the lexicon tables.

```text
SenseCandidate {
  schema_id: "melm.sense_candidate.v1",
  batch_id,                      # ingestion run, or runtime event id
  lemma, language, pos,
  source: {provenance, source_ref, retrieved_at, license},
  definition,                    # human-readable gloss (auditable)
  genus_lemma?,                  # extracted or source-given hypernym head
  semantic_class_candidates: [{class_id, method, confidence}],
                                 # class_id ∈ closed semantic_classes registry
                                 # method ∈ {supersense_map, genus_walk,
                                 #           verbnet_map, llm_assigned,
                                 #           seed_authored}
  argument_template_id?,         # verbs: from VerbNet→template map
  forms: [{surface, morph_features, provenance}],
  frequency_rank?,               # zipf class (wordfreq, MIT) or null
  relations: [{relation, target_lemma, target_sense_ref}],   # optional
                                 # atlas fodder (hypernym/holonym), quarantined
                                 # like any atlas write
  safety: {reserved_conflict: bool, policy_term_overlap: bool},
  suggested_status: active|dormant|quarantined,
  confidence_prior               # per-source prior, table below
}
```

**The ingestion gate** (one function, `lexicon_ingest(candidate)`):
schema-validate → reserved-namespace check → class_id ∈ closed registry →
genus resolvable in lexicon (or confidence cut) → collision check (same
(lemma,pos,class) merges provenance and takes max prior; *different* class =
new sense_id — polysemy is normal, never a conflict) → write
lexemes/word_forms/lexical_senses (+quarantined relation edges). Identical
behavior at build time and runtime; the poisoning suite tests this gate, so
the seeders are covered by the same defenses as user teaching.

**Per-source adapters:**

| Source | Adapter input | Class mapping method | Default status | Prior |
|---|---|---|---|---|
| Open English WordNet | synset: lemma, supersense (45 lexnames), gloss, hypernyms | `wn_supersense_map.v1.json`: 45 supersenses → MELM classes (noun.artifact→physical_object, noun.food→food_item, verb.communication→communicate-class, …); hypernym chain fills `genus_lemma` + relations | active if zipf ≥ 3.0, else dormant | 0.85 |
| Wiktextract (kaikki.org en) | entry: senses, **inflected forms**, POS | gloss head-noun genus_walk, cross-checked against WordNet | dormant unless WordNet-corroborated | 0.70 |
| VerbNet 3.4 | class members + thematic roles | `verbnet_map.v1.json`: curated map of the ~40 device-relevant classes → MELM verb classes + argument templates; rest → generic class, dormant | active for mapped classes | 0.85 |
| Legacy code constants (`_VERBS`, nominal domains, intent keyword sets) | hand-authored pairs | direct (`seed_authored`) | active | 0.95 |
| User teaching (runtime) | definition-frame parse | genus_walk through active lexicon | **quarantined always** | 0.60 |
| Cloud LLM lookup (runtime) | strict-JSON prompt, below | `llm_assigned`, but class_id must be in the closed enum AND genus must cross-check | **quarantined always** | 0.50 |

**spaCy's role — tooling, not a sense source.** spaCy contributes no senses;
it is (a) the ingestion-time generator of `forms` for candidates that lack
them (its MIT lemmatizer/morph rules run in the build pipeline, provenance
`spacy_rules`), (b) a POS sanity-check on candidates, and (c) later the E2
teacher (§17.4). This keeps "where meanings come from" and "where
morphology/syntax tooling comes from" cleanly separated.

**Cloud prompt contract** (the only sanctioned shape for LLM-sourced
vocabulary):

```text
system: You are a dictionary service. Output JSON only, conforming to the
        provided melm.sense_candidate.v1 schema. Return up to 4 senses for
        the word. semantic_class_candidates.class_id MUST be one of the
        enumerated class IDs provided. Definitions must be generic
        dictionary glosses; include no usage examples and no references to
        any person or device.
user:   {"word": "kalimba", "pos_hint": "noun",
         "class_enum": [...closed registry IDs...]}
```

Membrane rule unchanged: the bare word only, never utterance context.
**Polysemy resolution is therefore split: the cloud defines the word (all
senses, like a dictionary page); the device picks the sense** — at next use,
context selects whichever quarantined sense's class is compatible with the
bound frame slot, and only that sense proceeds toward promotion. Response
handling: JSON parse → full ingestion gate → on any validation failure the
response is discarded entirely (no partial salvage).

**Activation tiers** (precision control for a 100k+-sense lexicon on-device):

```text
active      participates in parsing and frame binding
dormant     resident in SQLite; consulted only when active lookup misses
            (one extra indexed query); promotion to active by frequency,
            device-domain class, or first successful use
quarantined never parses; §21.6 state machine governs promotion
```

Initial targets (measured, not assumed): active ≈ 25–40k senses (zipf ≥ 3.0
∪ device-domain classes ∪ VerbNet-mapped verbs); dormant = remainder of the
WordNet/Wiktextract import. The treebank ambiguity rate gates the threshold:
if activating breadth degrades parse precision, the zipf floor rises — a
data change, not a code change.

**Worked candidates, one per source family:**

```json
{"schema_id":"melm.sense_candidate.v1","lemma":"kalimba","pos":"noun",
 "source":{"provenance":"wordnet","source_ref":"oewn:kalimba%1:06:00",
           "license":"BSD-style"},
 "definition":"a Lamellophone of African origin; thumb piano",
 "genus_lemma":"piano",
 "semantic_class_candidates":[{"class_id":"physical_object.instrument",
   "method":"supersense_map","confidence":0.86}],
 "forms":[{"surface":"kalimbas","morph_features":"Number=Plur",
           "provenance":"spacy_rules"}],
 "frequency_rank":48211,
 "relations":[{"relation":"hypernym","target_lemma":"piano"}],
 "safety":{"reserved_conflict":false,"policy_term_overlap":false},
 "suggested_status":"dormant","confidence_prior":0.85}
```

```json
{"schema_id":"melm.sense_candidate.v1","lemma":"stream","pos":"verb",
 "source":{"provenance":"verbnet","source_ref":"verbnet:send-11.1?",
           "license":"CU Boulder"},
 "definition":"transmit media continuously",
 "semantic_class_candidates":[{"class_id":"verb.perform_media",
   "method":"verbnet_map","confidence":0.84}],
 "argument_template_id":"tmpl.agent_theme_medium",
 "suggested_status":"active","confidence_prior":0.85}
```

```json
{"schema_id":"melm.sense_candidate.v1","lemma":"kalimba","pos":"noun",
 "source":{"provenance":"user_taught","source_ref":"event:e_123"},
 "definition":"a small thumb piano from Africa",
 "genus_lemma":"piano",
 "semantic_class_candidates":[{"class_id":"physical_object.instrument",
   "method":"genus_walk","confidence":0.72}],
 "safety":{"reserved_conflict":false,"policy_term_overlap":false},
 "suggested_status":"quarantined","confidence_prior":0.60}
```

The user-taught and WordNet candidates for the same word illustrate the merge
rule: same (lemma, pos, class) → one sense, merged provenance, max prior —
the user's teaching *corroborates* the dictionary instead of duplicating it.

## 22. Drift catalogue wired to CI

| Drift | Detection / guard |
|---|---|
| Contract or label drift | Registry validation; compatibility tests; behavior gates ignore debug strings |
| Vocabulary returns to code | Audit router/parser constants; migrated families must load seeds from stores |
| Capability escalation through learning | Attempted lexicon/atlas writes cannot alter manifest hash or enable handler |
| Shortcut/demo drift | Scan prompts, grammars, and route code for fixture literals; held-out paraphrases |
| Unsupported model authority | Plan verifier + empty-evidence probes + `unsupported_applied_claims == 0` |
| Parser/linker blending | Only ownership registry may select producer; shadow output excluded from decisions |
| Acquisition poisoning | Reserved-word, homonym, contradiction, action/policy-definition suite |
| Data contamination | Sealed split hashes, provenance, overlap audit, external/user-derived labels |
| Scope drift | Packaging freeze until M3; milestone reports classify language-core vs packaging work |
| Integration/performance drift | Full-pipeline cold/warm p50/p95 TTFT, tok/s, RSS, fallback and rejection rates |

## 23. First implementation slices

1. **Truth slice:** D1 behavioral fix, CI, v1 supersession, clean baseline
   artifacts.
2. **Contract slice:** registry + `UOLParse`/`RouteDecision` validators and
   adapters around current output; no routing change.
3. **Authority slice:** `EvidencePacket` + `AnswerPlan` + deterministic
   verifier applied to current templates before any model is introduced.
4. **Meaning slice:** semantic classes, factored lexicon, capability manifest,
   and one media-family seed migration with bit-identical behavior.
5. **Learning slice:** red `kalimba` fixture through acquisition, restart,
   honest inventory miss, correction, and rollback.
6. **Expert slice:** run E4 feasibility in parallel; then E3 and E2 only after
   the contracts they consume are stable.

This order produces a useful, testable vertical result at each step and keeps
neural work replaceable. The first genuinely thesis-proving artifact is M3:
the system learns a word as meaning-bearing state, reuses it after restart,
changes no weights, grants no capability, and can undo the learning.

## 24. Vocabulary bootstrap mini-plan (executes inside M2–M3)

The focused, self-contained plan for building the initial vocabulary. Five
slices, each a day-to-days unit with a hard gate. Everything flows through
§21.10's single contract and gate; there is no other write path.

**V1 — Contracts and class registry** (with M1 contract work)

- `melm/contracts/sense_candidate.v1.json` (validator for §21.10).
- `melm/contracts/semantic_classes.v1.json` — the closed registry, seeded
  from three places and frozen at version: 45 WordNet supersenses (mapped,
  not copied), ~40 VerbNet-derived verb classes the device domain touches,
  ~12 MELM device classes (media_descriptor, food_item, weather_phenomenon,
  contact_person, …). Policy flags on classes that touch privacy/action
  surfaces.
- `melm/contracts/wn_supersense_map.v1.json`, `verbnet_map.v1.json`.
- Gate: registry validates; every class_id referenced by maps exists;
  enum frozen (additions = new version).

**V2 — Seed builder** (`scripts/build_lexicon_seed.py`)

- Downloads pinned releases — Open English WordNet (current annual release),
  kaikki.org English JSONL (one pinned monthly dump), VerbNet 3.4 — into
  `local_data/lexicon_sources/`, SHA-256s recorded in a data manifest.
  wordfreq (MIT) supplies zipf ranks.
- Runs adapters → `candidates.jsonl` → ingestion gate →
  `local_data/lexicon_seed.sqlite`.
- Gate: per-source counts report (lexemes/forms/senses); 0 reserved
  violations; collision report generated and reviewed; **deterministic
  rebuild** — same source hashes in, same DB hash out.

**V3 — Legacy migration through the same gate**

- `_VERBS`, `_KNOWN_NOMINAL_DOMAINS`, all 11 intent classifier keyword sets
  in `local_assistant_router.py`, `_secondary_meaning_hint_groups`,
  `_semantic_object_role_tokens`, auto-bio sub-frame vocabulary
  (`session_objects`, `summary_actions`, `event_object`), and all remaining
  inline vocabulary sets in helper functions are exported as `seed_authored`
  candidates (prior 0.95, status active) and ingested through the same gate.
  No special path: if the gate rejects a legacy entry, that is a finding, not
  an exception.
- Gate: **bit-identical routing** on the full regression suite with the
  parser/classifiers reading the store (M2's exit criterion); deleting a
  seed row changes behavior predictably. No vocabulary remains in code
  constants — every token a classifier checks comes from the store.

**V4 — Store-backed parsing with activation tiers**

- Lemmatizer reads `word_forms`; unknown-word path consults dormant tier on
  active miss; ambiguity-rate measurement on the normative UOL set tunes the
  zipf activation floor (data change only).
- Gate: treebank UOL parses unchanged for active vocabulary; lexicon lookup
  p95 < 5 ms on Pi-class hardware; ambiguity rate within the floor set at V1.

**V5 — Runtime channels join** (= M3 entry)

- User-teaching definition frames, the offline Wiktextract subset lookup,
  and the §21.10 cloud prompt all emit candidates into the same gate.
- Gate: `kalimba` fixture end-to-end (teach → quarantine → minimal pairs →
  promote → reuse after restart → correct → rollback); poisoning suite green
  — including against the *batch* adapters (a malicious source dump must be
  rejected by the same defenses as a malicious user definition).

Out of scope here, by design: E1 similarity fallback (optional M3 add-on),
E2 parser replacement (M6), anything that writes the capability manifest.