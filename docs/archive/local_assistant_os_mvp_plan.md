# Local Assistant OS MVP Plan

Date: 2026-06-07

Status: superseded by `docs/local_assistant_os_mvp_plan_v2.md`.

This v1 document is retained as historical v0.1 implementation context only.
For current execution order, defect priorities, milestone gates, and governance,
`docs/local_assistant_os_mvp_plan_v2.md` wins.

## Goal

Build `MELM Local Assistant OS v0.1` as a local assistant OS kernel:

```text
membrane policy
  + homeostatic state
  + autobiographical memory
  + user model
  + self model
  + opportunity planner
  + local inventory builders
  + budgeted evidence/state runtime
  + local/tool/action/cloud triage
```

The project direction does not change. The refinement is the underlying
structure: the assistant should behave less like a chatbot and more like a
personal immune/homeostatic cell inside the user's device.

It protects the local organism, recognizes self/user/outside boundaries,
remembers encounters, calls outside help only when needed, and prepares local
capability after repeated gaps.

The base loop is:

```text
observe event
  -> classify self/user/world meaning
  -> enforce membrane boundary
  -> update autobiographical memory
  -> project homeostatic state
  -> detect needs, risks, gaps, opportunities
  -> choose local/tool/action/fetch/cloud/clarify/reject
  -> act only through a typed gate
  -> remember outcome
  -> prepare future capability
```

Earlier shorthand:

```text
Autobiographical memory
  + user model
  + self model
  + opportunity planner
  + local inventory builders
  + budgeted evidence/state runtime
  + local/tool/action/cloud triage
```

This is stronger than a chatbot, a tool router, or a small model alone. It is an
assistant that remembers what the user needs, remembers what it has and has not
done, notices recurring gaps, and prepares local capability so future requests
need less cloud.

The assistant is framed as a white-blood-cell-like caretaker:

```text
self/non-self boundary
local-first response
memory of prior encounters
reinforcement calls only when needed
adaptive preparation
strict avoidance of autoimmune mistakes
```

In product terms, "autoimmune mistake" means exposing private facts to cloud,
taking an action without confirmation, inventing a memory, or blocking the user
because the assistant overgeneralized a safety rule.

This metaphor must compile to code, not vibe:

```text
cell membrane        -> MembranePolicy
surface receptors    -> intent/fact/action classifiers
internal state       -> HomeostaticState
memory markers       -> autobiographical memory + user facts
stress signals       -> risk, uncertainty, cloud dependence, stale cache
metabolism           -> budgeted evidence/state runtime
adaptive preparation -> OpportunityPlanner + inventory builders
effector behavior    -> typed local/tool/action/cloud gates
```

The fundamental object is therefore not "a chat loop." It is a protected local
stateful organism that uses chat as one interface.

Evidence labels in this plan:

```text
measured evidence       -> reproduced by CLI commands and/or tests
initial implementation  -> first-class code exists, with known scope limits
remaining gap           -> required before calling the v0.1 architecture complete
```

## Root Documentation Audit

Only the root `README.md` is allowed to steer current product work. The other
root Markdown files are retained as supporting evidence and now carry explicit
status guards on their first screen:

| Root doc | Current role | Drift guard |
|---|---|---|
| `README.md` | Landing page and routing contract. | Names v2 as authoritative, points to `pi-smoke`, and rejects model-first, tokenizer-first, or generic-chatbot drift. |
| `MELM_whitepaper.md` | Research thesis and validation ladder. | Supporting only; superseded if it conflicts with the Local Assistant OS plan. |
| `MELM_validation_implementation_plan.md` | Validation workstream map. | Workstreams must feed the OS kernel before becoming MVP direction. |
| `MELM_implementation_research_review_2026.md` | Historical de-risking review. | Model-first recommendations are background unless new measured evidence changes this plan. |

Root-level Word drafts are not active product docs; the legacy Word whitepaper
is archived under `docs/archive/`.

Current root-level drift scan scope: `README.md`, `MELM_whitepaper.md`,
`MELM_validation_implementation_plan.md`, and
`MELM_implementation_research_review_2026.md`. None of those files should add
new product goals unless the evidence is promoted into this plan.

## Evidence So Far

### Child-world architecture proof

The grounded child-chat micro-MVP proves the inner architecture:

```text
UOL parser -> semantic atlas -> typed frame -> bound plan -> state algebra
-> Memory OS -> evidence admission / projection -> SSM/attention boundary
```

### Experience-built frame non-regression rule

The local assistant ChatFrame is inherited from the `db-claw` / SemanticSQL
lesson: scoped language should flow into a typed frame first, then a bound plan
or bounded local answer. In SemanticSQL terms, `QueryFrame` is the bridge between
the natural-language request and the SQL the runtime is willing to render; its
intent-pattern library is only an additive bias layer. MELM must keep the same
discipline for chat, with one extra constraint: chat frames are primarily built
from experience.

A UOL parse is a candidate meaning graph, not yet a trusted frame. A ChatFrame is
accepted only when that candidate graph binds to the assistant's lived local
world: past chat sessions, referenced objects, user facts, self-state, action
state, local inventories, memory digests, and a growing assistant world atlas of
semantic/relational strengths. This keeps frames aligned with MELM's thesis that
meaning is learned from use and observation rather than from a static phrase or
lemma table.

The required order is:

```text
utterance
-> word extraction and weighted functional-role analysis
-> ranked predicate candidates and relation graph
-> UOL slots: subject, action, object, complement, recipient, source, target, modifiers
-> experience lookup: past sessions + referenced objects + user/self memory
   + local inventories + pending actions + world-atlas relation strengths
-> AssistantFrameRegistry match: frame_id, source policy, secondary-hint policy
-> ChatFrame: domain, route, capability gates, safety/memory/tool boundary,
   and cited experience/atlas support
-> answer, action, fetch, clarify, reject, or typed cloud handoff
```

The assistant world atlas is not a synonym for vocabulary. It is the local
semantic map built from what the assistant has observed, done, failed to do,
prepared, and repeatedly used. Example atlas edges include:

```text
user -> likes -> bedtime stories
story_gap -> caused -> build_story_inventory
weather_cache -> supports -> morning planning
trusted_contact -> requires -> explicit local setup
private_memory -> blocked_from -> external_cloud_model
career_goal -> lacks_local_frame -> typed_cloud_handoff
```

Grammar/UOL can infer that "can you help me grow in my career?" is a
`user -> request_help -> career_growth` relation. The frame decision must then
ask the atlas whether MELM has past experience, local inventory, policy support,
or a proven local capability for that relation. If not, the turn is understood
but cloud-bound or clarified; it must not become a fake local frame because the
words matched a goal/advice pattern.

The parser must run before capability routing. It may know what a request means
while also concluding that the device cannot answer it:

```text
understood + local evidence       -> local/tool/action frame
understood + no local capability  -> typed cloud handoff
insufficient predicate structure -> unknown/clarify or cloud interpretation
```

### Database growth and learning lifecycle

The assistant database should grow like a child building a local world, but with
an immune-system safety boundary. A raw conversation is not automatically
knowledge. It becomes usable knowledge only after it passes through typed
interpretation, ownership/policy, contradiction handling, grounding, and
promotion.

The durable growth loop is:

```text
chat/session event
-> UOL interpretation with confidence, unknowns, and candidate relations
-> frame attempt with cited session/object/action/memory context
-> outcome record: answer/action/cloud/clarify/reject/user correction
-> homeostatic update: uncertainty, trust, risk, local capability, mood/stance
-> atlas-edge proposal: subject/action/object/frame relation with provenance
-> quarantine if low confidence, contradicted, private, or externally uncertain
-> grounding: local evidence, user correction, repeated use, or bounded research
-> promotion into user fact, self-state, inventory, frame rule, or atlas edge
-> regression/minimal-pair evaluation before any router/SLM behavior changes
-> rollback/tombstone if later correction or contradiction defeats the edge
```

The database therefore needs four different memory classes, not one generic
"memory" bucket:

```text
event memory          raw turns, routes, answers, actions, and evidence keys
model memory          user facts, preferences, self-state, consent, policies
world-atlas memory    relational strengths between objects/actions/frames/events
learning memory       quarantined candidates, corrections, research, promotions
```

Learning the wrong thing must be a first-class state. If the user says "you got
that wrong," if a later event contradicts an older fact, or if internet/local
research fails to verify a learned relation, the system should record a negative
edge rather than silently delete the history:

```text
candidate relation: user -> prefers -> bananas
source: event_123
status: contradicted
defeated_by: correction_event_129
route_effect: cannot answer as stable preference
next_action: ask clarifier or keep as uncertain preference
```

This is how the assistant avoids pretending its first interpretation is truth.
Corrections are not just user facts; they are training signals for UOL,
ChatFrame construction, world-atlas edge weights, synthesis style, and future
abstention. They should lower confidence in the defeated interpretation, add a
negative/minimal-pair case, and require a new promotion gate before a similar
frame can route locally.

Grounding has the same quarantine rule. Internet or cloud research can propose
definitions, corpora, source candidates, and verification notes, but it must not
edit live grammar, atlas edges, frame rules, user facts, or router code directly.
Promotion requires source provenance, redaction status, policy status,
conflict/contradiction checks, held-out examples, and a rollback artifact. For
time-sensitive knowledge, the promoted edge needs freshness metadata and must
expire back to fetch/cloud/clarify when stale.

The assistant's "mood" or response stance is not decorative personality. It is
the verbal surface of homeostatic state: high uncertainty should produce
clarifying or cautious language; high privacy risk should become protective
wording; repeated successful local use can become warmer and more concise; high
action risk must become confirmation-seeking. A guided SLM may render that
stance, but the stance must come from state and frame evidence, not from an
unguarded free-chat prompt.

The white-blood-cell analogy maps to a concrete "strange detector":

```text
known self/user/world edge + low risk       -> local/cached/tool action
known edge + private boundary              -> block or ask consent
unknown predicate structure                -> clarify or cloud interpretation
known language shape + missing capability  -> typed cloud handoff/opportunity
contradiction or correction                -> quarantine/negative edge/relearn
stale external fact                        -> fetch/research before answering
```

The next implementation milestone should make this explicit in SQLite, not only
in prose: add first-class atlas and learning ledgers for candidate edges,
corrections, contradictions, research artifacts, promotion decisions, and
rollback state. Existing `events`, `user_facts`, `self_state`, `inventories`,
`opportunities`, `response_integrity`, and `improvement_candidates` are useful
raw material, but they do not yet prove a complete learning architecture.

`melm/appliance/functional_grammar.py` implements the first dependency-free
weighted functional layer. It ranks predicates, resolves conversational deixis,
and tracks subjects, objects, complements, recipients, possessors,
prepositional relations, modality, frequency, equivalence, and negation before
projecting UOL. Its weights are functional association biases, not sentence
matches. Debug output must expose predicate candidates, relation edges,
syntactic coverage, parse score, and semantic unknown tokens.

The seven-turn failure transcript now measures:

```text
before: structured=2/7, local=2/7
after:  structured=7/7, local=5/7
```

The two career turns remain cloud-bound because no grounded local career-advice
capability exists, but now parse as `want/help -> grow -> career`. Held-out
paraphrases and negative controls verify relation-level generalization rather
than a transcript phrase table.

Phrase or vocabulary tables may exist only as secondary meaning hints. They can
boost or explain noun, verb, analogy, idiom, or domain interpretation after the
candidate words are already grounded, but they must not create the route, object,
or action by themselves. A new capability must be expressed as reusable UOL slot
logic and ChatFrame transitions, not as a one-off string shortcut for a benchmark
or demo sentence.
For this MVP, the built-in secondary hint table is deliberately limited to
concept-token hints. Request-shaped phrases such as `who are you`, `what have
you done`, `about me`, or `talk to someone` must live in UOL/ChatFrame
composition coverage, not in secondary lexical tables.
Self-awareness composition should be maintained as identity/status frame logic,
not exact surface text: `who/what/how` operators, copula/modal relations,
assistant deixis, allowed scope/capability roles, and explicit task-frame
rejection must explain why `who exactly are you on this device` is local while
`who are you calling` is not an assistant-identity shortcut.
Bare fragments such as `your name` do not contain enough relation structure and
must stay unsupported unless a question or request frame supplies the missing
identity relation.

Implementation guard: the primary intent classifier and post-route UOL
slot/object helpers must not call phrase/marker-table or secondary-hint helpers
such as `_secondary_meaning_*`. Marker matching is allowed only in secondary
evidence, debug hints, requested-inventory resolution, or external source
matching. A primary route must come from token-role analysis plus UOL/ChatFrame
composition, and regression tests should fail if the classifier or slot helpers
start depending on phrase hits again.
In code, the current positive ownership boundary is `AssistantFrameRegistry`:
every accepted local, cached-tool, device-action, or private-boundary route must
expose `frame_registry=melm.assistant_frame_registry.v1`, a concrete `frame_id`,
`source_policy=primary_uol_chatframe_only`, and
`secondary_hint_policy=debug_only_never_primary_route` in parse-debug,
capability, transcript, and shortcut-audit evidence. The next architectural
cleanup must make this boundary explicitly experience-grounded by adding
world-atlas/event-memory support evidence to the same debug trace. A useful
answer without registry-owned frame evidence and atlas/memory support is still
drift because the route could have come from a hidden shortcut.
The same rule covers noun/verb shortcuts: bare domain words such as `story`,
`bedtime`, `doctor`, `medicine`, `dinner`, `naked`, `call`, or `family` are not
accepted behavior until token roles form a compatible question, request, object,
target, or safety frame.
Generic capability and status words must also stay subordinate to the task
object. `what can you do to improve my health` is a health-advice frame, not
self-identity. `can you explain cloud computing`, `can you explain story
structure`, and `can you explain weather systems` remain unsupported/open-domain
asks unless a real local capability frame is present; `cloud`, `story`, or
`weather` alone cannot create a runtime-status, story, or weather-cache route.
Weather and meal frames need the same discipline: `what is weather` and `how
does weather work` are concept questions, while `what is the weather` can use the
weather cache; `can you cook dinner` is not a local meal recommendation, while
`what can I cook for dinner` is a user-choice meal frame. Autobiographical recall
must be driven by shared UOL/ChatFrame scope composition, so `what was the last
thing I asked you` can recall the latest local event, but statements such as `I
dropped the last thing yesterday` must remain unsupported. The persistent kernel
must not own a separate exact recall-phrase table for this path.
Private-cloud export asks such as `send my favorite color to the cloud` must
surface through the Basic NLP -> UOL -> ChatFrame debug map as a memory-policy
route instead of bypassing the parser as a hidden pre-parse shortcut.
The accepted parse is a private memory export boundary frame:

```text
user / send / facts.favorite_color
  source=local_memory
  target=external_cloud_model
  pattern=request_private_memory_cloud_boundary
  policy=private_memory_requires_boundary_gate
```

It is a regression if that request becomes a generic `recall / user_profile`
frame or if the router applies the cloud policy before the primary UOL
classifier has identified a `personal_memory` frame.

For example, `"who are you"` must decompose as `who` = identity
interrogative, `are` = copula/state relation, and `you` = second-person deixis
resolved to the assistant in conversation context. The UOL projection is
`assistant / identify / self_model -> user`; the ChatFrame route is local because
the `self_model` source is available. It is a regression if this appears as
a `who are you` phrase-level primary route instead of compositional UOL evidence.

The same evidence shape now applies beyond identity: story, weather, safety,
media, health, memory, meal, and trusted-contact turns must expose
`slot_role_relation` decomposition in `parse-debug`, with primary routing basis
items such as `composition:question_weather_cache` or
`composition:command_trusted_contact` plus a `frame_id` such as
`weather.question_weather_cache` or `social_contact.command_trusted_contact`.
Phrase tables may still appear as
`secondary_meaning_hints`, but those hints must not be the primary route.
Debug output must keep `primary_domain_evidence` separate from
`secondary_domain_hints` so a phrase-table hint cannot masquerade as the accepted
UOL/ChatFrame path. It must also expose `secondary_hint_policy=
debug_only_never_primary_route` and `secondary_debug_hints` rather than any
route-adjacent secondary field. Secondary domain hints are not allowed to erase
semantic uncertainty. `"what is a story"` is now an understood open-domain
copular frame, not a local story request; `story` remains visible in
`semantic_unknown_tokens` for research/integrity scoring while the primary route
stays a typed cloud handoff.

The same rule applies to memory ownership. `"my child's school"` must resolve to
owned local evidence such as `facts.child_school` with `child_local` scope. It is
a regression if child, household, or routine facts collapse into generic
`profile.age`, `facts.school`, or phrase-level route flags.

The same rule also applies to topic nouns: `music` in "explain music theory" is
not a media action, and `phone` in "I bought a phone" is not a contact action.
They become actionable only when a compatible UOL action frame and target are
present, such as `play/song` or `phone/<profile-backed trusted contact>`.
Exact personal names must be loaded from `UserModel`/trusted-contact memory for
the active profile, not from a global classifier name list.

### Drift risks that remain

The current plan is strongest when it treats language, memory, state, and action
as one local organism. It can drift if any component is allowed to impersonate
the whole organism. These are the major failure modes to audit against:

| Drift risk | Why it breaks MELM | Required guard |
|---|---|---|
| Grammar becomes the frame owner. | UOL role parsing can understand a sentence shape, but it cannot prove the assistant has lived context or capability for that frame. | Accepted frames must cite atlas/event/user/self/inventory/action evidence, not only token roles. |
| World atlas becomes a vocabulary table. | A table of nouns/verbs is static knowledge, not learned relational strength from use. | Atlas edges need source event IDs, confidence, frequency/recency, policy, and negative evidence. |
| User correction becomes blind truth. | Users can misspeak, joke, or correct only one local context. | Corrections should defeat specific edges, add uncertainty/minimal pairs, and require confirmation when scope is unclear. |
| Internet research becomes live mutation. | Cloud or web output can be wrong, stale, unsafe, or privacy-sensitive. | Research artifacts stay quarantined until source, redaction, contradiction, eval, and promotion gates pass. |
| SLM renders outside the frame. | A fluent local model can hallucinate unsupported memories or actions. | Guided SLM input must be compact UOL + frame + state + admitted evidence; blocked/cloud/action routes refuse free synthesis. |
| Mood becomes personality decoration. | Random warmth/caution does not reflect the system's actual safety or uncertainty. | Response stance must derive from homeostatic state, action risk, privacy risk, trust, and frame confidence. |
| Memory digest replaces provenance. | Summaries can hide the original event that supports or defeats a claim. | Digests must cite source sessions/events and remain local-only; important facts need explicit keys and policy metadata. |
| Repeated failures are ignored. | The assistant will keep paying cloud/clarification cost for predictable needs. | Opportunity planning must convert repeated gap edges into setup requests, inventory jobs, or research candidates. |
| Negative evidence is discarded. | The assistant will relearn mistakes after deletion or contradiction. | Store tombstones, defeated edges, stale facts, and rejected promotions as durable learning memory. |
| Benchmarks become scripted demos. | Static traces can be overfit without proving adaptive behavior. | Use imported/redacted or live sessions, no stored answer/route expectations, and same-UOL minimal-pair regressions. |
| "Local first" becomes "local always." | The assistant may answer unsupported, stale, medical/legal, or open-domain questions locally. | Unknown, stale, high-risk, or no-capability frames must clarify, fetch, block, or hand off. |
| Database growth becomes unbounded hoarding. | A child learns patterns; it does not stuff every utterance forever into prompts. | Keep event ledgers queryable, compact into cited digests/atlas edges, preserve deletion/consent, and budget evidence. |

Secondary lexical evidence must be token-sequence bounded, not raw substring
matching: `weather` must not create an `eat` hint, `yesterday` must not confirm
an action because it starts with `yes`, confirmation target extraction must not
recover `play` from inside `replay` or `display`, `play chess` must not become
media playback from the bare verb `play`, and `bring my phone` must not become a
call request from the noun `phone`. Lifecycle, transcript, CLI, and browser probes
must exercise the same kernel setup extraction and router path; they must not
carry shadow parsers that special-case scenario words such as locations,
contacts, or story themes outside the kernel.

Generated capability probe:

```text
cases=50
accepted=22
answered=4
abstained=11
rejected=13
```

### Context-budget proof

Long story/history does not need to enter model context.

```text
Where is the red block?, 128 moves:
  raw transcript=5617 chars
  compact payload=319 chars
  compression=17.61x

Positive evidence check, 128 moves:
  unbudgeted payload=4053 chars, 64 attended events, 1.39x
  budgeted payload=323 chars, 1 attended event, 64 matching events, 17.46x
```

### Realistic assistant routing proof

Eight realistic user asks:

```text
Tell me a story.
What is the weather today?
Should I go to school dressed naked?
Play a song for me.
What do you think I should do to improve my health?
Tell me something about myself.
What do you think I should eat today?
I need to talk to someone.
```

Measured comparison:

```text
strategy                    local/device  cloud  fetch  clarify  privacy  memory
memory_centric_local_triage 8/8           0      0      0        0        7
thin_tools_plus_cloud       4/8           3      0      1        3        3
cloud_first_assistant       2/8           5      1      0        3        2
secondary_lexical_baseline  1/8           6      0      1        0        0
```

`secondary_lexical_baseline` is intentionally a secondary-lexical baseline. It
must not borrow `_classify_intent` / UOL / ChatFrame composition; otherwise the
comparison stops measuring static vocabulary expansion against the assistant OS
architecture.

The stronger transcript-level comparison now runs over the same authored
25-user-turn replay fixture that exercises identity, profile facts, weather,
story inventory, media/contact setup, private-cloud blocking, status, and
long-horizon recall:

```text
strategy                         local/device  cloud  fetch  clarify  privacy
memory_os_kernel_with_learning   17/25         4      1      2        0
local_state_router_no_lifecycle  7/25          11     2      5        5
thin_tools_plus_cloud            1/25          15     2      7        15
cloud_first_assistant            0/25          18     2      5        4
secondary_lexical_baseline       1/25          21     0      3        0
```

Measured win over the best static baseline:

```text
local_resolution_rate_gain=+0.40
cloud_handoff_reduction=7
clarification_reduction=3
profile_updates_advantage=2
private_cloud_blocks_advantage=1
long_horizon_digest_advantage=1
```

### Self-model learning proof

The new assistant OS kernel remembers repeated story misses and creates an
opportunity to build a local story inventory:

```text
before_route=cloud_handoff
cloud_handoffs_before=3
opportunity=build_story_inventory
executed_job=build_story_inventory
story_inventory_count=3
after_route=local_answer
cloud_handoffs_after=0
```

This is the strongest signal: the assistant can improve future local capability
from its own autobiographical memory.

### Realistic lifecycle proof

The lifecycle probes now have three levels:

1. a 17-step cold-start scenario used for database/resource evidence; and
2. a 3-scenario / 34-turn multi-profile suite used for broader architecture
   evidence; and
3. a 37-turn household-week trace used to pressure-test the full loop across
   setup, memory, inventory, action, offline, privacy, cancellation, and digest
   behavior.

The cold-start scenario covers onboarding, school morning, repeated bedtime
story requests, next-day personal memory, meal advice, contact action,
confirmation, and offline behavior.

Measured result:

```text
steps=17
local_or_device=12
local_resolution_rate=0.706
cloud_handoffs=3
external_fetches=1
blocked_offline=1
confirmations_required=1
actions_executed=1
jobs_executed=refresh_weather_cache, build_story_inventory
story_before=cloud_handoff
story_after=local_answer
cloud_story_handoffs_before_inventory=3
inventory=stories:3, weather_days:3, contacts:1
```

Lifecycle outline:

```text
day 0: user gives age, location, story interests, trusted contact
day 1: weather cache miss -> refresh_weather_cache
day 1: school clothing advice uses cached weather
day 1: unsafe naked-school question answers locally
day 1: three story requests go to cloud because inventory is empty
day 1: reflection creates and executes build_story_inventory
day 2: story request routes locally
day 2: "tell me about myself" uses profile memory
day 2: food suggestion uses weather cache and food inventory
day 2: contact request creates a pending device action
day 2: confirmation executes call action
day 3 offline: story still works locally
day 3 offline: latest-news request is blocked/clarified instead of faked
```

This is the first genuinely end-to-end signal. It shows the OS kernel handling
missing state, cache creation, compounding inventory, personal memory, safety,
action confirmation, and offline limits in one lifecycle.

The multi-profile lifecycle suite extends that proof beyond one child profile:

```text
scenarios=3
steps=34
local_resolution_rate=0.647
cloud_handoffs=3
external_fetches=1
clarifications=7
blocked_offline=3
confirmations_required=3
actions_executed=3
opportunities=build_story_inventory, refresh_weather_cache,
              build_media_index, ask_routine_memory,
              ask_household_memory, request_trusted_contact
safety_flags:
  cloud_private_inclusions=0
  unconfirmed_executed_actions=0
  fake_latest_news_local_answers=0
  low_quality_applied_synthesis=0
  dangling_memory_links=0
```

Suite scenarios:

```text
child_cold_start_story_weather_action_offline
adult_media_routine_household_setup
elder_sparse_offline_contact_story
```

This is the stronger current architecture signal. It shows the same OS kernel
handling cold-start learning, local media setup, routine-memory gaps,
household-memory gaps, trusted-contact setup, privacy rejection, confirmation,
and offline tool/cloud unavailability without static capability claims. Routine,
household, and contact setup opportunities now record setup requests; future
answers/actions change only after the user supplies the routine, household
context, or trusted contact.

The household-week trace adds a longer single-life proof:

```text
steps=37
local_resolution_rate=0.676
cloud_handoffs=3
external_fetches=1
clarifications=7
blocked_offline=1
confirmations_required=3
actions_executed=2
memory_digest=session_count:6, event_count:33, local_only:true
architecture_checks=11/11
safety_flags:
  cloud_private_inclusions=0
  unconfirmed_executed_actions=0
  fake_latest_news_local_answers=0
  low_quality_applied_synthesis=0
  dangling_memory_links=0
  action_replay_blocks=1
  cancelled_pending_actions=1
  consent_revocations=1
```

It proves the same kernel can notice empty household/routine/contact/media/story
state, create setup or inventory opportunities, change future routing only after
real local evidence arrives, use a typed dry-run action executor, block private
conversation export, refuse offline latest-news fabrication, and answer a
detailed "last few days" question from a local memory digest.

### Structural Gate Implemented

The lifecycle probe no longer merely implies membrane/homeostasis through
routes and reasons. v0.1 now persists them as first-class code objects.

Do not add assistant skills unless each route continues to emit these records:

```text
MembraneDecision:
  route
  allowed/blocked
  boundary crossed
  personal facts included/excluded
  confirmation requirement
  reason

HomeostaticState:
  privacy_risk
  cloud_dependence
  local_capability
  uncertainty
  cache_freshness
  action_risk
  inventory_coverage
```

Otherwise the project drifts into a capable but unsafe tool router.

## Reviewed Architecture Evidence Map

Each component below has been rechecked against the current runnable code,
tests, and CLI probes. The strongest evidence is the 17-step cold lifecycle,
the 3-scenario / 34-turn lifecycle suite, the 37-turn household-week trace, the
2-scenario / 29-turn open trace gate, the 25-turn transcript replay gate, the
105-case/12-profile eval, `pi-smoke`, `synthesis-variant-smoke`,
`synthesis-stress-smoke`,
the resource report, the scheduled-refresh metadata import smoke, the local
media manifest/directory importer regressions, and the chat-native
autobiographical recall/session/digest regression tests.
The standalone `dataset-audit` gate now validates the seed/source fixtures,
their hashes, source coverage, open traces, transcript replay fixture, and
SQLite bootstrap before the runtime and bundle gates claim readiness.

| Component | v0.1 role | Current evidence | Remaining gap |
|---|---|---|---|
| `MembranePolicy` | Decide what crosses local/device/tool/cloud boundaries. | `melm/appliance/assistant_os_kernel.py`, `assistant_os_store.py`, and `assistant_dashboard.py` persist one `MembraneDecision` per turn. `eval --json` currently shows `105` membrane decisions, `10` blocked private-cloud routes, `0` cloud private inclusions, `12` confirmation-required actions, and `0` unsafe local actions. Tests cover private-cloud blocking, parent/child privacy, action replay, cancellation, consent revocation, stale-cache exclusion, invented confirmation targets, and blocked cloud export of prior conversation memory. `autoimmune-smoke --reset --json` now packages those boundaries into a compact 26-turn SQLite/kernel gate with `34/34` checks: private-cloud block, generic cloud handoff without private evidence, explicitly shareable `facts.public_profile` cloud allowance only when its stored policy is `consent=true`, `local_only=false`, and `cloud_eligible=true`, mixed public-profile plus private household cloud blocking without partial leakage, policy-indexed dashboard accounting that keeps `cloud_private_inclusions=0`, household/shared-device memory cloud blocking, household and personal consent revocation with reload, child-owned age/school setup, child-school revocation without generic `facts.school` fallback, stale-cache exclusion, contact and media confirmation target mismatch, cancel/replay protection with no final pending action, parent/child private-cloud block, child-location private-cloud block, prior-conversation export block, and clean blocking safety flags. | Add richer multi-user consent and parent/child ownership cases from longer real transcripts. |
| `HomeostaticState` | Track privacy risk, cloud dependence, local capability, uncertainty, cache freshness, action risk, trust, and inventory coverage. | The cold lifecycle logs `17` snapshots for `17` turns; the lifecycle suite logs matching homeostatic records across `34` turns and `3` profiles; the household-week trace logs `37` matching membrane/homeostasis rows while preserving `0` missing ledger records; the open trace logs `29` turns through the same membrane/homeostasis path. The eval logs `105` snapshots and dashboard averages/max values. `resource-report` confirms this runs on stdlib Python plus SQLite. `schedule-refreshes` and kernel reflection now read recent homeostatic averages/deltas when setting priority. | Compare pressure choices against user-derived transcripts and expand the state model as new integrations appear. |
| `WeightedFunctionalGrammar` | Bootstrap UOL decomposition by parsing reusable grammatical and semantic relations before deciding whether the device has a capability. This is an internal UOL method, not a separate frame owner. | `melm/appliance/functional_grammar.py` emits `melm.weighted_functional_grammar.v1` with speech act, subject, ranked predicates, object, complement, recipient, possessive/prepositional relations, modifiers, syntactic coverage, semantic unknowns, and parse score. The original seven-turn failure transcript improves from `2/7` structured turns and `2/7` local turns to `7/7` structured and `5/7` local: greeting and assistant-behavior questions become grounded local self-model frames, while both career turns remain typed cloud handoffs with `want/help -> grow -> career` UOL. Held-out paraphrases and negative controls pass, the source contains none of the transcript phrases, and `shortcut-audit --json` enforces that boundary. | Rename/refactor this into a first-class `UOLDecomposer`, move lemma/role/concept inventories into data-backed sources, migrate every older bounded domain recognizer to consume the shared UOL relation graph, evaluate against a substantially larger user-derived dependency/minimal-pair corpus, calibrate relation weights, add coordination/coreference/tense scope, and compare accuracy/latency against a compact proven parser before claiming broad English coverage. |
| `AssistantWorldAtlas` | Build accepted frames from accumulated assistant experience: past sessions, referenced objects, user/self memory, inventories, pending actions, policy boundaries, and semantic/relational strengths. | Current evidence exists as fragments rather than a first-class module: SQLite `events` link sessions, `memory_digest.*` compacts long-horizon threads/capability transitions/open loops, `user_facts` and `self_state` preserve local user/self context, inventories store story/weather/media/contact/food/setup objects, and opportunity jobs record repeated gaps. These already prove frame behavior can change after experience: story asks move from cloud to local after inventory import, weather moves from fetch to cache after refresh, contact/media asks move from clarify to confirmation after local setup, and private-memory cloud export stays blocked by remembered policy. | Create a first-class atlas runtime/index that records relation edges with source event/inventory/fact keys, confidence/recency/frequency/policy metadata, and negative capability evidence. Make `AssistantFrameRegistry` require atlas or cited memory support for accepted local/tool/action/private-boundary frames, expose atlas support in parse-debug, and add tests where the same UOL parse changes route only when new experience creates or removes atlas support. |
| `LearningLedger` | Preserve the full lifecycle from interpretation to correction, contradiction, quarantine, research, promotion, and rollback. | Initial pieces exist: `response_integrity` scores every turn, `session_improvement_consent` gates opted-in learning, `improvement_candidates` queues low-confidence turns without live mutation, consent revocation creates durable fact tombstones, and transcript calibration drops static answer/route fields. | Add durable tables/artifacts for candidate atlas edges, user corrections, contradiction links, defeated interpretations, research artifacts, promotion decisions, regression fixtures, and rollback state. Promotion must require provenance, policy, held-out/minimal-pair tests, and no direct LLM/router mutation. |
| `GroundingResearchAdapter` | Verify or expand learned knowledge through bounded local/web/cloud research while protecting privacy and freshness. | Current queue records research topics and keeps `cloud_export_allowed=false`; inventory importers already preserve source/license/provenance and failure observability for story/weather/media sources. | Implement a redaction-bound research path that can gather source candidates, compare claims, mark stale/time-sensitive facts, and write quarantined research artifacts only. No researched claim becomes a user fact, atlas edge, grammar rule, or frame transition until it passes promotion/evaluation gates. |
| `GuidedResponseState` | Convert homeostatic state into response stance/temperature for the local SLM/verbalizer. | `HomeostaticState` persists privacy risk, uncertainty, local capability, cache freshness, action risk, trust, and inventory coverage; `BoundedLocalSynthesizer` already refuses blocked/cloud/action routes and cites evidence. | Add an explicit response-state packet derived from frame confidence, uncertainty, privacy risk, action risk, trust, age/profile, and recent corrections. The guided SLM should receive this packet plus UOL/frame/evidence, and should be allowed to emit words, a short phrase, silence, clarification, or action proposal only inside the selected frame. |
| `ResponseIntegrityAndImprovementLoop` | Measure what the assistant understood, whether the response was grounded, and which opted-in turns deserve research without allowing online mutation. | `melm/appliance/assistant_integrity.py` computes separate understanding and response-integrity scores. Understanding combines syntactic and semantic unknown-token coverage, UOL parse score, primary composition strength, resolved intent, and route agreement. Response integrity combines route confidence, synthesis/evidence grounding, route discipline, and membrane/privacy integrity. SQLite persists every assessment in `response_integrity`; browser sessions carry stable local session IDs; `session_improvement_consent` records explicit opt-in; and only opted-in research-recommended turns enter `improvement_candidates`. `/ask`, the browser debug frame, CLI chat, the dashboard, and `improvement-queue --json` expose the score, components, flags, topics, consent, and queue state. Regression tests prove a fully understood identity turn is reliable and not queued, an unknown `quasar/algebra/zorbulator` turn is scored separately from its safe cloud handoff and queued only with consent, and revocation moves queued work to `consent_revoked`. The queue declares `live_router_mutation=false` and `cloud_export_allowed=false`, preserving the db-claw/UOL rule against phrase shortcuts. | Add a redaction-bound external research adapter, candidate corpus schema, held-out minimal-pair and lifecycle evaluation, signed artifact binding, and explicit promotion/rollback. No LLM proposal may directly edit the live vocabulary, frame registry, grammar weights, or router. |
| `UserModel` | Store local facts with source, confidence, consent, and local-only scope. | SQLite `user_facts` load into the profile; consent tombstones prevent revoked facts from reloading. Facts now carry explicit `local_only`, `cloud_eligible`, and `scope` policy metadata. Existing seed/profile facts remain `private_local`, revoked facts clear cloud eligibility while preserving existing scope or deriving child/household/routine scope from the fact key, and profile sync preserves an explicitly shareable fact's policy instead of flattening it back to local-only. Child facts such as `facts.child_age` and `facts.child_school` persist with `child_local` scope, recall through owned evidence keys, reload from SQLite, and block cloud export without falling back to generic `profile.age` or `facts.school`. Regressions now prove revoking `facts.child_school` leaves `facts.child_age` usable, keeps the child-school tombstone scoped as `child_local`, and makes later child-school recall clarify instead of using generic `facts.school`. Membrane tests prove private facts are excluded from cloud while a user-approved `facts.public_profile` can cross only when `consent=true`, `local_only=false`, and `cloud_eligible=true`. Eval includes `4` consent revocations and current checks preserve `0` invented user-memory answers. The transcript replay caught and fixed a real weakness where "I am 8 years old" and "I live in Lagos" were cloud handoffs; those now route locally as `personal_memory / profile_update` and persist to `profile.age`/`profile.location` before later memory/status/digest use. | Add richer multi-user provenance and scoped ownership for more household members beyond current child/household/routine policy metadata. |
| `SelfModel` | Track assistant name, purpose, local limits, inventory counts, runtime status, and job outcomes. | Identity questions such as "Who are you?" and "What is your name?" now route locally as `assistant_identity` with `self_model.name`, `self_model.purpose`, `self_model.local_capabilities`, and `self_model.limits` citations instead of falling through to cloud. That route must be proven by UOL composition (`who/what` interrogative + copula + second-person/possessive deixis -> `assistant / identify|name / self_model -> user`), not by a static phrase hit. Runtime-status questions such as "What have you done so far?", "What do you need next?", and "Are you using cloud?" route locally as `assistant_status`; answers cite `self_status.counts`, `self_status.routes`, `self_status.safety_flags`, `self_status.self_observation`, and `self_status.next_steps` from the persisted dashboard/ledger. Repeated story misses update self-state enough to create `build_story_inventory`; restart/reload preserves inventory counts and executed jobs. Self-state now counts story models, weather days, contacts, media items, routine facts, and household facts. `runtime_health_trends` and bounded `runtime_health_history` are persisted in SQLite `self_state` after turns, reflection, job execution, direct story/media/weather imports, and digest/bootstrap paths; they record local-resolution rate, cache readiness, queued/completed/failed jobs, importer cycles/quality, synthesis warnings, action health, safety cleanliness, next observed needs, local-resolution deltas, cache-gap persistence, and weather/story readiness transitions. Kernel reflection and `schedule-refreshes` now include self-observation history pressure in story/weather priority signals. Regressions prove a cold weather miss stores `weather_cache=missing`, queued refresh pressure, history points, and a cited status answer; `refresh-weather` and `run-jobs` then flip persisted cache health to ready and status reports `weather_cache_transition=ready`. The 29-turn open trace validates the same trend pressure under a messier session: weather miss-to-hit, story cloud-to-local, media/contact setup-to-action, `history_points=24`, and priority signals with `weather_cache_gap_persistence=1.0` and `story_inventory_gap_persistence=1.0`. The dashboard reports job priority, retryable queued work, importer health, source coverage, and story-quality floor compliance from persisted job/inventory ledgers. | Validate trend weights against real user-derived transcripts and longer household traces. |
| `AutobiographicalMemory` | Record every meaningful turn as structured experience for reflection and bounded recall. | SQLite `events` records utterance, intent, route, answer, reason, flags, evidence keys, job/action effects, `session_id`, `previous_event_id`, and `next_event_id`. The cold lifecycle dashboard reports `linked_previous=16`, `linked_next=16`, and `dangling_memory_links=0`; the 34-turn lifecycle suite reports `dangling_memory_links=0` across child, adult, and elder scenarios; the household-week trace samples `33` events across `6` sessions into a local-only memory digest and still reports `0` dangling links. Reload tests prove later sessions link back to prior events. `memory-replay` queries linked event memory by text, intent, route, session, or bounded recent-session windows; dashboard memory summaries include recent sessions and per-session intent/route counts. Recent-session chat summaries now keep cited `events.*` evidence grouped by session while extracting capability transitions, open local gaps, action state, and boundary controls from the same event reasons/routes. `memory-digest` compacts bounded multi-session history into a local-only `memory_digest.*` inventory row with remembered threads, per-session summaries, capability transitions, active limits, open loops, and deterministic quality scoring over local-only discipline, long-horizon coverage, event density, thread coverage, session-summary coverage, key moments, resolution awareness, and intent/route diversity. The household-week digest scores `1.0` over the `0.72` floor with no warnings, while thin one-event digests fail with `insufficient_long_horizon`. Long-horizon chat recall cites that digest instead of raw event keys and now verbalizes transitions such as story cloud-to-local, weather miss-to-cache, media/contact setup-to-action, privacy blocking, offline non-fabrication, and revoked routine memory. Chat-native recall answers "what did we talk about," "what was my last question," "summarize our recent sessions," and "what happened over the last few days" from local-only evidence, and conversation-memory cloud export is blocked. | Evaluate digest-quality scores on messier user-derived household transcripts. |
| `OpportunityPlanner` | Convert repeated gaps into background preparation. | Three story cloud handoffs create `build_story_inventory`; weather cache miss creates `refresh_weather_cache`; profile-memory empties create `ask_profile_memory`; missing-contact turns create `request_trusted_contact`; empty media asks create `build_media_index`; routine-memory gaps create `ask_routine_memory`; household-memory gaps create `ask_household_memory`. `schedule-refreshes` queues Pi-budgeted `import_story_metadata` and `refresh_weather_cache` jobs for thin/stale inventory. `run-jobs` now executes weather refresh jobs through the Open-Meteo-shaped weather adapter, defaulting to the bundled offline fixture and allowing live Open-Meteo HTTP with `--weather-live`. Scheduled refresh and kernel reflection priorities include gap counts, recent homeostatic averages/deltas, failed job counts, local-capability pressure, and expected local-resolution gain. Regressions show trusted-contact pressure outranking profile-memory (`0.799` vs `0.691`) and media/routine/household opportunity signals on true cold start. The 34-turn suite, 37-turn household-week trace, 29-turn open trace, and `setup-integration-smoke --reset --json` exercise all current opportunity classes: story, weather, media, routine, household, and trusted contact; the setup smoke proves cold setup gaps create setup requests without fake facts, and later routes change only after explicit user-supplied routine, household, and trusted-contact statements. | Validate priority choices against real user-derived traces and implement real integration policies beyond the current local setup request/action proof. |
| `LocalInventoryBuilders` | Build local story, weather, food, contact, media, and routine/household setup inventories. | Lifecycle creates `stories=3`, `weather_days=3`, `contacts=1`. Offline replay importers ingest Project Gutenberg CSV and Internet Archive search metadata with source/license provenance. `dataset-audit --reset --json` validates the seed, public-domain story metadata, local media manifest, Open-Meteo-shaped weather fixture, Gutenberg CSV, Internet Archive JSON, open-trace fixture, and transcript replay fixture with SHA-256 file reports; it also verifies `4` story metadata items, `3` local media items, `7` weather days, `2+` Gutenberg story candidates, `2+` Internet Archive story candidates, `29` open-trace turns, `25` transcript replay user turns, and a SQLite bootstrap profile with `6` user facts and `8` inventories. Imported story payloads now persist `topics`, `cultures`, `quality_score`, `local_fit_score`, and `metadata_quality`. `refresh-weather --offline-json benchmarks/sample_open_meteo_forecast.json --json` replays an Open-Meteo-shaped forecast into weather inventory rows with source/license/provenance; `--live` uses stdlib HTTP against Open-Meteo. A cold weather ask first routes to `external_fetch / weather_cache_miss`; after refresh, the next ask routes to `cached_tool / weather_cache_hit` with cited `weekly_weather.today`. `import-media` ingests `benchmarks/local_media_manifest.json` or a scanned local media directory into SQLite with `local_device` provenance, tags, paths, and path-existence metadata. `build_media_index` now uses that manifest inventory path instead of hardcoded media names; the next media ask changes from clarify to a gated device action. Routine, household, and trusted-contact setup opportunities persist `setup_request` inventory rows instead of writing fake facts; only explicit user statements such as "My morning routine is..." or "Ada is my trusted contact..." store local facts/contacts and change later routes. `setup-integration-smoke --reset --json` proves this full loop through the real kernel/store/action path: cold routine/household/contact gaps create user-value-required setup requests, no fake fact/contact appears before setup, explicit routine/household/contact statements create scoped local memory/contact rows, later memory recalls become local answers, and trusted-contact calling remains confirmation-gated. Importers now apply retry/backoff on live metadata fetches, canonical-title dedupe before ranking, `MIN_STORY_METADATA_QUALITY` filtering, bounded Internet Archive cursor walking with explicit page-size/max-page/rate-limit controls, and result observability for candidate/quality/duplicate rejection counts, page counts, fetch attempts, byte budget exhaustion, and between-page sleeps. The dashboard reports import job health, pagination/rate-limit health, story quality floor compliance, and multi-cycle import trends; the reference scheduler smoke imports `4` story metadata items and the next story answer is local with citations to imported metadata. The two-cycle offline `inventory-soak` readiness gate preserves completed refresh history, adds `4` story rows, covers both required sources, records `0` failed cycles, and exposes failure observability. `inventory-diversity-smoke --reset --json` now runs folktale, bedtime, and adventure Internet Archive query niches through the same scheduler/job/importer path, reports each query from the executed import job, keeps quality/source checks clean, and proves each resulting DB routes the next story ask locally from inventory. `inventory-retry-smoke --reset --json` runs local Project Gutenberg and Internet Archive-shaped HTTP sources that fail once, requires both importer retry paths to recover, records `4` fetch attempts and `2` transient failures for dashboard observability, proves no external network was used, and confirms a future story ask changes from `cloud_handoff / missing_story_model` to `local_answer / local_story_inventory` only after imported inventory reload. `inventory-failure-smoke --reset --json` runs malformed Internet Archive JSON, source byte-budget exhaustion, and empty source fixtures through the same job/importer path, then requires observable failure/completion ledgers, `0` added story inventory, no local synthesis, and future story asks to remain `cloud_handoff / missing_story_model` instead of fabricating from a shortcut. Live source soaks remain optional hardening. | Extend live story/weather-source soak to longer runs and real network/source retry modes; connect routine/household/contact setup to real device/app integrations beyond the current local proof. |
| `TypedActionGate` | Keep side effects separate from ordinary answers. | Device actions create pending action rows; confirmation executes exactly the pending target through a typed local executor; dry-run mode records prepared media/contact execution with no side effect, while real mode blocks without an explicitly configured command. Replay without a pending action clarifies; cancellation prevents later confirmation from resurrecting the action; target mismatch stays pending. Eval covers `12` confirmation-required actions with `0` unconfirmed executed actions. CLI regressions prove media dry-run execution resolves the imported item/path and real media mode is confirmed but not executed when no player command is configured. `action-smoke` now exercises media and trusted-contact actions through the same confirmation gate; dry-run prepares both actions, and regression tests prove real mode executes configured commands, resolves media file paths, and resolves contact targets from local inventory. `host-action-smoke --reset --json` adds a harmless real command-mode host proof: local recorder subprocesses receive an existing media file and resolved trusted-contact target through argv-list execution with no shell, while safety flags stay clean. `host-app-probe --reset --json` now reports real target-device app-command readiness without pretending commands are configured; when media/call commands are supplied by args, `MELM_MEDIA_PLAYER_COMMAND` / `MELM_CALL_COMMAND`, or `--config-json config/host_actions.json`, it executes them through the same typed confirmation gate. `write-host-actions-demo-config --out config/host_actions.local_recorder.json --overwrite --json` creates a safe local recorder config so each target can rehearse the configured gate with `host-app-probe --config-json config/host_actions.local_recorder.json --require-configured --json` before swapping in real apps. The portable bundle includes `config/host_actions.example.json` as a target-device configuration template; the config supplies argv prefixes and optional media directory only, never a route shortcut or confirmation bypass. Target acceptance can require this with `--require-configured` or `v01-acceptance --host-app-config-json config/host_actions.json --require-host-app-configured --json`. The household-week trace covers `3` confirmation gates, `2` dry-run executions, one media cancellation, one replay block, and `0` unconfirmed executed actions through the store-backed gate rather than the older simulator shortcut. | Run `host-app-probe --config-json config/host_actions.json --require-configured --json` with actual media/call apps on each target platform and keep richer rollback/cancel semantics on every future side-effect path. |
| `TriageRuntime` | Route each ask to local answer, cache/tool, action, fetch, cloud, clarify, or reject. | The original eight realistic asks favored memory-centric triage (`8/8` local/device) over cloud-first, thin-tools, and vocabulary-only baselines. The current eval routes `105` broader utterances with `66` local/device resolutions, `9` cloud handoffs, `1` external fetch, `19` clarifications, and `10` rejects while preserving `0` privacy leaks and `0` wrong local answers. The 34-turn lifecycle suite adds profile-specific sequencing with `0.647` local resolution, `3` cloud handoffs, `1` external fetch, `7` clarifications, `3` offline blocks, `3` confirmed actions, and zero privacy/action/fake-news/memory-link safety flags. The 37-turn household-week trace adds `0.676` local resolution, `3` cloud handoffs, `1` fetch, `7` clarifications, `1` offline block, `1` reject, `3` confirmation gates, and `11/11` architecture checks across one longer life. The 29-turn open trace adds `0.655` local/device resolution across identity, weather, story, safety, media, contact, routine, household, meal, health, memory recall, private-cloud rejection, and offline latest-news refusal, with required miss-to-hit/setup-to-action transitions and `0` privacy/action/fake-news failures. The 25-turn transcript replay adds `0.68` local/device resolution over messier user-turn JSONL, with `profile_update` for simple age/location facts, all required route/reason coverage, complexity and unknown-token scoring, clean safety flags, and no static per-turn expected answers/routes. Its baseline comparison scores the same `25` user turns as current kernel `17/25` local/device versus best static baseline `7/25`, with `+0.40` local-resolution gain, `7` fewer cloud handoffs, `3` fewer clarifications, `0` private-cloud exposure, `2` profile-update advantage, one private-cloud block advantage, and one long-horizon digest advantage. `calibrate-transcript-replay` now imports one or more raw chat JSONL files, strips assistant/system rows and static answer/route fields, redacts private-looking tokens, replays the imported fixtures, and aggregates route, complexity, safety, redaction, debug-map evidence, and `primary_uol_chatframe_not_secondary_phrase_route`; local answers, cached tools, and device actions must prove `token_role_relation` or `slot_role_relation` primary evidence rather than secondary phrase hints. | Grow beyond authored open trace/transcript fixtures into user-derived household transcripts and calibrate baseline failure-rate thresholds on real usage. |
| `LocalApiSurface` | Expose the assistant as a local app/API surface, not only a CLI. | `serve --host 127.0.0.1 --port 8771` exposes a stdlib browser chat UI at `GET /`, health at `GET /health`, local dashboard evidence at `GET /dashboard`, parse-only debugging at `GET/POST /parse-debug`, non-static event-ledger export at `GET /event-transcript-replay`, live event-ledger replay calibration at `POST /calibrate-event-ledger`, and chat at `POST /ask`. `api-smoke --reset --json` starts the same handler on localhost, verifies seeded `/health`, posts an identity challenge to `/parse-debug` without writing an event, posts `"Tell me a story."` to `/ask`, confirms local story synthesis plus membrane/homeostasis records, checks the event persisted in SQLite, verifies dashboard counts, and proves the API event export omits stored answers/routes/reasons and static expectations. `api-session-smoke --reset --json` runs an 11-turn realistic local API session over assistant identity, story, weather, school-safety, media confirmation, health, profile memory, meal, and trusted-contact confirmation; it preserves `0` cloud/fetch routes, completes two dry-run action confirmations, logs `11` events/membrane/homeostasis rows, keeps safety flags clean, exports all `11` user turns without static expectations, and replays them through `POST /calibrate-event-ledger` with local-resolution and intent-diversity thresholds. `parse-debug --utterance ... --json`, `GET/POST /parse-debug`, and every `/ask` response expose a bounded basic NLP -> UOL -> assistant ChatFrame trace with tokens, token roles, compositional parse details, `primary_domain_evidence`, secondary domain hints, domain hints, secondary meaning hints, unknown-token lists, slots, UOL slot sources, stage mapping, route/reason, parse score, complexity score, ChatFrame capabilities, primary routing basis, secondary debug hints, and notes. Identity challenges map to `self_model` through token-role/UOL composition rather than phrase-level primary routing; status maps to `runtime_status` or `next_steps`, routine/household memory maps to `routine_memory`/`household_memory`, simple age/location statements map to `user_profile` with `profile_update`, and session recall maps to `conversation_events`. The browser UI renders the same trace under a collapsible debug frame and auto-opens it for suspicious, unknown, or high-unknown-token turns; it now includes operator `Export` and `Calibrate` controls that call the local event-ledger endpoints without adding route shortcuts. `run-open-traces --json` and `run-transcript-replay --json` preserve that debug map for every trace turn and summarize complexity/unknown-token pressure. `ui-smoke --reset --json` loads `/` and `/index.html`, verifies the dependency-free chat shell is wired to `/health`, `/parse-debug`, `/ask`, `/event-transcript-replay`, and `/calibrate-event-ledger`, posts local identity/status/story/action turns, verifies the Basic NLP -> UOL -> ChatFrame frame and action confirmation controls, and confirms the same persisted SQLite calibration path. `chat --turn ... --json` gives the same kernel/store path to cross-platform terminal sessions. | Use the live export/calibration endpoints on real browser/CLI sessions to retire the user-derived lifecycle/calibration blockers; keep the endpoints local-only and non-static. |
| `ReadinessGate` | Prove the current v0.1 pieces run together inside a small-device/cross-platform software envelope. | `pi-smoke --reset --json` now verifies required datasets, the full `dataset-audit`, seeded SQLite memory, one local story ask with membrane/homeostasis and bounded synthesis, the 17-step lifecycle, typed media/contact action preparation with a tiny local media file created in the smoke workspace and imported through the directory scanner, a resolved trusted-contact target, the `setup-integration-smoke` gap-to-user-setup proof, the 10-turn `synthesis-variant-smoke` and 24-turn `synthesis-stress-smoke` gates, the 29-turn open trace debug-parser gate, the 25-turn transcript replay gate with baseline comparison, the offline both-source `inventory-soak` gate, the cold-start multi-source `inventory-soak-matrix` gate, the multi-niche `inventory-diversity-smoke` source-query gate, the transient-source `inventory-retry-smoke` retry gate, the negative `inventory-failure-smoke` source-failure gate, complete ledgers, clean safety flags, and stdlib Python + SQLite with no required network, vector DB, or ML framework. Development-machine reference runs pass all `22/22` checks; the setup-integration gate passes `9/9` checks over routine, household, and trusted-contact gaps, setup requests, explicit local setup, UOL/ChatFrame debug mapping, scoped local memory, and dry-run action confirmation. The synthesis-variant gate passes `15/15` checks over story variants, health variants, urgent health, cached weather, meal choice, recent-session summary, and long-horizon digest recall while requiring primary `slot_role_relation` UOL/ChatFrame evidence and clean cited synthesis. The synthesis-stress gate passes `14/14` checks over `24` turns and `3` sessions, keeps all routes local or cached (`22` local answers, `2` cached-tool answers), proves identity/status/self-progress, story, health, meal, weather, safety, recent-session recall, and long-horizon digest synthesis, and preserves primary UOL/ChatFrame evidence without phrase-table primary routing. The inventory soak adds `4` story rows from Project Gutenberg CSV plus Internet Archive metadata fixtures, covers both required sources, exposes failure observability, and records `0` failed cycles. The inventory matrix runs `both_extended`, `internet_archive_query`, and `gutenberg_replay` cold-start profiles for at least `9` total cycles, covers both source families, records `0` failed import cycles, and proves each future story ask routes locally from imported inventory with primary UOL/ChatFrame evidence. The diversity smoke runs folktale, bedtime, and adventure source queries through executed import jobs and proves each resulting DB routes the next story ask locally from inventory. The retry smoke forces localhost source 503s, verifies both source importers retry, records fetch-attempt observability, uses no external network, and proves future story routing changes only after imported inventory reload. The failure smoke runs malformed JSON, byte-budget exhaustion, and empty-source cases through the same import path and proves there is no fabricated inventory or local story route when sources fail or are empty. The open trace and transcript replay preserve the basic NLP -> UOL -> ChatFrame mapping, prove identity maps to `self_model`, status maps to `runtime_status`/`next_steps`, simple age/location facts map to `user_profile`, long-horizon digest quality passes, and the current kernel beats the best static transcript baseline by `+0.40` local-resolution rate. `host-action-smoke --reset --json` separately proves harmless real subprocess handoff for media/contact commands through the same action gate. `host-app-probe --reset --json` separately reports host media/call command readiness and can execute supplied target commands through the typed action gate; target-specific acceptance should use `--require-configured`. `capability-probe --reset --json` runs an 18-case realistic surface probe through the real SQLite/kernel path: original eight assistant asks resolve local/device, 4 open-domain/unsupported examples hand off to cloud, 2 private-cloud examples are blocked, action confirmations stay dry-run, and every case reports mapping, complexity, unknown tokens, route/reason, primary routing basis, and secondary debug hints. | Run configured app probes on each target platform and keep cross-platform browser/CLI acceptance green. |
| `PortableDeploymentBundle` | Make the runnable v0.1 assistant copyable without hidden repo state. | `pi-bundle --reset --zip --json` copies the runnable CLI, local `melm` Python modules, seed/source/media/weather/open-trace/transcript-replay/raw-transcript fixtures, `config/host_actions.example.json`, root/docs plan files, portable runbook, generic Unix launchers, Windows PowerShell/`.cmd` launchers, Raspberry/Linux launchers, and a systemd user-service example into a portable bundle; it writes a SHA-256 manifest and self-check JSON, optionally writes a zip archive, runs `dataset-audit`, `pi-smoke`, `autoimmune-smoke`, `synthesis-variant-smoke`, `synthesis-stress-smoke`, `setup-integration-smoke`, `host-action-smoke`, `host-app-probe`, `capability-probe`, `shortcut-audit`, `v01-audit`, `v01-progress`, `api-smoke`, `api-session-smoke`, `ui-smoke`, `bootstrap-runtime`, `launcher-smoke`, `run-open-traces`, `run-transcript-replay`, and `calibrate-transcript-replay` from inside the copied bundle root, and cleans generated smoke/bootstrap/launcher/open-trace/transcript-replay/transcript-calibration/autoimmune/synthesis-variant/synthesis-stress/setup-integration/host-action/host-app/capability/dataset/progress artifacts before packaging. The runbook tells target installs to generate `config/host_actions.local_recorder.json` with `write-host-actions-demo-config`, prove the configured gate with `host-app-probe --config-json config/host_actions.local_recorder.json --require-configured --json`, then copy the example to `config/host_actions.json`, fill `media_player_command`, `call_command`, and optional `media_dir`, and run `host-app-probe --config-json config/host_actions.json --require-configured --json` or the matching `v01-acceptance --host-app-config-json ...` gate for real apps. `shortcut-audit --json`, `v01-audit --json`, and `v01-progress --json` run from the copied bundle; the shortcut audit proves source/behavior anti-shortcut discipline directly, while the v01 audit/progress reports keep `architecture_complete=false` with six real-world validation blockers and browser/CLI core evidence. `launcher-smoke --reset --json` starts the copied app through the platform start launcher, verifies localhost `/health` through the platform health launcher, checks the browser shell plus parse endpoint, and shuts down. `verify-bundle --json` then verifies every manifest-listed file exists with matching byte count and SHA-256 hash, required portable commands are present, including `portable_dataset_audit_command`, `portable_inventory_soak_matrix_command`, `portable_autoimmune_command`, `portable_synthesis_variant_command`, `portable_synthesis_stress_command`, `portable_setup_integration_command`, `portable_host_action_command`, `portable_host_actions_demo_config_command`, `portable_host_app_probe_command`, `portable_host_app_demo_config_probe_command`, `portable_capability_probe_command`, `portable_shortcut_audit_command`, `portable_v01_audit_command`, `portable_v01_progress_command`, `portable_v01_evidence_pack_command`, `portable_launcher_smoke_command`, `portable_first_run_smoke_command`, `portable_open_traces_command`, `portable_transcript_replay_command`, `portable_transcript_calibration_command`, and `portable_refresh_weather_command`, required launcher files are present, stdlib-only declarations are intact, and the self-check was not skipped; its self-check summary includes `pi_smoke_inventory_soak_matrix_passed`, `shortcut_audit_passed`, and `v01_progress_passed`. Regression coverage tampers with `README.md` and proves the verifier fails on byte/hash mismatch. `first-run-smoke --json` then executes the packaged platform first-run launcher from the completed bundle and verifies the nested bundle integrity check, dataset audit, target report, bootstrap runtime, UI smoke, and launcher smoke all pass from copied files. `archive-smoke --json` treats the zip as the deployable artifact: it rejects unsafe archive entries, extracts into a fresh work directory, finds exactly one portable bundle root, runs `verify-bundle`, and executes `first-run-smoke` from the extracted copy. Reference bundle: `passed=true`, self-check includes dataset audit `19/19`, readiness `22/22` including synthesis-variant smoke, synthesis-stress smoke, setup-integration smoke, inventory soak, inventory soak matrix, inventory diversity smoke, inventory retry smoke, and inventory failure smoke, autoimmune `34/34`, synthesis-variant `15/15`, synthesis-stress `14/14`, setup-integration `9/9`, host-action command checks, explicit host-app configuration reporting, capability probe `18` cases, shortcut audit behavior/source checks, v01 audit core checks with blocker count `6`, v01 progress remaining blocker count `6`, API parser/chat checks, realistic API session checks, browser UI checks, usable-runtime bootstrap checks, launcher checks, open-trace parser checks, transcript-replay parser/digest/baseline checks, and transcript-calibration redaction/debug checks; post-build first-run smoke passes `16/16` checks; archive smoke passes all extraction/integrity/first-run checks; stdlib Python + SQLite/HTTP/HTML, no required network/vector DB/ML framework. | Keep the bundle green on Windows/macOS/Linux and treat Raspberry hardware checks as optional appliance validation. |
| `BootstrapRuntime` | Create the actual usable assistant DB after package verification, not only disposable smoke DBs. | `bootstrap-runtime --reset --json` initializes `artifacts/local_assistant_os/assistant_v01.sqlite` from the seed dataset, imports initial local media metadata, runs story/weather/school-safety turns against that runtime DB, verifies local/cached routes with bounded synthesis, writes `3` event/membrane/homeostasis rows, checks clean safety flags, reports DB size and media import observability, and prints next `ask`, `serve`, and `dashboard` commands. Regression coverage verifies the command creates a usable local DB with `3` events, `3` membrane decisions, `3` homeostatic snapshots, local story, cached weather, local school-safety answer, and zero unconfirmed actions. | Optionally rerun with a real `--media-dir` and `--require-media-files` for host-local media readiness. |
| `TargetDeviceReport` | Produce one host evidence artifact instead of relying on manual notes. | `target-report --reset --json` records Python version, SQLite version, platform, machine, disk space, Linux memory facts when available, optional Raspberry Pi model detection when available, and then runs `dataset-audit`, `pi-smoke`, `autoimmune-smoke`, `synthesis-variant-smoke`, `synthesis-stress-smoke`, `setup-integration-smoke`, `host-action-smoke`, `host-app-probe`, `capability-probe`, `v01-audit`, `api-smoke`, `api-session-smoke`, `ui-smoke`, `bootstrap-runtime`, `run-open-traces`, `run-transcript-replay`, and `calibrate-transcript-replay`. The nested `pi-smoke` evidence now includes `inventory_soak_matrix_passed` and the matrix profile summary. Development-machine reference passes with `raspberry_pi_detected=false`; `--require-raspberry-pi` remains available only when the target is specifically Raspberry hardware. `--require-host-app-configured` can make actual configured media/call app execution part of target acceptance. | Keep this as the cross-platform host evidence artifact and run configured app probes on target platforms when validating appliance integrations. |
| `V01AcceptanceMatrix` | Collapse runnable evidence into one release-candidate decision without pretending full completion. | `v01-acceptance --reset --json` runs `target-report`, a real three-turn scripted `chat` session, and `v01-audit`, then emits requirement rows for target smokes, datasets/bootstrap, `pi-smoke` plus `inventory-soak-matrix`, terminal CLI chat, API/browser UI, transcript/synthesis gates, setup/action gates, direct `shortcut-audit --json` anti-static UOL/ChatFrame discipline, optional portable bundle status, and explicit completion blockers. It also accepts `--host-app-config-json config/host_actions.json --require-host-app-configured` to include configured target app commands in the same evidence matrix. A passing report sets `release_candidate=true` while preserving `architecture_complete=false` and blocker count `6`. | Keep this as the first host command for browser/CLI v0.1 release-candidate acceptance; use `--include-bundle` when packaging must be part of the same report and add `--host-app-config-json config/host_actions.json --require-host-app-configured` for target-app validation. |
| `BudgetedEvidenceBoundary` | Feed synthesis compact state plus admitted evidence, not raw life history. | Grounded child-chat probes show positive evidence compression from `1.21x` to `4.55x` at 32 moves and from `1.39x` to `17.46x` at 128 moves. The assistant synthesizer receives cited evidence keys rather than full transcripts. | Keep top-k evidence, matching counts, and source provenance as synthesis grows richer. |
| `BoundedLocalSynthesis` | Produce local/cached answers only from admitted evidence and refuse blocked/cloud/action paths. | `BoundedLocalSynthesizer` now builds multi-sentence local stories from title/summary/topics/cultures, answers assistant identity from cited `self_model.*` state, answers runtime status from cited `self_status.*` dashboard facts, summarizes profile memory across multiple cited local facts/preferences, groups recent-session event memory while extracting capability transitions/open local gaps/action state/boundary controls, verbalizes structured local long-horizon memory digests, gives richer bounded health/meal/weather/media-cancel/contact-cancel/safety wording, and cites story inventory, self-state, runtime status, profile facts, event memory, memory digests, weather, food, health goals, contacts, media, and policy inputs. Digest answers now say what the assistant remembers as threads, what capability changed, what stayed blocked/local-only, and what still needs setup, all from one cited `memory_digest.*` row rather than raw transcript stuffing; dashboard `latest_memory_digest` exposes `quality_score`, `quality_floor`, `quality_passed`, and warnings. Consent revocation synthesis cites a privacy policy rather than echoing the revoked fact value; private-cloud blocks refuse synthesis. Each trace now carries a deterministic quality score over route discipline, citation coverage, evidence strength, answer specificity, source diversity, and local privacy discipline; the dashboard reports `97` eval synthesis traces with `0` low-quality applied traces and empty warning counts after meal and contact-cancellation synthesis were made inventory/action-specific. `synthesis-variant-smoke --reset --json` adds compact runtime evidence over 10 non-canonical local/cached turns: bedtime/read/tale story variants, two health-advice variants, urgent health safety, cached weather, local meal choice, recent-session summary, and long-horizon digest recall. It passes `15/15` checks with `9` local answers, `1` cached-tool answer, cited synthesis on every turn, `0` low-quality applied synthesis, complete ledgers, and primary `slot_role_relation` UOL/ChatFrame evidence with phrase hints excluded from primary routing. `synthesis-stress-smoke --reset --json` extends this to a 24-turn / 3-session lifecycle with identity/status self-awareness, story niches, health and urgent-health advice, weather and meal decisions, school-safety, last-question recall, recent-session summary, long-horizon digest recall, and status-next answers; it passes `14/14` checks with all routes local/cached, complete ledgers, clean safety flags, min quality `0.799`, max complexity `0.57`, and the same primary UOL/ChatFrame discipline. | Expand to user-derived story/advice/summarization transcripts while preserving citations and membrane/action gates. |

Portable command boundary: `shortcut-audit --json` is the direct runnable
anti-shortcut evidence for UOL/ChatFrame source and behavior drift.
It is not only a behavior smoke: it also scans the primary classifier,
post-route slot helpers, the concept-token secondary hint table, identity
composition, self-status composition, the shared autobiographical-memory
composer, and the kernel recall gate. A phrase-list shortcut in any of those
blocks invalidates the evidence even if the current demo sentences still appear
to pass.
`portable_shortcut_audit_command`, `portable_v01_progress_command`,
`portable_v01_evidence_pack_command`, and `portable_v01_audit_command` are
mandatory; any bundle/runbook command list that omits shortcut audit, progress,
or evidence-pack is stale.

Auto-lifecycle replay evidence: `run-transcript-replay`,
`calibrate-event-ledger`, `calibrate-transcript-replay`, and
`v01-blocker-evidence` can run with `--auto-lifecycle`. In that mode replayed
user turns trigger refresh scheduling, safe offline job execution, and
memory-digest creation from runtime state. This is the preferred micro-MVP
proof for assistant self-awareness because planner priority signals and digest
quality are produced by the assistant's own lifecycle loop instead of authored
per-turn route/answer controls.

Blocker evidence matrix: `write-source-attestation` writes a hash-bound source
attestation for real local event-ledger sessions. `v01-blocker-evidence`
reports the six remaining blocker rows without claiming architecture
completion. In that report, `passed=true` means the report assembled without
read/runtime errors; candidate readiness is represented separately by
`report_valid`, `candidate_evidence_complete`, `candidate_blockers_satisfied`,
and `remaining_blocker_count`. `v01-evidence-pack --db <assistant.sqlite> --work-dir <dir>
--auto-lifecycle --json` is the preferred wrapper when starting from a local
session DB: it writes event-ledger export, calibration, blocker evidence,
progress, and source-note artifacts while preserving the
development-vs-attested-user boundary. `v01-blocker-evidence` accepts an
event-ledger calibration source, source-attestation
JSON, a strict `calibrate-transcript-replay --out <calibration-report.json>`
report through `--transcript-calibration-report-json`, optional inventory-soak
report, optional configured host-app probe, and optional host-app attestation
JSON. The digest/route threshold row cannot be inferred from loose event-ledger
fields, static transcript expectations, or a standalone strict-looking JSON
report; it requires strict digest/baseline gates plus calibration-report binding
to the same attested replay/event SQLite DB path and SHA-256 with
imported-redacted transcript capture provenance. Source attestation is valid for
candidate user evidence only when the event ledger has matching imported
redacted transcript, interactive CLI, served browser UI (`browser_ui` with the
served page capture token), or
target-device capture provenance covering every packaged turn; scripted
CLI/API/UI smoke ledgers, or mixed ledgers where only some turns are candidate
capture, remain development evidence even if human-review and redaction flags
are supplied. `calibrate-transcript-replay` now emits audit-first
`next_candidate_commands` for each replay DB: run `candidate-session-audit`
with the same replay DB/session, then write source attestation with
`--event-ledger-session all`, then package through `v01-evidence-pack` or
`v01-blocker-evidence`. That recipe is guidance only; the generated calibration
report does not become candidate proof until the audit, attestation, DB hash,
and blocker-evidence gates all pass. `candidate-session-audit` also includes a
projection-only blocker view built from the same `v01-blocker-evidence` rows so
an operator can see whether the session is likely to satisfy lifecycle,
synthesis, planner, digest, live-inventory, or host-app rows after attestation;
optional transcript-calibration, inventory-soak, and host-app artifacts can be
included in that projection, but projection rows are not evidence until the
written attestation and artifact-bound blocker report exist. The live-inventory row cannot
be inferred from a report-shaped JSON claim either; it requires
`inventory-soak-matrix --live` artifact binding with schema
`melm.inventory_soak_matrix.v1`, `mode=live_metadata`, live fetch counters,
strict matrix checks, both source families, matching per-run DB SHA-256 hashes,
verified story inventory rows, verified future local story events, and primary
UOL/ChatFrame story evidence. `write-host-app-attestation` binds
`config/host_actions.json` by SHA-256 and records human review that real
media/call app commands are configured, not the local recorder/demo. The
`configured_target_device_apps` row can be candidate evidence only when the
probe passes, the host-app attestation is valid, and the config analyzer does
not detect recorder markers such as `host-action-recorder` or
`action_recorder.py`. `api-session-smoke --action-mode real
--host-app-config-json config/host_actions.json` and `serve --action-mode real
--host-app-config-json config/host_actions.json` use the same typed action gate;
dry-run remains the portable default.
Development sessions remain separate from candidate `redacted_user_session` or
`target_device_user_session` evidence; candidate user-derived rows require a
valid `--source-attestation-json` matching the current event-ledger DB hash. The
synthesis and planner rows additionally require positive
`--min-synthesis-traces` and `--min-priority-signal-samples` floors; a zero
floor is a smoke/default convenience, not evidence that can retire a blocker. The
portable bundle manifest exposes the same path through
`portable_v01_evidence_pack_command`,
`portable_candidate_session_audit_command`, `portable_source_attestation_command`,
`portable_host_app_attestation_command`, and
`portable_v01_blocker_evidence_command`.
`v01-blocker-rehearsal --reset --json` is the development-only guard for this
workflow. It creates a real scripted chat ledger, exports/replays it through
`v01-blocker-evidence`, writes a development source note, and feeds the report
into `v01-progress`; the expected pass condition keeps
`candidate_blockers_satisfied=0`, leaves digest/live-inventory/target-app rows
unclaimed, and only marks lifecycle/synthesis/planner rows as development
evidence.
`v01-progress --json` summarizes the current audit plus blocker evidence into a
single non-mutating status report for bundle and target operators.

Inventory matrix evidence: `inventory-soak-matrix --reset --json` is now the
stronger cold-start source-growth gate and is included in `pi-smoke`,
`target-report`, `pi-bundle`, and `verify-bundle`. It runs `both_extended`,
`internet_archive_query`, and `gutenberg_replay` profiles for at least `9` total
cycles, covers both Project Gutenberg CSV and Internet Archive metadata source
families, requires `0` failed import cycles and clean story-quality/failure
observability checks, and proves each future story ask routes
`local_answer / local_story_inventory` from imported inventory with primary
UOL/ChatFrame evidence instead of secondary phrase hints.

Portable target-app evidence has two separate commands by design. The default
`portable_host_app_probe_command` stays unconfigured so bundle self-checks can
run on any browser/CLI host, while `portable_host_app_configured_probe_command`
and `portable_v01_acceptance_configured_host_app_command` name the real
target-device proof using `config/host_actions.json`. `verify-bundle` requires
those command entries to be present, but real app execution remains a
target-platform validation step.

The biggest remaining gap is no longer basic explicitness. Membrane,
homeostasis, SQLite persistence, resource reporting, dashboarding, replay/cancel
action gates, consent/stale-cache/parent-child privacy gates, scheduled refresh
jobs, and cited local synthesis are first-class v0.1 objects. The architecture
is now gated by quality and scale: longer/user-derived bounded-synthesis coverage,
more diverse live inventory soak, target-device app integration smokes,
routine/household integrations, redacted user-derived transcript traces imported
with `import-transcript-replay` beyond the authored open-trace/transcript-replay
fixtures, and digest-quality calibration on real household transcripts.

## Current Runnable Artifacts

```text
database store: melm/appliance/assistant_os_store.py
dashboard:      melm/appliance/assistant_dashboard.py
evaluation:     melm/appliance/assistant_eval.py
inventory:      melm/appliance/assistant_inventory.py
actions:        melm/appliance/assistant_actions.py
kernel wiring:  melm/appliance/assistant_os_kernel.py
lifecycle:      melm/appliance/assistant_lifecycle.py
synthesis:      melm/appliance/assistant_synthesis.py
open traces:    melm/appliance/assistant_open_traces.py
transcript import:
                melm/appliance/assistant_transcript_import.py
seed dataset:   benchmarks/local_assistant_os_seed.json
metadata:       benchmarks/public_domain_story_metadata.json
source samples: benchmarks/sample_gutenberg_catalog.csv
                benchmarks/sample_internet_archive_search.json
media manifest: benchmarks/local_media_manifest.json
open fixture:   benchmarks/local_assistant_open_traces.json
transcript:     benchmarks/local_assistant_transcript_replay.jsonl
raw transcript: benchmarks/sample_local_assistant_raw_transcript.jsonl
CLI/API:        scripts/local_assistant_os_cli.py
tests:          tests/test_assistant_os_store_mvp.py
                tests/test_local_assistant_os_cli_mvp.py
                tests/test_assistant_lifecycle_mvp.py
                tests/test_assistant_os_jobs_mvp.py
                tests/test_assistant_os_eval_mvp.py
                tests/test_assistant_synthesis_mvp.py
                tests/test_assistant_inventory_importers_mvp.py
                tests/test_assistant_open_traces_mvp.py
```

Runnable commands:

```powershell
python scripts\local_assistant_os_cli.py init --reset --json
python scripts\local_assistant_os_cli.py ask --utterance "Tell me a story." --json
python scripts\local_assistant_os_cli.py run-jobs --json
python scripts\local_assistant_os_cli.py schedule-refreshes --offline-samples --json
python scripts\local_assistant_os_cli.py run-lifecycle --reset --json
python scripts\local_assistant_os_cli.py run-lifecycle-suite --json
python scripts\local_assistant_os_cli.py run-household-week --reset --json
python scripts\local_assistant_os_cli.py run-open-traces --reset --json
python scripts\local_assistant_os_cli.py run-transcript-replay --reset --json
python scripts\local_assistant_os_cli.py export-transcript-replay --db artifacts\local_assistant_os\assistant_v01.sqlite --out artifacts\local_assistant_os\event_ledger_transcript_replay.jsonl --json
python scripts\local_assistant_os_cli.py calibrate-event-ledger --db artifacts\local_assistant_os\assistant_v01.sqlite --work-dir artifacts\local_assistant_os\event_ledger_calibration --controls-json config\safe_lifecycle_controls.example.json --min-total-turns 4 --min-local-resolution-rate 0.5 --json
python scripts\local_assistant_os_cli.py v01-evidence-pack --db artifacts\local_assistant_os\assistant_v01.sqlite --work-dir artifacts\local_assistant_os\v01_evidence_pack --auto-lifecycle --json
python scripts\local_assistant_os_cli.py import-transcript-replay --input path\raw_chat.jsonl --out artifacts\local_assistant_os\imported_transcript_replay.jsonl --controls-json config\safe_lifecycle_controls.example.json --replace "Maya=<person_1>" --json
python scripts\local_assistant_os_cli.py calibrate-transcript-replay --input benchmarks\sample_local_assistant_raw_transcript.jsonl --controls-json config\safe_lifecycle_controls.example.json --replace "Maya=<person_1>" --min-total-turns 4 --min-local-resolution-rate 0.2 --min-route-kinds 3 --min-intent-kinds 3 --require-redaction --require-static-drop --out artifacts\local_assistant_os\sample_transcript_calibration.json --reset --json
python scripts\local_assistant_os_cli.py autoimmune-smoke --reset --json
python scripts\local_assistant_os_cli.py synthesis-variant-smoke --reset --json
python scripts\local_assistant_os_cli.py synthesis-stress-smoke --reset --json
python scripts\local_assistant_os_cli.py setup-integration-smoke --reset --json
python scripts\local_assistant_os_cli.py host-action-smoke --reset --json
python scripts\local_assistant_os_cli.py host-app-probe --reset --json
python scripts\local_assistant_os_cli.py capability-probe --reset --json
python scripts\local_assistant_os_cli.py shortcut-audit --json
python scripts\local_assistant_os_cli.py v01-audit --json
python scripts\local_assistant_os_cli.py candidate-session-audit --db artifacts\local_assistant_os\assistant_v01.sqlite --session all --capture-surface cli_chat --json
python scripts\local_assistant_os_cli.py write-source-attestation --event-ledger-db artifacts\local_assistant_os\assistant_v01.sqlite --event-ledger-session all --source-kind redacted_user_session --capture-surface cli_chat --redaction-applied --static-expectations-absent --answers-routes-reasons-absent --human-reviewed --out artifacts\local_assistant_os\source_attestation.json --json
python scripts\local_assistant_os_cli.py write-host-app-attestation --host-app-config-json config\host_actions.json --capture-surface target_device_cli --media-app-configured --call-app-configured --not-demo-recorder --real-app-commands-acknowledged --human-reviewed --out artifacts\local_assistant_os\host_app_attestation.json --json
python scripts\local_assistant_os_cli.py inventory-soak-matrix --live --reset --out artifacts\local_assistant_os\live_inventory_soak_matrix.json --json
python scripts\local_assistant_os_cli.py v01-blocker-evidence --event-ledger-db artifacts\local_assistant_os\assistant_v01.sqlite --event-ledger-session all --event-source-kind redacted_user_session --source-attestation-json artifacts\local_assistant_os\source_attestation.json --transcript-calibration-report-json artifacts\local_assistant_os\sample_transcript_calibration.json --inventory-soak-report-json artifacts\local_assistant_os\live_inventory_soak_matrix.json --host-app-config-json config\host_actions.json --host-app-attestation-json artifacts\local_assistant_os\host_app_attestation.json --run-host-app-probe --out artifacts\local_assistant_os\v01_blocker_evidence.json --json
python scripts\local_assistant_os_cli.py v01-blocker-rehearsal --reset --json
python scripts\local_assistant_os_cli.py v01-progress --json
python scripts\local_assistant_os_cli.py v01-acceptance --reset --json
python scripts\local_assistant_os_cli.py parse-debug --utterance "wow you don't know who you are?" --json
python scripts\local_assistant_os_cli.py dataset-audit --reset --json
python scripts\local_assistant_os_cli.py resource-report --reset --json
python scripts\local_assistant_os_cli.py pi-smoke --reset --json
python scripts\local_assistant_os_cli.py pi-bundle --reset --zip --json
python scripts\local_assistant_os_cli.py verify-bundle --bundle-root artifacts\local_assistant_os\melm_local_assistant_os_v01_pi_bundle --json
python scripts\local_assistant_os_cli.py bootstrap-runtime --reset --json
python scripts\local_assistant_os_cli.py api-smoke --reset --json
python scripts\local_assistant_os_cli.py api-session-smoke --reset --json
python scripts\local_assistant_os_cli.py ui-smoke --reset --json
python scripts\local_assistant_os_cli.py launcher-smoke --bundle-root artifacts\local_assistant_os\melm_local_assistant_os_v01_pi_bundle --reset --json
python scripts\local_assistant_os_cli.py first-run-smoke --bundle-root artifacts\local_assistant_os\melm_local_assistant_os_v01_pi_bundle --json
python scripts\local_assistant_os_cli.py archive-smoke --archive artifacts\local_assistant_os\melm_local_assistant_os_v01_pi_bundle.zip --reset --json
python scripts\local_assistant_os_cli.py chat --turn "Tell me a story." --turn "What is the weather today?" --json
python scripts\local_assistant_os_cli.py target-report --reset --json
python scripts\local_assistant_os_cli.py dashboard --json
python scripts\local_assistant_os_cli.py memory-replay --query story --json
python scripts\local_assistant_os_cli.py memory-replay --sessions 3 --events-per-session 1 --json
python scripts\local_assistant_os_cli.py memory-digest --sessions 20 --events-per-session 3 --json
python scripts\local_assistant_os_cli.py eval --json
python scripts\local_assistant_os_cli.py import-stories --source both --json
python scripts\local_assistant_os_cli.py import-media --cold-start --manifest benchmarks\local_media_manifest.json --limit 2 --json
python scripts\local_assistant_os_cli.py ask --utterance "Play calm piano." --json
python scripts\local_assistant_os_cli.py ask --utterance "Yes, play calm piano." --action-mode dry-run --json
python scripts\local_assistant_os_cli.py action-smoke --reset --json
python scripts\local_assistant_os_cli.py inventory-soak --offline-samples --source both --cycles 2 --story-limit 3 --min-story-models 12 --json
python scripts\local_assistant_os_cli.py inventory-soak-matrix --reset --json
python scripts\local_assistant_os_cli.py inventory-diversity-smoke --reset --json
python scripts\local_assistant_os_cli.py inventory-retry-smoke --reset --json
python scripts\local_assistant_os_cli.py inventory-failure-smoke --reset --json
python scripts\local_assistant_os_cli.py serve --host 127.0.0.1 --port 8771
```

Current cold lifecycle database evidence:

```text
steps=17
events=17
membrane_decisions=17
homeostatic_snapshots=17
synthesis_traces=11
opportunities=2
inventories=12
jobs=2
pending_actions=1
local_resolution_rate=0.706
cloud_handoffs=3
external_fetches=1
blocked_offline=1
jobs=refresh_weather_cache, build_story_inventory
```

Current resource report evidence on the development machine:

```text
runtime=stdlib_python_sqlite
dependency_class=stdlib_only
ask_ms~=5.3-9.4
lifecycle_ms~=76.3-113.5
reference_resource_run=ask_ms:9.394, lifecycle_ms:113.494, peak_traced_kb:110.514
peak_traced_kb~=110.5-111.5
seeded_db_bytes~=143360
lifecycle_db_bytes~=151552
required network/vector DB/ML framework: no
```

Current dataset/source/bootstrap audit evidence on the development machine:

```text
command=dataset-audit --reset --json
passed=true
checks=19/19 true
files=8/8 with SHA-256 hashes
seed_facts=6
seed_inventories=8
story_metadata_items=4
media_items=3
weather_days=7
gutenberg_story_candidates=2
internet_archive_story_candidates=2
open_trace_scenarios=2
open_trace_turns=29
transcript_replay_rows=26
transcript_replay_user_turns=25
sqlite_bootstrap_created=true
bootstrap_profile=Maya age 7 Lagos Yoruba
bootstrap_inventories=story, weather, contact, media, food
runtime=stdlib_python_sqlite_json_csv
dependency_class=stdlib_only
```

Reference Pi readiness smoke evidence on the development machine:

```text
command=pi-smoke --reset --json
passed=true
checks=22/22 true
datasets_present=8/8
dataset_audit_passed=true
dataset_audit_checks=19/19 true
runtime=stdlib_python_sqlite
dependency_class=stdlib_only
ask_ms~=35
lifecycle_ms~=287
open_trace_ms~=931
transcript_replay_ms~=1006
inventory_soak_ms~=428
inventory_soak_matrix_passed=true
inventory_soak_matrix_profiles=3
inventory_soak_matrix_failed_cycles=0
inventory_diversity_ms~=2757
inventory_failure_ms~=2783
total_ms~=8213
peak_traced_kb~=2123
db_bytes=143360
lifecycle_db_bytes=167936
action_db_bytes=151552
ask_route=local_answer
ask_reason=local_story_inventory
ask_synthesis_applied=true
lifecycle_steps=17
lifecycle_actions_executed=1
lifecycle_story_route_after_inventory=local_answer
action_smoke_passed=true
media_target_path_exists=true
contact_target=+234-000-ADA
synthesis_variant_smoke_passed=true
synthesis_variant_checks=15/15 true
synthesis_variant_turns=10
synthesis_variant_routes=local_answer:9, cached_tool:1
synthesis_variant_primary_uol_chatframe_not_secondary_phrase_route=true
synthesis_variant_quality_clean=true
synthesis_stress_smoke_passed=true
synthesis_stress_checks=14/14 true
synthesis_stress_turns=24
synthesis_stress_sessions=3
synthesis_stress_routes=local_answer:22, cached_tool:2
synthesis_stress_autobiographical_summaries_use_events_and_digest=true
synthesis_stress_primary_uol_chatframe_not_secondary_phrase_route=true
synthesis_stress_quality_min=0.799
synthesis_stress_complexity_max=0.57
open_trace_debug_gate_passed=true
open_trace_turns=29
open_trace_identity_maps_to_self_model=true
transcript_replay_gate_passed=true
transcript_replay_turns=25
transcript_replay_local_resolution_rate=0.68
transcript_replay_profile_updates=2
inventory_soak_passed=true
inventory_soak_mode=offline_fixture
inventory_soak_source=both
inventory_diversity_smoke_passed=true
inventory_diversity_niches=folktale,bedtime,adventure
inventory_diversity_future_story_routes_local=true
inventory_retry_smoke_passed=true
inventory_retry_transient_failures=2
inventory_retry_future_story_routes_local=true
inventory_retry_external_network_used=false
inventory_failure_smoke_passed=true
inventory_failure_cases=3
inventory_failure_no_fake_story=true
inventory_failure_future_story_routes_missing_inventory=true
inventory_failure_errors_observable=true
inventory_soak_cycles=2/2
inventory_soak_story_inventory_added=4
inventory_soak_required_sources=project_gutenberg_catalog_csv, internet_archive_search_metadata
inventory_soak_missing_sources=0
inventory_soak_failed_import_cycles=0
inventory_soak_failure_observability_present=true
required network/vector DB/ML framework: no
```

Current local API smoke evidence on the development machine:

```text
command=api-smoke --reset --json
passed=true
checks=9/9 true
runtime=stdlib_python_sqlite_http
dependency_class=stdlib_only
endpoint=temporary localhost HTTP server
health_ok=true
seeded_inventory_available=true
parse_debug_endpoint_identity_frame=true
parse_debug_does_not_write_event=true
parse_debug_mapping=basic_nlp -> uol_parse -> chat_frame
ask_route=local_answer
ask_reason=local_story_inventory
ask_synthesis_applied=true
membrane_boundary=none
event_persisted_after_request=true
external network required: no
```

Current realistic API session smoke evidence on the development machine:

```text
command=api-session-smoke --reset --json
passed=true
checks=15/15 true
turns=11
route_counts:
  cached_tool=1
  device_action=4
  local_answer=6
covered:
  identity=self_model_identity
  story=local_story_inventory
  weather=weather_cache_hit
  school_safety=local_common_sense_policy
  media=request + confirmed dry-run play_media
  health=bounded_general_health_guidance
  profile_memory=personal_memory_summary
  meal=memory_plus_weather_cache
  contact=request + confirmed dry-run call_contact
events=11
membrane_decisions=11
homeostatic_snapshots=11
cloud_or_fetch_routes=0
unconfirmed_executed_actions=0
action_without_confirmation_gate=0
low_quality_applied_synthesis=0
```

Current target-device report evidence on the development machine:

```text
command=target-report --reset --json
passed=true
python_supported=true
sqlite_available=true
dataset_audit_passed=true
pi_smoke_passed=true
autoimmune_smoke_passed=true
synthesis_variant_smoke_passed=true
host_action_smoke_passed=true
host_app_probe_reported=true
host_app_requirement_satisfied=true
host_app_probe_configured=false
host_app_probe_skipped=true
api_smoke_passed=true
api_session_smoke_passed=true
capability_probe_passed=true
capability_probe_cases=18
capability_probe_local_device_rate=0.667
capability_probe_bucket_counts=local:8, device_action:4, cloud_handoff:4, blocked:2
bootstrap_runtime_passed=true
open_traces_passed=true
transcript_replay_passed=true
runtime_python=3.13.4
runtime_sqlite=3.49.1
platform=Windows-11-10.0.26200-SP0
raspberry_pi_detected=false
hardware_policy.raspberry_pi_required=false
hardware_policy.raspberry_pi_hardware_optional_for_v01=true
raspberry_pi_requirement_satisfied=true
note=--require-raspberry-pi is optional appliance-specific validation
note=--require-host-app-configured is optional target-app validation
```

Current portable browser/CLI bundle evidence on the development machine:

```text
command=pi-bundle --reset --zip --json
passed=true
self_check=dataset-audit, pi-smoke, autoimmune-smoke, synthesis-variant-smoke, synthesis-stress-smoke, setup-integration-smoke, host-action-smoke, host-app-probe, capability-probe, shortcut-audit, v01-audit, v01-progress, api-smoke, api-session-smoke, ui-smoke, bootstrap-runtime, launcher-smoke, run-open-traces, run-transcript-replay, and calibrate-transcript-replay executed from copied bundle root
dataset_audit_self_check_checks=19/19 true
pi_smoke_self_check_checks=22/22 true
inventory_soak_self_check=true
inventory_soak_matrix_self_check=true
inventory_diversity_self_check=true
inventory_retry_self_check=true
inventory_failure_self_check=true
autoimmune_self_check_checks=34/34 true
synthesis_variant_self_check_checks=15/15 true
synthesis_stress_self_check_checks=14/14 true
setup_integration_self_check_checks=9/9 true
host_action_self_check_checks=7/7 true
host_app_probe_self_check_reported=true
host_app_probe_self_check_skipped=true
capability_probe_self_check_cases=18
capability_probe_self_check_buckets=local:8, device_action:4, cloud_handoff:4, blocked:2
api_self_check_checks=9/9 true
api_session_self_check=true
ui_self_check=true
bootstrap_runtime_self_check=true
launcher_self_check_checks=12/12 true
open_traces_self_check=true
transcript_replay_self_check=true
transcript_calibration_self_check=true
v01_audit_self_check=true
v01_audit_static_shortcut_guard=true
v01_audit_architecture_complete=false
v01_audit_blocker_count=6
v01_progress_self_check=true
v01_progress_remaining_blockers=6
verify_bundle=passed true with all manifest file byte counts and SHA-256 hashes matching
launcher_files=bin/first_run.sh, bin/start_app.sh, bin/health_check.sh, bin/first_run.ps1, bin/start_app.ps1, bin/health_check.ps1, bin/first_run.cmd, bin/start_app.cmd, bin/health_check.cmd, bin/first_run_on_raspberry_pi.sh, bin/start_api.sh, systemd/melm-local-assistant.service.example
tamper_detection=README.md edit fails byte_counts_match and sha256_match
archive_written=true
manifest=bundle_manifest.json with SHA-256 file records
runbook=RUN_PORTABLE_APP.md
generated_smoke_artifacts_removed=true
required network/vector DB/ML framework: no
```

Current packaged first-run launcher evidence on the development machine:

```text
command=first-run-smoke --bundle-root artifacts/local_assistant_os/melm_local_assistant_os_v01_pi_bundle_verify --json
passed=true
checks=16/16 true
platform_launcher=bin/first_run.ps1
returncode=0
json_reports=6
nested_verify_bundle_passed=true
nested_dataset_audit_passed=true
nested_target_report_passed=true
nested_bootstrap_runtime_passed=true
nested_ui_smoke_passed=true
nested_launcher_smoke_passed=true
runtime_db_created=artifacts/local_assistant_os/assistant_v01.sqlite
target_report_artifacts_created=true
completion_message_present=true
```

Current zip archive handoff evidence on the development machine:

```text
command=archive-smoke --archive artifacts/local_assistant_os/melm_local_assistant_os_v01_pi_bundle_verify.zip --work-dir artifacts/archive_extract_verify --reset --json
passed=true
archive_entries_safe=true
top_level_roots=melm_local_assistant_os_v01_pi_bundle_verify
windows_path_budget_ok=true
single_extracted_bundle_root=true
verify_bundle_passed=true
first_run_smoke_passed=true
first_run_json_reports=6
extracted_runtime_db_created=true
required network/vector DB/ML framework: no
```

Current multi-profile lifecycle suite evidence:

```text
scenarios=3
steps=34
local_resolution_rate=0.647
cloud_handoffs=3
external_fetches=1
clarifications=7
blocked_offline=3
confirmations_required=3
actions_executed=3
opportunities_by_kind:
  ask_household_memory=1
  ask_routine_memory=1
  build_media_index=1
  build_story_inventory=1
  refresh_weather_cache=1
  request_trusted_contact=1
safety_flags:
  cloud_private_inclusions=0
  unconfirmed_executed_actions=0
  fake_latest_news_local_answers=0
  low_quality_applied_synthesis=0
  dangling_memory_links=0
scenario_reports:
  child_cold_start_story_weather_action_offline: steps=17, local_resolution_rate=0.706
  adult_media_routine_household_setup: steps=10, local_resolution_rate=0.600
  elder_sparse_offline_contact_story: steps=7, local_resolution_rate=0.571
```

Current household-week lifecycle evidence:

```text
steps=37
local_resolution_rate=0.676
cloud_handoffs=3
external_fetches=1
clarifications=7
blocked_offline=1
confirmations_required=3
actions_executed=2
route_counts:
  cached_tool=1
  clarify=7
  cloud_handoff=3
  device_action=5
  external_fetch=1
  local_answer=19
  reject=1
opportunities_by_kind:
  ask_household_memory=1
  ask_routine_memory=2
  build_media_index=1
  build_story_inventory=1
  refresh_weather_cache=1
  request_trusted_contact=1
digest:
  local_only=true
  session_count=6
  event_count=33
architecture_checks=11/11
safety_flags:
  cloud_private_inclusions=0
  unconfirmed_executed_actions=0
  fake_latest_news_local_answers=0
  low_quality_applied_synthesis=0
  dangling_memory_links=0
  action_replay_blocks=1
  cancelled_pending_actions=1
  consent_revocations=1
```

Current multi-profile eval evidence:

```text
profiles=12
cases=105
passed=105
pass_rate=1.000
local_or_device_resolved=66
local_resolution_rate=0.629
cloud_handoffs=9
external_fetches=1
clarifications=19
blocked_routes=10
confirmations_required=12
action_replay_blocks=3
confirmation_target_mismatches=3
consent_revocations=4
privacy_exposures=0
privacy_blocks=10
wrong_local_answers=0
unsafe_local_actions=0
overblocks=0
fake_latest_news_local_answers=0
memory_events=105
memory_recent_sessions=1
memory_recent_session_event_count=105
memory_dangling_links=0
synthesis_traces=97
synthesis_avg_quality=0.969
synthesis_min_quality=0.833
synthesis_low_quality_applied=0
synthesis_warning_counts={}
```

Covered profiles:

```text
child_lagos_inventory_and_boundaries
adult_professional_routine
elder_care_low_connectivity
traveler_offline_local_first
accessibility_action_memory
child_ready_phrase_variants
adult_health_meal_cloud_variants
elder_sparse_offline_variants
traveler_offline_action_privacy_variants
accessibility_action_privacy_variants
household_shared_device_variants
open_domain_and_health_safety_variants
```

Current cold lifecycle dashboard evidence:

```text
events=17
membrane_decisions=17
homeostatic_snapshots=17
memory_sessions=1
memory_linked_previous=16
memory_linked_next=16
dangling_memory_links=0
synthesis_traces=11
synthesis_avg_quality=0.989
synthesis_min_quality=0.944
synthesis_low_quality_applied=0
synthesis_warnings=0
ledger_complete=true
cloud_private_inclusions=0
unconfirmed_executed_actions=0
action_without_confirmation_gate=0
fake_latest_news_local_answers=0
completed_jobs=2
executed_pending_actions=1
```

Current action replay/cancel smoke evidence:

```text
replay_without_pending=no_pending_action_to_confirm
first_action_route=device_action
first_action_confirmation_required=1
cancel_reason=cancelled_pending_action
replay_after_cancel=no_pending_action_to_confirm
pending_total=1
pending_cancelled=1
pending_executed=0
action_replay_blocks=2
cancelled_pending_actions=1
ledger_complete=true
```

Current typed action command smoke evidence:

```text
command=action-smoke --reset --json
mode=dry-run
passed=true
action_results:
  play_media: status=prepared, side_effect_executed=false,
              resolved_target=benchmarks/media/calm_piano.mp3
  call_contact: status=prepared, side_effect_executed=false,
                resolved_target=+234-000-ADA
safety_flags:
  unconfirmed_executed_actions=0
  action_without_confirmation_gate=0
  cloud_private_inclusions=0

real_command_regression:
  mode=real
  configured_media_command_status=executed
  configured_call_command_status=executed
  media_command_receives_existing_file=true
  call_command_receives_resolved_contact_target=+234-000-ADA
  shell_used=false
```

Current autoimmune boundary smoke evidence:

```text
command=autoimmune-smoke --reset --json
passed=true
checks=34/34 true
turns=26
private_cloud_block=blocked_private_facts_to_cloud
generic_cloud_allowed=cloud_handoff without private evidence
public_profile_cloud_allowed=facts.public_profile with consent/local_only/cloud_eligible policy
mixed_public_household_cloud=blocked_private_facts_to_cloud with no partial private cloud inclusion
cloud_private_inclusions=0
forget_reason=consent_revoked_user_fact
revoked_fact_absent_after_reload=true
child_school_consent_revoke=consent_revoked_user_fact
child_school_after_revoke=personal_memory_empty
child_age_after_school_revoke=facts.child_age
child_school_generic_fallback=false
mismatch_reason=confirmation_target_mismatch
media_mismatch_reason=confirmation_target_mismatch
pending_total=2
pending_cancelled=2
pending_executed=0
final_pending_actions=0
confirmation_target_mismatches=2
consent_revocations=1
stale_weather_route=external_fetch
stale_weather_reason=weather_cache_miss
stale_weather_cache_freshness=0.0
parent_child_private_cloud=blocked_private_facts_to_cloud
child_location_private_cloud=facts.child_location blocked, profile.location not used
conversation_export_block=blocked_private_facts_to_cloud
cloud_private_inclusions=0
unconfirmed_executed_actions=0
```

Current bounded synthesis variant smoke evidence:

```text
command=synthesis-variant-smoke --reset --json
passed=true
checks=15/15 true
turns=10
route_counts=local_answer:9, cached_tool:1
reason_counts:
  local_story_inventory=3
  bounded_general_health_guidance=2
  urgent_health_safety_escalation=1
  weather_cache_hit=1
  memory_plus_weather_cache=1
  autobiographical_session_summary=1
  autobiographical_memory_digest=1
story_tale_not_exact_story_phrase=true
health_sleep_not_exact_health_phrase=true
session_summary_citations=events.*
long_horizon_digest_citation=memory_digest.long_horizon_latest
primary_parse_basis=uol_chat_frame
primary_domain_evidence=slot_role_relation
secondary_phrase_hints_in_primary_route=false
quality_clean=true
low_quality_applied_synthesis=0
cloud_private_inclusions=0
```

Current bounded synthesis stress smoke evidence:

```text
command=synthesis-stress-smoke --reset --json
passed=true
checks=14/14 true
turns=24
sessions=3
route_counts=local_answer:22, cached_tool:2
reason_count=14
intent_count=14
quality_min=0.799
quality_avg=0.961
quality_max=1.0
complexity_max=0.57
story_variants_remain_local_and_cited=true
health_variants_and_urgent_policy_cited=true
weather_meal_and_safety_use_local_state=true
self_status_and_identity_use_self_evidence=true
autobiographical_summaries_use_events_and_digest=true
primary_parse_basis=uol_chat_frame
primary_domain_evidence=slot_role_relation
secondary_phrase_hints_in_primary_route=false
low_quality_applied_synthesis=0
cloud_private_inclusions=0
```

Current bounded synthesis evidence:

```text
seeded story ask:
  route=local_answer
  synthesis_applied=true
  synthesis_quality=0.978
  synthesis_warnings=none
  citations=story_models.public_domain_folktale_lagos_age7, profile.location
  story_source=local_seed_public_domain_story_frame
  answer_shape=multi_sentence_story_from_title_and_bound_profile_location

health advice:
  synthesis_quality>=0.65
  citations=health_goals.0, health_goals.1, local_health_safety_policy
  answer_shape=bounded_general_guidance_with_one_small_local_goal

urgent health:
  answer_shape=symptom_specific_escalation_without_diagnosis
  synthesis_warnings=none

weather/cache:
  answer_shape=cached_forecast_with_refresh_boundary
  synthesis_warnings=none

meal scorer:
  route=local_answer
  scorer_inputs=food_inventory, preferences, utterance_scope, cached_weather
  static_combo_ladder=false
  synthesis_warnings=none

media cancellation:
  answer_shape=cancelled_pending_media_action_with_no_replay
  synthesis_warnings=none

public clothing safety:
  answer_shape=proper_clothes_public_boundary
  synthesis_warnings=none

personal memory summary:
  route=local_answer
  reason=personal_memory_summary
  synthesis_quality>=0.65
  citations=profile.age, profile.location, profile.culture, facts.*, preferences.*
  admitted_evidence_count=7-8

autobiographical chat recall:
  route=local_answer
  reason=autobiographical_memory_summary
  citations=events.*
  answer_shape=bounded_summary_from_local_conversation_memory
  last_question_limit=1 event
  prior_conversation_cloud_export=blocked_private_facts_to_cloud

consent revocation:
  reason=consent_revoked_user_fact
  synthesis_citation=local_privacy_policy.consent_revocation
  revoked_fact_value_echoed=false

private-cloud block:
  route=reject
  synthesis_refused=true
  synthesis_reason=membrane_blocked
  synthesis_expected_refusal=true
```

Current metadata import evidence:

```text
offline replay samples:
  command=import-stories --cold-start --source both
  imported_items=4
  inventories=4
  gutenberg_selected=2
  internet_archive_selected=2
  imported story ask route=local_answer
  imported story synthesis_applied=true
  imported story answer starts with=I picked The Rain Map Bedtime Story
  imported story static_catalog_frame_used=false
  persisted_metadata_fields=narrative_frame, topics, cultures, quality_score, local_fit_score, metadata_quality
  importer_hardening=retry_backoff, canonical_title_dedupe, metadata_quality_floor

replayable Internet Archive live-path pagination:
  pages_fetched=2
  max_pages=3
  page_size=2
  source_count=3
  selected_count=3
  fetch_attempts_total=2
  rate_limit_sleep_count=1
  rate_limit_delay_total_seconds=2.0
  next_cursor=
  byte_budget_exhausted=false

live Project Gutenberg smoke:
  source=project_gutenberg_catalog_csv
  source_count=78601
  imported_items=2
  network_used=true
  first_title=Aesop's fables in words of one syllable
  first_topics=folktale, animal
```

Current scheduled refresh evidence:

```text
schedule_command=schedule-refreshes --cold-start --offline-samples --min-story-models 4 --story-limit 3
recommendations=2
recommendation_kinds=import_story_metadata, refresh_weather_cache
story_priority=0.88
story_priority_signals=inventory_gap:4, expected_local_resolution_gain:1.0
weather_priority=0.74
weather_priority_signals=weather_today_cached:false, expected_local_resolution_gain:1.0
kernel_reflection_pressure:
  ask_profile_memory_priority=0.691
  ask_profile_memory_signals=profile_memory_misses:1, avg_uncertainty:0.31, avg_local_capability:0.333
  request_trusted_contact_priority=0.799
  request_trusted_contact_signals=trusted_contact_misses:2, avg_uncertainty:0.31, avg_local_capability:0.333
  priority_order=trusted_contact_above_profile_memory
future_opportunity_pressure:
  build_media_index_priority=0.708
  build_media_index_signals=media_misses:1, avg_uncertainty:0.333, avg_local_capability:0.5
  ask_routine_memory_priority=0.680
  ask_routine_memory_signals=routine_memory_misses:1, avg_uncertainty:0.333, avg_local_capability:0.5
  ask_household_memory_priority=0.675
  ask_household_memory_signals=household_memory_misses:1, avg_uncertainty:0.333, avg_local_capability:0.5
cold_start_media_cli:
  route=clarify
  reason=empty_media_library
  opportunity=build_media_index
story_inventory_count_before=0
weather_today_cached_before=false
queued_jobs_after_schedule=2
run_command=run-jobs
run_jobs_executed=import_story_metadata, refresh_weather_cache
imported_story_metadata_items=4
completed_jobs=2
queued_jobs_after_run=0
inventories_after_run=12
dashboard_importer_health=completed_import_jobs:1, imported_items:4, selected_items:4, raw_rejected_items:3
dashboard_pagination_health=pages_fetched:0, fetch_attempts_total:0, rate_limit_sleep_count:0
dashboard_story_quality=with_quality_scores:4, below_metadata_quality_floor:0, avg_metadata_quality:0.838
story_route_after_run=local_answer
story_synthesis_applied=true
story_citations=story_models.ia_rainmapbedtimestory00test, profile.location

pagination_budget_smoke=schedule-refreshes --cold-start --offline-samples --min-story-models 4 --story-limit 3 --internet-archive-page-size 100 --internet-archive-max-pages 2 --internet-archive-rate-limit-delay-seconds 0.125
queued_story_budget=internet_archive_page_size:100, internet_archive_max_pages:2, internet_archive_rate_limit_delay_seconds:0.125

two_cycle_refresh_trend_smoke:
  first_job=import_story_metadata:inventory_scheduler_story_model_thin
  second_job=import_story_metadata:inventory_scheduler_story_model_thin:cycle_2
  completed_import_jobs=2
  importer_trend_cycles=2
  imported_items_total=8
  selected_items_total=8
  avg_metadata_quality=0.837
  quality_delta=0.0
  story_quality_below_floor=0

offline_inventory_soak_readiness:
  command=inventory-soak --offline-samples --source both --cycles 2 --story-limit 3 --min-story-models 12 --json
  passed=true
  checks=12/12 true
  mode=offline_fixture
  network_used=false
  source=both
  successful_import_cycles=2
  failed_import_cycles=0
  completed_import_jobs=2
  imported_items_total=8
  selected_items_total=8
  story_inventory_added=4
  source_coverage=project_gutenberg_catalog_csv:2, internet_archive_search_metadata:2
  missing_sources=0
  failure_observability_present=true
  recent_cycle_count=2
  failed_import_jobs=0
  byte_budget_exhausted_results=0
  story_quality_below_floor=0

optional_live_internet_archive_soak:
  command=inventory-soak --source internet-archive --cycles 2 --story-limit 3 --internet-archive-page-size 100 --internet-archive-max-pages 2 --internet-archive-rate-limit-delay-seconds 0.1
  mode=live_metadata
  network_used=true
  use=optional importer hardening beyond the repeatable readiness gate
```

Current autobiographical replay/query evidence:

```text
command=memory-replay --query story --json
matches=1
first_event_id=os_e1
last_event_id=os_e1
returned_events=1
linked_next_in_ledger=1
dangling_previous=0
dangling_next=0
event_intents=story
dashboard_memory_events=3
dashboard_memory_sessions=3
recent_session_summaries=3

chat_recall:
  prompt=What did we talk about earlier?
  route=local_answer
  intent=autobiographical_memory
  reason=autobiographical_memory_summary
  evidence=events.*
  synthesis_applied=true
  local_only=true

last_question_recall:
  prompt=What was my last question?
  route=local_answer
  evidence_limit=1
  excludes_older_story_event=true

private_conversation_export:
  prompt=Send our previous conversation to the cloud.
  route=reject
  reason=blocked_private_facts_to_cloud
  evidence=events.local_conversation
  synthesis_refused=true

recent_session_replay:
  command=memory-replay --sessions 3 --events-per-session 1 --json
  session_count=3
  matches=3
  session_ids=session_1, session_2, session_3

chat_session_summary:
  prompt=Summarize our recent sessions.
  route=local_answer
  intent=autobiographical_memory
  reason=autobiographical_session_summary
  citations=3 events.*
  synthesis_applied=true
  answer_groups_by_session=true

long_horizon_digest:
  command=memory-digest --sessions 20 --events-per-session 3 --json
  inventory_kind=memory_digest
  digest_id=long_horizon_latest
  local_only=true
  source=assistant_event_ledger_compactor
  evidence_key=memory_digest.long_horizon_latest
  dashboard_memory_digests>=1

chat_long_horizon_recall:
  prompt=What happened over the last few days?
  route=local_answer
  intent=autobiographical_memory
  reason=autobiographical_memory_digest
  citations=memory_digest.*
  raw_event_keys_returned=false
  synthesis_applied=true
```

## External Inventory Feasibility

Project Gutenberg explicitly provides
[machine-readable catalog metadata](https://www.gutenberg.org/ebooks/offline_catalogs.html)
and directs automated tools to use catalog files rather than crawling the
website. It also documents
[robot-access harvest endpoints](https://www.gutenberg.org/policy/robot_access.html)
for certain file types.

Internet Archive provides
[item search APIs](https://doc-tools.readthedocs.io/en/ia-test-gsod/item-search-apis.html)
and
[metadata APIs](https://doc-tools.readthedocs.io/en/ia-test-gsod/metadata.html). Its search APIs
support metadata search, advanced search, and cursor-based scraping for deep
paging; its scrape API documents `count` with a minimum of `100`, so the v0.1
importer normalizes lower requested page sizes to `100` before live calls. Its
metadata API exposes per-item metadata and file lists.

MVP implication:

```text
Public-domain inventory building is feasible as a background job, but must use
official catalog/API paths, source metadata, cache budgets, and copyright/license
filters. The MVP should ingest metadata first, not blindly scrape text.
```

## Architecture

### 1. MembranePolicy

The membrane decides what may cross boundaries:

```text
local-only user facts
cloud-eligible context
tool-fetch context
device-action context
confirmation-required actions
blocked/unsafe actions
child-safety boundaries
consent requirements
offline behavior
```

This is the assistant OS nucleus. The assistant can be smart only after it knows
what must stay inside the device and what is allowed to leave.

Initial policy shape:

```text
local_only:
  age, child profile, contacts, private preferences, health goals

cloud_allowed_if_needed:
  generic story prompt without private profile
  open-domain factual question without personal context

fetch_allowed:
  weather by coarse location
  public-domain catalog metadata

requires_confirmation:
  call contact
  message contact
  play media at volume
  purchase/order/open external app

never_fake:
  latest news while offline
  medical diagnosis
  personal fact not in memory
```

Gate: every local/tool/action/cloud route must carry a membrane decision.

### 2. HomeostaticState

Homeostasis is the assistant's live health dashboard for the user/device
relationship:

```text
user_safety
privacy_risk
cloud_dependence
local_capability
uncertainty
cache_freshness
action_risk
user_trust
inventory_coverage
```

The planner should optimize this state over time:

```text
reduce cloud dependence
reduce privacy exposure
increase local capability
keep uncertainty explicit
keep action risk gated
refresh stale caches
build missing inventories only when evidence justifies it
```

Gate: lifecycle reports must include homeostatic deltas, not only route counts.

### 3. UserModel

Stores facts with source and confidence:

```text
age
location
culture/language hints
likes/dislikes
routine
contacts
food preferences
health goals
story/music interests
consent and privacy flags
```

Rule: the assistant must distinguish remembered fact, inferred preference, and
unknown. It should not silently invent stable user facts.

### 4. SelfModel

The assistant's operational self-awareness:

```text
purpose
local capabilities
cloud capabilities
known limits
inventory counts
tool/cache status
resource budget
safety policies
```

This is what lets it reason:

```text
I can answer this locally.
I lack a story inventory.
I should fetch weather, not ask a large model.
I should ask for a trusted contact before I can call someone.
```

### 5. AutobiographicalMemory

Every meaningful turn becomes an event:

```text
utterance
intent
route
answer/action
evidence keys
cloud handoff yes/no
failure or gap reason
privacy exposure
```

This is not chat transcript replay. It is structured memory for reflection and
future routing.

### 6. OpportunityPlanner

Periodically scans autobiographical memory and self model:

```text
repeated story cloud handoffs -> build_story_inventory
weather cache miss -> refresh_weather_cache
personal memory empty -> ask profile question
missing trusted contact -> request trusted contact
frequent food questions -> build meal inventory
frequent local language requests -> build translation/phrase inventory
```

Each opportunity includes:

```text
priority
priority signals
reason
evidence event ids
expected cloud reduction
proposed action
source candidates
```

### 7. Inventory Builders

Background jobs that prepare local capability:

```text
story inventory
weather cache
meal plans
music/media index
contacts and social graph
school/work routine
local safety/common-sense policies
```

Story inventory builder should start with metadata:

```text
source -> metadata -> license/public-domain filter -> age/topic/culture tags
-> short summaries -> local retrieval index -> safe remix frames
```

### 8. TypedActionGate

Device actions are not ordinary answers. They need a separate execution
contract:

```text
action type
target
source user request
preconditions
membrane decision
confirmation state
execution result
rollback/cancel state
```

Initial actions:

```text
call trusted contact
message trusted contact
play local media
open external app
start background inventory job
```

Gate: no side effect leaves the assistant until the action plan, membrane
decision, and confirmation state all agree. A confirmation utterance without a
live pending action must clarify instead of executing, and a cancelled pending
action must not be resurrected by a later confirmation.

### 9. Triage Runtime

Every request routes through:

```text
local answer
cached tool
device action
external fetch
cloud handoff
clarify
reject/unsafe
```

Cloud is a capability, not the operating center.

### 10. Budgeted Evidence Boundary

The final verbalizer receives:

```text
compact state
top-k admitted evidence
matching evidence counts
bound plan
route intent
```

It does not receive the whole life transcript.

## Why This Is the Biggest MVP

### Compared with more vocabulary

More words increase interpretation coverage, but do not create useful action.
The vocabulary-only probe handled `1/8` realistic assistant asks locally.
That probe must remain secondary-lexical only and must not borrow UOL/ChatFrame
composition from the winning architecture.

### Compared with cloud-first

Cloud-first can answer broadly but loses the OS-layer thesis:

```text
more latency
more cost
more privacy exposure
less local compounding
weaker offline behavior
```

The cloud-first probe handled only `2/8` locally and sent `5/8` to cloud.

### Compared with thin tools

Thin tools handle weather/music/calls but miss the user's personal memory and
the assistant's ability to prepare future local capability.

Thin tools handled `4/8` locally; memory-centric triage handled `8/8`.

### Compared with budgeted evidence alone

Budgeted evidence is necessary but not sufficient. It makes memory scalable; the
self-model and opportunity planner make memory useful.

## Worst Case

Even if no disruptive local SLM emerges, the MVP still becomes valuable as:

```text
private user memory
cloud minimizer
cache/tool/action router
background inventory planner
offline-first assistant shell
```

That alone can reduce cloud calls for routine assistant traffic and make any
cloud LLM more useful by sending compact, typed context instead of raw history.

## Best Case

The assistant becomes a base OS layer:

```text
local memory substrate
personal context graph
self-improving inventory manager
tool/action policy layer
small-model runtime
cloud fallback broker
```

Apps do not each build their own assistant memory. They plug into the local
assistant OS memory and triage layer.

## Build Phases

| Phase | Build | Evidence carried forward | Exit gate |
|---|---|---|---|
| 0. Alignment | Keep root docs, roadmap, and supporting memos pointed at this OS plan. | Drift scan shows no older MVP wording competing with `MELM Local Assistant OS v0.1`. | Root docs name the OS plan as product direction and validation docs as support. |
| 1. Explicit kernel | Add first-class `MembranePolicy`, `MembraneDecision`, `HomeostaticState`, and typed action plans. | Lifecycle already shows offline block, confirmation, local/cloud/fetch choices, and no fake latest news. | Every route emits membrane and homeostatic records. |
| 2. Persistent memory | Add SQLite stores for user facts, self state, events, opportunities, inventories, membrane decisions, and homeostatic snapshots. | Current probes already produce structured events and inventory counts in memory. | Restart/reload preserves profile, inventory, events, and pending opportunities. |
| 3. Inventory loop | Add job queue and metadata-only public-domain/weather/media/contact inventory builders. | Repeated story cloud handoffs already become local story inventory; weather miss already becomes cache fill. | Repeated gaps create jobs, jobs update inventory, future routes change from cloud/fetch to local/cache. |
| 4. Evaluation harness | Expand lifecycle suite across realistic profiles and autoimmune failures. | Current evidence combines the 17-step cold lifecycle, 3-scenario / 34-turn lifecycle suite, 37-turn household-week trace, 2-scenario / 29-turn open trace, 25-turn transcript replay, and 105-case/12-profile eval. Together they cover onboarding, weather, safety, story, personal memory, autobiographical recall, meal, contact, confirmation, action replay, cancellation, consent revocation, invented action targets, offline behavior, private-cloud blocking, conversation-memory cloud-export blocking, parent/child private-cloud blocking, urgent health escalation, media setup, routine setup, household setup, simple profile fact updates, overblocking, wrong-local checks, richer long-horizon digest compaction, and a repaired `history` vs `story` substring failure. | Grow from authored traces into user-derived transcript gates while preserving `0` privacy leaks, `0` unsafe local actions, and `0` fake latest-news local answers. |
| 5. Bounded synthesis | Put local story/advice/summarization behind the same memory/evidence/membrane boundary. | Micro SLM boundary already verbalizes from compact state plus admitted evidence instead of raw transcript; `BoundedLocalSynthesizer` now emits richer cited local/cache answers, cited event-memory recall summaries, refuses blocked/action/cloud routes, and writes quality-scored traces to the SQLite ledger. Current eval/lifecycle dashboards preserve `0` low-quality applied synthesis, and the 105-case eval now reports empty synthesis warning counts. | Broaden story/advice/summarization and longer recall variants while preserving citations and never bypassing membrane/action gates. |

## Next MVP Gate

Name:

```text
MELM Local Assistant OS v0.1
```

Core gate status:

| Gate | Status | Evidence / remaining work |
|---|---|---|
| Every route has membrane and homeostatic records. | Met | `eval --json` emits `105` membrane decisions and `105` homeostatic snapshots across local, cached tool, action, fetch, cloud, clarify, and reject routes. |
| Structured autobiographical memory logs meaningful turns. | Met | Cold lifecycle writes `17` events with matching membrane/homeostasis rows; lifecycle suite extends this to `34` turns across child/adult/elder scenarios; household-week samples `33` events across `6` sessions into a local-only digest and preserves `0` dangling links. Events include session and previous/next links; dashboards report ledger completeness. `memory-replay` gives bounded local replay/query over linked event memory and recent sessions. `memory-digest` stores a local-only long-horizon `memory_digest.*` inventory row with remembered threads, per-session summaries, capability transitions, active limits, open loops, and a deterministic quality score. The household-week digest passes at `1.0 >= 0.72`; one-event thin digests fail with `insufficient_long_horizon`. Chat-native recall now answers prior-conversation, last-question, recent-session summary, and last-few-days asks from cited local evidence instead of raw transcript stuffing. Remaining: calibrate digest quality on user-derived household transcripts. |
| User model stores facts with source/confidence/consent/local-only scope. | Met for v0.1 | Consent revocation and stale reload prevention are implemented. Seed facts now declare `cloud_eligible=false` and `scope=private_local`; SQLite adds backward-compatible `cloud_eligible` and `scope` columns; profile sync preserves existing policy metadata; and membrane tests prove explicitly shareable facts can cross cloud while ordinary private facts remain excluded. Transcript replay proves simple explicit facts like "I am 8 years old" and "I live in Lagos" now route to local `profile_update` instead of cloud handoff. Remaining: richer household/child ownership and provenance calibration. |
| Self model tracks capabilities, limits, inventory counts, and runtime health trends. | Met for v0.1 | Story gaps create inventory work and persisted counts; `runtime_health_trends` and bounded `runtime_health_history` now persist tool/cache/importer/action/synthesis health in SQLite `self_state`; status answers cite `self_status.self_observation`; weather miss/refresh regressions prove the history changes from `weather_cache=missing` to ready and status reports the transition. Kernel reflection and `schedule-refreshes` use history-derived local-resolution deltas and cache/story gap persistence in priority signals. The 29-turn open trace validates weather/story trend pressure under messier lifecycle use, and the 25-turn transcript replay validates status/digest use after profile facts, inventory growth, actions, privacy blocking, and long-horizon recall. Remaining: validate trend weights against user-derived transcripts. |
| Opportunity planner turns repeated gaps into preparation. | Met for current opportunity kinds | Story handoffs create `build_story_inventory`; weather miss creates `refresh_weather_cache`; profile-memory miss creates `ask_profile_memory`; contact miss creates `request_trusted_contact`; empty media asks create `build_media_index`; routine and household gaps create local setup opportunities. Scheduler queues `import_story_metadata` and weather refresh, while kernel reflection assigns pressure-scored priorities for all current opportunity kinds. The 34-turn suite, 37-turn household-week trace, and 29-turn open trace exercise story, weather, media, routine, household, and contact opportunities; setup opportunities no longer fabricate facts or contacts. Remaining: validate against user-derived traces and connect setup requests to real integrations. |
| Inventory loop changes future routing. | Met | Running story/weather jobs converts future story asks from cloud to local and weather from fetch to cache. `import-media` writes manifest or directory media metadata into SQLite, and `build_media_index` changes future media asks from clarify to gated device action without hardcoded media names. Initial live fetch retry/backoff, dedupe, quality thresholds, importer observability, bounded multi-page IA cursor walking, page/fetch/rate-limit dashboards, quality dashboards, completed-cycle preservation, import trend dashboards, the two-cycle offline both-source `inventory-soak` readiness gate, the cold-start multi-source `inventory-soak-matrix` gate, the multi-niche `inventory-diversity-smoke` growth gate, the localhost transient-failure `inventory-retry-smoke` gate, and the offline `inventory-failure-smoke` no-fabrication gate are implemented. Remaining: longer live source soak/retry modes and target-device app playback smoke with actual media players. |
| Seed/source/bootstrap dataset audit exists. | Met | `dataset-audit --reset --json` validates `8/8` required files with SHA-256 hashes, `19/19` checks, explicit seed privacy policy fields (`local_only`, `cloud_eligible`, `scope`), `4` story metadata items, `3` media manifest items, `7` weather days, `2+` Gutenberg and Internet Archive story candidates, `2` scenarios / `29` open-trace turns, `25` transcript replay user turns, and a SQLite bootstrap profile with `6` user facts, `8` inventories, Maya age `7`, Lagos/Yoruba, story/weather/contact/media/food coverage. `pi-smoke`, `pi-bundle`, `verify-bundle`, `target-report`, and first-run launchers include this gate. |
| Memory-centric triage beats baseline directions. | Met | Original eight-ask probe: memory-centric triage `8/8` local/device vs thin tools `4/8`, cloud-first `2/8`, vocabulary-only `1/8`. The vocabulary-only baseline is locked to secondary lexical hints and must not borrow UOL/ChatFrame composition. The transcript-level comparison over the same 25 authored user turns is stronger: current kernel `17/25` local/device versus best static baseline `7/25`, `+0.40` local-resolution gain, `7` fewer cloud handoffs, `3` fewer clarifications, `0` private-cloud exposure, `2` profile-update advantage, one private-cloud block advantage, and one long-horizon digest advantage. Current eval preserves `0` leaks/unsafe/wrong/fake-news failures. |
| MVP capability surface exposes can-handle and cannot-handle cases. | Met | `capability-probe --reset --json` runs `18` realistic cases through the real SQLite/kernel path. It reports `bucket_counts={local:8, device_action:4, cloud_handoff:4, blocked:2}`, local/device rate `0.667`, `6` unsupported/blocked examples with route/reason, primary routing basis, and secondary debug hints, dry-run-only action confirmations, clean safety flags, and Basic NLP -> UOL -> ChatFrame mapping with complexity and unknown-token counts for every case. `target-report`, `pi-bundle`, and `verify-bundle` now include this gate through `portable_capability_probe_command`. |
| Budgeted evidence shows at least 4x compression by 32 events. | Met | Positive evidence check improves to `4.55x` at 32 moves and `17.46x` at 128 moves. |
| Device actions require typed plan, membrane approval, and confirmation. | Met for current actions | Eval includes `12` confirmation-required actions and `0` unconfirmed executions; replay, cancel, and target mismatch are covered. Confirmed actions now pass through `LocalDeviceActionExecutor`, defaulting to dry-run and blocking real mode without a configured command. `action-smoke` prepares media/contact in dry-run and regression-tests configured real commands for both action kinds. `host-action-smoke` runs harmless real recorder commands and proves an existing media file plus resolved contact target are passed by argv-list subprocess execution with clean safety flags. `host-app-probe` reports target app-command configuration and can execute supplied media/call commands through the same typed gate from CLI args, environment, or `--config-json config/host_actions.json`; target-specific runs can require this with `--require-configured`. |
| Local API surface can run a chat turn and parse-only debug request. | Met | `api-smoke --reset --json` starts the stdlib localhost API, verifies `/health`, posts an identity challenge to `/parse-debug` and confirms Basic NLP -> UOL -> ChatFrame mapping without writing an event, posts one story ask to `/ask`, confirms local story synthesis plus membrane/homeostasis records, verifies the event persisted in SQLite, checks `GET /dashboard`, and proves `GET /event-transcript-replay` exports user turns without stored answers/routes/reasons or static expectations. |
| Local API surface can run and calibrate a realistic assistant session. | Met | `api-session-smoke --reset --json` runs `11` localhost `/ask` turns covering assistant identity, story, weather, school safety, media confirmation, health, profile memory, meal, and trusted-contact confirmation. It produces route counts `local_answer=6`, `device_action=4`, `cached_tool=1`, uses no cloud/fetch routes, prepares two dry-run actions behind confirmation, logs `11` events/membrane/homeostasis rows, keeps safety flags clean, exports all `11` turns through `GET /event-transcript-replay`, and passes `POST /calibrate-event-ledger` over that live ledger with no answer/route/reason export. Those smoke turns are labeled `scripted_api_smoke`, so they verify the API path but remain development evidence unless replaced by imported-redacted, interactive CLI, served browser UI, or target-device capture. |
| Local browser chat UI can run through the same assistant and calibration path. | Met | `ui-smoke --reset --json` loads the stdlib-served browser UI at `/` and `/index.html`, verifies the chat form, `/health`, `/parse-debug`, `/ask`, `/event-transcript-replay`, and `/calibrate-event-ledger` wiring, Basic NLP -> UOL -> ChatFrame debug display, auto-open hook for suspicious parser output, operator export/calibration controls, identity/status/story/action-confirmation turns, and the same persisted SQLite event/calibration path. Manual `curl.exe` checks also verified health, HTML shell, JS wiring, local story route, and persisted event. |
| Cross-platform CLI can run a multi-turn chat session. | Met | `chat --turn ... --json` runs scripted local terminal sessions through the same kernel/store path. Regression coverage verifies a three-turn story/weather/safety session writes `3` events, `3` membrane decisions, and `3` homeostatic snapshots. Interactive `chat` is available for terminal use on Windows, macOS, Linux, and Raspberry Pi OS with Python 3.11+. |
| One-command readiness gate exists. | Met | `pi-smoke --reset --json` checks required datasets, `dataset-audit`, seeded SQLite memory, local story synthesis, 17-step lifecycle, typed media/contact action preparation with resolved local targets, the `setup-integration-smoke` routine/household/trusted-contact gap-to-user-setup proof, the 10-turn `synthesis-variant-smoke` gate, the 24-turn / 3-session `synthesis-stress-smoke` gate, the 29-turn open trace debug-parser gate, the 25-turn transcript replay gate with baseline comparison, the offline both-source inventory-soak gate, the cold-start multi-source `inventory-soak-matrix` gate, the multi-niche `inventory-diversity-smoke` source-query gate, the transient-source `inventory-retry-smoke` retry gate, the negative `inventory-failure-smoke` no-fabrication gate, complete ledgers, clean safety flags, and stdlib Python + SQLite with no required network/vector DB/ML framework. Reference development runs pass all `22/22` checks. |
| One-command release-candidate matrix exists. | Met | `v01-acceptance --reset --json` runs `target-report`, a real scripted `chat` session over story/weather/safety, and `v01-audit`; it reports requirement rows for target smokes, datasets/bootstrap, readiness smoke plus inventory matrix, cross-platform CLI chat, API/browser UI, transcript/synthesis gates, setup/action gates, direct `shortcut-audit --json` anti-static UOL/ChatFrame discipline, optional bundle evidence, and the explicit blocker boundary. `--host-app-config-json config/host_actions.json --require-host-app-configured` extends the same matrix to configured target media/call apps without changing the browser/CLI release-candidate path. Passing means browser/CLI `release_candidate=true`, not `architecture_complete=true`. |
| Portable browser/CLI bundle exists and its launchers are runnable. | Met | `pi-bundle --reset --zip --json` builds a portable bundle with the CLI, local package, fixtures, host-action template, runbook, manifest, generic Unix launchers, Windows PowerShell/`.cmd` launchers, Raspberry/Linux launchers, systemd example, optional zip archive, and a self-check that runs `dataset-audit`, `pi-smoke`, `autoimmune-smoke`, `synthesis-variant-smoke`, `synthesis-stress-smoke`, `setup-integration-smoke`, `host-action-smoke`, `host-app-probe`, `capability-probe`, `shortcut-audit`, `v01-audit`, `v01-progress`, `api-smoke`, `api-session-smoke`, `ui-smoke`, `bootstrap-runtime`, `launcher-smoke`, `run-open-traces`, `run-transcript-replay`, and `calibrate-transcript-replay` from inside the copied bundle root. `verify-bundle --json` validates manifest-listed files, byte counts, SHA-256 hashes, launcher presence, stdlib-only declarations, non-skipped self-check status, `pi_smoke_inventory_soak_matrix_passed`, `shortcut_audit_passed`, `v01_progress_passed`, and required portable commands including `portable_capability_probe_command`, `portable_shortcut_audit_command`, `portable_v01_audit_command`, `portable_v01_progress_command`, `portable_v01_evidence_pack_command`, transcript/replay/calibration commands, launcher/first-run commands, host-action/host-app commands, and dataset/inventory commands. `first-run-smoke --json` verifies nested bundle integrity, dataset audit, target report, bootstrap runtime, UI smoke, launcher smoke, runtime DB creation, and first-run output. `archive-smoke --json` rejects unsafe zip entries, extracts one bundle root, runs `verify-bundle`, and executes `first-run-smoke`; regression coverage proves a tampered `README.md` fails byte/hash checks. |
| Usable runtime database can be bootstrapped. | Met | `bootstrap-runtime --reset --json` creates the actual runtime DB, imports `3` local media metadata rows, verifies story/weather/school-safety local chat turns, writes `3` event/membrane/homeostasis rows, keeps pending actions at `0`, reports clean safety flags, and prints next `ask`, `serve`, and `dashboard` commands. `target-report`, `pi-bundle`, and the generic first-run launchers run this command as part of their evidence paths. |
| Cross-platform target report artifact exists. | Met | `target-report --reset --json` records Python/SQLite/platform/disk/memory facts and runs `dataset-audit`, `pi-smoke`, `autoimmune-smoke`, `synthesis-variant-smoke`, `synthesis-stress-smoke`, `setup-integration-smoke`, `host-action-smoke`, `host-app-probe`, `capability-probe`, `v01-audit`, `api-smoke`, `api-session-smoke`, `ui-smoke`, `bootstrap-runtime`, `run-open-traces`, `run-transcript-replay`, and `calibrate-transcript-replay`; its nested `pi-smoke` summary includes `inventory_soak_matrix_passed` and matrix profile evidence. It no longer needs Raspberry hardware for the v0.1 browser/CLI acceptance path. `--require-raspberry-pi` remains available as an optional appliance-specific check, and `--host-app-config-json config/host_actions.json --require-host-app-configured` is available for real target media/call app validation. |
| Lifecycle covers onboarding, cache miss/fill, repeated content, inventory, memory, action confirmation, and offline limits. | Met | 17-step cold lifecycle has `0.706` local resolution, `3` story cloud handoffs before inventory, local story after inventory, confirmed call action, and offline latest-news clarification instead of fabrication. The 34-turn lifecycle suite adds adult media/routine/household setup and elder sparse/offline contact behavior with `3` blocked-offline turns, `3` confirmed actions, explicit user-supplied routine/household/contact setup, and zero privacy/action/fake-news/memory-link safety flags. The 37-turn household-week trace adds cancellation/replay, private-cloud rejection, detailed six-session digest recall, `11/11` architecture checks, and `0` privacy/action/fake-news/dangling-link/low-quality-synthesis failures. The 29-turn open trace adds messier identity/status/weather/story/media/contact/routine/household/meal/health/memory/offline coverage with required route transitions and clean safety flags. The 25-turn transcript replay adds authored user-turn JSONL coverage without static answer/route expectations and validates profile updates, complexity scoring, debug mapping, memory-digest quality, and transcript-level baseline superiority over static alternatives. |
| Blocked routes exist and prevent unsafe/unsupported behavior. | Met | Eval includes `10` blocked private-cloud routes, `10` privacy blocks, and `0` cloud private inclusions. |

Remaining blockers before calling the v0.1 architecture complete:

```text
1. User-derived bounded local story/advice/summarization traces beyond the compact 10-turn synthesis gate and the 24-turn authored stress gate.
2. Longer live inventory soak across more cycle counts and real source retry modes. Candidate blocker evidence must be bound to an actual `inventory-soak-matrix --live` artifact set: `melm.inventory_soak_matrix.v1`, `mode=live_metadata`, live fetch counters, strict matrix checks, both source families, per-run DB artifacts with matching SHA-256 hashes, verified story inventory rows, verified future local story events, and primary UOL/ChatFrame story evidence. A report-shaped JSON claim without those generated artifacts is still missing evidence. Current live artifact evidence: `artifacts/local_assistant_os/live_inventory_matrix_probe_escalated.json` passed with 3 cold-start profiles, 9 completed live cycles, 0 failed import cycles, both Gutenberg and Internet Archive source families, 10 story inventory rows added, per-run DB hash matches, and future local story events verified; `artifacts/local_assistant_os/live_inventory_blocker_evidence.json` maps this row to `candidate_evidence_present`, bringing progress-with-that-report to `candidate_blockers_satisfied=1` and `remaining_blocker_count=5`. Full architecture completion still requires the remaining user-derived, digest, planner, lifecycle, and target-app blockers, plus review of live inventory evidence.
3. Planner priority comparison on user-derived traces and real local
   lifecycle sessions beyond the current authored open trace, setup-request,
   media manifest/directory import, browser UI, CLI chat, and user-supplied
   memory capture paths.
4. Real user-derived transcript lifecycle traces beyond the current authored 29-turn open trace and 25-turn transcript replay fixtures.
5. Real user-derived threshold calibration for the richer scored long-horizon
   compactor and route/complexity floors now that the raw-transcript calibration
   gate exists. Strict calibration must include
   `primary_uol_chatframe_not_secondary_phrase_route`, so a report with only
   Basic NLP -> UOL -> ChatFrame objects is not enough; local/device turns must
   cite primary `token_role_relation` or `slot_role_relation` evidence and keep
   secondary phrase hints out of primary routes. Current audit evidence also
   requires those local/cached/device routes to be owned by
   `melm.assistant_frame_registry.v1` with a non-empty `frame_id`.
6. Configured target-device media/call apps passing the typed action gate with
   real command configuration. The target-app proof should run
   `host-app-probe --config-json config/host_actions.json --require-configured --json`
   or the matching `v01-acceptance --host-app-config-json ...` gate against
   actual local media/call apps.
```

## Immediate Build Order

Strategic review now ranks live inventory acquisition as the next headline MVP
proof. The current router, memory, browser/CLI, and dry-run action surfaces are
useful, but the strongest remaining question is whether the assistant can grow
local capability from real public sources without fabricating, leaking private
state, or regressing into phrase/static shortcuts. Target-device apps and
user-derived digest calibration remain important next blockers; SLM-efficiency
work should stay secondary until the live inventory and evidence-binding proof
is stronger.

Completed in the first runnable v0.1 slices:

1. Add persistent SQLite tables for membrane decisions, homeostatic snapshots,
   user facts, self state, events, opportunities, inventories, and pending
   actions.
2. Add consent/source/confidence/local-only fields to stored user facts.
3. Add explicit `MembranePolicy`, `MembraneDecision`, `HomeostaticState`, and
   `TypedActionPlan` primitives.
4. Add initial assistant OS seed dataset and runnable CLI/API.
5. Add restart/reload tests proving profile, inventory, events, opportunities,
   membrane decisions, homeostatic snapshots, and pending actions persist.
6. Add resource-budgeted SQLite job queue with queued/running/completed/error
   states.
7. Add metadata-backed public-domain story inventory builder over a local
   Gutenberg/Internet Archive-style metadata fixture.
8. Add Pi-class resource report for CLI ask, lifecycle, and database size.
9. Add initial autoimmune-failure tests for private-cloud exclusion, invented
   user memory, and unconfirmed actions.
10. Add route/evidence/membrane dashboard over the SQLite ledger.
11. Add realistic assistant eval with privacy, wrong-local, fake-news,
    overblocking, urgent-health, and action-risk metrics; the seed began as
    five profiles and now runs 105 cases across 12 profiles.
12. Move school-clothing safety and urgent-health escalation into the kernel
    router instead of leaving them only in the lifecycle simulator.
13. Add bounded local synthesis traces for membrane-approved local/cache routes
    with citations to inventory, memory, weather, food, health goals, and
    policy inputs.
14. Preserve existing inventory provenance when profile sync reloads seed or
    builder-produced rows, so citations do not degrade to generic sources.
15. Add metadata-only Project Gutenberg CSV and Internet Archive search
    importers with replayable source-response fixtures and `import-stories` CLI
    ingestion into SQLite.
16. Add kernel, CLI, dashboard, and eval coverage for action confirmation replay
    and pending-action cancellation.
17. Add initial kernel, CLI, dashboard, store, and eval coverage for consent
    revocation, stale weather cache misuse, invented confirmation targets, and
    parent/child private-cloud blocking.
18. Scale the assistant eval to 105 realistic utterances across 12 profiles,
    preserving `0` privacy leaks, `0` unsafe local actions, `0` wrong-local
    answers, and `0` fake latest-news local answers after fixing a `history`
    versus `story` substring misroute.
19. Add `schedule-refreshes` and `import_story_metadata` job execution so thin
    story inventory and stale/missing weather queue Pi-budgeted refresh work,
    run offline replay importers, and convert the next story ask to local.
20. Add richer bounded local synthesis from admitted evidence: multi-sentence
    story synthesis from metadata fields, personal memory summaries across
    multiple local facts/preferences, richer health/meal wording, story metadata
    quality fields, and a regression preventing consent-revocation synthesis
    from echoing revoked fact values.
21. Harden metadata importers with stdlib retry/backoff for live source fetches,
    canonical-title dedupe before ranking, and a minimum metadata-quality floor
    before story candidates enter local inventory selection.
22. Add importer/job observability and quality dashboards: import results now
    expose candidate, quality-reject, duplicate-reject, fetch, page/cursor,
    byte-budget, and rate-limit signals; the dashboard summarizes importer
    health, pagination/rate-limit health, priority by kind, retryable queued
    work, and story metadata quality floor compliance.
23. Use homeostatic/job pressure in refresh scheduling: story and weather
    refresh priorities now include inventory gap, recent cloud handoffs/cache
    misses, homeostatic averages/deltas, failed jobs, and expected
    local-resolution gain.
24. Add autobiographical session links: each event now stores `session_id`,
    `previous_event_id`, and `next_event_id`; dashboards report session counts,
    linked previous/next counts, dangling links, and a safety flag for broken
    memory chains.
25. Add bounded-synthesis quality scoring and persistence: each trace now
    records route discipline, citation coverage, evidence strength, answer
    specificity, source diversity, local privacy discipline, warnings, and a
    quality score; dashboards and eval report low-quality applied synthesis as a
    safety signal.
26. Reduce applied synthesis quality failures by enriching urgent-health,
    weather-cache, meal, media/contact-cancel, public-clothing-safety, and
    consent-revocation answers while keeping them deterministic, cited, and
    membrane-bound; the current 105-case eval keeps low-quality applied
    synthesis at `0` and warning counts empty.
27. Add bounded multi-page Internet Archive cursor walking for story metadata
    refreshes, carry page-size/max-page/cursor/rate-limit budgets through the
    scheduler and CLI, and expose pages/fetch attempts/rate-limit sleeps/byte
    exhaustion in importer health dashboards.
28. Preserve completed import refreshes as trendable cycles while keeping
    queued/running work idempotent; dashboard import trends now report recent
    cycles, imported/selected totals, metadata-quality averages/deltas,
    page/fetch totals, failures, and byte-budget exhaustion.
29. Add `inventory-soak` to run repeated resource-bounded refresh cycles from
    the CLI. The repeatable readiness path now runs two offline fixture cycles
    across both Project Gutenberg CSV and Internet Archive metadata, verifies
    story inventory growth, metadata-quality floor compliance, source coverage,
    failure-mode observability, `0` failed import cycles, and `0` network use;
    live Internet Archive fetches remain optional importer hardening.
30. Add bounded `memory-replay` over linked autobiographical events with text,
    intent, route, and session filters; dashboard memory now reports recent
    sessions with per-session intent/route counts.
31. Add chat-native autobiographical recall over the same local event ledger:
    prior-conversation and last-question asks route to local answers with
    cited `events.*` evidence, while conversation-memory cloud export is
    blocked as private local event memory.
32. Add bounded recent-session replay and chat summaries: `memory-replay`
    can return recent sessions with per-session event limits, and
    "Summarize our recent sessions" groups cited `events.*` evidence by
    session without exporting conversation memory.
33. Extend kernel reflection priority pressure beyond story/weather:
    `build_story_inventory`, `refresh_weather_cache`, `ask_profile_memory`, and
    `request_trusted_contact` now carry `priority_signals`; repeated trusted
    contact misses outrank a single profile-memory miss based on homeostatic
    uncertainty and local-capability pressure.
34. Add first future opportunity classes: `build_media_index`,
    `ask_routine_memory`, and `ask_household_memory`. True cold-start media
    asks clarify and surface `build_media_index`; executing it now imports the
    local media manifest path and changes the next media ask to a gated device
    action.
35. Add `run-lifecycle-suite` for a 3-scenario / 34-turn architecture proof:
    child cold-start story/weather/action/offline behavior, adult
    media/routine/household setup, and elder sparse offline/contact behavior.
    The suite exercises all current opportunity classes and reports zero
    privacy/action/fake-news/synthesis/memory-link safety flags. Routine,
    household, and trusted-contact setup requests do not create facts by
    themselves; user-supplied setup statements are required before future
    memory/action routes change.
36. Add consented setup capture for routine, household, and trusted-contact
    memory. Executing setup opportunities now persists `setup_request` rows
    instead of invented facts; explicit user statements such as "My morning
    routine is..." and "Ada is my trusted contact..." store local-only facts or
    contacts, survive reload, and change later memory/action routes. The CLI
    `ask --execute-jobs` path can execute these safe setup requests.
37. Add long-horizon autobiographical memory digests. `memory-digest` compacts
    bounded multi-session event memory into a local-only `memory_digest.*`
    inventory row; dashboards report digest coverage, and chat asks like "What
    happened over the last few days?" cite the digest instead of raw event keys.
    The digest now stores remembered threads, per-session summaries, capability
    transitions, active limits, open loops, and an inspectable quality score so
    local recall can say what got better, what stayed blocked, what still needs
    setup, and whether the compaction is strong enough.
38. Replace the hardcoded media-index demo with a real local media inventory
    importer. `import-media` ingests a JSON manifest or scanned media directory
    into SQLite with `local_device` provenance, tags, paths, and path-existence
    metadata; `build_media_index` uses that path before enabling a gated media
    action.
39. Add a typed local action executor behind confirmation. Confirmed call/media
    actions now write structured execution results; default dry-run records the
    prepared target without side effects, while `--action-mode real` refuses to
    run unless an explicit command is configured.
40. Add `run-household-week` for a 37-turn longer-life proof. It exercises
    household/routine/contact setup, weather refresh/reuse, story
    cloud-to-local conversion, media manifest import, media/contact action
    confirmation, cancellation/replay blocking, private-cloud rejection,
    offline latest-news refusal, and detailed six-session memory-digest recall
    with remembered threads, capability transitions, active limits, and `11/11`
    architecture checks.
41. Add `action-smoke` for media/contact execution readiness. The command
    imports media inventory, stores a trusted contact, requests and confirms
    media/contact actions, and reports structured execution results. Dry-run
    prepares both actions with no side effects; real-mode regressions execute
    configured commands and prove media commands receive existing file paths
    while call commands receive resolved local contact targets.
42. Add `pi-smoke` as a compact v0.1 readiness gate. The command verifies
    required datasets, seeded SQLite memory, local story synthesis behind
    membrane/homeostasis records, the 17-step lifecycle, dry-run typed
    media/contact actions with resolved local targets, the `setup-integration-smoke`
    routine/household/trusted-contact setup proof, the 10-turn
    `synthesis-variant-smoke` bounded synthesis gate, the 24-turn / 3-session
    `synthesis-stress-smoke` bounded synthesis stress gate, the 29-turn open
    trace debug-parser gate, the 25-turn transcript replay gate with same-turn
    baseline comparison, the offline both-source inventory-soak gate, the
    multi-niche inventory-diversity gate, the transient-source inventory-retry
    gate, the negative inventory-failure gate, complete ledgers, clean safety
    flags, stdlib Python + SQLite, and no required network/vector DB/ML
    framework in one JSON report.
43. Add `pi-bundle` as a portable Raspberry Pi package gate. The command copies
    the runnable CLI, local Python modules, required seed/source/media fixtures,
    root/docs plan files, target-device runbook, first-run/API/health launcher
    scripts, and a systemd user-service example; writes a SHA-256 manifest and
    self-check JSON; optionally creates a zip archive; and proves the copied
    tree by running `pi-smoke` including its inventory-soak,
    inventory-diversity, inventory-retry, and inventory-failure checks,
    `autoimmune-smoke`, `synthesis-variant-smoke`,
    `synthesis-stress-smoke`, `setup-integration-smoke`, `host-action-smoke`, `host-app-probe`,
    `api-smoke`, `api-session-smoke`, `ui-smoke`, `bootstrap-runtime`,
    `launcher-smoke`, `run-open-traces`, `run-transcript-replay`, and
    `calibrate-transcript-replay` from inside the bundle root.
44. Add `api-smoke` as a local app/API surface gate. The command starts the
    same stdlib HTTP handler used by `serve`, verifies `GET /health`, posts one
    non-mutating identity challenge to `POST /parse-debug`, posts one local
    story ask to `POST /ask`, checks membrane/homeostasis/event persistence,
    and shuts the server down. `pi-bundle` now runs this smoke from the copied
    bundle root as part of self-check.
45. Add `target-report` as the target-hardware evidence artifact. The command
    records Python, SQLite, platform, machine, disk, Linux memory facts when
    available, Raspberry Pi model detection when available, and runs `pi-smoke`,
    `autoimmune-smoke`, `synthesis-variant-smoke`,
    `synthesis-stress-smoke`, `setup-integration-smoke`, `host-action-smoke`, `host-app-probe`,
    `capability-probe`, `api-smoke`,
    `api-session-smoke`, `ui-smoke`, `bootstrap-runtime`, and
    `run-open-traces`/`run-transcript-replay`/`calibrate-transcript-replay`. On real hardware,
    `--require-raspberry-pi` optionally turns Raspberry hardware detection into
    an appliance-specific pass/fail check.
46. Add `api-session-smoke` as the first realistic working-MVP API session
    gate. The command starts the same stdlib HTTP handler used by `serve`, runs
    eleven local `/ask` turns covering identity, story, weather, school safety, media
    confirmation, health, profile memory, meal, and trusted-contact
    confirmation, verifies no cloud/fetch routes, checks two dry-run confirmed
    actions, and confirms clean persisted ledgers. `pi-bundle` and
    `target-report` now include this session smoke.
47. Add `ui-smoke` as the dependency-free browser-chat gate. The command
    starts the same stdlib HTTP handler used by `serve`, loads `/` and
    `/index.html`, verifies chat form plus `/health`, `/parse-debug`, and
    `/ask` wiring, verifies the Basic NLP -> UOL -> ChatFrame debug frame,
    posts identity/status/story/action-confirmation turns, and confirms SQLite
    event persistence. `pi-bundle` and `target-report` now include this UI
    smoke.
48. Add `verify-bundle` as the portable bundle integrity gate. The command
    checks `bundle_manifest.json`, verifies every listed file exists with the
    expected byte count and SHA-256 hash, requires portable
    Pi/autoimmune/synthesis-variant/host-action/host-app/API/session/UI/bootstrap/
    launcher/open-trace/transcript-replay/target commands, requires the launchers and
    systemd example, checks stdlib-only declarations, and fails by default if
    the bundle self-check was skipped. Regression coverage proves a tampered
    manifest-listed file fails byte/hash verification.
49. Add `bootstrap-runtime` as the usable first-run database gate. The command
    initializes the actual assistant runtime DB, imports initial local media
    metadata, verifies local story/weather/school-safety chat turns, writes
    event/membrane/homeostasis ledgers, checks clean safety flags, and prints
    next `ask`, `serve`, and `dashboard` commands. `pi-bundle` and
    `target-report` now run this command in their evidence paths.
50. Add persisted self-observation trends. `runtime_health_trends` is saved in
    SQLite `self_state` after turns, reflection, queued job execution, direct
    story/media/weather imports, memory digest, and bootstrap paths. Status
    answers now cite `self_status.self_observation`, and regressions prove a
    cold weather gap records missing cache/job pressure while refresh paths
    update the same self-state to ready cache health.
51. Add bounded self-observation history pressure. `runtime_health_history`
    keeps a compact Pi-friendly window of self-observation points, dedupes
    repeated identical snapshots, records local-resolution deltas and
    weather/story gap persistence, and feeds those signals into kernel
    reflection plus `schedule-refreshes` story/weather priorities. Regressions
    prove weather gap persistence raises scheduled refresh pressure and status
    reports the later weather-cache transition.
52. Add `run-open-traces` as a transcript-like architecture gate. The fixture
    runs 2 scenarios / 29 turns through the real kernel, SQLite store,
    scheduler, weather fixture, inventory jobs, action gate, self-observation
    history, and debug parser. Current evidence passes with `0.655`
    local/device resolution, required weather/story/media/contact transitions,
    clean safety flags, and priority signals for weather/story pressure.
    `pi-smoke`, `pi-bundle`, and `target-report` now include this gate so the
    basic NLP -> UOL -> ChatFrame mapping cannot regress outside the portable
    evidence path.
53. Add `autoimmune-smoke` as a compact boundary gate. The command runs 26
    turns / 34 checks through the real seeded SQLite/kernel path and proves private-cloud
    rejection, generic cloud handoff without private evidence, public-profile
    cloud allowance only when policy marks it shareable,
    mixed public-profile plus private household cloud blocking without partial
    leakage, household/shared-device memory cloud blocking, household and
    personal consent revocation across reload, child-owned age/school setup,
    child-school revocation without generic `facts.school` fallback, stale
    weather-cache exclusion, contact and media confirmation-target blocking,
    cancel/replay protection with no final pending action, parent/child
    private-cloud rejection, child-location private-cloud rejection,
    prior-conversation export blocking, and clean blocking safety flags.
    `pi-bundle` and `target-report` now include this gate so these safety
    boundaries stay portable.
54. Add parse-only API debugging, `host-action-smoke`, and `host-app-probe`. `GET/POST
    /parse-debug` now expose the same Basic NLP -> UOL -> ChatFrame mapping as
    the CLI without writing events, and the browser auto-opens that frame for
    suspicious, unknown, or high-unknown-token turns. `host-action-smoke` runs
    harmless real command-mode media/contact actions through local recorder
    subprocesses, proving existing media paths and resolved contact targets
    flow through the confirmation gate with no shell execution. `host-app-probe`
    reports target media/call command configuration and can execute supplied
    commands through that same gate. Self-awareness debug now has two primary
    composition schemas: `melm.identity_uol_composition.v1` for identity/name
    asks and `melm.self_status_uol_composition.v1` for status, ledger,
    local/cloud, and next-step asks. The persistent kernel reuses the same
    self-status composer, so these routes cannot be accepted by a duplicate
    static phrase list while merely displaying UOL/ChatFrame debug afterward.
    commands through the same typed gate; `--require-configured` turns that into
    a target-app acceptance requirement. `api-smoke`, `ui-smoke`, `pi-bundle`,
    and `target-report` now carry these proofs.
55. Add `launcher-smoke` as the portable launcher runtime proof. The command
    starts the copied app through `bin/start_app.ps1`/`bin/start_app.sh`,
    verifies localhost health through the platform health launcher, checks the
    browser shell and parse endpoint, and terminates the server process tree.
    `pi-bundle` now includes this gate in its self-check and `verify-bundle`
    requires `portable_launcher_smoke_command`.
56. Add `dataset-audit`, `first-run-smoke`, and `archive-smoke` as final portable evidence
    guards. `dataset-audit` validates seed/source fixture hashes, story/media/
    weather/open-trace/transcript-replay coverage, and SQLite bootstrap before
    readiness gates can pass. `first-run-smoke` executes the generated platform first-run
    launcher after bundle creation and verifies nested bundle integrity,
    dataset audit, target report, bootstrap runtime, UI smoke, launcher smoke,
    runtime DB creation, and completion output. `archive-smoke` rejects unsafe
    zip paths, extracts a fresh copy, verifies the manifest, and runs the
    extracted first-run launcher. `verify-bundle` now requires
    `portable_first_run_smoke_command`.
57. Add `run-transcript-replay` as the non-static transcript architecture gate.
    The JSONL fixture is explicitly authored and carries no per-turn expected
    answers, routes, reasons, or response text. The runner converts user
    transcript rows into the real open-trace path, scores route diversity,
    complexity, unknown tokens, Basic NLP -> UOL -> ChatFrame debug maps,
    memory-digest quality, and real ledger writes. The gate caught and fixed
    simple age/location facts that were previously cloud handoffs; they now
    persist locally as `profile_update`. It now also compares the same 25 user
    turns against static structural baselines: current kernel `17/25`
    local/device versus best static baseline `7/25`, with `+0.40`
    local-resolution gain, `7` fewer cloud handoffs, and `3` fewer
    clarifications. `pi-smoke`, `pi-bundle`, and `target-report` now include
    this gate.
58. Add `import-transcript-replay` as the real-chat calibration path. It reads
    raw local chat JSONL, keeps only user turns, skips assistant/system rows,
    strips static expected answer/route/reason text, redacts email, phone, URL,
    and long-number tokens plus optional manual replacement rules, and emits
    the same replay schema for Basic NLP -> UOL -> ChatFrame debug scoring. It
    also accepts `--controls-json` as a separate safe lifecycle overlay for
    fields such as `run_reflection`, `schedule_refreshes`, `execute_jobs`,
    `network_available`, `execute_opportunities`, and aggregate thresholds.
    Static route/intent/reason/answer expectations are rejected in both raw logs
    and controls. Imported replay fixtures default to calibration-only baseline comparison;
    the authored 25-turn replay remains the strict architecture win gate unless
    imported metadata explicitly sets `required_baseline_win`. The companion
    `export-transcript-replay` command now converts the assistant's own SQLite
    event ledger into the same replay format with user utterances, labels,
    sessions, days, capture provenance, and safe controls only; stored answers,
    routes, reasons, and assistant responses are not exported as expectations,
    so replay must rediscover behavior through the kernel. Capture provenance is
    required evidence: scripted CLI, scripted API/UI smokes, interactive CLI,
    and served browser UI turns must stay distinguishable before a session can
    be packaged as trustworthy v0.1 evidence. `calibrate-event-ledger` wraps that export, replay, and aggregate
    threshold scoring in one command for real local browser/CLI sessions.
59. Add `calibrate-transcript-replay` as the aggregate real-chat calibration
    loop. It accepts repeated `--input` files or an `--input-dir`, imports each
    raw chat JSONL into redacted replay fixtures, runs the real kernel/store/
    debug replay path for each imported fixture, persists
    `imported_redacted_transcript` capture provenance into the replayed SQLite
    event ledger, and summarizes redaction counts, stripped static fields,
    route/intent counts, complexity, safety totals, provenance, debug-map
    coverage, and baseline required/strict-pass counts. It
    now emits explicit aggregate `checks` and `thresholds`, including total-turn
    floor, local-resolution floor, route/intent diversity floors, persisted
    synthesis trace floor, planner priority-signal sample floor, optional
    memory-digest quality, optional strict static-baseline win,
    redaction/static-field-drop requirements, debug-map coverage, and critical
    safety cleanliness. The portable bundle and target-report self-checks run a
    lightweight thresholded command against a fake raw transcript fixture so the
    copied appliance proves the import/replay calibration gate, not just loose
    metrics. Real user-derived blocker-clearing runs should add
    `--controls-json config/safe_lifecycle_controls.example.json`,
    `--min-synthesis-traces`, `--require-priority-signals`,
    `--require-redaction`, `--require-static-drop`,
    `--require-memory-digest-quality`, `--require-strict-baseline-win`, and
    `--out artifacts/local_assistant_os/user_transcript_calibration.json` when
    the transcript scope is large enough to support those claims, then pass that
    report to `v01-blocker-evidence --transcript-calibration-report-json`.
60. Add `inventory-failure-smoke` as the no-fabrication source-failure gate.
    Malformed Internet Archive JSON, source byte-budget exhaustion, and empty
    source fixtures run through the real scheduler/job/importer path. The gate
    requires observable job failure/completion ledgers, error visibility, zero
    story inventory growth, no local story synthesis, and future story asks
    that remain `cloud_handoff / missing_story_model`. `pi-smoke`,
    `target-report`, and `pi-bundle` now surface this proof so source failures
    cannot be hidden by phrase tables, static catalogs, or fallback templates.
61. Add `inventory-retry-smoke` as the transient-source recovery gate.
    Localhost Project Gutenberg and Internet Archive-shaped sources return a
    transient failure before valid metadata. The command requires both importers
    to retry, records fetch-attempt health for dashboards, proves no external
    network was used, and verifies future story routing changes from cloud to
    local only after imported inventory reload.
62. Expand recent-session bounded synthesis beyond event listing.
    Chat-native recent-session recall still cites `events.*` rows, but now
    derives capability transitions, open local gaps, action state, and boundary
    controls from the same event routes/reasons. Unit and CLI regressions prove
    story/weather gaps and pending action state are summarized without raw
    transcript stuffing or uncited claims.
63. Add `synthesis-variant-smoke` as a bounded local synthesis breadth gate.
    The command runs 10 real SQLite/kernel turns over story variants, health
    variants, urgent health, cached weather, meal choice, recent-session
    summary, and long-horizon digest recall. It requires cited synthesis,
    quality scores above the local floor, complete ledgers, clean synthesis
    safety flags, and primary `slot_role_relation` UOL/ChatFrame evidence with
    phrase hints excluded from primary routing. `pi-smoke`, `pi-bundle`,
    `target-report`, and `verify-bundle` now surface this proof.
64. Add `synthesis-stress-smoke` as the longer bounded local synthesis stress
    gate. The command runs 24 real SQLite/kernel turns across 3 sessions over
    assistant identity/status, story niches, health/urgent-health advice,
    cached weather, meal decisions, school safety, last-question recall,
    recent-session summary, long-horizon digest recall, and status-next asks.
    It requires 14 checks over local/cached-only routing, cited synthesis,
    quality/complexity scoring, complete ledgers, clean safety flags,
    multi-reason/multi-intent diversity, autobiographical event/digest use, and
    primary UOL/ChatFrame evidence with phrase hints excluded from primary
    routing. `pi-smoke`, `pi-bundle`, `target-report`, and `verify-bundle` now
    surface this proof.
65. Add `inventory-soak-matrix` as the cold-start multi-source inventory growth
    gate. The command runs both-source, Internet Archive-only, and
    Gutenberg-only profiles from fresh databases for at least 9 total cycles,
    requires both source families, clean quality/failure observability, zero
    failed import cycles, and proves future story asks route locally from
    imported inventory with primary UOL/ChatFrame evidence.

Next implementation order:

1. Keep scaling the autoimmune-failure suite beyond the new 26-turn
   `autoimmune-smoke` with longer-trace multi-user consent, richer stale-cache
   expiry modes, cross-session pending-action edge cases, and additional
   parent/child ownership cases from real transcripts.
2. Extend the new `inventory-soak-matrix --live` path across more query niches,
   longer cycle counts, and real source retry modes. The next candidate run
   should target at least 3 cold-start profiles, 10+ cycles per profile,
   30+ completed cycles total, both Gutenberg and Internet Archive source
   families, zero failed import cycles, verified per-run DB hashes, future
   `local_answer / local_story_inventory` story events in every run DB, and
   primary UOL/ChatFrame story evidence with secondary hints kept out of the
   primary route.
3. Expand bounded local story/advice/summarization variants beyond the compact
   10-turn gate and authored 24-turn / 3-session stress gate into user-derived
   traces while preserving citations to inventory/policy/memory inputs, keeping
   low-quality applied synthesis and warning counts at zero with evidence rather
   than hidden suppressions.
4. Grow the assistant eval beyond the deterministic 105-case, 34-turn,
   37-turn, authored 29-turn open trace, and authored 25-turn transcript replay
   suites by importing redacted user-derived transcript traces with
   `import-transcript-replay` and running thresholded
   `calibrate-transcript-replay` gates, then tune profile-specific failure
   rates, route thresholds, digest quality, and baseline deltas on those traces.
5. Keep the portable browser/CLI bundle green through `verify-bundle --json`,
   `target-report --reset --json`, `bootstrap-runtime --reset --json`,
   `autoimmune-smoke --reset --json`, `synthesis-variant-smoke --reset --json`,
   `synthesis-stress-smoke --reset --json`,
   `host-action-smoke --reset --json`,
   `host-app-probe --reset --json`,
   `launcher-smoke --reset --json`, `first-run-smoke --json`,
   `archive-smoke --reset --json`, `run-open-traces --reset --json`,
   `run-transcript-replay --reset --json`, thresholded
   `calibrate-transcript-replay` runs, the browser UI smoke, and scripted
   `chat --turn` sessions; calibrate digest-quality scores on user-derived
   stress traces for the richer long-horizon compactor, run
   `host-app-probe --config-json config/host_actions.json --require-configured --json`
   or `v01-acceptance --host-app-config-json config/host_actions.json
   --require-host-app-configured --json` with actual media/call apps on target
   platforms, and connect routine setup, household setup, and contact setup to
   real local integrations.

This is the clearest next big MVP because it can be useful in the worst case and
become the assistant OS substrate in the best case.
