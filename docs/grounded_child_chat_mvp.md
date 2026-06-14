# Grounded Child Chat Micro-MVP

This is a supporting micro-MVP for the MELM Local Assistant OS architecture, not
the authoritative product plan. The world is tiny, but the internal joints match
the scaled target:

```text
UOL parser
  -> semantic atlas
  -> typed ChatFrame
  -> BoundChildPlan
  -> state algebra
  -> Memory OS
  -> evidence admission / state projection
  -> hybrid SSM/attention SLM boundary
```

Implementation: `melm/appliance/grounded_child_chat.py`  
Tests: `tests/test_grounded_child_chat_mvp.py`  
Demo: `python scripts/run_grounded_child_chat_mvp.py`

## World

The bounded world is deliberately five-year-old-scale:

- people: `maya`, `leo`;
- objects: `red block`, `blue box`, `green basket`;
- initial state: the blue box is `closed`, the green basket is `open`;
- aliases: `block -> red block`, `box -> blue box`, `basket -> green basket`.

The MVP is not trying to parse broad English. It is trying to prove that a
small local chat appliance can be built around the same contracts the larger
system would use.

## Components

`UOLSentence` is the sentence object:

- purpose: `inform`, `question`, or `command`;
- subject/action/object/source/target slots;
- parse score, parse complexity, and parse notes;
- raw source text preserved for evidence and rejection packets.

`ChildWorldAtlas` is the micro SemanticAtlas:

- resolves aliases into canonical nouns;
- rejects unknown actors, objects, and actions;
- owns action-frame templates such as `open`, `close`, `put`, and `move`;
- supplies base action complexity and required slots used by the scorer;
- marks objects by role, so containers can be opened/closed and targets must be
  containers.

`ChatFrame` and `BoundChildPlan` are the db-claw-shaped middle:

- the model never receives raw language directly;
- accepted routes must bind to an executable plan;
- unsupported routes return `RejectionPacket`.

`ChildStateAlgebra` is not assumed from `nameless_vector`:

- it applies `StatePatch(required, remove, add)`;
- `closed -> open` succeeds only when `closed` is explicitly removed;
- adding `open` without removing `closed` remains a contradiction.

`ChildMemoryOS` is the micro MELM Memory OS:

- stores committed events in `EventMemory`;
- owns current object state;
- projects compact state for the SSM path;
- admits evidence packets for the attention path;
- supports an `EvidencePolicy` so action evidence can be top-k budgeted while
  still carrying the total matching evidence count;
- rejects impossible memory writes, such as putting something into a closed box.

`MicroHybridSlm` is the replaceable model boundary:

- `MicroSsmState` receives compact projected state;
- `MicroAttentionSlice` receives admitted evidence IDs and packed context;
- generation only verbalizes a bound plan plus admitted state/evidence;
- rejected paths do not call the renderer.

## Context Budget Probe

`run_child_context_budget_probe()` measures the route/budget sub-gate:
whether a longer story can be answered from compact SSM state plus a bounded
attention slice instead of from the whole transcript.

The probe compares default action evidence admission with
`EvidencePolicy(action_top_k=1)`.

Representative results:

```text
location, 32 moves:
  raw transcript: 1441 chars
  compact payload: 318 chars
  compression: 4.53x

positive evidence check, 32 moves:
  raw transcript: 1465 chars
  unbudgeted payload: 1206 chars, 16 attended events, 1.21x
  budgeted payload: 322 chars, 1 attended event, 16 matching events, 4.55x

positive evidence check, 128 moves:
  raw transcript: 5641 chars
  unbudgeted payload: 4053 chars, 64 attended events, 1.39x
  budgeted payload: 323 chars, 1 attended event, 64 matching events, 17.46x
```

This is sub-gate evidence for the broader Local Assistant OS plan. It proves why
budgeted evidence/state matters, but it is not the full product direction. The
authoritative product plan is `docs/local_assistant_os_mvp_plan_v2.md`.

## Realistic Assistant Direction Probe

`compare_assistant_mvp_directions()` tests the broader on-device assistant shape
against eight realistic asks:

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

Current comparison:

```text
strategy                    local/device  cloud  fetch  clarify  privacy  memory
memory_centric_local_triage 8/8           0      0      0        0        7
thin_tools_plus_cloud       4/8           3      0      1        3        3
cloud_first_assistant       2/8           5      1      0        3        2
secondary_lexical_baseline  1/8           6      0      1        0        0
```

The vocabulary-only baseline is deliberately limited to secondary lexical hints.
It must not borrow the UOL/ChatFrame classifier, otherwise this comparison stops
measuring the secondary-lexical baseline honestly.

This is the clearer practical win for the routing layer: the MVP should
prioritize local memory, cached facts, policy, device actions, and cloud handoff
decisions before simply adding more child-room vocabulary. The OS plan adds the
missing structural foundation: membrane policy and homeostatic state.

## Happy Path

```text
Maya opened the blue box.
Maya put the red block in the blue box.
Leo moved the red block to the green basket.
Where is the red block?
```

Expected answer:

```text
The red block is in the green basket.
```

The answer uses:

- `BoundChildPlan(schema="melm.bound_child_plan.v1")`;
- SSM state containing `red block:location=green basket`;
- attention evidence containing `child_e3`, the validated move event.

## Fail-Closed Cases

- `Maya put the red block in the blue box.` fails before the box is opened.
- `Open the blue box.` fails once the box is already open.
- `Maya put the silver train in the blue box.` fails at the semantic atlas.
- `Did Maya move the red block to the green basket?` abstains because only Leo
  has matching evidence.

Rejections produce typed `RejectionPacket`s and do not call the SLM boundary.
Abstentions may call the SLM boundary, but with an empty attention slice and a
`not_enough_evidence` response intent.

## Generated Capability Probe

The demo also runs `run_child_capability_probe()`. It generates cases from the
atlas inventory rather than from hand-picked transcript prompts:

- state actions over every known person/object pair;
- put/move combinations over the bounded object vocabulary;
- location and evidence-check questions after a fixed setup story;
- unsupported sentence shapes such as `painted`, `hide`, and `beside`.

Current measured envelope:

```text
cases=50
accepted=22
answered=4
abstained=11
rejected=13
average_parse_score=0.863
average_complexity=1.234
max_complexity=1.93
```

Current reasons:

```text
evidence_check: 3
memory_read: 1
memory_write: 16
no_location_observation: 2
no_matching_action_evidence: 9
state_transition: 6
state_transition_invalid: 2
unsupported_object_action: 4
unsupported_target: 4
unsupported_uol_shape: 3
```

This is the micro-MVP's honest boundary: within the atlas verbs/nouns it can
bind, score, execute, answer, or abstain through the same architecture. Outside
those UOL shapes, it rejects instead of pretending the renderer knows what to do.

## Verification

Run:

```powershell
python scripts\run_grounded_child_chat_mvp.py
python -m unittest tests.test_grounded_child_chat_mvp
python -m unittest discover -s tests
```

The MVP tests assert:

- UOL parsing keeps subject/action/object/target slots;
- state algebra replaces old state before checking conflicts;
- every happy-path answer includes the scaled-stage trace;
- Memory OS, not the renderer, rejects closed-target writes;
- semantic atlas rejects unknown nouns before memory/model calls;
- scored parse candidates rank known grounded slots above unknown slots;
- generated capability probing measures accepted, answered, abstained, and
  rejected cases across the MVP vocabulary;
- context-budget probing shows the SSM/attention boundary improves with story
  length and that top-k evidence admission fixes repeated positive evidence
  checks;
- the model boundary receives distinct SSM and attention inputs;
- components are swappable, not hidden globals.

## What This Proves

This proves a minimal architecture-homomorphic chat appliance can run in a
bounded child-level world while preserving fail-closed behavior and evidence
contracts.

## What This Still Does Not Prove

- It does not benchmark Raspberry-class latency.
- It does not include a real neural SSM/attention checkpoint.
- It does not prove broad grammar or open-vocabulary grounding.
- It does not prove the atlas/frame inventory can scale without better authoring
  and extraction tools.

See `docs/local_assistant_os_mvp_plan_v2.md` for the authoritative product plan.
See `docs/grounded_child_chat_mvp_direction.md` only as retained supporting
evidence for the route/budget sub-gates.
