# Grounded Child Chat MVP Direction

Date: 2026-06-06

## Status

This memo is retained as supporting evidence. The authoritative product plan is
`docs/local_assistant_os_mvp_plan.md`.

The conclusion here still holds, but it is incomplete on its own: memory-centric
local triage and budgeted evidence are mechanisms inside the broader Local
Assistant OS. The refined foundation is membrane/homeostasis:

```text
membrane policy
  -> homeostatic state
  -> autobiographical memory
  -> user/self model
  -> opportunity planner
  -> inventories/tools/actions/cloud
```

The micro world should stay small for now. The next proof should show that a
Raspberry-class chat loop can answer from:

```text
compact Memory OS state -> SSM path
bounded admitted evidence -> attention path
```

instead of replaying the full transcript through the model or sending routine
assistant tasks to a cloud LLM.

For a realistic on-device bot, most user asks are not pure language generation.
They are routing problems over memory, tools, policy, actions, and cloud
fallback.

## Evidence From Current MVP

Generated capability probe:

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

The largest non-green bucket is not unknown grammar. It is evidence behavior:

```text
no_matching_action_evidence: 9
no_location_observation: 2
unsupported_object_action: 4
unsupported_target: 4
unsupported_uol_shape: 3
```

The current location path already compresses long context well:

```text
Where is the red block?, 32 moves:
  raw transcript=1441 chars
  SSM/attention payload=318 chars
  compression=4.53x

Where is the red block?, 128 moves:
  raw transcript=5617 chars
  SSM/attention payload=319 chars
  compression=17.61x
```

Positive evidence checks expose the bottleneck. Without a budget, attention
admits every matching event:

```text
Did Leo move the red block to the green basket?, 32 moves:
  raw transcript=1465 chars
  unbudgeted payload=1206 chars
  attended events=16
  compression=1.21x
```

With `EvidencePolicy(action_top_k=1)`, the model receives one evidence span plus
the total matching count:

```text
Did Leo move the red block to the green basket?, 32 moves:
  budgeted payload=322 chars
  attended events=1
  matching events=16
  compression=4.55x

Did Leo move the red block to the green basket?, 128 moves:
  budgeted payload=323 chars
  attended events=1
  matching events=64
  compression=17.46x
```

That is the clearest attainable win at this level.

## Broader Assistant Comparison

The realistic assistant probe uses these eight asks:

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

Memory-centric local triage resolves them through local inventory, cached tool
data, safety policy, device actions, personal memory, and contact memory:

```text
story -> local_answer from local story inventory
weather -> cached_tool from weekly forecast cache
naked school -> local_answer from common-sense safety policy
song -> device_action from media library
health -> local_answer from bounded health policy + user goals
about myself -> local_answer from personal memory
eat today -> local_answer from food inventory + weather cache
talk to someone -> device_action from trusted contacts
```

Comparison against other attainable MVP directions on the same eight asks:

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

This is the clearer win: adding more grammar or more vocabulary does not make
the assistant useful unless it can bind user meaning to local memory, cached
facts, policies, device actions, and cloud handoff rules.

The same sentence changes route based on local memory state:

```text
weather cache hit -> answer locally
weather cache miss -> fetch weather

story model present -> fabricate locally from inventory
story model missing -> cloud handoff

personal facts present -> answer "something about myself"
personal facts missing -> clarify or say memory is empty

trusted contact present -> prepare call action
trusted contact missing -> clarify who to contact
```

## Ranked Opportunities

1. Memory-centric local triage with budgeted evidence/state

This yields the largest proof value and the largest practical assistant win. It
shows the bot can decide local answer vs cached tool vs device action vs fetch
vs cloud handoff.

Budgeted evidence/state is still essential because personal memory and tool
history will grow. The assistant needs compact state plus small admitted
evidence, not full transcript replay.

Attainable now:

- keep the current tiny world;
- add realistic assistant route frames;
- distinguish local answer, cached tool, device action, external fetch, cloud
  handoff, and clarify;
- use `EvidencePolicy` for action evidence top-k;
- expose matching evidence counts in the SLM boundary;
- probe compression over repeated stories and realistic assistant tasks.

Expected outcome:

- eight realistic assistant asks resolve without cloud when local memory/cache
  exists;
- positive evidence checks move from near-transcript-sized payloads to flat
  payloads;
- the micro-MVP can make a concrete Raspberry-class context argument without a
  neural checkpoint.

2. Evidence verdicts for no/unknown/contradicted

This is the next chat-quality win. The current system abstains on absence of
matching evidence. A careful `EvidenceVerdict` layer could distinguish:

```text
positive: evidence matches
negative: closed-world evidence excludes the claim
unknown: trace is incomplete or open-world
contradicted: state/event history conflicts
unsafe: query cannot be answered from admitted evidence
```

Potential outcome in the current 50-case probe:

```text
answered could rise from 4 to about 13
abstained could fall from 11 to about 2
```

Risk: this must be gated by an explicit closed-world policy. Otherwise absence
of evidence can become a false denial.

3. ActionSpec-generated parser transitions

The parser has improved scoring, but it still has separate action-specific parse
functions. The db-claw lesson says not to grow phrase tables. Runtime code should
hold generic transitions, while meaning comes from the atlas.

Next shape:

```text
ActionSpec
  -> verbs
  -> required slots
  -> allowed prepositions
  -> route
  -> scoring
  -> generated parser cases
```

This is lower immediate user-visible gain than budgeted evidence, but it is the
right scalability path.

4. Vocabulary expansion

Adding more child-room nouns and verbs would improve demo breadth but prove less.
It should come after the memory/tool/policy/action loop is stronger.

Good candidates later:

- `give`, `take`, `bring`;
- ownership and possession;
- `on`, `under`, `beside`;
- simple emotional/mental state words.

Constraint: vocabulary expansion should count only when it exercises membrane,
state, memory, inventory, or action gates. Otherwise it is breadth without
structural proof.

## Borrowed db-claw Lessons

Use these patterns:

- typed frames before execution;
- deterministic legality checks;
- compact rejected/handoff packets;
- generated probes over source vocabulary;
- explicit fail-closed reasons.

Avoid these traps:

- static phrase fixes;
- hand-picked green transcripts;
- accepting partial plans;
- letting the renderer infer missing evidence.

## Retained Sub-Gates

This older gate is superseded by `MELM Local Assistant OS v0.1` in
`docs/local_assistant_os_mvp_plan.md`. The retained route/budget gates remain
useful sub-gates:

```text
Local Assistant route/budget sub-gate
```

Gate requirements:

```text
1. Existing 50-case capability probe still passes.
2. The eight realistic assistant asks are routed through local answer, cached
   tool, device action, fetch, cloud handoff, or clarify.
3. Memory-centric triage resolves 8/8 with the default local profile, with 0
   cloud handoffs and 0 privacy exposures.
4. Thin-tools-plus-cloud, cloud-first, and vocabulary-only baselines remain
   worse on the same asks, with vocabulary-only restricted to secondary lexical
   hints rather than UOL/ChatFrame composition.
5. Context-budget probe covers 32 and 128 repeated moves.
6. Location queries show at least 4x compression by 32 moves.
7. Positive evidence checks show less than 2x compression unbudgeted and at
   least 4x compression with top-k evidence at 32 moves.
8. Budgeted positive evidence keeps one attended event and the full matching
   event count.
9. Negative evidence checks remain abstentions until a closed-world verdict
   policy is added.
```

These gates are necessary but not sufficient. The OS plan adds membrane
decisions, homeostatic state, lifecycle probes, and autoimmune-failure checks so
the project does not drift into a merely capable router.

The authoritative product plan is `docs/local_assistant_os_mvp_plan.md`.
