# Support/Refunds Authored Dataset Protocol

This protocol exists to keep the MELM Guard + Memory OS dataset falsifiable rather
than merely decorative. The first corpus is an internally authored seed batch
(`benchmarks/support_refunds_authored.jsonl`) whose main purpose is to validate
schema, coverage, scoring, and reporting. It is not yet an external human-labeled
benchmark.

## Unit Of Annotation

Each scenario should contain:

- transcript turns: natural-language customer, agent, system, billing, risk, or
  manager notes;
- fact-bearing events: one explicit event per atomic fact used by Guard or Memory
  OS;
- guard cases: fixed action proposals with expected `allow`, `deny`, `warn`, or
  `abstain`;
- memory cases: fixed queries for event recall, current-state resolution,
  contradiction handling, and negative abstention.

## Required Coverage

Each authored batch must include:

- low-value valid refunds;
- high-value valid refunds with fresh manager approval;
- high-value refunds missing approval;
- stale approval cases;
- identity failure;
- fraud flag;
- not-yet-delivered order;
- duplicate refund;
- outside return window;
- unknown order;
- malformed action;
- stale-state traps where an earlier fact is superseded.

Memory cases must include positive and negative examples for:

- latest order status;
- latest refund status;
- manager approval recall;
- fraud/risk recall;
- policy recall;
- contradiction resolution;
- unknown-order abstention.

## Labeling Rules

Guard labels are policy labels, not model-preference labels.

- `allow`: all known required facts support the action and no blocking rule fires.
- `deny`: a hard block is present or a required hard fact is missing.
- `warn`: the action may proceed only with a visible warning, such as stale
  approval evidence.
- `abstain`: the action needs missing external approval rather than a direct
  denial.

Memory labels use the latest fact at or before the query time unless a case asks
for a specific historical event. Unknown state queries should expect `null`, and
systems should be rewarded for abstaining rather than retrieving a nearby order.

## Publication Upgrade Path

To become publication-grade, the next batch should be authored by someone who did
not implement MELM Guard or Memory OS. Preferred protocol:

- preregister the category counts before authoring;
- hide rule-engine failure cases from annotators;
- have two annotators label at least 20% overlap;
- adjudicate disagreements in a separate JSONL record;
- freeze the dataset hash before running MELM, vector RAG, and temporal/entity
  RAG scores;
- report both the internal seed batch and the external blind batch separately.
