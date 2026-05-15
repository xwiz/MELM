# MELM vs Letta Comparison Plan

Letta, formerly MemGPT, is the right external bar for MELM Memory OS. It is a
stateful-agent platform, not just a retriever. Its core pieces include persistent
agents, memory blocks, archival memory, message history, compaction, tools, runs,
conversations, SDKs, server deployment, and Letta Evals.

## Current MELM Status

What MELM has now:

- deterministic event/state memory;
- support/refunds Guard with traceable rules and facts;
- authored support/refunds JSONL validation;
- public LoCoMo retrieval and bounded-context benchmark;
- local appliance JSONL memory store, retrieval, context packing, and extractive
  answers;
- Letta Evals-style export pack for LoCoMo.

What MELM does not yet have:

- official Letta agent runs;
- agent-managed editable memory blocks;
- database-backed runs/messages/conversations;
- semantic embeddings in the appliance runtime;
- real local SLM answer generation, which is now the main appliance bottleneck;
- tool-use loops and self-directed memory update trajectories;
- Letta Evals state-inspection graders for memory updates.

## Current Evidence

The public LoCoMo benchmark in `reports/melm_public_memory_locomo.md` shows:

- MELM Memory OS recall@5: 88.23%;
- Letta/MemGPT-style tiered proxy recall@5: 87.74%;
- retrieval confidence interval vs the Letta/MemGPT-style proxy crosses zero;
- bounded-context answer support is stronger for MELM: 61.41% vs 56.78%.

Interpretation: MELM has a stronger appliance-relevant signal under bounded
context, but the retrieval-only claim is not statistically settled against a
strong Letta/MemGPT-style tiered memory proxy.

The local Letta-style pack in `reports/melm_letta_style_eval.md` is more
candid:

- MELM local exact contains accuracy: 20.40%;
- MELM local answer support rate: 23.60%;
- mean answer-token recall: 32.39%;
- mean citation evidence recall: 74.08%;
- mean retrieval evidence recall: 84.39%.

Interpretation: the memory layer frequently surfaces the correct gold evidence,
but the current deterministic extractive answer composer is weak. This is not
yet a Letta-beating agent. It is a useful appliance memory substrate that needs
a real local SLM or a stronger structured answer synthesis layer before we can
claim end-to-end superiority.

The current appliance answerer now includes a small structured synthesis layer:
observation facts are parsed into candidate facts, temporal phrases are resolved
against memory timestamps, and simple list/title/status/direct-object answers are
generated deterministically. This improved the local Letta-style exact score, but
multi-hop and open-domain answer composition remain weak compared with what a
real agentic model should do.

## Letta Eval Pack

Export the shared eval pack:

```powershell
python scripts\export_letta_eval_pack.py --download --max-questions 250
```

Run MELM locally on the same pack:

```powershell
python scripts\run_melm_letta_style_eval.py --dataset artifacts\letta_eval\locomo_letta_dataset.jsonl --memory artifacts\letta_eval\locomo_memory.jsonl
```

Run Letta later after configuring a real Letta agent file or server:

```powershell
pip install letta-evals
letta-evals validate artifacts\letta_eval\locomo_letta_suite.yaml
letta-evals run artifacts\letta_eval\locomo_letta_suite.yaml
```

The generated suite uses a `contains` grader over `last_assistant`. That is a
baseline, not the final scientific comparison. The fairer final suite should add:

- answer contains / answer F1;
- citation coverage against `metadata.evidence_session_ids`;
- memory-state inspection for whether facts were inserted into memory;
- tool-call graders for archival search and memory update operations;
- latency and token-budget metrics.

## What Would Prove Superiority

MELM would have a credible superiority claim over Letta only if it wins on at
least one constrained axis where Letta is given a fair setup:

- same public dataset;
- same or smaller local model;
- same context budget;
- same memory ingestion budget;
- same grading script;
- official Letta agent target, not only a proxy;
- repeated runs or deterministic target where possible.

The most plausible MELM wedge is not "more general than Letta." It is:

> A smaller local appliance that performs event/state projection and rule-aware
> action gating with stronger traceability and lower operational complexity.

The next implementation priority is therefore not another retrieval-only score.
It is a bounded local answer layer: either a small local model behind the
existing `MelmAppliance.answer` interface or a structured synthesis module that
turns cited event/state records into concise, graded answers. Without that,
Letta remains ahead as a complete stateful-agent product even if MELM has
interesting memory/retrieval mechanics.

## Sources

- Letta repository: https://github.com/letta-ai/letta
- Letta stateful agents: https://docs.letta.com/guides/core-concepts/stateful-agents
- Letta memory blocks: https://docs.letta.com/guides/core-concepts/memory/memory-blocks/
- Letta archival memory: https://docs.letta.com/guides/core-concepts/memory/archival-memory/
- Letta Evals datasets: https://docs.letta.com/guides/evals/concepts/datasets/
- Letta suite YAML: https://docs.letta.com/evals/configuration/suite-yaml-reference
