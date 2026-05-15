# MELM SLM Appliance Validation Status

This document tracks the practical evidence needed before MELM can be credibly
packaged as a local SLM appliance.

## Implemented Evidence

The repo now has two reproducible validation layers:

- support/refunds Guard + Memory OS: deterministic action gates, event/state
  memory, authored JSONL seed data, and no-API reports;
- public LoCoMo session-evidence retrieval: a dependency-light public benchmark
  adapter that compares local architecture-family implementations.

Run the public benchmark with:

```powershell
python scripts\run_public_memory_benchmark.py --download
```

Current LoCoMo result in `reports/melm_public_memory_locomo.md`:

- vector RAG recall@5: 50.65%;
- Mem0-style additive memory proxy recall@5: 86.00%;
- MemGPT-style tiered memory proxy recall@5: 87.74%;
- Zep-style temporal graph proxy recall@5: 87.21%;
- MELM Memory OS recall@5: 88.23%.

The important candid detail: MELM has a positive point estimate against all
local proxies, but the paired bootstrap interval against the MemGPT-style proxy
crosses zero. The stronger claim is statistically clear against vector RAG,
Mem0-style additive memory, and the Zep-style temporal proxy; it is not yet
statistically settled against a strong tiered-summary memory.

The MemGPT-specific stress test is bounded context support. With a 1,200-token
context budget, MELM now uses compact observations/summaries/event summaries
plus question-guided raw snippets. Current answer-support results:

- Mem0-style additive proxy: 55.41%;
- MemGPT-style tiered plus raw paging proxy: 56.78%;
- Zep-style observation/temporal proxy: 54.50%;
- MELM Memory OS adaptive context: 61.41%.

This context-budget result is the stronger appliance-relevant signal: the paired
bootstrap interval for MELM vs MemGPT-style answer support is positive. The
caveat is still real: these are local architecture-family proxies, not official
MemGPT/Letta service runs.

## Scope Of Comparison

These are not official vendor scores. The comparison isolates architecture ideas:

- Mem0-style: additive long-term memory retrieval over raw session memories;
- MemGPT-style: tiered raw plus session-summary memory;
- Zep-style: extracted observations plus temporal neighbor expansion;
- MELM Memory OS: raw turns plus observations, session summaries, event
  summaries, entity boosts, and temporal query routing.

The public benchmark is LoCoMo:

- https://github.com/snap-research/locomo
- https://snap-research.github.io/locomo/

Related systems to compare against with official adapters later:

- Mem0 benchmark repo: https://github.com/mem0ai/memory-benchmarks
- MemGPT paper: https://arxiv.org/abs/2310.08560
- Zep/Graphiti paper: https://arxiv.org/abs/2501.13956

## Next Appliance Milestones

The next implementation milestones should be:

- package `melm.guard`, `melm.memory`, and the public-memory adapter behind a
  local appliance CLI/API;
- add an official adapter path for Mem0, Graphiti/Zep, and Letta/MemGPT where
  those packages are installed, while keeping local proxies for no-API CI;
- connect a small local model for answer generation after retrieval, with
  retrieval evidence IDs preserved in the response;
- add latency and memory-footprint reports for 1k, 10k, and 100k memories;
- freeze an external blind support/refunds batch before making investability or
  publication claims.
