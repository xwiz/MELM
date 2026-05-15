# MELM Authored Support/Refunds Benchmark

Dataset: `benchmarks\support_refunds_authored.jsonl`
Schema validation passed: `True`
Authored batch gate passed: `True`
Publication-grade ready: `False`
Recommendation: `use_this_as_seed_and_author_external_blind_batch`

- Turns/events/facts: `22` / `56` / `56`
- Guard cases: `13`
- Memory cases: `40`
- Bootstrap samples: `1000`

## Guard

- Gate passed: `True`
- MELM accuracy: `100.00%`
- Schema-only accuracy: `30.77%`
- Prompt-only accuracy: `30.77%`
- MELM false-allow rate: `0.00%`
- Schema-only false-allow rate: `90.00%`
- False-allow reduction vs schema: `100.00%`
- Valid-action allow rate: `100.00%`
- Traceability: `100.00%`

| Estimate | Mean | 95% CI |
|---|---:|---:|
| Guard MELM accuracy | 100.00% | [100.00%, 100.00%] |
| Guard MELM - schema accuracy | 69.23% | [46.15%, 92.31%] |
| Guard MELM - prompt accuracy | 69.23% | [46.15%, 92.31%] |

## Memory OS

- Gate passed: `True`
- Vector RAG accuracy: `62.50%`
- Temporal/entity RAG accuracy: `72.50%`
- Memory OS accuracy: `100.00%`
- Memory OS gain vs vector: `37.50%`
- Positive recall: `100.00%`
- Negative abstention: `100.00%`

| Estimate | Mean | 95% CI |
|---|---:|---:|
| Memory OS accuracy | 100.00% | [100.00%, 100.00%] |
| Memory OS - vector accuracy | 37.50% | [22.50%, 52.50%] |
| Memory OS - temporal/entity accuracy | 27.50% | [15.00%, 42.50%] |

## Dataset Checks

- No schema or coverage validation errors.

## Candid Interpretation

This is a stronger authored seed batch, not yet a publishable external benchmark. It validates that the MELM pipeline can ingest non-generator JSONL, preserve evidence provenance, score Guard and Memory OS together, and report uncertainty. Publication-grade evidence still requires an independently authored blind batch or real support logs with human labels.
