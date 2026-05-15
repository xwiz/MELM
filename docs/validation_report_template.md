# MELM Validation Report Template

## Summary

- Report date:
- Experiment window:
- Primary conclusion:
- Recommendation:

## Tokenizer Results

| Arm | Data | Model Size | Loss | BabyLM Avg | Compression | Boundary F1 | Notes |
|---|---|---:|---:|---:|---:|---:|---|
| BPE | | | | | | | |
| Unigram | | | | | | | |
| Morphology-aware | | | | | | | |
| Byte/patch | | | | | | | |

Decision:

- [ ] Use morphology-aware tokenization.
- [ ] Use morphology as auxiliary supervision only.
- [ ] Drop morphology from the critical path.

## Event Memory Results

| Arm | Recall | Temporal Neighbor Recall | Causal QA | Context Budget | Latency | Notes |
|---|---:|---:|---:|---:|---:|---|
| RAG | | | | | | |
| RAG + temporal | | | | | | |
| Event memory | | | | | | |

Decision:

- [ ] Event memory passes the 15% improvement gate.
- [ ] Simplify to RAG plus temporal/entity metadata.
- [ ] Redesign the event schema.

## Model Results

| Arm | Params | Tokenizer | Memory | BabyLM Avg | Episodic Score | Notes |
|---|---:|---|---|---:|---:|---|
| BPE baseline | | | | | | |
| Best tokenizer | | | | | | |
| Best tokenizer + memory | | | | | | |

## Failure Analysis

- Main false positives:
- Main false negatives:
- Data quality issues:
- Contamination risks:
- Compute/runtime issues:

## Release Decision

- [ ] Positive demo release.
- [ ] Negative-results release.
- [ ] Re-run required before release.

Rationale:

