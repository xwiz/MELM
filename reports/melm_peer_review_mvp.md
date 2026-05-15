# MELM Guard + Memory OS Peer-Review MVP

No API required: `True`
Scenario repeats per split: `20`
Bootstrap samples: `1000`
Peer-review gate passed: `True`
Recommendation: `advance_to_external_human_labeled_support_logs`

## Development Split

- Seed: `17`
- Events/facts: `1183` / `1183`
- Guard cases: `240`
- Memory cases: `396`
- Guard gate: `True`
- Guard MELM accuracy: `100.00%`
- Guard schema-only accuracy: `25.00%`
- Guard false-allow reduction vs schema: `100.00%`
- Guard traceability: `100.00%`
- Memory gate: `True`
- Vector RAG accuracy: `81.06%`
- Temporal/entity RAG accuracy: `81.06%`
- Memory OS accuracy: `100.00%`
- Memory OS gain vs vector: `18.94%`
- Negative abstention: `100.00%`

| Estimate | Mean | 95% CI |
|---|---:|---:|
| Guard MELM accuracy | 100.00% | [100.00%, 100.00%] |
| Guard MELM - schema accuracy | 75.00% | [69.58%, 80.42%] |
| Guard MELM - prompt accuracy | 75.00% | [69.17%, 80.00%] |
| Memory OS accuracy | 100.00% | [100.00%, 100.00%] |
| Memory OS - vector accuracy | 18.94% | [15.15%, 22.98%] |
| Memory OS - temporal/entity accuracy | 18.94% | [15.40%, 22.47%] |

### Evidence Dropout Robustness

| Dropped | Guard accuracy | Guard false-allow | Valid allow | Memory OS accuracy | OS-vector gain | 95% CI |
|---:|---:|---:|---:|---:|---:|---:|
| 5% (59) | 95.00% | 1.00% | 85.00% | 95.71% | 18.94% | [15.15%, 22.73%] |
| 10% (118) | 88.75% | 3.00% | 75.00% | 90.66% | 18.94% | [15.15%, 22.73%] |
| 20% (236) | 79.17% | 7.00% | 57.50% | 82.83% | 18.94% | [15.15%, 22.73%] |

## Heldout Seed Split

- Seed: `91`
- Events/facts: `1183` / `1183`
- Guard cases: `240`
- Memory cases: `396`
- Guard gate: `True`
- Guard MELM accuracy: `100.00%`
- Guard schema-only accuracy: `25.00%`
- Guard false-allow reduction vs schema: `100.00%`
- Guard traceability: `100.00%`
- Memory gate: `True`
- Vector RAG accuracy: `81.06%`
- Temporal/entity RAG accuracy: `81.06%`
- Memory OS accuracy: `100.00%`
- Memory OS gain vs vector: `18.94%`
- Negative abstention: `100.00%`

| Estimate | Mean | 95% CI |
|---|---:|---:|
| Guard MELM accuracy | 100.00% | [100.00%, 100.00%] |
| Guard MELM - schema accuracy | 75.00% | [69.58%, 80.42%] |
| Guard MELM - prompt accuracy | 75.00% | [69.58%, 80.00%] |
| Memory OS accuracy | 100.00% | [100.00%, 100.00%] |
| Memory OS - vector accuracy | 18.94% | [15.40%, 22.73%] |
| Memory OS - temporal/entity accuracy | 18.94% | [14.90%, 22.47%] |

### Evidence Dropout Robustness

| Dropped | Guard accuracy | Guard false-allow | Valid allow | Memory OS accuracy | OS-vector gain | 95% CI |
|---:|---:|---:|---:|---:|---:|---:|
| 5% (59) | 94.17% | 1.00% | 90.00% | 96.97% | 18.94% | [15.15%, 22.73%] |
| 10% (118) | 91.25% | 4.00% | 82.50% | 91.67% | 18.94% | [15.15%, 22.73%] |
| 20% (236) | 82.50% | 4.00% | 65.00% | 81.57% | 18.94% | [15.15%, 22.73%] |

## Limitations

- Synthetic support/refund cases are deterministic and policy-derived.
- No live LLM baseline is required; prompt-only is simulated by fixed proposals.
- Temporal/entity RAG and Memory OS share the same raw event annotations.
- The next peer-review step must add external human-labeled logs or independently authored fixtures.

Interpretation: this is now a stronger internal benchmark, not a publication by itself. It adds balanced categories, held-out seed replication, paired bootstrap intervals, and explicit limitations. The next publishable step is independent human-labeled support logs or fixtures authored by someone who did not implement the system.
