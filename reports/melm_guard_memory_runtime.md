# MELM Guard + Memory OS Runtime Benchmark

Source: `support_refund_fixture`
Runtime gate passed: `True`
Recommendation: `advance_to_external_review_dataset`

## Guard Signal

- Gate passed: `True`
- False-allow reduction vs schema: `100.00%`
- Valid-action allow rate: `100.00%`
- Traceability: `100.00%`

## Memory OS Signal

- Gate passed: `True`
- Vector RAG accuracy: `75.00%`
- Temporal/entity RAG accuracy: `75.00%`
- Memory OS accuracy: `100.00%`
- Memory OS gain vs vector: `25.00%`
- Positive recall: `100.00%`
- Negative abstention: `100.00%`

## Resource Probe

Implementation: `pure-Python indexed SupportMemoryOS over EventMemory`
Queries per size: `20`

| Events | Build s | Vector RAG ms | Temporal/entity ms | State lookup ms |
|---:|---:|---:|---:|---:|
| 100 | 0.0011 | 0.589 | 0.635 | 0.001 |
| 1000 | 0.0077 | 6.235 | 6.053 | 0.001 |
| 5000 | 0.0380 | 29.199 | 31.151 | 0.001 |

This MVP reports latency honestly. The support indexes reduce state lookups, but vector and structured retrieval still keep a Python EventMemory path; a production CPU win needs a tighter indexed graph or native sidecar.
