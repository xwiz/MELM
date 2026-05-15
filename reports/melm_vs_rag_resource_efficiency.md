# MELM vs RAG Resource Efficiency

Implementation: `current pure-Python brute-force EventMemory`
Queries per size: `20`

Current MELM event memory is more evidence/context efficient than RAG on the validation tasks, but this Python implementation is not more CPU efficient than RAG because both retrievers scan the full event list.

| Events | Build s | RAG ms/query | Event-memory ms/query | Overhead |
|---:|---:|---:|---:|---:|
| 12 | 0.0002 | 0.087 | 0.089 | 1.02x |
| 100 | 0.0011 | 0.678 | 0.646 | 0.95x |
| 1000 | 0.0124 | 7.371 | 6.658 | 0.90x |
| 5000 | 0.0658 | 41.037 | 48.614 | 1.19x |

Interpretation: the current win is in answerability, context budget, and false-answer control. A true embedded resource win requires an indexed event/state graph or Rust/C sidecar rather than brute-force Python scans.
