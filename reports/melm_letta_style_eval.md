# MELM Letta-Style Evaluation

Target: `melm_appliance_local`
Dataset: `artifacts\letta_eval\locomo_letta_dataset.jsonl`
Memory: `artifacts\letta_eval\locomo_memory.jsonl`
Samples: `250`
Contains accuracy: `20.40%`
Answer support rate: `23.60%`
Mean answer-token recall: `32.39%`
Mean citation evidence recall: `74.08%`
Mean retrieval evidence recall: `84.39%`

## By Category

| Category | Samples | Contains accuracy | Answer support | Token recall | Citation evidence recall | Retrieval evidence recall |
|---|---:|---:|---:|---:|---:|---:|
| adversarial | 2 | 0.00% | 0.00% | 0.00% | 100.00% | 100.00% |
| multi_hop | 50 | 6.00% | 8.00% | 21.27% | 29.73% | 50.63% |
| open_domain | 114 | 13.16% | 15.79% | 22.71% | 87.72% | 94.74% |
| single_hop | 70 | 47.14% | 52.86% | 62.55% | 87.14% | 97.14% |
| temporal | 14 | 0.00% | 0.00% | 4.83% | 52.38% | 54.76% |

This local run consumes the same JSONL dataset exported for Letta Evals. It is not an official Letta comparison; it gives MELM's score on the shared eval pack before running a real Letta target.

Interpretation note: contains accuracy grades the final extractive answer. Evidence recall grades whether the memory layer surfaced the gold sessions. A large gap between the two means the retrieval appliance is ahead of the current no-LLM answer composer.
